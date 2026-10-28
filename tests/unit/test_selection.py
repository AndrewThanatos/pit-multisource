import dataclasses
import gzip
import hashlib
import http.client
import io
import json
import os
import urllib.error
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pit_multisource import cli
from pit_multisource.acquire.gdelt import (
    GdeltRunAborted,
    fetch_slot,
    run_slots,
    slot_url,
)
from pit_multisource.build.selection import ledger_line, select_lines, shard_bytes
from pit_multisource.config import load_config

CONFIG = load_config()
SLOT = datetime(2019, 1, 1, 0, 15, tzinfo=UTC)
FETCHED_AT = datetime(2026, 9, 7, 12, 26, 26, 138715, tzinfo=UTC)
LEDGER_KEYS = [
    "slot", "status", "zip_sha256", "vendor_md5", "vendor_digest_state", "rows_total",
    "rows_matched", "rows_malformed", "rows_duplicate_in_slot", "records_retained",
    "shard_sha256", "shard_bytes", "attempts", "error", "fetched_at", "ingest_version",
]  # fmt: skip


def line(record_id="r1", *, themes="", enhanced="", orgs="", url="", names="", amounts="") -> str:
    """A 27-field GKG line with the given fields filled in and the rest empty."""
    fields = [""] * 27
    fields[0], fields[4] = record_id, url
    fields[7], fields[8], fields[13] = themes, enhanced, orgs
    fields[23], fields[24] = names, amounts
    return "\t".join(fields)


def tsv(*lines: str) -> bytes:
    return "\n".join(lines).encode("utf-8")


def select(*lines: str):
    return select_lines(tsv(*lines), CONFIG)


def zipped(payload: bytes, member: str = "20190101001500.gkg.csv") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, payload)
    return buffer.getvalue()


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "err", http.client.HTTPMessage(), None)


def test_theme_branch_selects():
    result = select(
        line("a", themes="TAX_FNCACT,10;ECON_BITCOIN,42"),
        line("b", enhanced="econ_bitcoin,7"),
        line("c", themes="ECON_BITCOINS,1;XECON_BITCOIN,2;ECON_BITCOIN_X"),
        line("d", themes="ECON_BITCOIN"),
    )

    assert [row.split(b"\t")[0] for row in result.retained] == [b"a", b"b", b"d"]
    assert result.rows_total == 4 and result.rows_matched == 3


def test_entity_branch_selects_on_names_orgs_url():
    result = select(
        line("names", names="Mt Gox,10;Bitcoin,55"),
        line("orgs", orgs="coinbase,3"),
        line("url", url="https://example.com/news/kraken-exchange"),
        line("amounts", amounts="5,bitcoin,12"),
        line("plural", names="stablecoins,1;bitcoins,2;bitcoinist,3;tethered,4"),
        line("suffix", names="cryptocurrencies,1"),
        line("possessive", names="bitcoin's,1"),
        line("upper", url="https://example.com/ETHEREUM"),
        line("accented", names="ethereumé,1"),
    )

    kept = [row.split(b"\t")[0].decode() for row in result.retained]
    assert kept == ["names", "orgs", "url", "suffix", "possessive", "upper"]


def test_either_branch_suffices():
    result = select(
        line("theme", themes="ECON_BITCOIN,1"),
        line("entity", names="binance,9"),
        line("both", themes="ECON_BITCOIN,1", names="binance,9"),
        line("neither", themes="TAX_ETHNICITY,1", names="Berlin,3"),
    )

    assert [row.split(b"\t")[0] for row in result.retained] == [b"theme", b"entity", b"both"]
    assert result.rows_matched == 3 and result.rows_duplicate_in_slot == 0


def test_malformed_line_counted():
    short = "\t".join(["x"] * 26)
    long = line("long", themes="ECON_BITCOIN,1") + "\textra"
    result = select(line("ok", themes="ECON_BITCOIN,1"), short, "", "   ", long)

    assert result.rows_total == 3  # blank and whitespace-only lines are not counted
    assert result.rows_malformed == 2
    assert result.rows_matched == 1
    assert len(result.retained) == 1


def test_carriage_return_stays_in_the_last_field():
    row = line("cr", themes="ECON_BITCOIN,1") + "\r"

    result = select(row)

    assert result.retained == [row.encode("utf-8")]


def test_first_duplicate_kept():
    first = line("same", themes="ECON_BITCOIN,1", names="first")
    second = line("same", themes="ECON_BITCOIN,1", names="second")
    result = select(first, line("other", names="bitcoin"), second)

    assert result.retained == [first.encode(), line("other", names="bitcoin").encode()]
    assert (result.rows_matched, result.rows_duplicate_in_slot) == (3, 1)


def test_unmatched_line_does_not_claim_its_id():
    result = select(line("same", names="Berlin"), line("same", themes="ECON_BITCOIN,1"))

    assert len(result.retained) == 1 and result.rows_duplicate_in_slot == 0


def test_invalid_utf8_is_replaced_in_the_kept_line():
    raw = line("bad", themes="ECON_BITCOIN,1", names="x").encode("utf-8").replace(b"x", b"\xff")

    result = select_lines(raw, CONFIG)

    assert result.retained == [raw.replace(b"\xff", "�".encode())]


def test_shard_bytes_deterministic():
    lines = [b"a\tb", b"c\td"]

    first, second = shard_bytes(lines), shard_bytes(lines)

    assert first == second
    assert first[:10] == bytes.fromhex("1f8b08000000000002ff")
    assert gzip.decompress(first) == b"a\tb\nc\td\n"
    assert first == _expected_shard(lines)


def _expected_shard(lines: list[bytes]) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        handle.write(b"".join(row + b"\n" for row in lines))
    return buffer.getvalue()


def test_slot_url_is_built_without_a_master_list():
    assert (
        slot_url(datetime(2019, 1, 1, 0, 15, tzinfo=UTC))
        == "http://data.gdeltproject.org/gdeltv2/20190101001500.gkg.csv.zip"
    )


def test_404_is_missing_source():
    def fetch(url: str) -> bytes:
        raise http_error(404)

    fetched = fetch_slot(SLOT, CONFIG, fetch, sleep=lambda s: None)
    record = ledger_line(
        SLOT, zip_sha256=None, selection=None, shard=None, attempts=fetched.attempts,
        error=fetched.error, fetched_at=FETCHED_AT, ingest_version="0.1.0",
    )  # fmt: skip

    assert fetched.archive is None and fetched.error is None and fetched.attempts == 1
    assert record["status"] == "missing_source"
    assert record["records_retained"] is None
    assert (record["rows_total"], record["zip_sha256"], record["vendor_digest_state"]) == (
        0, None, None
    )  # fmt: skip


def test_retries_back_off_and_succeed():
    calls, sleeps = [], []
    archive = zipped(tsv(line("a", names="bitcoin")))

    def fetch(url: str) -> bytes:
        calls.append(url)
        if len(calls) == 1:
            raise http_error(503)
        if len(calls) == 2:
            raise http.client.IncompleteRead(b"par")
        if len(calls) == 3:
            return b"not a zip"
        return archive

    fetched = fetch_slot(SLOT, CONFIG, fetch, sleep=sleeps.append)

    assert fetched.archive == archive and fetched.attempts == 4 and fetched.error is None
    assert fetched.tsv == tsv(line("a", names="bitcoin"))
    # a pause before every request, then the doubling wait after each failure
    assert sleeps == [0.05, 5, 0.05, 10, 0.05, 20, 0.05]


def test_exhausted_retries_settle_as_unresolved():
    sleeps = []

    def fetch(url: str) -> bytes:
        raise http_error(429)

    fetched = fetch_slot(SLOT, CONFIG, fetch, sleep=sleeps.append)

    assert fetched.archive is None and fetched.attempts == 6 and fetched.error == "HTTP 429"
    assert [s for s in sleeps if s != 0.05] == [5, 10, 20, 40, 80]


def test_backoff_is_capped():
    sleeps = []
    config = dataclasses.replace(CONFIG, gdelt=dataclasses.replace(CONFIG.gdelt, max_retries=7))

    def fetch(url: str) -> bytes:
        raise urllib.error.URLError("down")

    fetch_slot(SLOT, config, fetch, sleep=sleeps.append)

    assert [s for s in sleeps if s != 0.05] == [5, 10, 20, 40, 80, 120, 120]


def test_other_http_errors_abort():
    def fetch(url: str) -> bytes:
        raise http_error(403)

    with pytest.raises(urllib.error.HTTPError):
        fetch_slot(SLOT, CONFIG, fetch, sleep=lambda s: None)


def test_zip_with_two_members_is_a_failed_attempt():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a", b"x")
        archive.writestr("b", b"y")

    def fetch(url: str) -> bytes:
        return buffer.getvalue()

    fetched = fetch_slot(SLOT, CONFIG, fetch, sleep=lambda s: None)

    assert fetched.archive is None and fetched.error is not None and fetched.attempts == 6


def test_corrupt_deflate_stream_is_a_failed_attempt():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("a", os.urandom(4000).hex().encode())
    corrupt = bytearray(buffer.getvalue())
    for i in range(50, 70):  # inside the compressed data, after the 31-byte local header
        corrupt[i] ^= 0xFF

    def fetch(url: str) -> bytes:
        return bytes(corrupt)

    fetched = fetch_slot(SLOT, CONFIG, fetch, sleep=lambda s: None)

    assert fetched.archive is None and fetched.attempts == 6
    assert fetched.error is not None and fetched.error.startswith("error: ")


def test_truncated_deflate_stream_is_a_failed_attempt(monkeypatch):
    def truncated(archive: bytes) -> bytes:
        raise EOFError("Compressed file ended before the end-of-stream marker was reached")

    monkeypatch.setattr("pit_multisource.acquire.gdelt.read_member", truncated)

    fetched = fetch_slot(SLOT, CONFIG, lambda url: b"any", sleep=lambda s: None)

    assert fetched.archive is None and fetched.attempts == 6
    assert fetched.error is not None and fetched.error.startswith("EOFError")


def test_ledger_line_for_a_retained_slot():
    archive = zipped(tsv(line("a", names="bitcoin"), line("a", names="bitcoin"), "bad"))
    selection = select_lines(
        tsv(line("a", names="bitcoin"), line("a", names="bitcoin"), "bad"), CONFIG
    )
    shard = shard_bytes(selection.retained)

    record = ledger_line(
        SLOT, zip_sha256=hashlib.sha256(archive).hexdigest(), selection=selection, shard=shard,
        attempts=1, error=None, fetched_at=FETCHED_AT, ingest_version="0.1.0",
    )  # fmt: skip

    assert list(record) == LEDGER_KEYS
    assert record["slot"] == "2019-01-01T00:15:00Z"
    assert record["fetched_at"] == "2026-09-07T12:26:26.138715Z"
    assert record["status"] == "retained"
    assert record["vendor_md5"] is None and record["vendor_digest_state"] == "NOT_PUBLISHED"
    assert (record["rows_total"], record["rows_matched"]) == (3, 2)
    assert (record["rows_malformed"], record["rows_duplicate_in_slot"]) == (1, 1)
    assert record["records_retained"] == 1
    assert record["shard_sha256"] == hashlib.sha256(shard).hexdigest()
    assert record["shard_bytes"] == len(shard)
    assert record["error"] is None


def test_ledger_line_for_no_match_and_unresolved():
    selection = select(line("a", names="Berlin"))
    served = ledger_line(
        SLOT, zip_sha256="ab" * 32, selection=selection, shard=None, attempts=1, error=None,
        fetched_at=FETCHED_AT, ingest_version="0.1.0",
    )  # fmt: skip
    failed = ledger_line(
        SLOT, zip_sha256=None, selection=None, shard=None, attempts=6, error="HTTP 503",
        fetched_at=FETCHED_AT, ingest_version="0.1.0",
    )  # fmt: skip

    assert served["status"] == "served_no_match"
    assert served["records_retained"] is None and served["shard_sha256"] is None
    assert served["vendor_digest_state"] == "NOT_PUBLISHED" and served["rows_total"] == 1
    assert failed["status"] == "unresolved" and failed["error"] == "HTTP 503"
    assert failed["attempts"] == 6 and failed["rows_total"] == 0


def _fake_archive_fetch(archives: dict[str, bytes | None], calls: list[str]):
    def fetch(url: str) -> bytes:
        calls.append(url)
        body = archives[url.rsplit("/", 1)[1][:14]]
        if body is None:
            raise http_error(404)
        return body

    return fetch


def test_run_writes_shards_and_ledger_in_slot_order_and_resumes(tmp_path: Path):
    hit = tsv(line("a", themes="ECON_BITCOIN,1"), line("b", names="Berlin"))
    miss = tsv(line("c", names="Berlin"))
    archives = {
        "20190101000000": zipped(hit),
        "20190101001500": zipped(miss),
        "20190101003000": None,
    }
    calls: list[str] = []
    start, end = datetime(2019, 1, 1, tzinfo=UTC), datetime(2019, 1, 1, 0, 45, tzinfo=UTC)
    now = iter(datetime(2026, 1, 1, 0, 0, s, tzinfo=UTC) for s in range(10))

    run_slots(CONFIG, start, end, tmp_path, _fake_archive_fetch(archives, calls),
              sleep=lambda s: None, now=lambda: next(now))  # fmt: skip

    ledger = [json.loads(x) for x in (tmp_path / "slot_ledger.jsonl").read_text().splitlines()]
    assert [r["slot"] for r in ledger] == [
        "2019-01-01T00:00:00Z", "2019-01-01T00:15:00Z", "2019-01-01T00:30:00Z"
    ]  # fmt: skip
    assert [r["status"] for r in ledger] == ["retained", "served_no_match", "missing_source"]
    shard = tmp_path / "shards" / "2019-01-01" / "20190101000000.tsv.gz"
    assert shard.read_bytes() == shard_bytes([line("a", themes="ECON_BITCOIN,1").encode()])
    assert ledger[0]["shard_sha256"] == hashlib.sha256(shard.read_bytes()).hexdigest()
    assert ledger[0]["zip_sha256"] == hashlib.sha256(archives["20190101000000"]).hexdigest()
    assert len(list((tmp_path / "shards").rglob("*"))) == 2  # one day directory, one shard
    assert len(calls) == 3

    run_slots(CONFIG, start, end, tmp_path, _fake_archive_fetch(archives, calls),
              sleep=lambda s: None, now=lambda: next(now))  # fmt: skip
    assert len(calls) == 3
    assert len((tmp_path / "slot_ledger.jsonl").read_text().splitlines()) == 3


def test_run_aborts_after_consecutive_unresolved_slots(tmp_path: Path):
    config = dataclasses.replace(
        CONFIG, gdelt=dataclasses.replace(CONFIG.gdelt, max_retries=0, max_consecutive_unresolved=2)
    )

    def fetch(url: str) -> bytes:
        raise urllib.error.URLError("down")

    with pytest.raises(GdeltRunAborted):
        run_slots(config, datetime(2019, 1, 1, tzinfo=UTC), datetime(2019, 1, 1, 1, tzinfo=UTC),
                  tmp_path, fetch, sleep=lambda s: None, now=lambda: FETCHED_AT)  # fmt: skip

    ledger = [json.loads(x) for x in (tmp_path / "slot_ledger.jsonl").read_text().splitlines()]
    assert [r["status"] for r in ledger] == ["unresolved"] * 3
    assert ledger[0]["error"].startswith("URLError")


def test_run_does_not_abort_at_the_limit(tmp_path: Path):
    config = dataclasses.replace(
        CONFIG, gdelt=dataclasses.replace(CONFIG.gdelt, max_retries=0, max_consecutive_unresolved=2)
    )

    def fetch(url: str) -> bytes:
        raise urllib.error.URLError("down")

    statuses = run_slots(config, datetime(2019, 1, 1, tzinfo=UTC),
                         datetime(2019, 1, 1, 0, 30, tzinfo=UTC), tmp_path, fetch,
                         sleep=lambda s: None, now=lambda: FETCHED_AT)  # fmt: skip

    assert statuses == {"unresolved": 2}


def test_acquire_gdelt_command_is_registered(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit):
        cli.main(["acquire", "gdelt", "--help"])

    out = capsys.readouterr().out
    assert "--start" in out and "--end" in out and "--out" in out
    assert "differ" in out
