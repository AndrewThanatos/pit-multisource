import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from pit_multisource.acquire.binance import download_month, monthly_zip_url
from pit_multisource.build.bars import BAR_SCHEMA, build_bars, months, parse_klines_zip, zips_for
from pit_multisource.config import load_config

HOUR_MS = 3_600_000


def _row(open_time: int, close_time: int, close: float = 2.5, count: int = 7) -> str:
    return f"{open_time},1.0,3.0,0.5,{close},10.0,{close_time},25.0,{count},4.0,10.0,0"


def _write_zip(path: Path, rows: list[str], member: str = "klines.csv") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, "\n".join(rows) + "\n")
    return path


def test_epoch_unit_detected_per_file(tmp_path: Path):
    t0 = 1_546_300_800_000  # 2019-01-01T00:00:00Z in ms
    ms_zip = _write_zip(tmp_path / "ms.zip", [_row(t0, t0 + HOUR_MS - 1)])
    us_zip = _write_zip(tmp_path / "us.zip", [_row(t0 * 1000, (t0 + HOUR_MS) * 1000 - 1)])

    from_ms = parse_klines_zip(ms_zip)
    from_us = parse_klines_zip(us_zip)

    assert from_ms["open_time"].to_list() == [datetime(2019, 1, 1, tzinfo=UTC)]
    assert from_us["open_time"].to_list() == from_ms["open_time"].to_list()
    assert from_ms["close_time"].to_list() == [datetime(2019, 1, 1, 0, 59, 59, 999000, tzinfo=UTC)]
    assert from_us["close_time"].to_list() == [datetime(2019, 1, 1, 0, 59, 59, 999999, tzinfo=UTC)]


def test_close_time_unit_is_judged_on_its_own_minimum(tmp_path: Path):
    t0 = 1_546_300_800_000
    path = _write_zip(tmp_path / "mix.zip", [_row(t0, (t0 + HOUR_MS - 1) * 1000)])

    frame = parse_klines_zip(path)

    assert frame["open_time"].to_list() == [datetime(2019, 1, 1, tzinfo=UTC)]
    assert frame["close_time"].to_list() == [datetime(2019, 1, 1, 0, 59, 59, 999000, tzinfo=UTC)]


def test_mixed_units_in_one_file_refused(tmp_path: Path):
    t0 = 1_546_300_800_000
    path = _write_zip(
        tmp_path / "bad.zip",
        [_row(t0, t0 + HOUR_MS - 1), _row((t0 + HOUR_MS) * 1000, (t0 + 2 * HOUR_MS) * 1000 - 1)],
    )

    with pytest.raises(ValueError, match="mixed epoch units"):
        parse_klines_zip(path)


def test_zip_without_exactly_one_csv_refused(tmp_path: Path):
    path = tmp_path / "two.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a.csv", _row(1_546_300_800_000, 1_546_304_399_999))
        archive.writestr("b.csv", _row(1_546_300_800_000, 1_546_304_399_999))

    with pytest.raises(ValueError, match="exactly one"):
        parse_klines_zip(path)


def test_duplicate_open_time_keeps_last(tmp_path: Path):
    t0 = 1_546_300_800_000
    first = _write_zip(tmp_path / "a.zip", [_row(t0, t0 + HOUR_MS - 1, close=1.0)])
    second = _write_zip(
        tmp_path / "b.zip",
        [_row(t0, t0 + HOUR_MS - 1, close=2.0), _row(t0 + HOUR_MS, t0 + 2 * HOUR_MS - 1)],
    )

    bars = build_bars([first, second])

    assert bars.height == 2
    assert bars["close"].to_list()[0] == 2.0


def test_months_outside_config_are_ignored(tmp_path: Path):
    config = load_config()
    listed = months(config)
    assert listed[0] == "2019-01" and listed[-1] == "2026-07" and len(listed) == 91
    for month in [*listed, "2018-12", "2026-08"]:
        _write_zip(
            tmp_path / f"BTCUSDT-1h-{month}.zip", [_row(1_546_300_800_000, 1_546_304_399_999)]
        )

    chosen = zips_for(tmp_path, "BTCUSDT", "1h", listed)

    assert [p.name for p in chosen] == [f"BTCUSDT-1h-{m}.zip" for m in listed]


def test_missing_month_refused(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        zips_for(tmp_path, "BTCUSDT", "1h", ["2019-01"])


def test_output_schema_and_order(tmp_path: Path):
    t0 = 1_546_300_800_000
    path = _write_zip(
        tmp_path / "a.zip",
        [_row(t0 + HOUR_MS, t0 + 2 * HOUR_MS - 1), _row(t0, t0 + HOUR_MS - 1)],
    )

    bars = build_bars([path])

    assert list(bars.schema.items()) == list(BAR_SCHEMA.items())
    assert bars.columns == [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    assert bars["open_time"].is_sorted()
    assert bars.schema["open_time"] == pl.Datetime("us", "UTC")
    assert bars.schema["trade_count"] == pl.Int64


def test_monthly_zip_url():
    assert monthly_zip_url("BTCUSDT", "1h", "2024-01") == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
    )
    with pytest.raises(ValueError):
        monthly_zip_url("BTCUSDT", "1h", "2024-13")


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("k.csv", _row(1_546_300_800_000, 1_546_304_399_999))
    return buffer.getvalue()


def _fake_fetch(payload: bytes, checksum: str):
    def fetch(url: str) -> bytes:
        if url.endswith(".CHECKSUM"):
            return f"{checksum}  {url.rsplit('/', 1)[1]}\n".encode()
        return payload

    return fetch


def test_download_month_writes_verified_zip_and_manifest(tmp_path: Path):
    payload = _zip_bytes()
    digest = hashlib.sha256(payload).hexdigest()

    path = download_month("BTCUSDT", "1h", "2019-01", tmp_path, fetch=_fake_fetch(payload, digest))

    assert path == tmp_path / "BTCUSDT-1h-2019-01.zip"
    assert path.read_bytes() == payload
    assert not list(tmp_path.glob("*.part"))
    line = json.loads((tmp_path / "download_manifest.jsonl").read_text().splitlines()[0])
    assert line["sha256"] == digest and line["size_bytes"] == len(payload)


def test_checksum_mismatch_refused(tmp_path: Path):
    payload = _zip_bytes()

    with pytest.raises(ValueError, match="checksum mismatch"):
        download_month("BTCUSDT", "1h", "2019-01", tmp_path, fetch=_fake_fetch(payload, "0" * 64))

    assert not list(tmp_path.iterdir())


def test_valid_local_zip_is_not_downloaded_again(tmp_path: Path):
    payload = _zip_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / "BTCUSDT-1h-2019-01.zip").write_bytes(payload)
    calls: list[str] = []
    fetch = _fake_fetch(payload, digest)

    download_month(
        "BTCUSDT", "1h", "2019-01", tmp_path, fetch=lambda u: calls.append(u) or fetch(u)
    )

    assert calls == [monthly_zip_url("BTCUSDT", "1h", "2019-01") + ".CHECKSUM"]
