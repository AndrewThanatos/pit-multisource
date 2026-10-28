import json
import os
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import polars as pl
import pytest

from pit_multisource.acquire.gdelt import read_member
from pit_multisource.build.selection import ledger_line, select_lines, shard_bytes
from pit_multisource.config import load_config

pytestmark = pytest.mark.equivalence

COUNTERS = ("rows_total", "rows_matched", "rows_malformed", "rows_duplicate_in_slot")


def _manifest() -> list[dict]:
    root = os.environ.get("PIT_SOURCES")
    path = Path(root or "") / "gdelt_sample" / "sample_manifest.jsonl"
    if not root or not path.exists():
        return []
    return [json.loads(text) for text in path.read_text(encoding="utf-8").splitlines()]


SAMPLE = _manifest()


def _slot(entry: dict) -> datetime:
    return datetime.fromisoformat(entry["slot"]).astimezone(UTC)


def _published_lines(dataset_root: Path, slot: datetime) -> list[bytes]:
    """The released records of one slot, each rebuilt as its 27 fields joined by tabs."""
    path = dataset_root / "news" / "gkg_records" / f"year={slot.year}.parquet"
    rows = pl.scan_parquet(path).filter(pl.col("slot_time") == slot).drop("slot_time").collect()
    assert rows.width == 27
    return ["\t".join(row).encode("utf-8") for row in rows.iter_rows()]


@cache
def _published_ledger(dataset_root: Path) -> dict[str, dict]:
    latest = {}
    for text in (dataset_root / "news" / "gdelt_slot_ledger.jsonl").read_text().splitlines():
        record = json.loads(text)
        latest[record["slot"]] = record
    return latest


def test_sample_has_ten_slots_of_each_status(sources_root: Path):
    statuses = [entry["expected_status"] for entry in SAMPLE]

    assert len(SAMPLE) == 30
    assert {s: statuses.count(s) for s in set(statuses)} == {
        "retained": 10, "served_no_match": 10, "missing_source": 10,
    }  # fmt: skip


@pytest.mark.parametrize(
    "entry", SAMPLE, ids=[f"{e['slot']}-{e['expected_status']}" for e in SAMPLE]
)
def test_sample_selection_equals_shards_and_ledger(
    entry: dict, sources_root: Path, dataset_root: Path
):
    config = load_config()
    slot = _slot(entry)
    published = _published_ledger(dataset_root)[entry["slot"]]
    slots = pl.scan_parquet(dataset_root / "news" / "gdelt_slots.parquet")
    slot_row = slots.filter(pl.col("slot_time") == slot).collect().to_dicts()[0]
    assert slot_row["status"] == entry["expected_status"] == published["status"]

    if entry["expected_status"] == "missing_source":
        assert entry["http_status"] == 404
        assert not (sources_root / "gdelt_sample" / f"{slot:%Y%m%d%H%M%S}.gkg.csv.zip").exists()
        assert _published_lines(dataset_root, slot) == []
        return

    archive = (sources_root / "gdelt_sample" / f"{slot:%Y%m%d%H%M%S}.gkg.csv.zip").read_bytes()
    selection = select_lines(read_member(archive), config)

    assert {name: getattr(selection, name) for name in COUNTERS} == {
        name: slot_row[name] for name in COUNTERS
    }
    assert selection.retained == _published_lines(dataset_root, slot)
    assert len(selection.retained) == (slot_row["records_retained"] or 0)

    shard = shard_bytes(selection.retained) if selection.retained else None
    rebuilt = ledger_line(
        slot,
        zip_sha256=entry["zip_sha256"],
        selection=selection,
        shard=shard,
        attempts=published["attempts"],
        error=None,
        fetched_at=datetime.fromisoformat(published["fetched_at"]),
        ingest_version=config.gdelt.ingest_version,
    )
    assert rebuilt == published
    assert list(rebuilt) == list(published)
