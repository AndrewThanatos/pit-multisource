"""Downloads 30 full GDELT slot archives chosen from the released slot table into the sample dir.

Usage: PIT_DATASET=... PIT_SOURCES=... python tools/fetch_gdelt_sample.py
Writes only to $PIT_SOURCES/gdelt_sample/: one zip per served slot and sample_manifest.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from pit_multisource.acquire.gdelt import slot_url, urllib_fetch
from pit_multisource.config import load_config

PER_STATUS = 10
STATUSES = ("retained", "served_no_match", "missing_source")


def pick_slots(slots: pl.DataFrame) -> list[tuple[datetime, str]]:
    """Ten slots per status, evenly spaced by slot_time within the status."""
    picked = []
    for status in STATUSES:
        times = slots.filter(pl.col("status") == status).sort("slot_time")["slot_time"].to_list()
        for i in range(PER_STATUS):
            picked.append((times[i * (len(times) - 1) // (PER_STATUS - 1)], status))
    return picked


def main() -> None:
    config = load_config()
    dataset, sources = Path(os.environ["PIT_DATASET"]), Path(os.environ["PIT_SOURCES"])
    out = sources / "gdelt_sample"
    out.mkdir(parents=True, exist_ok=True)
    fetch = urllib_fetch(config)
    slots = pl.read_parquet(dataset / "news" / "gdelt_slots.parquet")
    records = []
    for slot, expected in pick_slots(slots):
        record = {"slot": slot.strftime("%Y-%m-%dT%H:%M:%SZ"), "expected_status": expected}
        try:
            body = fetch(slot_url(slot, config))
        except urllib.error.HTTPError as exc:
            record |= {"http_status": exc.code, "zip_sha256": None, "zip_bytes": None}
        else:
            name = f"{slot.strftime('%Y%m%d%H%M%S')}.gkg.csv.zip"
            (out / name).write_bytes(body)
            digest = hashlib.sha256(body).hexdigest()
            record |= {"http_status": 200, "zip_sha256": digest, "zip_bytes": len(body)}
        record["fetched_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        records.append(record)
        print(record, flush=True)
    text = "".join(json.dumps(r) + "\n" for r in records)
    (out / "sample_manifest.jsonl").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
