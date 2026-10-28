"""Ledger counters and cluster shape of the news corpus.

Keys of the returned dict are `ledger.<name>`, `ledger.dev.<name>` and `cluster_shape.<name>`.
The ledger counts a line per fetch attempt: the final status of a slot is its last line in file
order, and a retry is any line that is not the last for its slot. `ledger.dev.*` sums the lines
with `slot < boundary`, null counters as 0. Cluster shape covers the records with
`slot_time < boundary`; a cluster that continues past the boundary is cut to its early members
and counted in `cluster_shape.spanning_boundary`. Domain counts use polars `n_unique`, so a null
source domain counts as one more domain.

Supplementary results (not in the article text): `cluster_shape.size.mean`,
`cluster_shape.size.max`, `cluster_shape.domains.mean` and `cluster_shape.distinct_domains`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl

from pit_multisource.config import load_config

STATUSES = ("retained", "served_no_match", "missing_source")
SERVED_STATUSES = ("retained", "served_no_match")
COUNTERS = (
    "rows_total",
    "rows_matched",
    "rows_malformed",
    "rows_duplicate_in_slot",
    "records_retained",
)


def _slot(text: str) -> datetime:
    return datetime.fromisoformat(text)


def ledger_counts(path: Path, boundary: datetime) -> dict[str, int | float | bool]:
    """Counters over the whole slot ledger and over the lines before `boundary`."""
    with Path(path).open(encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle]
    final: dict[str, dict] = {}
    for line in lines:
        final[line["slot"]] = line
    retries = [line for line in lines if final[line["slot"]] is not line]
    counts: dict[str, int | float | bool] = {
        "ledger.lines": len(lines),
        "ledger.slots": len(final),
    }
    for status in STATUSES:
        n = sum(1 for line in final.values() if line["status"] == status)
        counts[f"ledger.status.{status}"] = n
        counts[f"ledger.pct.{status}"] = 100 * n / len(final)
    counts["ledger.served"] = sum(counts[f"ledger.status.{s}"] for s in SERVED_STATUSES)
    counts["ledger.retry_lines"] = len(retries)
    counts["ledger.retry_lines.unresolved_to_retained"] = sum(
        1
        for line in retries
        if line["status"] == "unresolved"
        and final[line["slot"]]["status"] == "retained"
        and all(not line[c] for c in COUNTERS)
    )
    dev = [line for line in lines if _slot(line["slot"]) < boundary]
    counts["ledger.dev.lines"] = len(dev)
    counts["ledger.dev.slots"] = len({line["slot"] for line in dev})
    for counter in COUNTERS:
        counts[f"ledger.dev.{counter}"] = sum(line[counter] or 0 for line in dev)
    counts["ledger.dev.identity_matched_minus_duplicates_equals_retained"] = (
        counts["ledger.dev.rows_matched"] - counts["ledger.dev.rows_duplicate_in_slot"]
        == counts["ledger.dev.records_retained"]
    )
    return counts


def cluster_shape(records: pl.DataFrame, boundary: datetime) -> dict[str, int | float]:
    """Cluster size and domain figures over the records before `boundary`."""
    early = pl.col("slot_time") < boundary
    dev = records.filter(early)
    per = dev.group_by("cluster_id").agg(
        pl.len().alias("size"), pl.col("source_domain").n_unique().alias("domains")
    )
    spanning = (
        records.group_by("cluster_id")
        .agg(early.any().alias("early"), (~early).any().alias("late"))
        .filter(pl.col("early") & pl.col("late"))
        .height
    )
    singletons = per.filter(pl.col("size") == 1).height
    shape: dict[str, int | float] = {
        "cluster_shape.clusters": per.height,
        "cluster_shape.records": dev.height,
        "cluster_shape.spanning_boundary": spanning,
        "cluster_shape.singletons": singletons,
        "cluster_shape.singleton_share": singletons / per.height,
        "cluster_shape.distinct_domains": dev["source_domain"].n_unique(),
    }
    for column, prefix in (("size", "size"), ("domains", "domains")):
        series = per[column]
        shape[f"cluster_shape.{prefix}.mean"] = float(series.mean())
        shape[f"cluster_shape.{prefix}.median"] = float(series.median())
        shape[f"cluster_shape.{prefix}.p95"] = float(series.quantile(0.95))
        shape[f"cluster_shape.{prefix}.p99"] = float(series.quantile(0.99))
        shape[f"cluster_shape.{prefix}.max"] = int(series.max())
    return shape


def run_descriptive(dataset_root: Path) -> dict[str, int | float | bool]:
    """Ledger counters and cluster shape of the released news tables."""
    news = Path(dataset_root) / "news"
    boundary = load_config().targets.boundary
    counts = ledger_counts(news / "gdelt_slot_ledger.jsonl", boundary)
    records = pl.read_parquet(
        news / "news_records.parquet", columns=["cluster_id", "slot_time", "source_domain"]
    )
    return counts | cluster_shape(records, boundary)
