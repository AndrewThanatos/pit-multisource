"""Table 2: row counts of the released tables, keyed `sizes.<table>`."""

from __future__ import annotations

from pathlib import Path

import polars as pl

TABLES = {
    "bars_1h.btcusdt": "market/btcusdt_1h.parquet",
    "bars_1h.ethusdt": "market/ethusdt_1h.parquet",
    "bars_1m.btcusdt": "market/btcusdt_1m.parquet",
    "bars_1m.ethusdt": "market/ethusdt_1m.parquet",
    "news_records": "news/news_records.parquet",
    "news_clusters": "news/news_clusters.parquet",
    "entities": "news/entities.parquet",
    "source_domains": "diagnostics/source_domains/families.parquet",
    "source_domain_pairs": "diagnostics/source_domains/edges.parquet",
    "macro_releases": "calendar/macro_releases.parquet",
}


def run_sizes(dataset_root: Path) -> dict[str, int]:
    """Rows of each released table."""
    return {
        f"sizes.{name}": pl.scan_parquet(Path(dataset_root) / path)
        .select(pl.len())
        .collect()
        .item()
        for name, path in TABLES.items()
    }
