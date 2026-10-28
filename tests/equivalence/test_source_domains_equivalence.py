import time
from pathlib import Path

import polars as pl
import pytest

from pit_multisource.build.source_domains import RECORD_COLUMNS, build_source_domains

pytestmark = pytest.mark.equivalence


def _first_difference(built: pl.DataFrame, published: pl.DataFrame) -> str:
    """The first row and columns where two equally shaped frames differ."""
    for i, (a, b) in enumerate(zip(built.iter_rows(), published.iter_rows(), strict=True)):
        if a != b:
            columns = [c for c, x, y in zip(built.columns, a, b, strict=True) if x != y]
            return f"row {i}: columns {columns}\nbuilt     {a}\npublished {b}"
    return "no row differs"


def test_source_domain_tables_equal_dataset(dataset_root: Path):
    records = pl.read_parquet(
        dataset_root / "news" / "news_records.parquet", columns=RECORD_COLUMNS
    )

    started = time.perf_counter()
    tables = build_source_domains(records)
    print(f"build_source_domains: {time.perf_counter() - started:.1f} s")

    for name in ("families", "edges", "cluster_counts"):
        built = getattr(tables, name)
        published = pl.read_parquet(
            dataset_root / "diagnostics" / "source_domains" / f"{name}.parquet"
        )
        assert built.schema == published.schema, name
        assert built.height == published.height, name
        assert built.equals(published), f"{name}: {_first_difference(built, published)}"
