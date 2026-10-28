from pathlib import Path

import polars as pl
import pytest

from pit_multisource.build.news import NEWS_COLUMNS, build_news, read_gkg_records
from pit_multisource.config import load_config

pytestmark = pytest.mark.equivalence


def _first_difference(built: pl.DataFrame, published: pl.DataFrame) -> str:
    """The first row and columns where two equally shaped frames differ."""
    for i, (a, b) in enumerate(zip(built.iter_rows(), published.iter_rows(), strict=True)):
        if a != b:
            columns = [c for c, x, y in zip(built.columns, a, b, strict=True) if x != y]
            return f"row {i}: columns {columns}\nbuilt     {a}\npublished {b}"
    return "no row differs"


def test_news_records_and_clusters_equal_dataset(dataset_root: Path):
    tables = build_news(read_gkg_records(dataset_root, NEWS_COLUMNS), load_config())

    for built, name in ((tables.records, "news_records"), (tables.clusters, "news_clusters")):
        published = pl.read_parquet(dataset_root / "news" / f"{name}.parquet")
        assert built.schema == published.schema, name
        assert built.height == published.height, name
        assert built.equals(published), f"{name}: {_first_difference(built, published)}"
    assert (tables.records.height, tables.clusters.height) == (2_320_951, 779_559)
    assert tables.url_joins_refused_by_age_cap == 3_741
