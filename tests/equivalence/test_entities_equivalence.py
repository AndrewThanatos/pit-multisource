from pathlib import Path

import polars as pl
import pytest

from pit_multisource.build.entities import ENTITY_RECORD_COLUMNS, build_entities
from pit_multisource.build.news import read_gkg_records
from pit_multisource.config import load_config

pytestmark = pytest.mark.equivalence

PUBLISHED = {
    "entities": "entities",
    "surface_forms": "entity_surface_forms",
    "alias_candidates": "entity_alias_candidates",
}


def _first_difference(built: pl.DataFrame, published: pl.DataFrame) -> str:
    """The first row and columns where two equally shaped frames differ."""
    for i, (a, b) in enumerate(zip(built.iter_rows(), published.iter_rows(), strict=True)):
        if a != b:
            columns = [c for c, x, y in zip(built.columns, a, b, strict=True) if x != y]
            return f"row {i}: columns {columns}\nbuilt     {a}\npublished {b}"
    return "no row differs"


def test_entity_tables_equal_dataset(dataset_root: Path):
    tables = build_entities(read_gkg_records(dataset_root, ENTITY_RECORD_COLUMNS), load_config())

    for attribute, name in PUBLISHED.items():
        built = getattr(tables, attribute)
        published = pl.read_parquet(dataset_root / "news" / f"{name}.parquet")
        assert built.schema == published.schema, name
        assert built.height == published.height, name
        assert built.equals(published), f"{name}: {_first_difference(built, published)}"
    assert tables.entities["n_records_development"].sum() == 45_871_753
