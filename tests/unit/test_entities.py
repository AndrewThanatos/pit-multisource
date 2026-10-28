from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from pit_multisource import cli
from pit_multisource.build.entities import ENTITY_COLUMNS, build_entities
from pit_multisource.config import load_config

CONFIG = load_config()
BOUNDARY = CONFIG.targets.boundary
T0 = datetime(2020, 1, 1, tzinfo=UTC)
SLOT = timedelta(minutes=15)


def _frame(*rows: tuple[str, datetime, dict[str, str]]) -> pl.DataFrame:
    """Records from (record_id, slot, {record column: field text}); other columns are empty."""
    schema = {"slot_time": pl.Datetime("us", "UTC"), "gkg_record_id": pl.String()}
    schema.update(dict.fromkeys(ENTITY_COLUMNS, pl.String()))
    data = [
        {
            "slot_time": slot,
            "gkg_record_id": record_id,
            **{c: fields.get(c, "") for c in ENTITY_COLUMNS},
        }
        for record_id, slot, fields in rows
    ]
    return pl.DataFrame(data, schema=schema)


def _entity(tables, name: str, entity_type: str = "organisation") -> dict:
    rows = tables.entities.filter(
        (pl.col("normalized_name") == name) & (pl.col("entity_type") == entity_type)
    )
    assert rows.height == 1, name
    return rows.row(0, named=True)


def test_rows_from_boundary_on_are_dropped():
    records = _frame(
        ("early", BOUNDARY - SLOT, {"v21_all_names": "Kept Co,1"}),
        ("late", BOUNDARY, {"v21_all_names": "Dropped Co,1;Kept Co,2"}),
    )

    tables = build_entities(records, CONFIG)

    assert tables.entities["normalized_name"].to_list() == ["kept co"]
    assert tables.entities["n_records_development"].to_list() == [1]
    assert tables.entities["n_records_reserve"].to_list() == [None]
    assert tables.entities["corpus_cutoff"].to_list() == [BOUNDARY - timedelta(microseconds=1)]
    assert tables.entities["entity_type"].to_list() == ["untyped_name"]


def test_first_knowable_tie_break():
    records = _frame(
        ("b", T0, {"v21_all_names": "Xcorp,1"}),
        ("a", T0, {"v21_all_names": "Xcorp,1"}),
        ("0", T0 + SLOT, {"v21_all_names": "Xcorp,1"}),
        ("z", T0 - SLOT, {"v21_all_names": "Ycorp,1"}),
        ("y", T0 - SLOT, {"v21_all_names": "Ycorp,1"}),
    )

    tables = build_entities(records, CONFIG)

    x = _entity(tables, "xcorp", "untyped_name")
    assert (x["first_knowable_at"], x["first_knowable_earned_by"]) == (T0, "a")
    y = _entity(tables, "ycorp", "untyped_name")
    assert (y["first_knowable_at"], y["first_knowable_earned_by"]) == (T0 - SLOT, "y")
    form = tables.surface_forms.filter(pl.col("normalized_name") == "xcorp").row(0, named=True)
    assert (form["first_knowable_at"], form["first_knowable_earned_by"]) == (T0, "a")


def test_counts_are_token_occurrences():
    records = _frame(
        (
            "r1",
            T0,
            {
                "v1_organizations": "acme corp",
                "v2_enhanced_organizations": "Acme Corp,10;Acme Corp,40;ACME  Corp,70",
                "v21_all_names": "Acme Corp,5",
            },
        ),
        ("r2", T0 + SLOT, {"v1_organizations": "acme corp;"}),
    )

    tables = build_entities(records, CONFIG)

    org = _entity(tables, "acme corp")
    assert org["n_records_development"] == 5
    assert org["n_surface_forms"] == 3
    assert org["gkg_columns"] == "v1_organisations,v2_organisations"
    assert _entity(tables, "acme corp", "untyped_name")["n_records_development"] == 1
    counts = {
        (r["raw_name"], r["gkg_column"]): r["n_records_development"]
        for r in tables.surface_forms.filter(pl.col("entity_type") == "organisation").to_dicts()
    }
    assert counts == {
        ("acme corp", "v1_organisations"): 2,
        ("Acme Corp", "v2_organisations"): 2,
        ("ACME  Corp", "v2_organisations"): 1,
    }


def test_strip_comes_before_the_offset_rule():
    records = _frame(
        ("r1", T0, {"v2_enhanced_persons": " Jane Roe,12 ;Jane Roe,3; ,4 ;;"}),
    )

    tables = build_entities(records, CONFIG)

    person = _entity(tables, "jane roe", "person")
    assert person["n_records_development"] == 2
    assert person["n_surface_forms"] == 1
    assert tables.surface_forms["raw_name"].to_list() == ["Jane Roe"]
    assert tables.entities.height == 1


def test_alias_candidates_never_merged():
    records = _frame(
        ("r1", T0, {"v21_all_names": "Aite- Novarica Group,1;Aite Novarica Group,2;Solo Name,3"}),
        ("r2", T0, {"v1_organizations": "aite novarica group"}),
    )

    tables = build_entities(records, CONFIG)

    assert tables.entities.height == 4
    assert tables.alias_candidates.rows() == [
        (
            "untyped_name",
            "aite novarica group",
            2,
            "aite novarica group|aite- novarica group",
            "AMBIGUOUS_NO_MERGE",
        )
    ]
    assert _entity(tables, "aite- novarica group", "untyped_name")["alias_normalized_key"] == (
        "aite novarica group"
    )


def test_offset_in_an_offsetless_column_aborts():
    records = _frame(("r1", T0, {"v1_persons": "jane roe,12"}))

    with pytest.raises(ValueError, match="offsetless"):
        build_entities(records, CONFIG)


def test_name_that_is_not_a_fixed_point_aborts():
    records = _frame(("r1", T0, {"v21_all_names": "x,1,2"}))

    with pytest.raises(ValueError, match="own normalisation"):
        build_entities(records, CONFIG)


def test_cli_builds_entity_tables(tmp_path: Path):
    dataset = tmp_path / "dataset"
    root = dataset / "news" / "gkg_records"
    root.mkdir(parents=True)
    full = _frame(("r1", T0, {"v21_all_names": "Xcorp,1;Ycorp,2"}))
    full.write_parquet(root / "year=2020.parquet")

    status = cli.main(
        ["build", "entities", "--dataset", str(dataset), "--out", str(tmp_path / "o")]
    )

    assert status == 0
    assert pl.read_parquet(tmp_path / "o" / "entities.parquet").height == 2
    assert (tmp_path / "o" / "entity_surface_forms.parquet").exists()
    assert (tmp_path / "o" / "entity_alias_candidates.parquet").exists()
