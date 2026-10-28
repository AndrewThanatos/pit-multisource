import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from pit_multisource.build.news import GKG_FIELDS
from pit_multisource.experiments.agreement import (
    agreement_cells,
    cohen_kappa,
    positive_specific_agreement,
    run_agreement,
)
from pit_multisource.experiments.descriptive import cluster_shape, ledger_counts, run_descriptive

BOUNDARY = datetime(2025, 8, 1, tzinfo=UTC)
BEFORE = datetime(2025, 7, 31, tzinfo=UTC)
AFTER = datetime(2025, 8, 2, tzinfo=UTC)
_US = pl.Datetime("us", "UTC")


def _records(rows: list[tuple[str, datetime, str | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"cluster_id": pl.String(), "slot_time": _US, "source_domain": pl.String()},
        orient="row",
    )


def test_domain_count_counts_null_like_polars():
    records = _records(
        [
            ("c1", BEFORE, "a.com"),
            ("c1", BEFORE, None),
            ("c1", BEFORE, "a.com"),
            ("c2", BEFORE, None),
            ("c3", BEFORE, "b.com"),
            ("c3", AFTER, "z.com"),
        ]
    )
    shape = cluster_shape(records, BOUNDARY)
    assert shape["cluster_shape.clusters"] == 3
    assert shape["cluster_shape.records"] == 5
    assert shape["cluster_shape.distinct_domains"] == 3  # a.com, b.com and null
    assert shape["cluster_shape.domains.max"] == 2  # c1: a.com and null
    assert shape["cluster_shape.domains.mean"] == pytest.approx(4 / 3)
    assert shape["cluster_shape.spanning_boundary"] == 1
    assert shape["cluster_shape.size.max"] == 3
    assert shape["cluster_shape.singletons"] == 2


def _line(slot: str, status: str, total: int | None, matched: int | None, **extra: int | None):
    row = {
        "slot": slot,
        "status": status,
        "rows_total": total,
        "rows_matched": matched,
        "rows_malformed": 0,
        "rows_duplicate_in_slot": 0,
        "records_retained": matched,
    }
    return row | extra


def test_ledger_counts_use_last_line_per_slot_and_split_at_boundary(tmp_path: Path):
    lines = [
        _line("2025-07-31T23:45:00Z", "unresolved", None, None, records_retained=None),
        _line("2025-07-31T23:45:00Z", "retained", 100, 10, rows_duplicate_in_slot=1),
        _line("2025-08-01T00:00:00Z", "retained", 50, 5),
        _line("2025-08-01T00:15:00Z", "served_no_match", 40, 0),
        _line("2025-08-01T00:30:00Z", "missing_source", None, None, records_retained=None),
    ]
    path = tmp_path / "ledger.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    counts = ledger_counts(path, BOUNDARY)
    assert counts["ledger.lines"] == 5
    assert counts["ledger.slots"] == 4
    assert counts["ledger.status.retained"] == 2
    assert counts["ledger.status.served_no_match"] == 1
    assert counts["ledger.status.missing_source"] == 1
    assert counts["ledger.served"] == 3
    assert counts["ledger.retry_lines"] == 1
    assert counts["ledger.retry_lines.unresolved_to_retained"] == 1
    assert counts["ledger.dev.lines"] == 2
    assert counts["ledger.dev.slots"] == 1
    assert counts["ledger.dev.rows_total"] == 100
    assert counts["ledger.dev.rows_matched"] == 10
    assert counts["ledger.dev.rows_duplicate_in_slot"] == 1
    assert counts["ledger.dev.records_retained"] == 10


def test_kappa_known_values():
    # p_a = p_b = 0.5, p_o = 0.8, p_e = 0.5, kappa = 0.6, PSA = 2*40/(50+50)
    assert cohen_kappa(40, 10, 10, 40) == pytest.approx(0.6)
    assert positive_specific_agreement(40, 10, 10) == pytest.approx(0.8)
    # perfect disagreement on a balanced table
    assert cohen_kappa(0, 50, 50, 0) == pytest.approx(-1.0)
    assert positive_specific_agreement(0, 50, 50) == 0.0


def test_kappa_undefined_is_null():
    # annotator B never fires: kappa and PSA are absent, not 0
    assert cohen_kappa(0, 0, 7, 93) is None  # A constant negative
    assert cohen_kappa(0, 7, 0, 93) is None  # B constant negative
    assert cohen_kappa(5, 0, 0, 0) is None  # both constant positive (p_e = 1)
    assert cohen_kappa(0, 0, 0, 0) is None
    assert positive_specific_agreement(0, 0, 7) is None
    assert positive_specific_agreement(0, 0, 0) is None


def _gkg(rows: list[dict[str, str | None]]) -> pl.DataFrame:
    base = dict.fromkeys(GKG_FIELDS, "")
    full = [{"slot_time": BEFORE, **base, **row} for row in rows]
    schema = {"slot_time": _US, **{name: pl.String() for name in GKG_FIELDS}}
    return pl.DataFrame(full, schema=schema, orient="row")


def test_agreement_cells_follow_the_two_mappings():
    ok = {"date": "20250701000000", "document_identifier": "https://x.example/a"}
    frame = _gkg(
        [
            # theme and entity (corrected field 23) agree; deployed field 24 is empty
            ok
            | {"gkg_record_id": "1", "v1_themes": "ECON_BITCOIN,5", "v21_all_names": "Bitcoin,10"},
            # the entity sits in the deployed field 24 and in organizations
            ok
            | {
                "gkg_record_id": "2",
                "v21_amounts": "5,x",
                "v1_organizations": "Bitcoin Foundation,3",
            },
            # theme only; the offset after the comma is not part of the theme
            ok | {"gkg_record_id": "3", "v2_enhanced_themes": "WB_BITCOINX,1234"},
            # a marker after the first comma does not count
            ok | {"gkg_record_id": "4", "v1_themes": "TAX,BITCOIN"},
            # null columns count as empty
            ok | {"gkg_record_id": "5", "v1_themes": None, "v21_all_names": None},
        ]
    )
    deployed = agreement_cells(frame, "BITCOIN", "bitcoin", entity_column="v21_amounts")
    corrected = agreement_cells(frame, "BITCOIN", "bitcoin", entity_column="v21_all_names")
    # deployed field 24 holds "5,x" in record 2 only, and orgs hold bitcoin there: b fires in 2
    assert deployed == {"both": 0, "a_only": 2, "b_only": 1, "neither": 2}
    assert corrected == {"both": 1, "a_only": 1, "b_only": 1, "neither": 2}


def test_agreement_cells_skip_rows_the_parser_rejects():
    good = {"date": "20250701000000", "document_identifier": "https://x.example/a"}
    frame = _gkg(
        [
            good | {"gkg_record_id": "1", "v1_themes": "ECON_BITCOIN"},
            {"gkg_record_id": "2", "date": "not-a-date", "document_identifier": "u"},
            {"gkg_record_id": "3", "date": "20250701000000", "document_identifier": ""},
        ]
    )
    cells = agreement_cells(frame, "BITCOIN", "bitcoin", entity_column="v21_all_names")
    assert sum(cells.values()) == 1


def test_run_functions_are_exposed():
    assert callable(run_descriptive)
    assert callable(run_agreement)
