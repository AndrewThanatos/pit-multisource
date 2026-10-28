"""Ledger counters, cluster shape, kappa and positive specific agreement on the released tables."""

from pathlib import Path

import pytest

from pit_multisource.experiments.agreement import run_agreement
from pit_multisource.experiments.descriptive import run_descriptive

pytestmark = pytest.mark.equivalence

CELLS = {
    "btc.deployed": (123351, 2001386, 6303, 189911),
    "btc.corrected": (153490, 1971247, 15552, 180662),
    "eth.deployed": (0, 0, 24920, 2296031),
    "eth.corrected": (0, 0, 48265, 2272686),
}


def test_ledger_counters(dataset_root: Path):
    counts = run_descriptive(dataset_root)
    assert counts["ledger.lines"] == 265845
    assert counts["ledger.slots"] == 265824
    assert counts["ledger.status.retained"] == 247095
    assert counts["ledger.status.served_no_match"] == 13579
    assert counts["ledger.status.missing_source"] == 5150
    assert counts["ledger.served"] == 260674
    assert counts["ledger.retry_lines"] == 21
    assert counts["ledger.retry_lines.unresolved_to_retained"] == 21
    assert round(counts["ledger.pct.retained"], 3) == 92.954
    assert round(counts["ledger.pct.served_no_match"], 3) == 5.108
    assert round(counts["ledger.pct.missing_source"], 3) == 1.937
    assert counts["ledger.dev.lines"] == 230805
    assert counts["ledger.dev.slots"] == 230784
    assert counts["ledger.dev.rows_total"] == 335078226
    assert counts["ledger.dev.rows_matched"] == 2055318
    assert counts["ledger.dev.rows_malformed"] == 1517
    assert counts["ledger.dev.rows_duplicate_in_slot"] == 55
    assert counts["ledger.dev.records_retained"] == 2055263
    assert counts["ledger.dev.identity_matched_minus_duplicates_equals_retained"] is True


def test_cluster_shape(dataset_root: Path):
    shape = run_descriptive(dataset_root)
    assert shape["cluster_shape.clusters"] == 682652
    assert shape["cluster_shape.records"] == 2055263
    assert round(shape["cluster_shape.size.mean"], 3) == 3.011
    assert shape["cluster_shape.size.median"] == 1
    assert shape["cluster_shape.size.p95"] == 7
    assert shape["cluster_shape.size.p99"] == 27
    assert shape["cluster_shape.size.max"] == 3177
    assert shape["cluster_shape.singletons"] == 523300
    assert round(shape["cluster_shape.singleton_share"], 5) == 0.76657
    assert round(shape["cluster_shape.domains.mean"], 3) == 2.040
    assert shape["cluster_shape.domains.p99"] == 19
    assert shape["cluster_shape.domains.max"] == 1001
    assert shape["cluster_shape.distinct_domains"] == 15675
    assert shape["cluster_shape.spanning_boundary"] == 60


def test_kappa_and_positive_specific_agreement(dataset_root: Path):
    result = run_agreement(dataset_root)
    for name, cells in CELLS.items():
        got = tuple(
            result[f"agreement.{name}.{c}"] for c in ("both", "a_only", "b_only", "neither")
        )
        assert got == cells, name
        assert result[f"agreement.{name}.rows"] == 2320951
    assert round(result["agreement.btc.deployed.kappa"], 7) == 0.0046187
    assert round(result["agreement.btc.deployed.psa"], 7) == 0.1094318
    assert round(result["agreement.btc.corrected.kappa"], 7) == -0.0012711
    assert round(result["agreement.btc.corrected.psa"], 7) == 0.1338316
    for name in ("eth.deployed", "eth.corrected"):
        assert result[f"agreement.{name}.kappa"] is None
        assert result[f"agreement.{name}.psa"] is None
