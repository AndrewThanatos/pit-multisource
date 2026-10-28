"""Tables 4 and 5 and Table 6 rows 4–5 regrouped from the released GKG records."""

import os
from pathlib import Path

import pytest

from pit_multisource.config import load_config
from pit_multisource.experiments.grouping import run_grouping, run_sensitivity

pytestmark = pytest.mark.equivalence

# measure -> (unbounded, 72h-bounded)
TABLE_4 = {
    "clusters": (666635, 668519),
    "url_route_joins": (3492, 19),
    "url_joins_refused_by_age_cap": (0, 3473),
    "max_span_hours": (54436.0, 72.0),
    "clusters_over_published_cap": (2867, 0),
    "crossing_clusters": (2295, 0),
    "crossing_cluster_fold_pairs": (2545, 0),
    "url_fold_pairs": (2674, 2674),
    "size_median": (1.0, 1.0),
    "size_p95": (7.0, 7.0),
    "size_p99": (26.0, 26.0),
}
# asset -> changed hours, share (%), mean |d| (4 places), baseline mean
TABLE_6_ROWS_4_5 = {"btc": (14980, 25.97, 0.6353, 324.208), "eth": (10565, 18.32, 0.3098, 41.269)}
# (min shared, age cap) -> clusters, p99, singleton share, BTC mean, ETH mean
TABLE_5 = {
    (1, 24): (361365, 70, 0.623458, 181.171, 39.070),
    (1, 48): (329473, 81, 0.672501, 175.005, 39.627),
    (1, 72): (319088, 85, 0.691518, 172.884, 40.278),
    (2, 24): (728963, 23, 0.729912, 339.231, 48.494),
    (2, 48): (692988, 26, 0.757522, 332.545, 49.397),
    (2, 72): (682652, 27, 0.766569, 330.618, 49.994),
    (3, 24): (1103012, 14, 0.775277, 502.644, 52.134),
    (3, 48): (1056671, 16, 0.797969, 494.414, 52.444),
    (3, 72): (1042811, 16, 0.805353, 491.932, 52.762),
}
# BTC mean |d| against the published setting, quoted in the text
TABLE_5_BTC_MAD = {(2, 24): 8.939, (1, 72): 157.734, (3, 72): 161.314}


@pytest.fixture
def shared_ids() -> Path:
    value = os.environ.get("PIT_SHARED_IDS")
    if not value:
        pytest.skip("PIT_SHARED_IDS is not set")
    return Path(value)


def test_table_4_and_table_6_rows_4_5(dataset_root: Path, shared_ids: Path):
    result = run_grouping(dataset_root, shared_ids, load_config())

    assert result["grouping.detector_canary"] is True
    assert result["grouping.shared_ids"] == result["grouping.records"] == 2003579
    for policy in ("unbounded", "bounded"):
        assert result[f"grouping.{policy}.records"] == 2003579
        assert result[f"grouping.{policy}.folds_with_test_window"] == 17
        assert result[f"grouping.{policy}.folds_without_test_window"] == 1
    for measure, (unbounded, bounded) in TABLE_4.items():
        assert result[f"grouping.unbounded.{measure}"] == unbounded, measure
        assert result[f"grouping.bounded.{measure}"] == bounded, measure
    assert round(result["grouping.unbounded.singleton_share"], 7) == 0.7669114
    assert round(result["grouping.bounded.singleton_share"], 7) == 0.7695144
    over, crossing, clusters = (
        result[f"grouping.unbounded.{k}"]
        for k in ("clusters_over_published_cap", "crossing_clusters", "clusters")
    )
    assert (round(100 * over / clusters, 3), round(100 * crossing / clusters, 3)) == (0.430, 0.344)

    assert result["grouping.hours"] == 57672
    for asset, (changed, share, mad, baseline_mean) in TABLE_6_ROWS_4_5.items():
        key = f"grouping.features.{asset}"
        assert result[f"{key}.changed_hours"] == changed, asset
        assert round(100 * result[f"{key}.share"], 2) == share, asset
        assert round(result[f"{key}.mean_abs_diff"], 4) == mad, asset
        assert round(result[f"{key}.mean_reference"], 3) == baseline_mean, asset


def test_table_5(dataset_root: Path):
    result = run_sensitivity(dataset_root, load_config())

    assert result["sensitivity.records"] == 2055263
    assert result["sensitivity.hours"] == 57672
    for (m, a), (clusters, p99, singletons, btc, eth) in TABLE_5.items():
        key = f"sensitivity.min{m}_age{a}"
        assert result[f"{key}.records"] == 2055263, key
        assert result[f"{key}.clusters"] == clusters, key
        assert result[f"{key}.size_p99"] == p99, key
        assert round(result[f"{key}.singleton_share"], 6) == singletons, key
        assert round(result[f"{key}.btc.mean_series"], 3) == btc, key
        assert round(result[f"{key}.eth.mean_series"], 3) == eth, key
        assert result[f"{key}.crossing_clusters"] == 0, key
        assert result[f"{key}.clusters_over_own_cap"] == 0, key
    for (m, a), mad in TABLE_5_BTC_MAD.items():
        assert round(result[f"sensitivity.min{m}_age{a}.btc.mean_abs_diff"], 3) == mad
