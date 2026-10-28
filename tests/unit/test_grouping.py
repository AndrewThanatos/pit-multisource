from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from pit_multisource.build.news import GKG_FIELDS, ClusterParams, build_news
from pit_multisource.config import Fold, load_config
from pit_multisource.experiments.grouping import (
    cross_split,
    measure,
    run_grouping,
    run_sensitivity,
    sensitivity_settings,
)

CONFIG = load_config()
T0 = datetime(2020, 1, 1, tzinfo=UTC)
H = timedelta(hours=1)
_US = pl.Datetime("us", "UTC")


def _gkg(rows: list[tuple[str, datetime, str, str]]) -> pl.DataFrame:
    """GKG records from (record id, slot, url, names)."""
    out = []
    for record_id, slot, url, names in rows:
        row: dict[str, object] = {"slot_time": slot, **{name: "" for name in GKG_FIELDS}}
        row.update(gkg_record_id=record_id, document_identifier=url, v21_all_names=names)
        row.update(source_common_name="example.com", v15_tone="1,0,0")
        out.append(row)
    schema = {"slot_time": _US, **{name: pl.String() for name in GKG_FIELDS}}
    return pl.DataFrame(out, schema=schema, orient="row")


def _day(month: int, day: int) -> datetime:
    return datetime(2020, month, day, tzinfo=UTC)


def test_unbounded_url_route_can_exceed_72h():
    records = _gkg(
        [
            ("a", T0, "u", "Xcorp;Ycorp"),
            ("b", T0 + 100 * H, "u", ""),
        ]
    )
    bounded = ClusterParams()
    unbounded = ClusterParams(url_age_capped=False)

    capped = measure(build_news(records, CONFIG, bounded), bounded, CONFIG)
    uncapped = measure(build_news(records, CONFIG, unbounded), unbounded, CONFIG)

    assert (capped["clusters"], capped["url_route_joins"]) == (2, 0)
    assert capped["url_joins_refused_by_age_cap"] == 1
    assert (capped["max_span_hours"], capped["clusters_over_published_cap"]) == (0.0, 0)
    assert (uncapped["clusters"], uncapped["url_route_joins"]) == (1, 1)
    assert uncapped["url_joins_refused_by_age_cap"] == 0
    assert uncapped["max_span_hours"] == 100.0
    assert uncapped["clusters_over_published_cap"] == uncapped["clusters_over_own_cap"] == 1
    assert uncapped["singleton_share"] == 0.0
    assert capped["singleton_share"] == 1.0


def test_crossing_counted_per_record():
    fold = Fold(_day(1, 1), _day(2, 1), _day(2, 10), _day(3, 1))
    later = Fold(_day(1, 1), _day(3, 1), _day(3, 10), _day(4, 1))
    refit = Fold(_day(1, 1), _day(4, 1), None, None)
    rows = [
        ("crosses", _day(1, 15), "u1"),
        ("crosses", _day(2, 15), "u1"),
        ("crosses", _day(3, 15), "u2"),
        # First and last records lie outside both windows of the first fold.
        ("around", datetime(2019, 12, 31, tzinfo=UTC), "u3"),
        ("around", _day(3, 5), "u3"),
        # In train, then only the gap and after the test window.
        ("gap", _day(1, 20), "u4"),
        ("gap", _day(2, 5), "u5"),
        ("gap", _day(3, 2), "u6"),
        ("single", _day(1, 10), "u1"),
    ]
    records = pl.DataFrame(
        rows, schema={"cluster_id": pl.String(), "slot_time": _US, "url": pl.String()}, orient="row"
    )

    one = cross_split(records, [fold])
    both = cross_split(records, [fold, later, refit])

    assert one["crossing_clusters"] == 1
    assert one["crossing_cluster_fold_pairs"] == 1
    assert one["crossing_records"] == 2
    assert one["url_fold_pairs"] == 1
    # "crosses" holds records in both windows of both folds: one cluster, two pairs.
    assert both["crossing_clusters"] == 1
    assert both["crossing_cluster_fold_pairs"] == 2
    assert (both["folds_with_test_window"], both["folds_without_test_window"]) == (2, 1)


def test_nine_settings_enumerated():
    settings = sensitivity_settings(CONFIG)

    assert [(p.min_shared, p.age_cap_h) for p in settings] == [
        (m, a) for m in (1, 2, 3) for a in (24, 48, 72)
    ]
    assert {(p.join_window_h, p.url_age_capped) for p in settings} == {(24, True)}
    assert ClusterParams.from_config(CONFIG) in settings


def _dataset(root: Path) -> Path:
    gkg = root / "news" / "gkg_records"
    gkg.mkdir(parents=True)
    _gkg(
        [
            ("a", T0, "u", "Xcorp;Ycorp;Zcorp;bitcoin"),
            ("b", T0 + 2 * H, "v", "Xcorp;Ycorp"),
            ("c", T0 + 3 * H, "w", "Xcorp"),
            ("f", T0 + 90 * H, "v", ""),
            ("d", T0 + 100 * H, "u", ""),
            ("e", datetime(2025, 8, 1, tzinfo=UTC), "u", ""),
        ]
    ).write_parquet(gkg / "year=2020.parquet")
    return root


def test_sensitivity_runs_every_setting_against_the_published_one(tmp_path: Path):
    result = run_sensitivity(_dataset(tmp_path), CONFIG)

    assert result["sensitivity.records"] == 5
    assert result["sensitivity.min1_age72.clusters"] == 3
    assert result["sensitivity.min2_age72.clusters"] == 4
    assert result["sensitivity.min3_age72.clusters"] == 5
    assert result["sensitivity.min2_age72.btc.changed_hours"] == 0
    assert result["sensitivity.min2_age72.eth.mean_abs_diff"] == 0.0
    # Records a and b are one cluster at the published setting and two at min_shared 3.
    assert result["sensitivity.min3_age72.btc.changed_hours"] > 0
    assert result["sensitivity.hours"] == 57672
    assert len([k for k in result if k.endswith(".clusters")]) == 9


def test_grouping_runs_both_policies_on_the_shared_ids(tmp_path: Path):
    ids = tmp_path / "ids.parquet"
    pl.DataFrame({"record_id": ["a", "b", "f", "d", "zz"]}).write_parquet(ids)

    result = run_grouping(_dataset(tmp_path), ids, CONFIG)

    assert result["grouping.records"] == 4
    assert result["grouping.bounded.clusters"] == 3
    assert result["grouping.unbounded.clusters"] == 1
    assert result["grouping.unbounded.max_span_hours"] == 100.0
    assert result["grouping.bounded.url_joins_refused_by_age_cap"] == 2
    # Bounded, f and d are two clusters present together; unbounded they are one.
    assert result["grouping.detector_canary"] is True
    assert result["grouping.features.eth.changed_hours"] > 0
