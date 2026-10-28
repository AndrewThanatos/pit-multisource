"""Tables 4 and 5 and Table 6 rows 4–5: grouping policies and the parameter sensitivity grid.

Keys are `grouping.<bounded|unbounded>.<measure>` and `grouping.features.<asset>.<field>` (the
unbounded feature against the bounded one), and `sensitivity.min<m>_age<a>.<measure>` and
`sensitivity.min<m>_age<a>.<asset>.<field>` (against the published setting).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from pit_multisource.build.news import (
    ASSETS,
    NEWS_COLUMNS,
    ClusterParams,
    NewsTables,
    build_news,
    read_gkg_records,
)
from pit_multisource.config import Config, Fold
from pit_multisource.experiments.features import Convention, clusters_24h, compare

SENSITIVITY_MIN_SHARED = (1, 2, 3)
SENSITIVITY_AGE_CAPS_H = (24, 48, 72)
_US = pl.Datetime("us", "UTC")
_US_PER_HOUR = 3_600_000_000
_FEATURE = Convention(clock="slot_end", flag="dated")


def sensitivity_settings(config: Config) -> tuple[ClusterParams, ...]:
    """The nine (min shared entities, age cap) settings in grid order; other fields published."""
    published = ClusterParams.from_config(config)
    return tuple(
        ClusterParams(published.join_window_h, age, m, published.url_age_capped)
        for m in SENSITIVITY_MIN_SHARED
        for age in SENSITIVITY_AGE_CAPS_H
    )


def cross_split(records: pl.DataFrame, folds: Sequence[Fold]) -> dict[str, int]:
    """Clusters and addresses with a record in a fold's train window and one in its test window."""
    crossing: list[pl.Series] = []
    pairs = crossing_records = url_pairs = with_test = 0
    for fold in folds:
        if fold.test_start is None or fold.test_end is None:
            continue
        with_test += 1
        t = pl.col("slot_time")
        in_train = ((t >= fold.train_start) & (t < fold.train_end)).sum().alias("train")
        in_test = ((t >= fold.test_start) & (t < fold.test_end)).sum().alias("test")
        both = (pl.col("train") > 0) & (pl.col("test") > 0)
        clusters = records.group_by("cluster_id").agg(in_train, in_test).filter(both)
        urls = records.group_by("url").agg(in_train, in_test).filter(both)
        pairs += clusters.height
        crossing_records += int((clusters["train"] + clusters["test"]).sum())
        url_pairs += urls.height
        crossing.append(clusters["cluster_id"])
    return {
        "crossing_clusters": pl.concat(crossing).n_unique() if crossing else 0,
        "crossing_cluster_fold_pairs": pairs,
        "crossing_records": crossing_records,
        "url_fold_pairs": url_pairs,
        "folds_with_test_window": with_test,
        "folds_without_test_window": len(folds) - with_test,
    }


def assert_detector_can_fire() -> None:
    """Raises unless a planted crossing reads 1 and a single-record cluster reads 0."""
    fold = Fold(
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 2, 1, tzinfo=UTC),
        datetime(2020, 2, 10, tzinfo=UTC),
        datetime(2020, 3, 1, tzinfo=UTC),
    )
    schema = {"cluster_id": pl.String(), "slot_time": _US, "url": pl.String()}
    planted = pl.DataFrame(
        [
            ("c1", datetime(2020, 1, 15, tzinfo=UTC), "u1"),
            ("c1", datetime(2020, 2, 15, tzinfo=UTC), "u2"),
        ],
        schema=schema,
        orient="row",
    )
    single = planted.head(1)
    seen = cross_split(planted, [fold])["crossing_clusters"]
    clean = cross_split(single, [fold])["crossing_clusters"]
    if (seen, clean) != (1, 0):
        raise RuntimeError(f"crossing detector canary failed: planted {seen}, clean {clean}")


def measure(tables: NewsTables, params: ClusterParams, config: Config) -> dict[str, int | float]:
    """Counts, size shape, span and split crossings of one grouping, slot start as the instant."""
    records = tables.records
    spans = records.group_by("cluster_id").agg(
        (pl.col("slot_time").max() - pl.col("slot_time").min()).dt.total_microseconds().alias("us"),
        pl.len().alias("n"),
    )
    sizes = spans["n"].cast(pl.Float64)
    published_cap = config.news.age_cap_hours * _US_PER_HOUR
    return {
        "records": records.height,
        "clusters": tables.clusters.height,
        "url_route_joins": tables.url_joins,
        "url_joins_refused_by_age_cap": tables.url_joins_refused_by_age_cap,
        "max_span_hours": float(spans["us"].max() or 0) / _US_PER_HOUR,
        "clusters_over_published_cap": int((spans["us"] > published_cap).sum()),
        "clusters_over_own_cap": int((spans["us"] > params.age_cap_h * _US_PER_HOUR).sum()),
        "size_median": float(sizes.median() or 0),
        "size_p95": float(sizes.quantile(0.95, interpolation="nearest") or 0),
        "size_p99": float(sizes.quantile(0.99, interpolation="nearest") or 0),
        "size_max": int(spans["n"].max() or 0),
        "singleton_share": float((sizes == 1).mean() or 0),
        **cross_split(records, config.split_boundaries.folds()),
    }


def feature_series(tables: NewsTables, boundary: datetime) -> dict[str, pl.Series]:
    """Dated slot-end `clusters_24h` per asset over the grouping's own records."""
    records = tables.records.select("cluster_id", "slot_time")
    return {
        asset: clusters_24h(
            records,
            tables.clusters.select(
                "cluster_id", pl.col(f"{asset}_relevant_earned_at").alias("earned_at")
            ),
            _FEATURE,
            boundary=boundary,
        )
        for asset in ASSETS
    }


def _development_records(dataset_root: Path, boundary: datetime) -> pl.DataFrame:
    records = read_gkg_records(Path(dataset_root), NEWS_COLUMNS)
    return records.filter(pl.col("slot_time") < pl.lit(boundary, dtype=_US))


def _measure_variants(
    records: pl.DataFrame, variants: Sequence[tuple[str, ClusterParams]], config: Config
) -> tuple[dict[str, int | float], dict[str, dict[str, pl.Series]]]:
    """Per variant, its measures keyed `<name>.<field>` and its feature series per asset."""
    measures: dict[str, int | float] = {}
    series: dict[str, dict[str, pl.Series]] = {}
    for name, params in variants:
        tables = build_news(records, config, params)
        measures.update({f"{name}.{k}": v for k, v in measure(tables, params, config).items()})
        series[name] = feature_series(tables, config.targets.boundary)
    return measures, series


def run_grouping(
    dataset_root: Path, shared_ids_path: Path, config: Config
) -> dict[str, int | float | bool]:
    """Table 4 and Table 6 rows 4–5: the address route with and without the age cap."""
    assert_detector_can_fire()
    ids = pl.read_parquet(shared_ids_path, columns=["record_id"])["record_id"]
    records = _development_records(dataset_root, config.targets.boundary)
    records = records.filter(pl.col("gkg_record_id").is_in(ids.implode()))
    bounded = ClusterParams.from_config(config)
    unbounded = ClusterParams(bounded.join_window_h, bounded.age_cap_h, bounded.min_shared, False)
    measures, series = _measure_variants(
        records, [("bounded", bounded), ("unbounded", unbounded)], config
    )
    result: dict[str, int | float | bool] = {
        "grouping.detector_canary": True,
        "grouping.shared_ids": ids.len(),
        "grouping.records": records.height,
        "grouping.hours": series["bounded"][ASSETS[0]].len(),
    }
    result.update({f"grouping.{k}": v for k, v in measures.items()})
    for asset in ASSETS:
        diff = compare(series["unbounded"][asset], series["bounded"][asset])
        result.update({f"grouping.features.{asset}.{k}": v for k, v in diff.items()})
    return result


def run_sensitivity(dataset_root: Path, config: Config) -> dict[str, int | float | bool]:
    """Table 5: the nine settings over the whole development population, against the published."""
    assert_detector_can_fire()
    records = _development_records(dataset_root, config.targets.boundary)
    name = "min{0.min_shared}_age{0.age_cap_h}".format
    variants = [(name(p), p) for p in sensitivity_settings(config)]
    measures, series = _measure_variants(records, variants, config)
    baseline = series[name(ClusterParams.from_config(config))]
    result: dict[str, int | float | bool] = {
        "sensitivity.records": records.height,
        "sensitivity.hours": baseline[ASSETS[0]].len(),
    }
    result.update({f"sensitivity.{k}": v for k, v in measures.items()})
    for variant, _ in variants:
        for asset in ASSETS:
            diff = compare(series[variant][asset], baseline[asset])
            result.update({f"sensitivity.{variant}.{asset}.{k}": v for k, v in diff.items()})
    return result
