"""Table 6 rows 1–3: the 24-hour distinct-cluster count under clock and flag conventions.

`clusters_24h` at a whole hour H counts the distinct flagged clusters with a member visible in
(H − 24 h, H] whose flag has started by H. A record is visible at its slot start (`slot_start`)
or its slot end (`slot_end`); a flag counts for the cluster's whole life (`lifetime`) or from the
instant the earning record becomes visible (`dated`). Only clusters that earned the flag before
the boundary count.

Keys of the returned dict are `features.hours`, `features.clock.btc.<field>` (slot start against
slot end, lifetime flags) and `features.flag.<asset>.<field>` (lifetime against dated flags, slot
end), with field `changed_hours`, `share`, `mean_abs_diff`, `negative_hours`, `mean_series` and
`mean_reference`, plus `features.flag.<asset>.excluded_at_or_after_boundary`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from pit_multisource.config import load_config
from pit_multisource.pit.availability import SLOT_LENGTH

# First hour of the grid: one full window after the first slot.
FIRST_HOUR = datetime(2019, 1, 2, tzinfo=UTC)
WINDOW = timedelta(hours=24)
HOUR = timedelta(hours=1)
CLOCK_SHIFT = {"slot_start": timedelta(0), "slot_end": SLOT_LENGTH}
FLAGS = ("lifetime", "dated")
ASSETS = ("btc", "eth")
_US = pl.Datetime("us", "UTC")


@dataclass(frozen=True)
class Convention:
    """When a record becomes visible and from when a cluster's flag counts."""

    clock: str = "slot_end"
    flag: str = "dated"

    def __post_init__(self) -> None:
        if self.clock not in CLOCK_SHIFT:
            raise ValueError(f"unknown clock {self.clock!r}; expected one of {list(CLOCK_SHIFT)}")
        if self.flag not in FLAGS:
            raise ValueError(f"unknown flag {self.flag!r}; expected one of {list(FLAGS)}")


def hour_grid(first_hour: datetime, boundary: datetime) -> pl.Series:
    """Whole hours from `first_hour` to the last hour before `boundary`, inclusive."""
    return pl.datetime_range(
        first_hour, boundary - HOUR, HOUR, time_unit="us", time_zone="UTC", eager=True
    ).alias("decision_time")


def presence_segments(records: pl.DataFrame, convention: Convention) -> pl.DataFrame:
    """Per cluster, maximal runs of visible instants with no gap over 24 h, as [v_lo, pres_end)."""
    return (
        records.select(
            "cluster_id", (pl.col("slot_time") + CLOCK_SHIFT[convention.clock]).alias("vis")
        )
        .unique()
        .sort("cluster_id", "vis")
        .with_columns(
            (
                (pl.col("cluster_id") != pl.col("cluster_id").shift())
                | (pl.col("vis").diff() > WINDOW)
            )
            .fill_null(True)
            .cum_sum()
            .alias("segment")
        )
        .group_by("cluster_id", "segment")
        .agg(pl.col("vis").min().alias("v_lo"), pl.col("vis").max().alias("v_hi"))
        .select("cluster_id", "v_lo", (pl.col("v_hi") + WINDOW).alias("pres_end"))
    )


def _ceil_hour(col: pl.Expr) -> pl.Expr:
    truncated = col.dt.truncate("1h")
    return pl.when(truncated == col).then(col).otherwise(truncated + HOUR)


def clusters_24h(
    records: pl.DataFrame,
    clusters: pl.DataFrame,
    convention: Convention,
    *,
    boundary: datetime,
    first_hour: datetime = FIRST_HOUR,
) -> pl.Series:
    """The count on `hour_grid(first_hour, boundary)`.

    `records` holds `cluster_id` and `slot_time`; `clusters` holds `cluster_id` and `earned_at`,
    the slot start of the cluster's first flagged record (null when never flagged).
    """
    lo = pl.lit(first_hour, dtype=_US)
    hi = pl.lit(boundary, dtype=_US)
    flagged = clusters.select("cluster_id", "earned_at").filter(
        pl.col("earned_at").is_not_null() & (pl.col("earned_at") < hi)
    )
    segments = presence_segments(records, convention).join(flagged, on="cluster_id")
    if convention.flag == "dated":
        flag_start = pl.col("earned_at") + CLOCK_SHIFT[convention.clock]
        start = pl.max_horizontal("v_lo", flag_start)
    else:
        start = pl.col("v_lo")
    events = segments.select(
        pl.max_horizontal(_ceil_hour(start), lo).alias("open"),
        pl.min_horizontal(_ceil_hour(pl.col("pres_end")), hi).alias("close"),
    ).filter(pl.col("open") < pl.col("close"))
    n = pl.len().cast(pl.Int64)
    opens = events.group_by(pl.col("open").alias("decision_time")).agg(n.alias("d"))
    closes = events.group_by(pl.col("close").alias("decision_time")).agg((-n).alias("d"))
    delta = pl.concat([opens, closes]).group_by("decision_time").agg(pl.col("d").sum())
    return (
        hour_grid(first_hour, boundary)
        .to_frame()
        .join(delta, on="decision_time", how="left")
        .sort("decision_time")
        .select(pl.col("d").fill_null(0).cum_sum().alias("clusters_24h"))
        .to_series()
    )


def compare(series: pl.Series, reference: pl.Series) -> dict[str, int | float]:
    """Hours where the counts differ by at least one cluster, their share and the mean |d|."""
    d = series - reference
    return {
        "changed_hours": int((d.abs() >= 1).sum()),
        "share": float((d.abs() >= 1).sum()) / d.len(),
        "mean_abs_diff": float(d.abs().mean()),
        "negative_hours": int((d < 0).sum()),
        "mean_series": float(series.mean()),
        "mean_reference": float(reference.mean()),
    }


def run_features(
    dataset_root: Path,
    boundary: datetime | None = None,
    first_hour: datetime = FIRST_HOUR,
) -> dict[str, int | float]:
    """Row 1 (slot start against slot end, BTC) and rows 2–3 (lifetime against dated)."""
    boundary = boundary or load_config().targets.boundary
    news = Path(dataset_root) / "news"
    hi = pl.lit(boundary, dtype=_US)
    records = (
        pl.scan_parquet(news / "news_records.parquet")
        .filter(pl.col("slot_time") < hi)
        .select("cluster_id", "slot_time")
        .collect()
    )
    opened = (
        pl.scan_parquet(news / "news_clusters.parquet")
        .filter(pl.col("opened_at") < hi)
        .select("cluster_id", *(f"{a}_relevant_earned_at" for a in ASSETS))
        .collect()
    )

    def count(asset: str, clock: str, flag: str) -> pl.Series:
        clusters = opened.select(
            "cluster_id", pl.col(f"{asset}_relevant_earned_at").alias("earned_at")
        )
        return clusters_24h(
            records, clusters, Convention(clock, flag), boundary=boundary, first_hour=first_hour
        )

    result: dict[str, int | float] = {"features.hours": hour_grid(first_hour, boundary).len()}
    clock = compare(count("btc", "slot_start", "lifetime"), count("btc", "slot_end", "lifetime"))
    result.update({f"features.clock.btc.{k}": v for k, v in clock.items()})
    for asset in ASSETS:
        flag = compare(count(asset, "slot_end", "lifetime"), count(asset, "slot_end", "dated"))
        result.update({f"features.flag.{asset}.{k}": v for k, v in flag.items()})
        earned = pl.col(f"{asset}_relevant_earned_at")
        result[f"features.flag.{asset}.excluded_at_or_after_boundary"] = opened.filter(
            earned.is_not_null() & (earned >= hi)
        ).height
    return result
