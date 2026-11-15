"""Route B of the input check: the released tables over the whole period, restricted to a cutoff.

The released news records and clusters, hourly bars and calendar are read in full, records after
the diagnostic boundary included, and each cutoff is a filter on stored timestamps: a record is
visible when `slot_time + 15 min <= c`, a cluster flag when `<asset>_relevant_earned_at + 15 min
<= c`. Three switches inject one fault each, for the negative controls: flags shown from the
cluster's opening, the bar still forming at c admitted, and a release admitted before its
schedule was known.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from pit_multisource.pit.inputs import (
    ASSETS,
    BAR,
    BAR_COLUMNS,
    CALENDAR_SCHEMA,
    GKG_DELAY,
    MARKET_HISTORY,
    MIN_COMPLETENESS,
    SYMBOLS,
    WINDOWS_H,
    Snapshot,
)


class Projection:
    """Cutoff views of the released tables."""

    def __init__(self, dataset_root: Path) -> None:
        root = Path(dataset_root)
        self.records = pl.read_parquet(
            root / "news" / "news_records.parquet",
            columns=["record_id", "slot_time", "cluster_id", "source_domain"],
        ).with_columns((pl.col("slot_time") + GKG_DELAY).alias("available_at"))
        self.flags = pl.read_parquet(
            root / "news" / "news_clusters.parquet",
            columns=["cluster_id", *(f"{a}_relevant_earned_at" for a in ASSETS)],
        )
        self.bars = {
            a: pl.read_parquet(root / "market" / f"{SYMBOLS[a]}_1h.parquet", columns=BAR_COLUMNS)
            for a in ASSETS
        }
        self.releases = pl.read_parquet(root / "calendar" / "macro_releases.parquet")
        self.families = sorted(self.releases["family"].unique().to_list())

    def market(
        self, asset: str, cutoff: datetime, *, forming_bar: bool = False
    ) -> tuple[pl.DataFrame, float, bool]:
        """Bars in the 168-hour window, their completeness and the missing mark."""
        start = cutoff - MARKET_HISTORY
        ended = (
            pl.col("open_time") <= cutoff if forming_bar else pl.col("open_time") + BAR <= cutoff
        )
        bars = self.bars[asset].filter((pl.col("open_time") >= start) & ended).sort("open_time")
        first = start.replace(minute=0, second=0, microsecond=0)
        if first < start:
            first += BAR
        expected = (
            pl.datetime_range(first, cutoff - BAR, "1h", time_unit="us", eager=True).len()
            if first <= cutoff - BAR
            else 0
        )
        completeness = bars.height / expected
        return bars, completeness, completeness < MIN_COMPLETENESS

    def news(
        self, cutoff: datetime, *, lifetime_flags: bool = False
    ) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
        """Visible assignments, visible cluster states and the distinct-cluster counts."""
        visible = self.records.filter(pl.col("available_at") <= cutoff)
        membership = visible.select("record_id", "cluster_id")
        states = (
            visible.group_by("cluster_id", maintain_order=True)
            .agg(
                pl.len().alias("n_members"),
                pl.col("source_domain").drop_nulls().n_unique().alias("n_domains"),
                pl.col("available_at").max().alias("last_available_at"),
            )
            .join(self.flags, on="cluster_id", how="left", maintain_order="left")
            .with_columns(
                (
                    pl.col(f"{a}_relevant_earned_at").is_not_null()
                    if lifetime_flags
                    else (pl.col(f"{a}_relevant_earned_at") + GKG_DELAY <= cutoff).fill_null(False)
                ).alias(a)
                for a in ASSETS
            )
        )
        counts = {
            f"{a}.{hours}h": states.filter(
                pl.col(a) & (pl.col("last_available_at") > cutoff - hours * BAR)
            ).height
            for a in ASSETS
            for hours in WINDOWS_H
        }
        return membership, states.select("cluster_id", "n_members", "n_domains", *ASSETS), counts

    def calendar(self, cutoff: datetime, *, ignore_schedule_time: bool = False) -> pl.DataFrame:
        """Per family, the latest past release and the earliest future one known at the cutoff."""
        known = self.releases.filter(
            pl.lit(True) if ignore_schedule_time else pl.col("schedule_available_at") <= cutoff
        ).sort("released_at", "release_id")
        last = (
            known.filter(pl.col("released_at") <= cutoff)
            .group_by("family")
            .agg(
                pl.col("release_id").last().alias("last_id"),
                pl.col("released_at").last().alias("last_at"),
            )
        )
        upcoming = (
            known.filter(pl.col("released_at") > cutoff)
            .group_by("family")
            .agg(
                pl.col("release_id").first().alias("next_id"),
                pl.col("released_at").first().alias("next_at"),
            )
        )
        return (
            pl.DataFrame({"family": self.families})
            .join(last, on="family", how="left")
            .join(upcoming, on="family", how="left")
            .select(list(CALENDAR_SCHEMA))
            .cast(CALENDAR_SCHEMA)
            .sort("family")
        )

    def snapshot(self, cutoff: datetime) -> Snapshot:
        market, completeness, missing = {}, {}, {}
        for a in ASSETS:
            market[a], completeness[a], missing[a] = self.market(a, cutoff)
        membership, clusters, counts = self.news(cutoff)
        return Snapshot(
            market, completeness, missing, membership, clusters, counts, self.calendar(cutoff)
        )
