"""When built news facts become knowable."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

# GDELT publishes one archive per 15-minute slot; a slot's records are knowable at its end.
SLOT_LENGTH = timedelta(minutes=15)


def flags_state_as_of(clusters: pl.DataFrame, instant: datetime) -> pl.DataFrame:
    """Each cluster's asset flags as knowable at `instant`, from the end of the earning slot."""
    if instant.tzinfo is None:
        raise ValueError(f"instant must be timezone-aware, got {instant!r}")
    when = pl.lit(instant.astimezone(UTC), dtype=pl.Datetime("us", "UTC"))
    return clusters.select(
        "cluster_id",
        "opened_at",
        *(
            (pl.col(f"{asset}_relevant_earned_at") + SLOT_LENGTH <= when)
            .fill_null(False)
            .alias(f"{asset}_relevant")
            for asset in ("btc", "eth")
        ),
    )
