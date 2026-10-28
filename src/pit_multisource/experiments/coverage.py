"""Table 7: coverage state of the news window before each hourly decision time.

Keys of the returned dict are `coverage.<window>.<filter>.<field>` with window `1h` or `24h` and
filter `all`, `btc` or `eth`. Fields: `hours` (windows kept), `active`, `confirmed_quiet`,
`unavailable`, `partially_covered`, `partially_covered.with_records`,
`partially_covered.without_records` and `excluded_leading` (windows dropped at the start).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from pit_multisource.config import load_config
from pit_multisource.pit.windows import (
    ACTIVE,
    CONFIRMED_QUIET,
    GRID_START,
    PARTIALLY_COVERED,
    UNAVAILABLE,
    window_states,
)

WINDOW_HOURS = (1, 24)
FILTERS = {"all": None, "btc": "btc_relevant", "eth": "eth_relevant"}
STATES = {
    "active": ACTIVE,
    "confirmed_quiet": CONFIRMED_QUIET,
    "unavailable": UNAVAILABLE,
    "partially_covered": PARTIALLY_COVERED,
}


def run_coverage(
    dataset_root: Path,
    boundary: datetime | None = None,
    grid_start: datetime = GRID_START,
) -> dict[str, int]:
    """Window counts per window length and record filter."""
    boundary = boundary or load_config().targets.boundary
    news = Path(dataset_root) / "news"
    slots = pl.read_parquet(news / "gdelt_slots.parquet", columns=["slot_time", "status"])
    records = pl.read_parquet(
        news / "news_records.parquet", columns=["slot_time", "btc_relevant", "eth_relevant"]
    )
    decision_hours = int((boundary - grid_start).total_seconds() // 3600) - 1
    counts: dict[str, int] = {}
    for hours in WINDOW_HOURS:
        for name, flag in FILTERS.items():
            states = window_states(
                slots, records, hours, flag, boundary=boundary, grid_start=grid_start
            )
            key = f"coverage.{hours}h.{name}"
            counts[f"{key}.hours"] = states.height
            counts[f"{key}.excluded_leading"] = decision_hours - states.height
            for field, state in STATES.items():
                counts[f"{key}.{field}"] = states.filter(pl.col("status") == state).height
            partial = states.filter(pl.col("status") == PARTIALLY_COVERED)
            with_records = partial.filter(pl.col("count_n") > 0).height
            counts[f"{key}.partially_covered.with_records"] = with_records
            counts[f"{key}.partially_covered.without_records"] = partial.height - with_records
    return counts
