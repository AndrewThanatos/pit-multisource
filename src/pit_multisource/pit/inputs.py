"""The historical input X(u, c) of the article (Section 3.4) as both input routes produce it.

Market: the hourly bars of asset u whose interval has ended by c, `open_time + 1 h <= c`, and
started within the last 168 h, with their completeness against the whole hours of that window
and the missing mark below 0.90. News: the cluster of every record available by c, each visible
cluster's member count, distinct known source domains and dated asset flags, and per asset the
1 h and 24 h distinct-cluster counts (Eq. 24). Calendar: per release family, the latest past
release and the earliest future release whose schedule is known by c. Retrospective archive
coverage is diagnostic metadata, not an input (Section 3.4), and no optional numeric channel is
implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import polars as pl

ASSETS = ("btc", "eth")
SYMBOLS = {"btc": "btcusdt", "eth": "ethusdt"}
WINDOWS_H = (1, 24)
MARKET_HISTORY = timedelta(hours=168)
BAR = timedelta(hours=1)
MIN_COMPLETENESS = 0.90
# A GKG record becomes available at the end of its 15-minute slot.
GKG_DELAY = timedelta(minutes=15)
BAR_COLUMNS = (
    "open_time", "open", "high", "low", "close", "base_volume", "close_time", "quote_volume",
    "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume",
)  # fmt: skip
_US = pl.Datetime("us", "UTC")
CLUSTER_SCHEMA = {
    "cluster_id": pl.String(),
    "n_members": pl.UInt32(),
    "n_domains": pl.UInt32(),
    **{a: pl.Boolean() for a in ASSETS},
}
CALENDAR_SCHEMA = {
    "family": pl.String(),
    "last_id": pl.String(),
    "last_at": _US,
    "next_id": pl.String(),
    "next_at": _US,
}


@dataclass(frozen=True)
class Snapshot:
    """X(u, c) for both assets at one cutoff; `counts` is keyed `<asset>.<hours>h`."""

    market: dict[str, pl.DataFrame]
    completeness: dict[str, float]
    market_missing: dict[str, bool]
    membership: pl.DataFrame
    clusters: pl.DataFrame
    counts: dict[str, int]
    calendar: pl.DataFrame
