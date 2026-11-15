"""Route A of the input check: the input built from what is available at each cutoff.

Every source is consumed once, in the order its items become available, and the state is
captured at each cutoff before any later item is applied: a GKG record is clustered when its
slot ends, an hourly bar is added when its interval ends, a release becomes known at
`schedule_available_at` and past at `released_at`. The released news tables are not read, so the
cluster states come from the GKG records alone.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from pit_multisource.build.news import NEWS_COLUMNS, read_gkg_records, stream_records
from pit_multisource.config import Config
from pit_multisource.pit.inputs import (
    ASSETS,
    BAR,
    BAR_COLUMNS,
    CALENDAR_SCHEMA,
    CLUSTER_SCHEMA,
    GKG_DELAY,
    MARKET_HISTORY,
    MIN_COMPLETENESS,
    SYMBOLS,
    WINDOWS_H,
    Snapshot,
)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_US_PER_HOUR = 3_600_000_000


def _us(instant: datetime) -> int:
    return (instant - _EPOCH) // timedelta(microseconds=1)


def expected_bars(cutoff: datetime) -> int:
    """Whole UTC hours whose bar starts within the last 168 h and has ended by the cutoff."""
    first = -(-_us(cutoff - MARKET_HISTORY) // _US_PER_HOUR)
    last = (_us(cutoff - BAR)) // _US_PER_HOUR
    return max(last - first + 1, 0)


class _News:
    """Clusters records as they become available and keeps the per-cluster state."""

    def __init__(self, records: pl.DataFrame, config: Config) -> None:
        self._delay = _us(_EPOCH + GKG_DELAY)
        self._stream = stream_records(records, config)
        self._pending = next(self._stream, None)
        self.record_ids: list[str] = []
        self.record_clusters: list[str] = []
        self.index: dict[str, int] = {}
        self.ids: list[str] = []
        self.members: list[int] = []
        self.domains: list[set[str]] = []
        self.flag: dict[str, list[bool]] = {a: [] for a in ASSETS}
        self.last_available: list[int] = []

    def advance_to(self, cutoff_us: int) -> None:
        while self._pending is not None and self._pending.slot_us + self._delay <= cutoff_us:
            record = self._pending
            k = self.index.get(record.cluster_id)
            if k is None:
                k = self.index[record.cluster_id] = len(self.ids)
                self.ids.append(record.cluster_id)
                self.members.append(0)
                self.domains.append(set())
                for a in ASSETS:
                    self.flag[a].append(False)
                self.last_available.append(0)
            self.record_ids.append(record.record_id)
            self.record_clusters.append(record.cluster_id)
            self.members[k] += 1
            if record.source_domain is not None:
                self.domains[k].add(record.source_domain)
            self.flag["btc"][k] |= record.btc_relevant
            self.flag["eth"][k] |= record.eth_relevant
            self.last_available[k] = record.slot_us + self._delay
            self._pending = next(self._stream, None)

    def state(self, cutoff_us: int) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
        membership = pl.DataFrame(
            {"record_id": self.record_ids, "cluster_id": self.record_clusters},
            schema={"record_id": pl.String(), "cluster_id": pl.String()},
        )
        clusters = pl.DataFrame(
            {
                "cluster_id": self.ids,
                "n_members": self.members,
                "n_domains": [len(d) for d in self.domains],
                **{a: self.flag[a] for a in ASSETS},
                "last_available_us": self.last_available,
            },
            schema={**CLUSTER_SCHEMA, "last_available_us": pl.Int64()},
        )
        counts = {
            f"{a}.{hours}h": clusters.filter(
                pl.col(a) & (pl.col("last_available_us") > cutoff_us - hours * _US_PER_HOUR)
            ).height
            for a in ASSETS
            for hours in WINDOWS_H
        }
        return membership, clusters.drop("last_available_us"), counts


class _Bars:
    """The hourly bars of one symbol whose interval has ended, kept for the last 168 hours."""

    def __init__(self, bars: pl.DataFrame) -> None:
        bars = bars.select(BAR_COLUMNS).sort("open_time")
        self._schema = bars.schema
        self._rows = bars.iter_rows()
        self._pending = next(self._rows, None)
        self._window: deque[tuple] = deque()

    def advance_to(self, cutoff: datetime) -> None:
        while self._pending is not None and self._pending[0] + BAR <= cutoff:
            self._window.append(self._pending)
            self._pending = next(self._rows, None)
        while self._window and self._window[0][0] < cutoff - MARKET_HISTORY:
            self._window.popleft()

    def state(self) -> pl.DataFrame:
        return pl.DataFrame(list(self._window), schema=self._schema, orient="row")


class _Calendar:
    """Releases become known at their schedule time and past at their release time."""

    def __init__(self, releases: pl.DataFrame) -> None:
        events = []
        for row in releases.iter_rows(named=True):
            events.append((row["schedule_available_at"], 0, row["release_id"], row))
            events.append((row["released_at"], 1, row["release_id"], row))
        events.sort(key=lambda e: e[:3])
        self._events = iter(events)
        self._pending = next(self._events, None)
        self._families = sorted(releases["family"].unique().to_list())
        self._known: dict[str, dict[str, datetime]] = {f: {} for f in self._families}
        self._last: dict[str, tuple[datetime, str]] = {}

    def advance_to(self, cutoff: datetime) -> None:
        while self._pending is not None and self._pending[0] <= cutoff:
            _, kind, release_id, row = self._pending
            family = row["family"]
            if kind == 0:
                self._known[family][release_id] = row["released_at"]
            else:
                self._known[family].pop(release_id)
                latest = (row["released_at"], release_id)
                self._last[family] = max(self._last.get(family, latest), latest)
            self._pending = next(self._events, None)

    def state(self) -> pl.DataFrame:
        rows = []
        for family in self._families:
            last = self._last.get(family)
            upcoming = min(((at, rid) for rid, at in self._known[family].items()), default=None)
            rows.append(
                {
                    "family": family,
                    "last_id": None if last is None else last[1],
                    "last_at": None if last is None else last[0],
                    "next_id": None if upcoming is None else upcoming[1],
                    "next_at": None if upcoming is None else upcoming[0],
                }
            )
        return pl.DataFrame(rows, schema=CALENDAR_SCHEMA)


def snapshots(
    dataset_root: Path, config: Config, instants: Sequence[datetime]
) -> Iterator[Snapshot]:
    """The input at each cutoff, in ascending order, from one pass over every source."""
    if list(instants) != sorted(instants):
        raise ValueError("cutoffs must be ascending")
    root = Path(dataset_root)
    end = config.replay.period_end
    records = read_gkg_records(root, NEWS_COLUMNS).filter(pl.col("slot_time") < end)
    news = _News(records, config)
    bars = {a: _Bars(pl.read_parquet(root / "market" / f"{SYMBOLS[a]}_1h.parquet")) for a in ASSETS}
    calendar = _Calendar(pl.read_parquet(root / "calendar" / "macro_releases.parquet"))
    for cutoff in instants:
        if cutoff >= end:
            raise ValueError(f"cutoff {cutoff} is not before the period end {end}")
        cutoff_us = _us(cutoff)
        news.advance_to(cutoff_us)
        calendar.advance_to(cutoff)
        market, completeness, missing = {}, {}, {}
        for a in ASSETS:
            bars[a].advance_to(cutoff)
            market[a] = bars[a].state()
            completeness[a] = market[a].height / expected_bars(cutoff)
            missing[a] = completeness[a] < MIN_COMPLETENESS
        membership, clusters, counts = news.state(cutoff_us)
        yield Snapshot(
            market, completeness, missing, membership, clusters, counts, calendar.state()
        )
