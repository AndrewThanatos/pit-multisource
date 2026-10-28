"""What was knowable at one decision time from the market, news and calendar channels.

The article's worked example (§4.5) at 2023-10-18 12:00 UTC. A bar is usable once closed
(`close_time <= H`); a news record once its 15-minute slot has ended (`slot_time + 15 min <= H`),
the one-hour window keeping those that became available in (H − 1 h, H]; a future release only if
its schedule was known (`schedule_available_at <= H`).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from pit_multisource.commands import register

DECISION_TIME = datetime(2023, 10, 18, 12, tzinfo=UTC)
SLOT = timedelta(minutes=15)
WINDOW = timedelta(hours=1)


def point_in_time(dataset_root: Path, at: datetime = DECISION_TIME) -> dict[str, object]:
    """The last closed BTC bar, the one-hour news window and the releases around `at`."""
    root = Path(dataset_root)
    bar = (
        pl.scan_parquet(root / "market" / "btcusdt_1h.parquet")
        .filter(pl.col("close_time") <= at)
        .sort("close_time")
        .last()
        .collect()
        .row(0, named=True)
    )
    available_at = pl.col("slot_time") + SLOT
    news = (
        pl.scan_parquet(root / "news" / "news_records.parquet")
        .filter((available_at <= at) & (available_at > at - WINDOW))
        .select("cluster_id", "btc_relevant")
        .collect()
    )
    calendar = pl.read_parquet(root / "calendar" / "macro_releases.parquet")
    last = calendar.filter(pl.col("released_at") <= at).sort("released_at").row(-1, named=True)
    upcoming = (
        calendar.filter((pl.col("released_at") > at) & (pl.col("schedule_available_at") <= at))
        .sort("released_at")
        .row(0, named=True)
    )
    return {
        "example.decision_time": at.isoformat(),
        "example.bar.open_time": bar["open_time"].isoformat(),
        "example.bar.close_time": bar["close_time"].isoformat(timespec="milliseconds"),
        "example.bar.close": bar["close"],
        "example.news.records": news.height,
        "example.news.btc_relevant": int(news["btc_relevant"].sum()),
        "example.news.clusters": news["cluster_id"].n_unique(),
        "example.last_release.id": last["release_id"],
        "example.last_release.released_at": last["released_at"].isoformat(),
        "example.next_release.id": upcoming["release_id"],
        "example.next_release.family": upcoming["family"],
        "example.next_release.released_at": upcoming["released_at"].isoformat(),
        "example.next_release.date": upcoming["released_at"].date().isoformat(),
    }


def _run(args: argparse.Namespace) -> int:
    for key, value in point_in_time(args.dataset).items():
        print(f"{key.removeprefix('example.')}: {value}")
    return 0


@register
def _register_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "example", help="what was knowable at 2023-10-18 12:00 UTC (the article's example)"
    )
    parser.add_argument("--dataset", type=Path, required=True, help="released dataset directory")
    parser.set_defaults(func=_run)
