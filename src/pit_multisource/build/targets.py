"""Hourly 4 h / 24 h log-return targets from the one-minute bars that open before a boundary."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from pit_multisource.commands import command_group, register
from pit_multisource.config import TargetsConfig, load_config

TARGET_VERSION = "v3"
# A one-minute exit bar closes 60 s after it opens; its close is when the label becomes known.
LABEL_LAG = timedelta(seconds=60)
NO_ENTRY_BAR = "no_entry_bar"
ENTRY_TOO_LATE = "entry_too_late"
NO_EXIT_BAR = "no_exit_bar"

_TIME = pl.Datetime("us", "UTC")
TARGET_SCHEMA = pl.Schema(
    {
        "symbol": pl.String(),
        "decision_time": _TIME,
        "horizon": pl.String(),
        "execution_latency": pl.Float64(),
        "entry_ref_time": _TIME,
        "exit_ref_time": _TIME,
        "entry_ref_price": pl.Float64(),
        "exit_ref_price": pl.Float64(),
        "log_return": pl.Float64(),
        "simple_return": pl.Float64(),
        "target_close_time": _TIME,
        "label_availability_time": _TIME,
        "target_version": pl.String(),
    }
)
TARGET_COLUMNS = tuple(TARGET_SCHEMA.names())


def _hour_grid(open_times: pl.Series) -> pl.Series:
    """Every whole UTC hour from the first bar (rounded up) to the last bar (rounded down)."""
    first, last = open_times.min(), open_times.max()
    start = first.replace(minute=0, second=0, microsecond=0)
    if start != first:
        start += timedelta(hours=1)
    end = last.replace(minute=0, second=0, microsecond=0)
    return pl.datetime_range(start, end, "1h", time_unit="us", time_zone="UTC", eager=True).alias(
        "decision_time"
    )


def _entries(bars: pl.DataFrame, grid: pl.Series, config: TargetsConfig) -> pl.DataFrame:
    """The first bar at or after decision + latency; refused when it is more than the delay late."""
    latency = timedelta(seconds=config.execution_latency_seconds)
    max_delay = timedelta(seconds=config.max_entry_delay_seconds)
    entries = (
        grid.to_frame()
        .with_columns((pl.col("decision_time") + latency).alias("_earliest"))
        .sort("_earliest")
        .join_asof(
            bars.rename({"open_time": "entry_time", "open": "entry_price"}),
            left_on="_earliest",
            right_on="entry_time",
            strategy="forward",
        )
    )
    if not entries.filter(pl.col("entry_time") < pl.col("_earliest")).is_empty():
        raise AssertionError("an entry precedes decision_time + latency")
    return entries.with_columns(
        pl.when(pl.col("entry_time").is_null())
        .then(pl.lit(NO_ENTRY_BAR))
        .when(pl.col("entry_time") - pl.col("_earliest") > max_delay)
        .then(pl.lit(ENTRY_TOO_LATE))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("reason")
    ).with_columns(
        pl.when(pl.col("reason").is_null()).then(pl.col("entry_time")).alias("entry_time"),
        pl.when(pl.col("reason").is_null()).then(pl.col("entry_price")).alias("entry_price"),
    )


def _exits(bars: pl.DataFrame, entries: pl.DataFrame, hours: int) -> pl.DataFrame:
    """The first bar at or after the actual entry + horizon, for filled entries only."""
    filled = (
        entries.filter(pl.col("reason").is_null())
        .with_columns((pl.col("entry_time") + timedelta(hours=hours)).alias("_exit_target"))
        .sort("_exit_target")
        .join_asof(
            bars.rename({"open_time": "exit_time", "open": "exit_price"}),
            left_on="_exit_target",
            right_on="exit_time",
            strategy="forward",
        )
    )
    if not filled.filter(pl.col("exit_time") < pl.col("_exit_target")).is_empty():
        raise AssertionError("an exit precedes entry_time + horizon")
    filled = filled.with_columns(
        pl.when(pl.col("exit_time").is_null())
        .then(pl.lit(NO_EXIT_BAR))
        .otherwise(pl.col("reason"))
        .alias("reason"),
        (pl.col("exit_price") / pl.col("entry_price")).log().alias("log_return"),
    )
    unfilled = entries.filter(pl.col("reason").is_not_null()).with_columns(
        pl.lit(None, dtype=_TIME).alias("exit_time"),
        pl.lit(None, dtype=pl.Float64).alias("exit_price"),
        pl.lit(None, dtype=pl.Float64).alias("log_return"),
    )
    return pl.concat([filled.select(unfilled.columns), unfilled]).with_columns(
        pl.lit(f"{hours}h").alias("horizon")
    )


def build_targets(
    bars_1m: pl.DataFrame, symbol: str, config: TargetsConfig, boundary: datetime
) -> pl.DataFrame:
    """One row per decision hour and horizon, from the bars that open before the boundary only."""
    bars = (
        bars_1m.filter(pl.col("open_time") < boundary)
        .select("open_time", pl.col("open").cast(pl.Float64))
        .sort("open_time")
    )
    if bars.is_empty():
        raise ValueError(f"{symbol}: no bars open before {boundary.isoformat()}")
    grid = _hour_grid(bars["open_time"])
    entries = _entries(bars, grid, config)
    raw = pl.concat([_exits(bars, entries, hours) for hours in config.horizon_hours])
    if raw.height != grid.len() * len(config.horizon_hours):
        raise AssertionError("row count differs from decisions x horizons")
    if not raw.filter(pl.col("reason").is_null() == pl.col("log_return").is_null()).is_empty():
        raise AssertionError("a reason must be set exactly where the return is missing")

    targets = (
        raw.rename(
            {
                "entry_time": "entry_ref_time",
                "entry_price": "entry_ref_price",
                "exit_time": "exit_ref_time",
                "exit_price": "exit_ref_price",
            }
        )
        .with_columns(
            pl.lit(symbol).alias("symbol"),
            pl.lit(float(config.execution_latency_seconds)).alias("execution_latency"),
            (pl.col("log_return").exp() - 1).alias("simple_return"),
            pl.col("exit_ref_time").alias("target_close_time"),
            (pl.col("exit_ref_time") + LABEL_LAG).alias("label_availability_time"),
            pl.lit(TARGET_VERSION).alias("target_version"),
        )
        .sort("decision_time", "horizon")
        .select(*TARGET_COLUMNS, "reason")
    )
    cut_short = eligible(targets, config, boundary).filter(
        pl.col("reason").is_in([NO_ENTRY_BAR, NO_EXIT_BAR])
    )
    if not cut_short.is_empty():
        reasons = ", ".join(sorted(cut_short["reason"].unique().to_list()))
        raise ValueError(
            f"{cut_short.height} eligible rows have no bar ({reasons}): "
            "the bars end too early or a gap reaches the boundary"
        )
    return targets


def exposure_end(config: TargetsConfig) -> pl.Expr:
    """The label time, or for a row without one the latest label time its entry rule allows."""
    slack = (
        timedelta(seconds=config.execution_latency_seconds + config.max_entry_delay_seconds)
        + LABEL_LAG
    )
    hours = pl.col("horizon").str.strip_suffix("h").cast(pl.Int64)
    ceiling = pl.col("decision_time") + pl.duration(hours=hours) + slack
    return pl.col("label_availability_time").fill_null(ceiling)


def eligible(targets: pl.DataFrame, config: TargetsConfig, boundary: datetime) -> pl.DataFrame:
    """Rows whose exposure ends strictly before the boundary."""
    return targets.filter(exposure_end(config) < boundary)


def bars_path(dataset_root: Path, symbol: str) -> Path:
    return dataset_root / "market" / f"{symbol.lower()}_1m.parquet"


def read_bars(path: Path, boundary: datetime) -> pl.DataFrame:
    """Reads open_time and open of the bars that open before the boundary."""
    return (
        pl.scan_parquet(path)
        .select("open_time", "open")
        .filter(pl.col("open_time") < boundary)
        .collect()
    )


def _run(args: argparse.Namespace) -> int:
    config = load_config()
    boundary = config.targets.boundary
    for symbol in config.bars.symbols:
        bars = read_bars(bars_path(args.dataset, symbol), boundary)
        targets = build_targets(bars, symbol, config.targets, boundary)
        out = args.out / symbol / "targets.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        targets.write_parquet(out)
        print(f"{symbol}: {targets.height} rows, decisions before {boundary.isoformat()}")
    return 0


@register
def _register_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = command_group(subparsers, "build").add_parser(
        "targets", help="4 h / 24 h targets from the one-minute bars before the configured boundary"
    )
    parser.add_argument("--dataset", type=Path, required=True, help="released dataset directory")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.set_defaults(func=_run)
