"""Builds hourly and one-minute bar tables from Binance monthly kline archives."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import polars as pl

from pit_multisource.commands import command_group, register
from pit_multisource.config import Config, load_config

INTERVALS = ("1h", "1m")

_US = pl.Datetime("us", "UTC")

BAR_SCHEMA: dict[str, pl.DataType] = {
    "open_time": _US,
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "base_volume": pl.Float64(),
    "close_time": _US,
    "quote_volume": pl.Float64(),
    "trade_count": pl.Int64(),
    "taker_buy_base_volume": pl.Float64(),
    "taker_buy_quote_volume": pl.Float64(),
}

_RAW_SCHEMA: dict[str, pl.DataType] = {
    "open_time": pl.Int64(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "close_time": pl.Int64(),
    "quote_volume": pl.Float64(),
    "count": pl.Int64(),
    "taker_buy_volume": pl.Float64(),
    "taker_buy_quote_volume": pl.Float64(),
    "ignore": pl.String(),
}

_RENAMES = {
    "volume": "base_volume",
    "count": "trade_count",
    "taker_buy_volume": "taker_buy_base_volume",
}


def months(config: Config) -> list[str]:
    """Every month from the configured first to the configured last, as YYYY-MM."""
    first, last = config.bars.months_first, config.bars.months_last
    out = []
    year, month = int(first[:4]), int(first[5:])
    while f"{year:04d}-{month:02d}" <= last:
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def zips_for(root: Path, symbol: str, interval: str, month_list: list[str]) -> list[Path]:
    """The archives of the listed months in `root`, in order; other files are ignored."""
    paths = [root / f"{symbol}-{interval}-{month}.zip" for month in month_list]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing archives in {root}: {missing[:3]} ({len(missing)} total)")
    return paths


def _epoch_unit(value: int) -> str:
    """Milliseconds in [1e11, 1e14), microseconds in [1e14, 1e17); anything else is an error."""
    if 10**11 <= value < 10**14:
        return "ms"
    if 10**14 <= value < 10**17:
        return "us"
    raise ValueError(f"epoch value {value} is neither milliseconds nor microseconds")


def _to_utc(column: str, unit: str) -> pl.Expr:
    return (
        pl.from_epoch(pl.col(column), time_unit=unit)  # type: ignore[arg-type]
        .dt.replace_time_zone("UTC")
        .dt.cast_time_unit("us")
    )


def parse_klines_zip(path: Path) -> pl.DataFrame:
    """Parses one monthly archive; each time column's epoch unit is read from its own minimum."""
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{path.name}: expected exactly one CSV member, found {names}")
        data = archive.read(names[0])
    if not data.split(b",", 1)[0].strip().isdigit():
        raise ValueError(f"{path.name}: unexpected header row")
    raw = pl.read_csv(
        io.BytesIO(data),
        has_header=False,
        new_columns=list(_RAW_SCHEMA),
        schema_overrides=_RAW_SCHEMA,
    )
    open_min, open_max = raw["open_time"].min(), raw["open_time"].max()
    close_min = raw["close_time"].min()
    assert isinstance(open_min, int) and isinstance(open_max, int) and isinstance(close_min, int)
    open_unit = _epoch_unit(open_min)
    if open_unit != _epoch_unit(open_max):
        raise ValueError(f"{path.name}: mixed epoch units")
    return (
        raw.with_columns(
            _to_utc("open_time", open_unit),
            _to_utc("close_time", _epoch_unit(close_min)),
        )
        .drop("ignore")
        .rename(_RENAMES)
        .sort("open_time")
    )


def build_bars(zips: list[Path]) -> pl.DataFrame:
    """Concatenates archives in the given order; for a repeated open_time the later row wins."""
    frames = [parse_klines_zip(path) for path in zips]
    return (
        pl.concat(frames)
        .with_row_index("_seq")
        .sort(["open_time", "_seq"])
        .unique(subset=["open_time"], keep="last", maintain_order=True)
        .select(list(BAR_SCHEMA))
    )


def _run(args: argparse.Namespace) -> int:
    config = load_config()
    args.out.mkdir(parents=True, exist_ok=True)
    for symbol in config.bars.symbols:
        for interval in INTERVALS:
            root = args.sources / "binance" / symbol / interval
            bars = build_bars(zips_for(root, symbol, interval, months(config)))
            target = args.out / f"{symbol.lower()}_{interval}.parquet"
            bars.write_parquet(target)
            print(f"{target.name}: {bars.height} rows")
    return 0


@register
def _register_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = command_group(subparsers, "build").add_parser(
        "bars", help="hourly and one-minute spot bars from the Binance archives"
    )
    parser.add_argument("--sources", type=Path, required=True, help="raw sources directory")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.set_defaults(func=_run)
