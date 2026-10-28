from pathlib import Path

import polars as pl
import pytest

from pit_multisource.build.bars import build_bars, months, zips_for
from pit_multisource.config import load_config

pytestmark = pytest.mark.equivalence

CASES = [(symbol, interval) for symbol in ("BTCUSDT", "ETHUSDT") for interval in ("1h", "1m")]


@pytest.mark.parametrize(("symbol", "interval"), CASES)
def test_bars_equal_dataset(symbol: str, interval: str, sources_root: Path, dataset_root: Path):
    config = load_config()
    zips = zips_for(sources_root / "binance" / symbol / interval, symbol, interval, months(config))
    published = pl.read_parquet(dataset_root / "market" / f"{symbol.lower()}_{interval}.parquet")

    rebuilt = build_bars(zips)

    assert rebuilt.equals(published)
