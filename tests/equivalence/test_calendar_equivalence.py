from pathlib import Path

import polars as pl
import pytest

from pit_multisource.build.calendar import parse_calendar
from pit_multisource.config import load_config

pytestmark = pytest.mark.equivalence


def test_calendar_equals_dataset(sources_root: Path, dataset_root: Path):
    published = pl.read_parquet(dataset_root / "calendar" / "macro_releases.parquet")

    rebuilt = parse_calendar(sources_root / "calendar", load_config())

    assert published.height == 482
    assert rebuilt.equals(published)
