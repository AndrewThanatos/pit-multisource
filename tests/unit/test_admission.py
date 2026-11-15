from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from pit_multisource.build.targets import read_outcomes
from pit_multisource.config import TargetsConfig, load_config
from pit_multisource.experiments.admission import run_admission

B = datetime(2025, 8, 1, tzinfo=UTC)
H = timedelta(hours=1)
_US = pl.Datetime("us", "UTC")
CONFIG = TargetsConfig(
    execution_latency_seconds=30, max_entry_delay_seconds=300, horizon_hours=(4,), boundary=B
)


def _targets() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 3,
            "decision_time": [B - 30 * H, B - 20 * H, B - 2 * H],
            "horizon": ["4h"] * 3,
            "log_return": [0.01, None, 0.02],
            "label_availability_time": [B - 26 * H, None, B + H],
        },
        schema_overrides={"decision_time": _US, "label_availability_time": _US},
    )


def test_every_admission_case_passes():
    out = run_admission(load_config())
    assert out["admission.cases"] == 9
    assert out["admission.passed"] == 9


def test_reader_returns_the_admitted_rows_with_undefined_outcomes_left_null():
    read = read_outcomes(_targets(), CONFIG, B)
    assert read["decision_time"].to_list() == [B - 30 * H, B - 20 * H]
    assert read["log_return"].to_list() == [0.01, None]


def test_reader_refuses_a_key_that_is_not_admitted():
    keys = _targets().filter(pl.col("decision_time") == B - 2 * H)
    with pytest.raises(PermissionError, match="1 requested key"):
        read_outcomes(_targets(), CONFIG, B, keys)
