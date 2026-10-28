import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from pit_multisource.build.targets import TARGET_SCHEMA, build_targets, eligible
from pit_multisource.config import TargetsConfig, load_config
from pit_multisource.experiments.targets import run_targets

B = datetime(2025, 8, 1, tzinfo=UTC)
T = datetime(2024, 3, 1, 12, tzinfo=UTC)
S = timedelta(seconds=1)
M = timedelta(minutes=1)
H = timedelta(hours=1)


def _config(*horizons: int) -> TargetsConfig:
    return TargetsConfig(
        execution_latency_seconds=30,
        max_entry_delay_seconds=300,
        horizon_hours=horizons,
        boundary=B,
    )


def _bars(points: list[tuple[datetime, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        points,
        schema={"open_time": pl.Datetime("us", "UTC"), "open": pl.Float64},
        orient="row",
    )


def _build(points: list[tuple[datetime, float]], *horizons: int) -> pl.DataFrame:
    """Builds with the boundary one minute after the last bar, so no row is cut short."""
    bars = _bars(points)
    return build_targets(bars, "BTCUSDT", _config(*horizons), bars["open_time"].max() + M)


def _row(targets: pl.DataFrame, decision_time: datetime, horizon: str) -> dict:
    rows = targets.filter(
        (pl.col("decision_time") == decision_time) & (pl.col("horizon") == horizon)
    ).to_dicts()
    assert len(rows) == 1
    return rows[0]


def test_entry_boundary_inclusive():
    targets = _build(
        [(T, 100.0), (T + 29 * S, 101.0), (T + 30 * S, 102.0), (T + 4 * H + 30 * S, 110.0)], 4
    )

    row = _row(targets, T, "4h")
    assert row["entry_ref_time"] == T + 30 * S
    assert row["entry_ref_price"] == 102.0
    assert row["exit_ref_time"] == T + 4 * H + 30 * S
    assert row["log_return"] == math.log(110.0 / 102.0)
    assert row["label_availability_time"] == T + 4 * H + 90 * S
    assert row["reason"] is None


def test_entry_delay_300s_allowed_301s_refused():
    allowed = _row(
        _build([(T, 100.0), (T + 330 * S, 101.0), (T + 4 * H + 330 * S, 105.0)], 4), T, "4h"
    )
    refused = _row(
        _build([(T, 100.0), (T + 331 * S, 101.0), (T + 4 * H + 331 * S, 105.0)], 4), T, "4h"
    )

    assert allowed["entry_ref_time"] == T + 330 * S
    assert allowed["exit_ref_time"] == T + 4 * H + 330 * S
    assert allowed["reason"] is None
    assert refused["reason"] == "entry_too_late"
    for column in ("entry_ref_time", "entry_ref_price", "exit_ref_time", "log_return"):
        assert refused[column] is None


def test_exit_from_actual_entry():
    targets = _build(
        [(T, 100.0), (T + 5 * M, 100.0), (T + 4 * H + M, 200.0), (T + 4 * H + 5 * M, 300.0)], 4
    )

    row = _row(targets, T, "4h")
    assert row["entry_ref_time"] == T + 5 * M
    assert row["exit_ref_time"] == T + 4 * H + 5 * M
    assert row["exit_ref_price"] == 300.0
    assert row["target_close_time"] == row["exit_ref_time"]
    assert row["label_availability_time"] == T + 4 * H + 6 * M
    assert row["log_return"] == math.log(3.0)
    assert row["simple_return"] == math.exp(math.log(3.0)) - 1


def test_output_schema_and_order():
    targets = _build([(T, 100.0), (T + M, 101.0), (T + 25 * H, 102.0)], 4, 24)

    assert targets.columns == [
        "symbol",
        "decision_time",
        "horizon",
        "execution_latency",
        "entry_ref_time",
        "exit_ref_time",
        "entry_ref_price",
        "exit_ref_price",
        "log_return",
        "simple_return",
        "target_close_time",
        "label_availability_time",
        "target_version",
        "reason",
    ]
    assert targets.schema == pl.Schema({**TARGET_SCHEMA, "reason": pl.String})
    assert targets.height == 26 * 2
    assert targets["horizon"].head(2).to_list() == ["24h", "4h"]
    assert targets["execution_latency"].unique().to_list() == [30.0]
    assert targets["target_version"].unique().to_list() == ["v3"]


def _minute_bars(start: datetime, end: datetime) -> list[tuple[datetime, float]]:
    count = (end - start) // M
    return [(start + i * M, 100.0 + i / 100) for i in range(count)]


def test_no_bar_at_or_after_boundary_is_read():
    bars = _minute_bars(B - 24 * H, B)
    sentinels = [(B, 1.0), (B + 24 * H + M, 2.0)]
    config = _config(4, 24)

    clean = build_targets(_bars(bars), "BTCUSDT", config, B)
    with_sentinels = build_targets(_bars(bars + sentinels), "BTCUSDT", config, B)
    sentinels_read = build_targets(_bars(bars + sentinels), "BTCUSDT", config, B + 24 * H + 2 * M)

    assert not sentinels_read.equals(clean)
    assert with_sentinels.equals(clean)
    assert clean["decision_time"].max() < B


def test_eligible_row_without_exit_bar_is_refused():
    bars = _minute_bars(B - 48 * H, B - 24 * H)

    with pytest.raises(ValueError, match="no_exit_bar"):
        build_targets(_bars(bars), "BTCUSDT", _config(4), B)


def test_eligibility_coalesce_rule():
    cases = [
        # (case, decision_time, horizon, label_availability_time, admitted)
        ("label before the boundary", B - 30 * H, "4h", B - 26 * H + 2 * M, True),
        ("label after the boundary", B - H, "4h", B + 3 * H + 2 * M, False),
        ("label exactly on the boundary", B - 4 * H - 2 * M, "4h", B, False),
        ("no label, ceiling past the boundary", B - 2 * H, "4h", None, False),
        ("no label, far from the boundary", datetime(2025, 1, 1, tzinfo=UTC), "4h", None, True),
        ("no label, 24h ceiling before the boundary", B - 25 * H, "24h", None, True),
        ("no label, 24h ceiling past the boundary", B - 24 * H, "24h", None, False),
        ("no label, ceiling exactly on the boundary", B - 4 * H - 390 * S, "4h", None, False),
    ]
    targets = pl.DataFrame(
        [c[1:4] for c in cases],
        schema={
            "decision_time": pl.Datetime("us", "UTC"),
            "horizon": pl.String,
            "label_availability_time": pl.Datetime("us", "UTC"),
        },
        orient="row",
    )

    admitted = eligible(targets, _config(4, 24), B)

    expected = targets.filter(pl.Series([c[4] for c in cases]))
    assert admitted.equals(expected)


def test_run_targets_counts(tmp_path):
    bars = [
        point
        for point in _minute_bars(B - 48 * H, B)
        if not B - 38 * H <= point[0] < B - 38 * H + 7 * M
    ]
    (tmp_path / "market").mkdir()
    for name in ("btcusdt", "ethusdt"):
        _bars(bars).write_parquet(tmp_path / "market" / f"{name}_1m.parquet")

    counts = run_targets(tmp_path, load_config())

    assert counts["BTCUSDT.4h.candidates"] == 48
    assert counts["BTCUSDT.4h.eligible"] == 44
    assert counts["BTCUSDT.4h.defined"] == 43
    assert counts["BTCUSDT.4h.missing.entry_too_late"] == 1
    assert counts["ETHUSDT.24h.eligible"] == 24
    assert counts["ETHUSDT.24h.defined"] == 23
    assert counts["ETHUSDT.24h.defined_pct"] == round(100 * 23 / 24, 3)
    assert counts["total.candidates"] == 4 * 48
    assert counts["total.missing"] == 4
