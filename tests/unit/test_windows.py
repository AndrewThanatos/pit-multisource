from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from pit_multisource.experiments.coverage import run_coverage
from pit_multisource.pit.windows import window_states

START = datetime(2024, 1, 1, tzinfo=UTC)
SLOT = timedelta(minutes=15)
UTC_US = pl.Datetime("us", "UTC")


def _slots(statuses: dict[int, str], count: int = 12) -> pl.DataFrame:
    """A contiguous grid of `count` slots; unlisted positions are retained."""
    return pl.DataFrame(
        {
            "slot_time": [START + i * SLOT for i in range(count)],
            "status": [statuses.get(i, "retained") for i in range(count)],
        },
        schema={"slot_time": UTC_US, "status": pl.String},
    )


def _records(positions: list[tuple[int, bool, bool]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "slot_time": [START + i * SLOT for i, _, _ in positions],
            "btc_relevant": [b for _, b, _ in positions],
            "eth_relevant": [e for _, _, e in positions],
        },
        schema={"slot_time": UTC_US, "btc_relevant": pl.Boolean, "eth_relevant": pl.Boolean},
    )


def _states(slots, records, hours=1, flag=None, count=12) -> dict[datetime, str]:
    frame = window_states(
        slots, records, hours, flag, boundary=START + count * SLOT, grid_start=START
    )
    return dict(zip(frame["decision_time"], frame["status"], strict=True))


def _at(hour: int) -> datetime:
    return START + timedelta(hours=hour)


def test_status_priority():
    # window 1 (slots 0-3): nothing served beats a missing slot and records
    # window 2 (slots 4-7): one served slot among missing beats records being present
    # window 3 (slots 8-11) is dropped as the boundary-ending one
    slots = _slots(
        {
            0: "missing_source",
            1: "missing_source",
            2: "missing_source",
            3: "missing_source",
            4: "missing_source",
            5: "missing_source",
            6: "missing_source",
        }
    )
    states = _states(slots, _records([(7, True, True), (8, True, True)]))
    assert states == {_at(1): "UNAVAILABLE", _at(2): "PARTIALLY_COVERED"}


def test_partly_covered_when_any_slot_missing():
    slots = _slots({5: "missing_source"})
    states = _states(slots, _records([(0, True, False)]))
    assert states[_at(1)] == "ACTIVE"
    assert states[_at(2)] == "PARTIALLY_COVERED"


def test_confirmed_quiet_needs_all_slots_served():
    slots = _slots({}, count=12)
    # served_no_match counts as served; no records at all in window 2
    slots = slots.with_columns(
        pl.when(pl.col("slot_time") == START + 5 * SLOT)
        .then(pl.lit("served_no_match"))
        .otherwise(pl.col("status"))
        .alias("status")
    )
    states = _states(slots, _records([(0, True, False)]))
    assert states == {_at(1): "ACTIVE", _at(2): "CONFIRMED_QUIET"}


def test_flag_filters_count_only_flagged_records():
    slots = _slots({})
    records = _records([(0, True, False), (4, False, True)])
    assert _states(slots, records, flag="btc_relevant") == {
        _at(1): "ACTIVE",
        _at(2): "CONFIRMED_QUIET",
    }
    assert _states(slots, records, flag="eth_relevant") == {
        _at(1): "CONFIRMED_QUIET",
        _at(2): "ACTIVE",
    }
    assert _states(slots, records) == {_at(1): "ACTIVE", _at(2): "ACTIVE"}


def test_leading_incomplete_windows_dropped_for_long_horizon():
    slots = _slots({}, count=4 * 25)
    frame = window_states(
        slots, _records([]), 24, None, boundary=START + 100 * SLOT, grid_start=START
    )
    # decision hours 1..25 exist; 1..23 lack a full 24 h window; 25 closes on the boundary
    assert frame["decision_time"].to_list() == [_at(24)]


def test_unknown_status_is_refused():
    with pytest.raises(ValueError, match="status"):
        _states(_slots({3: "weird"}), _records([]))


def test_gap_in_grid_is_refused():
    slots = _slots({}).filter(pl.col("slot_time") != START + 3 * SLOT)
    with pytest.raises(ValueError, match="grid"):
        _states(slots, _records([]))


def test_slots_at_or_after_boundary_are_ignored():
    slots = _slots({}, count=16)
    frame = window_states(
        slots, _records([]), 1, None, boundary=START + 12 * SLOT, grid_start=START
    )
    assert frame["decision_time"].max() == _at(2)


def test_run_coverage_keys(tmp_path):
    news = tmp_path / "news"
    news.mkdir()
    slots = _slots({}, count=4 * 30)
    slots.write_parquet(news / "gdelt_slots.parquet")
    _records([(0, True, True)]).write_parquet(news / "news_records.parquet")
    counts = run_coverage(tmp_path, boundary=START + 120 * SLOT, grid_start=START)
    assert counts["coverage.1h.all.hours"] == 29
    assert counts["coverage.24h.eth.hours"] == 6
    assert counts["coverage.1h.all.active"] == 1
    assert counts["coverage.1h.all.confirmed_quiet"] == 28
    assert counts["coverage.1h.all.partially_covered.with_records"] == 0
    assert counts["coverage.24h.all.excluded_leading"] == 23
