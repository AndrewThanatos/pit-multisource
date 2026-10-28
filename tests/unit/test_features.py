import random
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from pit_multisource.experiments.features import (
    Convention,
    clusters_24h,
    compare,
    hour_grid,
    presence_segments,
)

_US = pl.Datetime("us", "UTC")
FIRST = datetime(2019, 1, 2, tzinfo=UTC)
BOUNDARY = datetime(2019, 1, 8, tzinfo=UTC)
HOUR = timedelta(hours=1)
DAY = timedelta(hours=24)
SLOT = timedelta(minutes=15)


def _records(rows: list[tuple[str, datetime]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"cluster_id": pl.String(), "slot_time": _US}, orient="row")


def _clusters(rows: list[tuple[str, datetime | None]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"cluster_id": pl.String(), "earned_at": _US}, orient="row")


def test_segment_split_at_24h_gap():
    t0 = datetime(2019, 1, 2, 3, 0, tzinfo=UTC)
    records = _records(
        [
            ("c1", t0),
            ("c1", t0),
            ("c1", t0 + DAY),
            ("c1", t0 + 2 * DAY + 2 * HOUR),
        ]
    )
    segments = presence_segments(records, Convention(clock="slot_end")).sort("v_lo")
    assert segments["v_lo"].to_list() == [t0 + SLOT, t0 + 2 * DAY + 2 * HOUR + SLOT]
    assert segments["pres_end"].to_list() == [t0 + 2 * DAY + SLOT, t0 + 3 * DAY + 2 * HOUR + SLOT]
    count = clusters_24h(
        records,
        _clusters([("c1", t0)]),
        Convention(clock="slot_end", flag="lifetime"),
        first_hour=FIRST,
        boundary=BOUNDARY,
    )
    hours = hour_grid(FIRST, BOUNDARY)
    by_hour = dict(zip(hours.to_list(), count.to_list(), strict=True))
    # The hours between the two segments have no member in the last 24 h.
    assert by_hour[t0 + 2 * DAY] == 1
    assert by_hour[t0 + 2 * DAY + HOUR] == 0
    assert by_hour[t0 + 2 * DAY + 2 * HOUR] == 0
    assert by_hour[t0 + 2 * DAY + 3 * HOUR] == 1
    assert max(count.to_list()) == 1


def test_changed_hours_threshold_is_one():
    series = pl.Series([3, 4, 2, 3, 5], dtype=pl.Int64)
    reference = pl.Series([3, 3, 3, 3, 5], dtype=pl.Int64)
    result = compare(series, reference)
    assert result["changed_hours"] == 2
    assert result["share"] == pytest.approx(2 / 5)
    assert result["mean_abs_diff"] == pytest.approx(2 / 5)
    assert result["negative_hours"] == 1


def test_flag_population_excludes_unflagged_and_late_earners():
    t0 = datetime(2019, 1, 3, tzinfo=UTC)
    records = _records([("a", t0), ("b", t0), ("c", t0)])
    clusters = _clusters([("a", t0), ("b", None), ("c", BOUNDARY)])
    count = clusters_24h(
        records,
        clusters,
        Convention(clock="slot_end", flag="lifetime"),
        first_hour=FIRST,
        boundary=BOUNDARY,
    )
    assert max(count.to_list()) == 1


def _closed_form(records, clusters, convention, hours):
    shift = SLOT if convention.clock == "slot_end" else timedelta(0)
    earned = {c: e for c, e in clusters.iter_rows() if e is not None and e < BOUNDARY}
    out = []
    for h in hours:
        present = set()
        for c, slot in records.iter_rows():
            vis = slot + shift
            if c in earned and h - DAY < vis <= h:
                if convention.flag == "lifetime" or earned[c] + shift <= h:
                    present.add(c)
        out.append(len(present))
    return out


@pytest.mark.parametrize("clock", ["slot_start", "slot_end"])
@pytest.mark.parametrize("flag", ["lifetime", "dated"])
def test_sweep_matches_closed_form(clock: str, flag: str):
    rng = random.Random(7)
    start = datetime(2019, 1, 1, 12, tzinfo=UTC)
    slots = [start + SLOT * rng.randrange(0, 4 * 24 * 7) for _ in range(120)]
    records = _records([(f"c{rng.randrange(12)}", s) for s in slots])
    first_slot = records.group_by("cluster_id").agg(pl.col("slot_time").min())
    clusters = _clusters(
        [
            (c, None if i % 5 == 0 else s + SLOT * rng.randrange(0, 4 * 48))
            for i, (c, s) in enumerate(sorted(first_slot.iter_rows()))
        ]
    )
    convention = Convention(clock=clock, flag=flag)
    hours = hour_grid(FIRST, BOUNDARY).to_list()
    got = clusters_24h(records, clusters, convention, first_hour=FIRST, boundary=BOUNDARY)
    assert got.to_list() == _closed_form(records, clusters, convention, hours)


def test_convention_rejects_unknown_names():
    with pytest.raises(ValueError):
        Convention(clock="slot_middle")
    with pytest.raises(ValueError):
        Convention(flag="final")
