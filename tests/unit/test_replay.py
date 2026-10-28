import hashlib
from datetime import UTC, datetime, timedelta

import polars as pl

from pit_multisource.config import load_config
from pit_multisource.experiments.replay import (
    compare_assignments,
    cutoffs,
    detector_controls,
    state_digest,
    state_digests,
)

_US = pl.Datetime("us", "UTC")
T0 = datetime(2023, 10, 18, 11, 0, tzinfo=UTC)
MICRO = timedelta(microseconds=1)
SLOT = timedelta(minutes=15)
EMPTY = "0" * 32


def _clusters(rows: list[tuple[str, datetime, datetime | None, datetime | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "cluster_id": pl.String(),
            "opened_at": _US,
            "btc_relevant_earned_at": _US,
            "eth_relevant_earned_at": _US,
        },
        orient="row",
    )


def _us(dt: datetime) -> int:
    return (dt - datetime(1970, 1, 1, tzinfo=UTC)) // MICRO


def test_cutoffs_are_seeded_and_207():
    config = load_config()
    got = cutoffs(config)
    assert len(got) == 207
    assert got == sorted(got)
    assert got == cutoffs(config)
    assert set(config.replay.boundary_cutoffs) <= set(got)
    drawn = [c for c in got if c not in config.replay.boundary_cutoffs]
    assert [sum(c.year == y for c in drawn) for y in range(2019, 2026)] == [28] * 7
    assert all(config.replay.first_cutoff <= c < config.replay.period_end for c in drawn)
    # Draws of the published run: the first, one in 2022 and the last.
    assert got[4] == datetime(2019, 1, 25, 3, 35, 36, 122839, tzinfo=UTC)
    assert datetime(2022, 8, 26, 4, 33, 43, 180486, tzinfo=UTC) in got
    assert got[-2] == datetime(2025, 7, 30, 23, 20, 19, 545091, tzinfo=UTC)


def test_digest_is_the_xor_of_blake2b_row_hashes():
    clusters = _clusters([("a", T0, T0, None), ("b", T0, None, T0 + SLOT)])
    o = _us(T0)
    rows = [f"a|{o}|{o}|", f"b|{o}||{_us(T0 + SLOT)}"]
    expected = 0
    for row in rows:
        expected ^= int.from_bytes(hashlib.blake2b(row.encode(), digest_size=16).digest(), "big")
    assert state_digest(clusters, T0 + 2 * SLOT) == f"{expected:032x}"


def test_digest_ignores_clusters_not_yet_visible():
    a = ("a", T0, None, None)
    both = _clusters([a, ("b", T0 + SLOT, None, None)])
    only_a = _clusters([a])
    assert state_digest(_clusters([]), T0 + 3 * SLOT) == EMPTY
    assert state_digest(only_a, T0 + SLOT - MICRO) == EMPTY
    assert state_digest(only_a, T0 + SLOT) != EMPTY
    assert state_digest(both, T0 + 2 * SLOT - MICRO) == state_digest(only_a, T0 + 2 * SLOT - MICRO)
    assert state_digest(both, T0 + 2 * SLOT) != state_digest(only_a, T0 + 2 * SLOT)


def test_digest_shows_a_flag_from_the_end_of_its_earning_slot():
    flagged = _clusters([("a", T0, T0 + SLOT, None)])
    unflagged = _clusters([("a", T0, None, None)])
    assert state_digest(flagged, T0 + 2 * SLOT - MICRO) == state_digest(
        unflagged, T0 + 2 * SLOT - MICRO
    )
    assert state_digest(flagged, T0 + 2 * SLOT) != state_digest(unflagged, T0 + 2 * SLOT)


def test_negative_control_detects_lifetime_flags():
    late = _clusters([("a", T0, T0 + 4 * SLOT, None), ("b", T0, None, None)])
    instants = [T0 + SLOT, T0 + 5 * SLOT - MICRO, T0 + 5 * SLOT]
    dated = state_digests(late, instants)
    lifetime = state_digests(late, instants, lifetime=True)
    assert [d == x for d, x in zip(dated, lifetime, strict=True)] == [False, False, True]

    at_opening = _clusters([("a", T0, T0, T0)])
    assert state_digests(at_opening, instants) == state_digests(at_opening, instants, lifetime=True)


def test_detector_controls_fire():
    controls = detector_controls()
    assert controls == {
        f"controls.{case}": True
        for case in (
            "clean_baseline",
            "record_removed_from_the_rebuild",
            "identifier_swapped_at_equal_row_count",
            "cluster_changed_for_one_record",
            "key_duplicated_in_the_rebuild",
            "cluster_id_nulled_in_the_rebuild",
        )
    }


def test_compare_assignments_counts_each_defect_by_name():
    base = pl.DataFrame({"record_id": ["r1", "r2", "r3"], "cluster_id": ["c1", "c1", "c2"]})
    assert compare_assignments(base, base.clone())["identical"] is True

    swapped = pl.DataFrame({"record_id": ["r1", "r2", "r9"], "cluster_id": ["c1", "c1", "c2"]})
    counts = compare_assignments(base, swapped)
    assert (counts["keys_only_in_left"], counts["keys_only_in_right"]) == (1, 1)
    assert counts["identical"] is False

    nulled = pl.DataFrame({"record_id": ["r1", "r2", "r3"], "cluster_id": ["c1", None, "c9"]})
    counts = compare_assignments(base, nulled)
    assert counts["null_cluster_id_right"] == 1
    assert counts["shared_keys_with_a_one_sided_null"] == 1
    assert counts["shared_keys_disagreeing"] == 1


def test_distinct_clusters_are_counted_per_side_over_its_own_ids():
    left = pl.DataFrame({"record_id": ["r1", "r2", "r3"], "cluster_id": ["c1", "c2", "c3"]})
    right = pl.DataFrame({"record_id": ["r1", "r2", "r4"], "cluster_id": ["c1", "c1", None]})
    counts = compare_assignments(left, right)
    assert counts["distinct_clusters_left"] == 3
    assert counts["distinct_clusters_right"] == 1
