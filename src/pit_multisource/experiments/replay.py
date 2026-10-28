"""Historical replay checks: cluster-state digests at seeded cutoffs and assignment comparison.

A cluster is visible at cutoff `c` when `opened_at + 15 min <= c`, and a flag is knowable when
`earned_at + 15 min <= c`. The state digest at `c` is the XOR, over visible clusters, of the
128-bit blake2b of `cluster_id|opened_us|btc_us|eth_us`, where a flag not yet knowable is empty.
Three routes are compared at every cutoff: the causal rebuild from the records before the period
end, the released cluster table under the dated flags, and the released table under lifetime
flags (each flag shown from the opening), which must disagree. Keys of the returned dict are
`replay.*` (cutoff counts, route agreement, rebuild size), `slice.*` (the first quarter of 2019
rebuilt on its own against the released assignment), `assignment.*` (the development assignment
comparison counters) and `controls.*` (the comparison on five planted defects and one clean copy).
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from pit_multisource.build.news import NEWS_COLUMNS, build_news, read_gkg_records
from pit_multisource.config import Config
from pit_multisource.pit.availability import SLOT_LENGTH, flags_state_as_of

SLICE_END = datetime(2019, 4, 1, tzinfo=UTC)
FLAG_COLUMNS = ("btc_relevant_earned_at", "eth_relevant_earned_at")
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MICRO = timedelta(microseconds=1)
_LOW64 = (1 << 64) - 1
# Row-hash variants by (btc shown, eth shown); the state code is btc + 2 * eth.
_SHOWN = ((False, False), (True, False), (False, True), (True, True))


def _us(instant: datetime) -> int:
    return (instant - _EPOCH) // _MICRO


def cutoffs(config: Config) -> list[datetime]:
    """Seeded random cutoffs per calendar year plus the fixed boundary cutoffs, ascending."""
    replay = config.replay
    rng = random.Random(replay.seed)
    drawn = []
    for year in range(replay.first_year, replay.last_year + 1):
        lo = max(_us(datetime(year, 1, 1, tzinfo=UTC)), _us(replay.first_cutoff))
        hi = min(_us(datetime(year + 1, 1, 1, tzinfo=UTC)), _us(replay.period_end)) - 1
        drawn += [_EPOCH + rng.randrange(lo, hi) * _MICRO for _ in range(replay.per_year)]
    return sorted([*drawn, *replay.boundary_cutoffs])


def _row_hash(cluster_id: str, opened: int, btc: int | None, eth: int | None) -> int:
    text = f"{cluster_id}|{opened}|{'' if btc is None else btc}|{'' if eth is None else eth}"
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=16).digest(), "big")


def _hash_table(clusters: pl.DataFrame) -> pl.DataFrame:
    """Per cluster, the high and low 64 bits of its row hash in each of the four flag states."""
    columns = clusters.select("cluster_id", pl.col("opened_at", *FLAG_COLUMNS).dt.epoch("us"))
    halves: dict[str, list[int]] = {f"h{s}_{p}": [] for s in range(4) for p in ("hi", "lo")}
    for cluster_id, opened, btc, eth in columns.iter_rows():
        for s, (show_btc, show_eth) in enumerate(_SHOWN):
            h = _row_hash(cluster_id, opened, btc if show_btc else None, eth if show_eth else None)
            halves[f"h{s}_hi"].append(h >> 64)
            halves[f"h{s}_lo"].append(h & _LOW64)
    return pl.DataFrame({k: pl.Series(v, dtype=pl.UInt64()) for k, v in halves.items()})


def state_digests(
    clusters: pl.DataFrame, instants: Sequence[datetime], *, lifetime: bool = False
) -> list[str]:
    """The state digest at each instant; with `lifetime` a flag is shown from the opening."""
    hashes = _hash_table(clusters)
    timing = clusters.select("cluster_id", "opened_at", *FLAG_COLUMNS)
    if lifetime:
        timing = timing.with_columns(
            pl.when(pl.col(c).is_not_null()).then(pl.col("opened_at")).alias(c)
            for c in FLAG_COLUMNS
        )
    digests = []
    for instant in instants:
        flags = flags_state_as_of(timing, instant)
        state = flags["btc_relevant"].cast(pl.UInt8) + 2 * flags["eth_relevant"].cast(pl.UInt8)
        visible = flags["opened_at"] + SLOT_LENGTH <= instant
        halves = (
            hashes.with_columns(state.alias("state"))
            .filter(visible)
            .select(
                pl.coalesce(
                    pl.when(pl.col("state") == s).then(pl.col(f"h{s}_{part}")) for s in range(4)
                )
                .bitwise_xor()
                .alias(part)
                for part in ("hi", "lo")
            )
            .row(0)
        )
        hi, lo = (h or 0 for h in halves)
        digests.append(f"{(hi << 64) | lo:032x}")
    return digests


def state_digest(clusters: pl.DataFrame, instant: datetime) -> str:
    """XOR of the row hashes of the clusters visible at `instant`, flags as knowable then."""
    return state_digests(clusters, [instant])[0]


def compare_assignments(left: pl.DataFrame, right: pl.DataFrame) -> dict[str, int | bool]:
    """Every way two (record_id, cluster_id) tables can differ, counted by name."""
    dup_left = left.height - left["record_id"].n_unique()
    dup_right = right.height - right["record_id"].n_unique()
    only_left = left.join(right, on="record_id", how="anti").height
    only_right = right.join(left, on="record_id", how="anti").height
    null_left = left["cluster_id"].null_count()
    null_right = right["cluster_id"].null_count()
    shared = left.join(right, on="record_id", how="inner", suffix="_right")
    both = shared.filter(
        pl.col("cluster_id").is_not_null() & pl.col("cluster_id_right").is_not_null()
    )
    disagree = both.filter(pl.col("cluster_id") != pl.col("cluster_id_right")).height
    one_sided_null = shared.height - both.height
    defects = (dup_left, dup_right, only_left, only_right, null_left, null_right)
    return {
        "rows_left": left.height,
        "rows_right": right.height,
        "duplicate_keys_left": dup_left,
        "duplicate_keys_right": dup_right,
        "keys_only_in_left": only_left,
        "keys_only_in_right": only_right,
        "shared_keys": shared.height,
        "null_cluster_id_left": null_left,
        "null_cluster_id_right": null_right,
        "shared_keys_with_a_one_sided_null": one_sided_null,
        "shared_keys_disagreeing": disagree,
        "distinct_clusters_left": left["cluster_id"].drop_nulls().n_unique(),
        "distinct_clusters_right": right["cluster_id"].drop_nulls().n_unique(),
        "identical": not any((*defects, one_sided_null, disagree)),
    }


def _replace(frame: pl.DataFrame, column: str, record_id: str, value: str | None) -> pl.DataFrame:
    hit = pl.col("record_id") == record_id
    return frame.with_columns(
        pl.when(hit).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)
    )


def detector_controls() -> dict[str, bool]:
    """Whether the comparison reports a clean copy as identical and catches five planted defects."""
    base = pl.DataFrame(
        {"record_id": ["r1", "r2", "r3", "r4"], "cluster_id": ["c1", "c1", "c2", "c3"]}
    )
    cases: dict[str, tuple[pl.DataFrame, tuple[str, ...]]] = {
        "clean_baseline": (base.clone(), ()),
        "record_removed_from_the_rebuild": (
            base.filter(pl.col("record_id") != "r3"),
            ("keys_only_in_left",),
        ),
        "identifier_swapped_at_equal_row_count": (
            _replace(base, "record_id", "r3", "r9"),
            ("keys_only_in_left", "keys_only_in_right"),
        ),
        "cluster_changed_for_one_record": (
            _replace(base, "cluster_id", "r2", "c7"),
            ("shared_keys_disagreeing",),
        ),
        "key_duplicated_in_the_rebuild": (
            pl.concat([base, base.filter(pl.col("record_id") == "r1")]),
            ("duplicate_keys_right",),
        ),
        "cluster_id_nulled_in_the_rebuild": (
            _replace(base, "cluster_id", "r4", None),
            ("null_cluster_id_right",),
        ),
    }
    result = {}
    for case, (right, expected) in cases.items():
        counts = compare_assignments(base, right)
        if expected:
            passed = all(counts[name] for name in expected) and not counts["identical"]
        else:
            passed = counts["identical"] is True
        result[f"controls.{case}"] = bool(passed)
    return result


def _assignments(records: pl.DataFrame) -> pl.DataFrame:
    return records.select("record_id", "cluster_id")


def released_clusters(dataset_root: Path, period_end: datetime) -> pl.DataFrame:
    """Released clusters opened before `period_end`, with flags earned from then on dropped."""
    return (
        pl.read_parquet(
            dataset_root / "news" / "news_clusters.parquet",
            columns=["cluster_id", "opened_at", *FLAG_COLUMNS],
        )
        .filter(pl.col("opened_at") < period_end)
        .with_columns(
            pl.when(pl.col(c) < period_end).then(pl.col(c)).alias(c) for c in FLAG_COLUMNS
        )
    )


def run_replay(dataset_root: Path, config: Config) -> dict[str, int | bool]:
    """Route agreement at every cutoff, the 2019 slice, the full assignment and the controls."""
    period_end = config.replay.period_end
    if period_end > config.targets.boundary:
        raise ValueError(f"period end {period_end} is after the boundary {config.targets.boundary}")
    news = Path(dataset_root) / "news"
    records = read_gkg_records(Path(dataset_root), NEWS_COLUMNS).filter(
        pl.col("slot_time") < period_end
    )
    released_records = pl.read_parquet(
        news / "news_records.parquet", columns=["record_id", "slot_time", "cluster_id"]
    ).filter(pl.col("slot_time") < period_end)
    released = released_clusters(Path(dataset_root), period_end)

    out: dict[str, int | bool] = {}
    instants = cutoffs(config)
    rebuilt = build_news(records, config)
    causal = state_digests(rebuilt.clusters, instants)
    out["replay.cutoffs"] = len(instants)
    out["replay.cutoffs.boundary"] = len(config.replay.boundary_cutoffs)
    for route, lifetime in (("dated", False), ("lifetime", True)):
        shipped = state_digests(released, instants, lifetime=lifetime)
        agree = sum(a == b for a, b in zip(causal, shipped, strict=True))
        out[f"replay.{route}.agree"] = agree
        out[f"replay.{route}.disagree"] = len(instants) - agree
    out["replay.rebuild.records"] = rebuilt.records.height
    out["replay.rebuild.clusters"] = rebuilt.clusters.height
    out["replay.rebuild.url_joins_refused_by_age_cap"] = rebuilt.url_joins_refused_by_age_cap

    in_slice = pl.col("slot_time") < SLICE_END
    piece = build_news(records.filter(in_slice), config)
    released_piece = _assignments(released_records.filter(in_slice))
    out["slice.records"] = piece.records.height
    out["slice.clusters.rebuilt"] = piece.clusters.height
    out["slice.clusters.released"] = released_piece["cluster_id"].n_unique()
    out["slice.assignments_equal"] = _assignments(piece.records).equals(released_piece)
    out["slice.url_joins_refused_by_age_cap"] = piece.url_joins_refused_by_age_cap

    comparison = compare_assignments(_assignments(released_records), _assignments(rebuilt.records))
    out |= {f"assignment.{k}": v for k, v in comparison.items()}
    return out | detector_controls()
