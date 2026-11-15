"""Full-input invariance: X(u, c) from the prefix route against the projection route.

At the 207 replay cutoffs and both assets, route A (`inputs_prefix`) builds the input from what
is available at the cutoff and route B (`inputs_projection`) restricts the released tables, later
records included, to the cutoff. Each component is compared field by field: a field mismatch is
a value that differs where both routes have one; a missingness mismatch is an item or value
present in one route only (a bar, a record, a cluster, a release, the market missing mark).
Three faults injected into route B check that the comparison would see a violation: final flags
shown from the cluster's opening, the forming bar admitted, a release admitted before its
schedule was known. For each, `changed` counts inputs the fault alters against clean route B and
`detected` counts inputs where the faulty route B disagrees with route A.

Keys: `inputs.cutoffs`, `inputs.compared` (asset x cutoff), `inputs.<component>.<count>` with
component `market`, `news`, `calendar`, `indicators` or `complete` and count `compared`,
`field_mismatches`, `missingness_mismatches` or `equal`; `inputs.rows.*` (items route A produced
over all cutoffs); `inputs.fault.<fault>.changed` and `.detected`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pit_multisource.config import Config
from pit_multisource.experiments.inputs_prefix import snapshots
from pit_multisource.experiments.inputs_projection import Projection
from pit_multisource.experiments.replay import cutoffs
from pit_multisource.pit.inputs import ASSETS, WINDOWS_H

COMPONENTS = ("market", "news", "calendar", "indicators", "complete")
FAULTS = ("final_relevance", "forming_bar", "unknown_schedule")
_Diff = tuple[bool, bool]


def diff(left: pl.DataFrame, right: pl.DataFrame, key: str) -> _Diff:
    """(a value differs where both have one, an item or value is present on one side only)."""
    if left.equals(right):
        return False, False
    joined = left.join(right, on=key, how="full", suffix="_r", coalesce=False)
    one_sided = joined.filter(pl.col(key).is_null() | pl.col(f"{key}_r").is_null()).height > 0
    both = joined.filter(pl.col(key).is_not_null() & pl.col(f"{key}_r").is_not_null())
    fields = [c for c in left.columns if c != key]
    values = both.select(
        pl.any_horizontal(
            pl.col(c).is_not_null()
            & pl.col(f"{c}_r").is_not_null()
            & (pl.col(c) != pl.col(f"{c}_r"))
            for c in fields
        ).any()
    ).item()
    nulls = both.select(
        pl.any_horizontal(pl.col(c).is_null() != pl.col(f"{c}_r").is_null() for c in fields).any()
    ).item()
    return bool(values), bool(one_sided or nulls)


def _either(*diffs: _Diff) -> _Diff:
    return any(d[0] for d in diffs), any(d[1] for d in diffs)


def _market(bars_a: pl.DataFrame, bars_b: pl.DataFrame) -> _Diff:
    return diff(bars_a, bars_b, "open_time")


def _indicators(a: tuple[float, bool], b: tuple[float, bool]) -> _Diff:
    return a[0] != b[0], a[1] != b[1]


def _news(asset: str, a: tuple, b: tuple) -> _Diff:
    """Assignments, the asset's cluster states and its 1 h / 24 h counts."""
    membership_a, clusters_a, counts_a = a
    membership_b, clusters_b, counts_b = b
    columns = ["cluster_id", "n_members", "n_domains", asset]
    counts = any(counts_a[f"{asset}.{h}h"] != counts_b[f"{asset}.{h}h"] for h in WINDOWS_H)
    return _either(
        diff(membership_a, membership_b, "record_id"),
        diff(clusters_a.select(columns), clusters_b.select(columns), "cluster_id"),
        (counts, False),
    )


def _changed(d: _Diff) -> bool:
    return d[0] or d[1]


def run_inputs(dataset_root: Path, config: Config) -> dict[str, int]:
    """Route agreement per component and the three injected faults, over the 207 cutoffs."""
    instants = cutoffs(config)
    projection = Projection(dataset_root)
    tally = {f"{c}.{n}": 0 for c in COMPONENTS for n in ("field", "missing", "equal")}
    faults = {f"{f}.{n}": 0 for f in FAULTS for n in ("changed", "detected")}
    rows = dict.fromkeys(
        ("market_bars", "record_assignments", "cluster_states", "calendar_rows"), 0
    )

    for cutoff, a in zip(instants, snapshots(dataset_root, config, instants), strict=True):
        b = projection.snapshot(cutoff)
        news_a = (a.membership, a.clusters, a.counts)
        news_b = (b.membership, b.clusters, b.counts)
        news_final = projection.news(cutoff, lifetime_flags=True)
        calendar = diff(a.calendar, b.calendar, "family")
        calendar_unknown = projection.calendar(cutoff, ignore_schedule_time=True)
        rows["record_assignments"] += a.membership.height
        rows["cluster_states"] += a.clusters.height
        rows["calendar_rows"] += a.calendar.height

        for asset in ASSETS:
            rows["market_bars"] += a.market[asset].height
            forming, rho, missing = projection.market(asset, cutoff, forming_bar=True)
            components = {
                "market": _market(a.market[asset], b.market[asset]),
                "news": _news(asset, news_a, news_b),
                "calendar": calendar,
                "indicators": _indicators(
                    (a.completeness[asset], a.market_missing[asset]),
                    (b.completeness[asset], b.market_missing[asset]),
                ),
            }
            components["complete"] = _either(*components.values())
            for name, (field, missing_) in components.items():
                tally[f"{name}.field"] += field
                tally[f"{name}.missing"] += missing_
                tally[f"{name}.equal"] += not (field or missing_)

            injected = {
                "final_relevance": (
                    _news(asset, news_final, news_b),
                    _news(asset, news_final, news_a),
                ),
                "forming_bar": (
                    _either(
                        _market(forming, b.market[asset]),
                        _indicators(
                            (rho, missing), (b.completeness[asset], b.market_missing[asset])
                        ),
                    ),
                    _either(
                        _market(forming, a.market[asset]),
                        _indicators(
                            (rho, missing), (a.completeness[asset], a.market_missing[asset])
                        ),
                    ),
                ),
                "unknown_schedule": (
                    diff(calendar_unknown, b.calendar, "family"),
                    diff(calendar_unknown, a.calendar, "family"),
                ),
            }
            for fault, (against_clean, against_prefix) in injected.items():
                faults[f"{fault}.changed"] += _changed(against_clean)
                faults[f"{fault}.detected"] += _changed(against_prefix)

    compared = len(instants) * len(ASSETS)
    out = {"inputs.cutoffs": len(instants), "inputs.compared": compared}
    for name in COMPONENTS:
        out[f"inputs.{name}.compared"] = compared
        out[f"inputs.{name}.field_mismatches"] = tally[f"{name}.field"]
        out[f"inputs.{name}.missingness_mismatches"] = tally[f"{name}.missing"]
        out[f"inputs.{name}.equal"] = tally[f"{name}.equal"]
    out |= {f"inputs.rows.{k}": v for k, v in rows.items()}
    out |= {f"inputs.fault.{k}": v for k, v in faults.items()}
    return out
