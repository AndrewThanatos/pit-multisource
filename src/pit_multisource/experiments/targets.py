"""Table 8: how many 4 h / 24 h targets of the diagnostic period are eligible and defined."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pit_multisource.build.targets import (
    ENTRY_TOO_LATE,
    NO_ENTRY_BAR,
    NO_EXIT_BAR,
    bars_path,
    build_targets,
    eligible,
    read_bars,
)
from pit_multisource.config import Config

REASONS = (ENTRY_TOO_LATE, NO_ENTRY_BAR, NO_EXIT_BAR)


def _percent(part: int, whole: int) -> float:
    return round(100 * part / whole, 3)


def run_targets(dataset_root: Path, config: Config) -> dict[str, int | float]:
    """Candidates, eligible, defined and missing targets per symbol and horizon, plus totals."""
    boundary = config.targets.boundary
    counts: dict[str, int | float] = {}
    totals = dict.fromkeys(("candidates", "eligible", "defined", "missing"), 0)
    for symbol in config.bars.symbols:
        bars = read_bars(bars_path(dataset_root, symbol), boundary)
        targets = build_targets(bars, symbol, config.targets, boundary)
        admitted = eligible(targets, config.targets, boundary)
        for hours in config.targets.horizon_hours:
            horizon = f"{hours}h"
            cell = admitted.filter(pl.col("horizon") == horizon)
            row = {
                "candidates": targets.filter(
                    (pl.col("horizon") == horizon) & (pl.col("decision_time") < boundary)
                ).height,
                "eligible": cell.height,
                "defined": cell.filter(pl.col("exit_ref_time").is_not_null()).height,
            }
            row["missing"] = row["eligible"] - row["defined"]
            key = f"{symbol}.{horizon}"
            for name, value in row.items():
                counts[f"{key}.{name}"] = value
                totals[name] += value
            for reason in REASONS:
                counts[f"{key}.missing.{reason}"] = cell.filter(pl.col("reason") == reason).height
            counts[f"{key}.defined_pct"] = _percent(row["defined"], row["eligible"])
    for name, value in totals.items():
        counts[f"total.{name}"] = value
    counts["total.defined_pct"] = _percent(totals["defined"], totals["eligible"])
    return counts
