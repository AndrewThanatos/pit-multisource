"""Author-only check: rebuilt targets equal the frozen research targets on the diagnostic period.

Only rows whose exposure ends before the boundary are read from the frozen files: their keys are
selected from clock columns first, and the remaining columns are read for those keys only. No
value is printed or stored; the report holds counts, equality flags and file digests.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from pit_multisource.build.targets import (
    TARGET_COLUMNS,
    bars_path,
    build_targets,
    eligible,
    exposure_end,
    read_bars,
)
from pit_multisource.config import TargetsConfig, load_config

pytestmark = pytest.mark.equivalence

KEY = ["symbol", "decision_time", "horizon"]
CLOCKS = [*KEY, "label_availability_time"]
REPORT = Path(__file__).resolve().parents[2] / "equivalence" / "targets_report.json"
RESTRICTION = (
    "Frozen rows are compared only when their exposure ends strictly before the boundary: "
    "coalesce(label_availability_time, decision_time + horizon + 390 s) < boundary. The keys are "
    "selected from the clock columns first; the other columns are read for those keys only."
)


@pytest.fixture
def frozen_root() -> Path:
    value = os.environ.get("PIT_FROZEN_TARGETS")
    if not value:
        pytest.skip("PIT_FROZEN_TARGETS is not set")
    return Path(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restricted_read(
    path: Path, config: TargetsConfig, boundary: datetime
) -> tuple[pl.DataFrame, dict[str, int | bool]]:
    """Frozen rows of the admitted keys, plus clock-only counts that check the restriction."""
    end = exposure_end(config)
    keys = pl.scan_parquet(path).select(CLOCKS).filter(end < boundary).select(KEY).collect()
    rows = pl.scan_parquet(path).join(keys.lazy(), on=KEY, how="semi").collect()
    extra = [c for c in rows.columns if c not in TARGET_COLUMNS]
    extra_constant = all(rows[c].n_unique() == 1 for c in extra)
    rows = rows.select(TARGET_COLUMNS)
    checks = (
        pl.scan_parquet(path)
        .join(keys.lazy(), on=KEY, how="semi")
        .select(
            (end >= boundary).sum().alias("admitted_rows_at_or_past_boundary"),
            (pl.col("exit_ref_time") >= boundary).sum().alias("admitted_exits_at_or_past_boundary"),
        )
        .collect()
        .row(0, named=True)
    )
    outside = (
        pl.scan_parquet(path)
        .select(*CLOCKS, "exit_ref_time")
        .filter((pl.col("exit_ref_time") < boundary) & (end >= boundary))
        .select(pl.len())
        .collect()
        .item()
    )
    checks["filled_rows_exiting_before_boundary_not_compared"] = outside
    checks["frozen_only_columns"] = len(extra)
    if rows.height != keys.height:
        raise AssertionError("the frozen read returned a different row count than the keys")
    if checks["admitted_rows_at_or_past_boundary"] or checks["admitted_exits_at_or_past_boundary"]:
        raise AssertionError("an admitted row reaches the boundary")
    if not extra_constant:
        raise AssertionError("a frozen column outside the public schema is not constant")
    counts = {name: int(value) for name, value in checks.items()}
    return rows, {**counts, "frozen_only_columns_constant": extra_constant}


def test_targets_equal_frozen_on_diagnostic_period(dataset_root: Path, frozen_root: Path):
    config = load_config()
    boundary = config.targets.boundary
    report: dict = {
        "boundary": boundary.isoformat().replace("+00:00", "Z"),
        "restriction_rule": RESTRICTION,
        "comparison": (
            "polars DataFrame.equals on the 13 public table columns (same dtypes, exact floats), "
            "rows sorted by decision_time within each horizon; frozen columns outside the public "
            "schema are not compared, only checked to hold one distinct value each"
        ),
        "rebuild": "build_targets from the released one-minute bars with open_time < boundary",
        "symbols": {},
    }
    all_equal = True
    for symbol in config.bars.symbols:
        path = frozen_root / symbol / "targets.parquet"
        frozen, checks = _restricted_read(path, config.targets, boundary)
        rebuilt = eligible(
            build_targets(
                read_bars(bars_path(dataset_root, symbol), boundary),
                symbol,
                config.targets,
                boundary,
            ),
            config.targets,
            boundary,
        ).select(TARGET_COLUMNS)
        entry: dict = {"frozen_file": f"{symbol}/targets.parquet", "frozen_sha256": _sha256(path)}
        entry.update(checks)
        horizons = {}
        for hours in config.targets.horizon_hours:
            horizon = f"{hours}h"
            built = rebuilt.filter(pl.col("horizon") == horizon).sort("decision_time")
            reference = frozen.filter(pl.col("horizon") == horizon).sort("decision_time")
            equal = bool(built.schema == reference.schema and built.equals(reference))
            horizons[horizon] = {
                "rows_compared": reference.height,
                "rebuilt_eligible_rows": built.height,
                "equal": equal,
            }
            all_equal = all_equal and equal
        entry["horizons"] = horizons
        report["symbols"][symbol] = entry
    report["equal"] = all_equal

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert all_equal, f"see {REPORT.name}"
