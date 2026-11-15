"""Target admission: synthetic cases through the target build, the Table 8 count and the reader.

One rule decides admission everywhere: a target is admitted when its label time, or for an
undefined target its nominal ceiling `t + h + 390 s`, is strictly before the boundary. Each case
builds 4 h targets from a few one-minute bars with the configured boundary and checks that the
Table 8 count (`eligible`) and the outcome reader (`read_outcomes`) both admit it exactly when
expected, that an undefined target stays undefined rather than becoming a zero return, that a bar
opening at or after the boundary is never used, and that the reader refuses a key outside the
admitted set and returns exactly the admitted rows.

Keys: `admission.cases`, `admission.passed` and `admission.<case>` (True when the case passed).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from pit_multisource.build.targets import OUTCOME_KEY, build_targets, eligible, read_outcomes
from pit_multisource.config import Config, TargetsConfig

S = timedelta(seconds=1)
M = timedelta(minutes=1)
H = timedelta(hours=1)
US = timedelta(microseconds=1)
SYMBOL = "BTCUSDT"


def _targets(points: list[tuple[datetime, float]], config: TargetsConfig) -> pl.DataFrame:
    bars = pl.DataFrame(
        points, schema={"open_time": pl.Datetime("us", "UTC"), "open": pl.Float64}, orient="row"
    )
    return build_targets(bars, SYMBOL, config, config.boundary)


def _key(decision_time: datetime) -> pl.DataFrame:
    return pl.DataFrame(
        {"symbol": [SYMBOL], "decision_time": [decision_time], "horizon": ["4h"]},
        schema={
            "symbol": pl.String,
            "decision_time": pl.Datetime("us", "UTC"),
            "horizon": pl.String,
        },
    )


def _admission_case(
    config: TargetsConfig,
    points: list[tuple[datetime, float]],
    t: datetime,
    admit: bool,
    defined: bool,
) -> bool:
    """Both consumers agree with `admit`; the build defines the target exactly when `defined`."""
    targets = _targets(points, config)
    row = targets.join(_key(t), on=list(OUTCOME_KEY), how="semi")
    counted = eligible(row, config, config.boundary).height == 1
    try:
        read = read_outcomes(targets, config, config.boundary, _key(t))
        returned = read.height == 1
        stays_undefined = read["log_return"].null_count() == read.height or defined
    except PermissionError:
        returned, stays_undefined = False, True
    is_defined = row["log_return"].null_count() == 0
    return counted == admit and returned == admit and is_defined == defined and stays_undefined


def run_admission(config: Config) -> dict[str, int | bool]:
    """Runs every case against the configured boundary."""
    c = config.targets
    b = c.boundary
    early, late = b - 5 * H, b - 4 * H
    far = b - 30 * 24 * H
    defined_early = [(early, 100.0), (early + 30 * S, 100.0)]
    cases = {
        "label_well_before_the_boundary": _admission_case(
            c, [*defined_early, (early + 4 * H + 30 * S, 110.0)], early, True, True
        ),
        "label_after_the_boundary": _admission_case(
            c, [*defined_early, (b - 30 * S, 110.0)], early, False, True
        ),
        "label_exactly_on_the_boundary": _admission_case(
            c, [*defined_early, (b - M, 110.0)], early, False, True
        ),
        "label_one_microsecond_before_the_boundary": _admission_case(
            c, [*defined_early, (b - M - US, 110.0)], early, True, True
        ),
        "exit_bar_at_the_boundary_is_never_read": _admission_case(
            c, [(late, 100.0), (late + 30 * S, 100.0), (b + 30 * S, 110.0)], late, False, False
        ),
        "undefined_ceiling_after_the_boundary": _admission_case(
            c, [(late, 100.0), (late + 400 * S, 100.0)], late, False, False
        ),
        "undefined_far_from_the_boundary_stays_undefined": _admission_case(
            c, [(far, 100.0), (far + 400 * S, 100.0)], far, True, False
        ),
    }

    targets = _targets([(far, 100.0), (far + 400 * S, 100.0), *defined_early, (b - M, 110.0)], c)
    try:
        read_outcomes(targets, c, b, _key(early))
        cases["stray_key_is_refused"] = False
    except PermissionError:
        cases["stray_key_is_refused"] = True
    admitted = eligible(targets, c, b).select(*OUTCOME_KEY, "log_return", "label_availability_time")
    cases["reader_returns_exactly_the_admitted_rows"] = read_outcomes(targets, c, b).equals(
        admitted
    )

    return {
        "admission.cases": len(cases),
        "admission.passed": sum(cases.values()),
        **{f"admission.{name}": passed for name, passed in cases.items()},
    }
