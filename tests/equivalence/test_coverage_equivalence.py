"""Table 7 counts rebuilt from the released slots and news records."""

from pathlib import Path

import pytest

from pit_multisource.experiments.coverage import run_coverage

pytestmark = pytest.mark.equivalence

# window, filter -> hours, active, confirmed quiet, unavailable, partially covered
TABLE_7 = {
    ("1h", "all"): (57695, 56093, 86, 1044, 472),
    ("1h", "btc"): (57695, 56088, 91, 1044, 472),
    ("1h", "eth"): (57695, 45531, 10648, 1044, 472),
    ("24h", "all"): (57672, 53932, 0, 403, 3337),
}


def test_table_7_counts(dataset_root: Path):
    counts = run_coverage(dataset_root)
    for (window, name), expected in TABLE_7.items():
        key = f"coverage.{window}.{name}"
        got = tuple(
            counts[f"{key}.{field}"]
            for field in ("hours", "active", "confirmed_quiet", "unavailable", "partially_covered")
        )
        assert got == expected, key
    assert counts["coverage.1h.all.partially_covered.with_records"] == 447
    assert counts["coverage.1h.all.partially_covered.without_records"] == 25
    assert counts["coverage.1h.all.excluded_leading"] == 0
    assert counts["coverage.24h.all.excluded_leading"] == 23
    for field in ("active", "confirmed_quiet", "unavailable", "partially_covered"):
        assert counts[f"coverage.24h.btc.{field}"] == counts[f"coverage.24h.all.{field}"]
        assert counts[f"coverage.24h.eth.{field}"] == counts[f"coverage.24h.all.{field}"]
