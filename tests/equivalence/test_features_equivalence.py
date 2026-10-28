"""Table 6 rows 1–3 rebuilt from the released news records and clusters."""

from pathlib import Path

import pytest

from pit_multisource.experiments.features import run_features

pytestmark = pytest.mark.equivalence

# row -> changed hours, share (%), mean |d|, mean of the series, mean of the reference
TABLE_6 = {
    "clock.btc": (47059, 81.60, 2.574, 330.804, 330.815),
    "flag.btc": (9723, 16.86, 0.198, 330.815, 330.618),
    "flag.eth": (54016, 93.66, 5.001, 54.995, 49.994),
}


def test_table_6_rows_1_to_3(dataset_root: Path):
    result = run_features(dataset_root)
    assert result["features.hours"] == 57672
    for row, (changed, share, mad, mean_series, mean_reference) in TABLE_6.items():
        key = f"features.{row}"
        assert result[f"{key}.changed_hours"] == changed, row
        assert round(100 * result[f"{key}.share"], 2) == share, row
        assert round(result[f"{key}.mean_abs_diff"], 3) == mad, row
        assert round(result[f"{key}.mean_series"], 3) == mean_series, row
        assert round(result[f"{key}.mean_reference"], 3) == mean_reference, row
    assert round(result["features.flag.btc.mean_abs_diff"], 4) == 0.1975
    for asset, excluded in (("btc", 1), ("eth", 9)):
        assert result[f"features.flag.{asset}.negative_hours"] == 0
        assert result[f"features.flag.{asset}.excluded_at_or_after_boundary"] == excluded
