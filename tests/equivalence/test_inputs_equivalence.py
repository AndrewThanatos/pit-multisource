"""The full input X(u, c) agrees between the prefix and projection routes at the 207 cutoffs."""

from pathlib import Path

import pytest

from pit_multisource.config import load_config
from pit_multisource.experiments.inputs import COMPONENTS, run_inputs

pytestmark = pytest.mark.equivalence


def test_inputs_agree_at_every_cutoff(dataset_root: Path):
    out = run_inputs(dataset_root, load_config())

    assert out["inputs.cutoffs"] == 207
    assert out["inputs.compared"] == 414
    for component in COMPONENTS:
        assert out[f"inputs.{component}.equal"] == 414, component
        assert out[f"inputs.{component}.field_mismatches"] == 0, component
        assert out[f"inputs.{component}.missingness_mismatches"] == 0, component
    assert out["inputs.rows.market_bars"] == 68_176
    assert out["inputs.rows.record_assignments"] == 193_675_640
    assert out["inputs.rows.cluster_states"] == 67_070_559
    assert out["inputs.rows.calendar_rows"] == 828
    faults = {"final_relevance": 232, "forming_bar": 414, "unknown_schedule": 32}
    for fault, changed in faults.items():
        assert out[f"inputs.fault.{fault}.changed"] == changed, fault
        assert out[f"inputs.fault.{fault}.detected"] == changed, fault
