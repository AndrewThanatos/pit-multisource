"""Table 3: replay agreement at 207 cutoffs, the 2019 slice, the assignment and the controls."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from pit_multisource.config import load_config
from pit_multisource.experiments.replay import released_clusters, run_replay, state_digests

pytestmark = pytest.mark.equivalence


def test_replay_checks(dataset_root: Path):
    result = run_replay(dataset_root, load_config())
    assert result["replay.cutoffs"] == 207
    assert result["replay.cutoffs.boundary"] == 11
    assert (result["replay.dated.agree"], result["replay.dated.disagree"]) == (207, 0)
    assert (result["replay.lifetime.agree"], result["replay.lifetime.disagree"]) == (15, 192)
    assert result["replay.rebuild.records"] == 2_055_263
    assert result["replay.rebuild.clusters"] == 682_652
    assert result["replay.rebuild.url_joins_refused_by_age_cap"] == 3_639

    assert result["slice.records"] == 59_822
    assert result["slice.clusters.rebuilt"] == 24_319
    assert result["slice.clusters.released"] == 24_319
    assert result["slice.assignments_equal"] is True
    assert result["slice.url_joins_refused_by_age_cap"] == 0

    assert result["assignment.rows_left"] == 2_055_263
    assert result["assignment.rows_right"] == 2_055_263
    assert result["assignment.shared_keys"] == 2_055_263
    assert result["assignment.distinct_clusters_left"] == 682_652
    assert result["assignment.distinct_clusters_right"] == 682_652
    for counter in (
        "duplicate_keys_left",
        "duplicate_keys_right",
        "keys_only_in_left",
        "keys_only_in_right",
        "null_cluster_id_left",
        "null_cluster_id_right",
        "shared_keys_with_a_one_sided_null",
        "shared_keys_disagreeing",
    ):
        assert result[f"assignment.{counter}"] == 0, counter
    assert result["assignment.identical"] is True

    controls = {k: v for k, v in result.items() if k.startswith("controls.")}
    assert len(controls) == 6
    assert all(controls.values())


def test_digests_equal_the_published_run(dataset_root: Path):
    clusters = released_clusters(dataset_root, load_config().replay.period_end)
    first = datetime(2019, 1, 1, 0, 15, tzinfo=UTC)
    last = datetime(2025, 7, 31, 23, 59, 59, 999999, tzinfo=UTC)
    assert state_digests(clusters, [first, last]) == [
        "e51e77f54e7b7e969f1b8213bf3e1fdb",
        "b4f5a06dff5fe51c40bcfc6cbbadd373",
    ]
    assert state_digests(clusters, [first], lifetime=True) == ["9275fdaecdafa13dbcc6a4185911d890"]
