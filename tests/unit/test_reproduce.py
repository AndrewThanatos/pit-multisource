import json
from pathlib import Path

import pytest

from pit_multisource import cli
from pit_multisource import reproduce as rp
from pit_multisource.experiments import EXPERIMENTS, Experiment


def _fake(value: object, needs: tuple[str, ...] = ()) -> Experiment:
    return Experiment(
        name="fake",
        run=lambda root, config, extra: {"fake.n": value},
        prefixes=("fake.",),
        needs=needs,
    )


def _values(tmp_path: Path, key: str = "fake.n") -> Path:
    entry = {"id": "n", "section": "Table 0", "experiment": "fake", "key": key, "value": 3}
    path = tmp_path / "values.json"
    path.write_text(json.dumps({"paper": [entry], "supplementary": [], "not_reproducible": []}))
    return path


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, experiment: Experiment, key="fake.n"):
    monkeypatch.setattr(rp, "EXPERIMENTS", (experiment,))
    monkeypatch.setattr(rp, "VALUES_PATH", _values(tmp_path, key))
    return cli.main(["reproduce", "--dataset", str(tmp_path), "--out", str(tmp_path / "results")])


def test_every_paper_value_maps_to_an_experiment_key():
    values = rp.load_values()
    by_name = {experiment.name: experiment for experiment in EXPERIMENTS}
    entries = values["paper"] + values["supplementary"]
    assert values["paper"] and values["supplementary"] and values["not_reproducible"]
    assert len({entry["id"] for entry in entries}) == len(entries)
    for entry in entries:
        experiment = by_name[entry["experiment"]]
        key = entry["key"]
        assert key in experiment.derived or key.startswith(experiment.prefixes), entry["id"]
        assert entry["section"], entry["id"]


def test_mismatch_exits_nonzero(tmp_path, monkeypatch, capsys):
    assert _run(tmp_path, monkeypatch, _fake(4)) == 1
    assert "NO" in capsys.readouterr().out
    assert json.loads((tmp_path / "results" / "fake.json").read_text()) == {"fake.n": 4}


def test_match_exits_zero(tmp_path, monkeypatch, capsys):
    assert _run(tmp_path, monkeypatch, _fake(3)) == 0
    assert "yes" in capsys.readouterr().out


def test_absent_key_is_a_mismatch(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, _fake(3), key="fake.other") == 1


def test_missing_input_skips_instead_of_failing(tmp_path, monkeypatch, capsys):
    assert _run(tmp_path, monkeypatch, _fake(4, needs=("shared_ids",))) == 0
    out = capsys.readouterr().out
    assert "skipped: needs the shared-id file from the Materials record" in out
    assert not (tmp_path / "results" / "fake.json").exists()


@pytest.mark.parametrize(
    ("value", "decimals", "scale", "reproduced", "expected"),
    [
        (76.69, 2, 100, 0.766949, True),
        (76.69, 2, 100, 0.766951, False),
        (0.430, 3, 100, 0.0043049, True),
        (99.879, 3, 1, 99.8794, True),
        (99.879, 3, 1, 99.8796, False),
        (54436, 0, 1, 54435.75, True),
        (72, 0, 1, 72.6, False),
        (-0.00127, 5, 1, -0.0012749, True),
        (0, 3, 1, -0.0001, True),
    ],
)
def test_rounding_matches_article_precision(value, decimals, scale, reproduced, expected):
    entry = {"value": value, "decimals": decimals, "scale": scale}
    assert rp.matches(entry, reproduced) is expected


@pytest.mark.parametrize(
    ("value", "reproduced", "expected"),
    [
        (True, True, True),
        (True, 1, False),
        (None, None, True),
        (None, 0.0, False),
        ("a", "a", True),
    ],
)
def test_flags_undefined_values_and_text_compare_exactly(value, reproduced, expected):
    assert rp.matches({"value": value}, reproduced) is expected
