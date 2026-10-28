"""One module per article table; each returns its numbers as a flat dict. EXPERIMENTS runs them."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pit_multisource.config import Config

Results = dict[str, Any]
Runner = Callable[[Path, Config, Mapping[str, Path]], Results]
Derivation = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class Experiment:
    """A named run over the dataset, the key prefixes it returns and the inputs it needs."""

    name: str
    run: Runner
    prefixes: tuple[str, ...]
    needs: tuple[str, ...] = ()
    derived: Mapping[str, Derivation] = field(default_factory=dict)

    def results(self, dataset_root: Path, config: Config, extra: Mapping[str, Path]) -> Results:
        """The run's dict plus the figures derived from it."""
        out = self.run(dataset_root, config, extra)
        return out | {key: derive(out) for key, derive in self.derived.items()}


def _call(target: str, *args: str) -> Runner:
    """A runner that imports `module:function` when run and passes it the named inputs."""

    def run(dataset_root: Path, config: Config, extra: Mapping[str, Path]) -> Results:
        module, function = target.split(":")
        inputs = {"dataset_root": dataset_root, "config": config, **extra}
        return getattr(importlib.import_module(module), function)(*(inputs[a] for a in args))

    return run


def _pct(part: str, whole: str) -> Derivation:
    return lambda r: 100 * r[part] / r[whole]


def _minus(a: str, b: str) -> Derivation:
    return lambda r: r[a] - r[b]


def _change_pct(new: str, old: str) -> Derivation:
    return lambda r: 100 * (r[new] - r[old]) / r[old]


_M = "pit_multisource.experiments."
_POLICIES = ("bounded", "unbounded")
_SETTINGS = tuple(f"min{m}_age{a}" for m in (1, 2, 3) for a in (24, 48, 72))
_CELLS = tuple(f"{s}.{h}h" for s in ("BTCUSDT", "ETHUSDT") for h in (4, 24))
_S = "sensitivity."

EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment("sizes", _call(_M + "sizes:run_sizes", "dataset_root"), ("sizes.",)),
    Experiment(
        "descriptive",
        _call(_M + "descriptive:run_descriptive", "dataset_root"),
        ("ledger.", "cluster_shape."),
    ),
    Experiment("agreement", _call(_M + "agreement:run_agreement", "dataset_root"), ("agreement.",)),
    Experiment(
        "coverage",
        _call(_M + "coverage:run_coverage", "dataset_root"),
        ("coverage.",),
        derived={
            "coverage.1h.eth.confirmed_quiet_pct": _pct(
                "coverage.1h.eth.confirmed_quiet", "coverage.1h.eth.hours"
            ),
            "coverage.24h_minus_1h.all.excluded_leading": _minus(
                "coverage.24h.all.excluded_leading", "coverage.1h.all.excluded_leading"
            ),
        },
    ),
    Experiment(
        "targets",
        _call(_M + "targets:run_targets", "dataset_root", "config"),
        ("BTCUSDT.", "ETHUSDT.", "total."),
        derived={
            "total.missing.entry_too_late": lambda r: sum(
                r[f"{cell}.missing.entry_too_late"] for cell in _CELLS
            )
        },
    ),
    Experiment(
        "replay",
        _call(_M + "replay:run_replay", "dataset_root", "config"),
        ("replay.", "slice.", "assignment.", "controls."),
        derived={
            "replay.cutoffs.sampled": _minus("replay.cutoffs", "replay.cutoffs.boundary"),
            "replay.lifetime.disagree_pct": _pct("replay.lifetime.disagree", "replay.cutoffs"),
            "controls.passed": lambda r: sum(
                v is True for k, v in r.items() if k.startswith("controls.")
            ),
        },
    ),
    Experiment("features", _call(_M + "features:run_features", "dataset_root"), ("features.",)),
    Experiment(
        "grouping",
        _call(_M + "grouping:run_grouping", "dataset_root", "shared_ids", "config"),
        ("grouping.",),
        needs=("shared_ids",),
        derived={
            **{
                f"grouping.{p}.{name}_pct": _pct(f"grouping.{p}.{name}", f"grouping.{p}.clusters")
                for p in _POLICIES
                for name in ("clusters_over_published_cap", "crossing_clusters")
            },
            "grouping.bounded_minus_unbounded.clusters": _minus(
                "grouping.bounded.clusters", "grouping.unbounded.clusters"
            ),
            "grouping.bounded_minus_unbounded.clusters_pct": _change_pct(
                "grouping.bounded.clusters", "grouping.unbounded.clusters"
            ),
        },
    ),
    Experiment(
        "sensitivity",
        _call(_M + "grouping:run_sensitivity", "dataset_root", "config"),
        (_S,),
        derived={
            _S + "min2_age24_minus_min2_age72.clusters": _minus(
                _S + "min2_age24.clusters", _S + "min2_age72.clusters"
            ),
            _S + "min2_age24_minus_min2_age72.clusters_pct": _change_pct(
                _S + "min2_age24.clusters", _S + "min2_age72.clusters"
            ),
            _S + "min3_age72_over_min1_age72.clusters": lambda r: (
                r[_S + "min3_age72.clusters"] / r[_S + "min1_age72.clusters"]
            ),
            _S + "all.crossing_clusters.max": lambda r: max(
                r[f"{_S}{s}.crossing_clusters"] for s in _SETTINGS
            ),
            _S + "all.size_median.min": lambda r: min(r[f"{_S}{s}.size_median"] for s in _SETTINGS),
            _S + "all.size_median.max": lambda r: max(r[f"{_S}{s}.size_median"] for s in _SETTINGS),
        },
    ),
    Experiment(
        "example", _call("pit_multisource.example:point_in_time", "dataset_root"), ("example.",)
    ),
)
