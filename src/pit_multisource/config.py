"""Loads and validates configs/pipeline.yaml into frozen dataclasses."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

import yaml

_PACKAGED = Path(__file__).parent / "configs" / "pipeline.yaml"
_SOURCE_TREE = Path(__file__).resolve().parents[2] / "configs" / "pipeline.yaml"
DEFAULT_CONFIG_PATH = _SOURCE_TREE if _SOURCE_TREE.exists() else _PACKAGED


@dataclass(frozen=True)
class BarsConfig:
    months_first: str
    months_last: str
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class NewsConfig:
    theme_tokens: tuple[str, ...]
    selection_terms: tuple[str, ...]
    suffix_terms: tuple[str, ...]
    entity_terms: Mapping[str, tuple[str, ...]]
    asset_theme_tokens: Mapping[str, tuple[str, ...]]
    join_window_hours: int
    age_cap_hours: int
    min_shared_entities: int
    stop_entities: frozenset[str]


@dataclass(frozen=True)
class GdeltConfig:
    base_url: str
    user_agent: str
    request_delay_seconds: float
    timeout_seconds: float
    max_retries: int
    backoff_base_seconds: float
    backoff_cap_seconds: float
    max_consecutive_unresolved: int
    ingest_version: str


@dataclass(frozen=True)
class TargetsConfig:
    execution_latency_seconds: int
    max_entry_delay_seconds: int
    horizon_hours: tuple[int, ...]
    boundary: datetime


@dataclass(frozen=True)
class Fold:
    train_start: datetime
    train_end: datetime
    test_start: datetime | None
    test_end: datetime | None


@dataclass(frozen=True)
class SplitBoundariesConfig:
    train_start: datetime
    first_train_end: datetime
    step_months: int
    gap_months: int
    test_months: int
    test_folds: int
    refit_train_end: datetime

    def folds(self) -> tuple[Fold, ...]:
        """The test folds in order, then the refit fold without a test window."""
        folds = []
        for k in range(self.test_folds):
            train_end = _add_months(self.first_train_end, k * self.step_months)
            test_start = _add_months(train_end, self.gap_months)
            test_end = _add_months(test_start, self.test_months)
            folds.append(Fold(self.train_start, train_end, test_start, test_end))
        folds.append(Fold(self.train_start, self.refit_train_end, None, None))
        return tuple(folds)


@dataclass(frozen=True)
class ReplayConfig:
    seed: int
    per_year: int
    first_year: int
    last_year: int
    first_cutoff: datetime
    period_end: datetime
    boundary_cutoffs: tuple[datetime, ...]


@dataclass(frozen=True)
class CalendarPages:
    fomc_year: str
    fomc_rolling: str
    fomc_press: str
    bls_year: str


@dataclass(frozen=True)
class CalendarConfig:
    first_year: int
    last_year: int
    timezone: str
    tiers: Mapping[str, int]
    bls_titles: Mapping[str, str]
    fomc_statement_time: time
    fomc_rolling_from_year: int
    urls: CalendarPages
    pages: CalendarPages
    request_delay_seconds: float
    timeout_seconds: float


@dataclass(frozen=True)
class Config:
    bars: BarsConfig
    news: NewsConfig
    gdelt: GdeltConfig
    targets: TargetsConfig
    split_boundaries: SplitBoundariesConfig
    replay: ReplayConfig
    calendar: CalendarConfig


def load_config(path: Path | None = None) -> Config:
    """Reads the pipeline config; unknown or missing keys raise ValueError."""
    path = DEFAULT_CONFIG_PATH if path is None else path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _build(
        Config,
        raw,
        bars=lambda v: _build(BarsConfig, v, symbols=tuple),
        news=lambda v: _build(
            NewsConfig,
            v,
            theme_tokens=tuple,
            selection_terms=tuple,
            suffix_terms=tuple,
            entity_terms=lambda terms: {k: tuple(t) for k, t in terms.items()},
            asset_theme_tokens=lambda tokens: {k: tuple(t) for k, t in tokens.items()},
            stop_entities=frozenset,
        ),
        gdelt=lambda v: _build(GdeltConfig, v),
        targets=lambda v: _build(TargetsConfig, v, horizon_hours=tuple, boundary=_parse_instant),
        split_boundaries=lambda v: _build(
            SplitBoundariesConfig,
            v,
            train_start=_parse_instant,
            first_train_end=_parse_instant,
            refit_train_end=_parse_instant,
        ),
        replay=lambda v: _build(
            ReplayConfig,
            v,
            first_cutoff=_parse_instant,
            period_end=_parse_instant,
            boundary_cutoffs=lambda xs: tuple(_parse_instant(x) for x in xs),
        ),
        calendar=lambda v: _build(
            CalendarConfig,
            v,
            tiers=dict,
            bls_titles=dict,
            fomc_statement_time=time.fromisoformat,
            urls=lambda u: _build(CalendarPages, u),
            pages=lambda p: _build(CalendarPages, p),
        ),
    )


def _build(cls: type, raw: Any, **convert: Callable[[Any], Any]) -> Any:
    """Builds a dataclass from a mapping, rejecting unknown and missing keys."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{cls.__name__}: expected a mapping, got {type(raw).__name__}")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - names)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {unknown}")
    missing = sorted(names - set(raw))
    if missing:
        raise ValueError(f"{cls.__name__}: missing keys {missing}")
    return cls(**{k: convert[k](v) if k in convert else v for k, v in raw.items()})


def _parse_instant(value: Any) -> datetime:
    """Parses a UTC instant written as an ISO-8601 string ending in Z."""
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"expected an ISO-8601 UTC string ending in Z, got {value!r}")
    return datetime.fromisoformat(value).astimezone(UTC)


def _add_months(instant: datetime, months: int) -> datetime:
    """Adds whole months to an instant that sits on the first day of a month."""
    index = instant.year * 12 + instant.month - 1 + months
    return instant.replace(year=index // 12, month=index % 12 + 1)
