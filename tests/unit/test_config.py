from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import pytest
import yaml

from pit_multisource.config import DEFAULT_CONFIG_PATH, load_config


def test_config_values_match_the_article():
    cfg = load_config()

    assert cfg.bars.months_first == "2019-01"
    assert cfg.bars.months_last == "2026-07"
    assert cfg.bars.symbols == ("BTCUSDT", "ETHUSDT")

    assert cfg.news.theme_tokens == ("ECON_BITCOIN",)
    assert cfg.news.selection_terms == (
        "bitcoin", "ethereum", "cryptocurrenc", "stablecoin", "coinbase", "binance", "kraken",
        "tether",
    )  # fmt: skip
    assert cfg.news.suffix_terms == ("cryptocurrenc",)
    assert cfg.news.entity_terms == {"btc": ("bitcoin",), "eth": ("ethereum",)}
    assert cfg.news.asset_theme_tokens == {"btc": ("ECON_BITCOIN",), "eth": ()}
    assert cfg.news.join_window_hours == 24
    assert cfg.news.age_cap_hours == 72
    assert cfg.news.min_shared_entities == 2
    assert cfg.news.stop_entities == frozenset(
        {
            "bitcoin", "ethereum", "cryptocurrency", "cryptocurrencies", "crypto",
            "blockchain", "stablecoin", "coinbase", "binance", "kraken", "tether",
        }
    )  # fmt: skip

    assert cfg.gdelt.base_url == "http://data.gdeltproject.org/gdeltv2"
    assert cfg.gdelt.request_delay_seconds == 0.05
    assert cfg.gdelt.timeout_seconds == 120
    assert (cfg.gdelt.max_retries, cfg.gdelt.backoff_base_seconds) == (5, 5)
    assert cfg.gdelt.backoff_cap_seconds == 120
    assert cfg.gdelt.max_consecutive_unresolved == 20
    assert cfg.gdelt.ingest_version == "0.1.0"

    boundary = datetime(2025, 8, 1, tzinfo=UTC)
    assert cfg.targets.execution_latency_seconds == 30
    assert cfg.targets.max_entry_delay_seconds == 300
    assert cfg.targets.horizon_hours == (4, 24)
    assert cfg.targets.boundary == boundary

    assert cfg.replay.seed == 20260913
    assert cfg.replay.per_year == 28
    assert cfg.replay.first_year == 2019
    assert cfg.replay.last_year == 2025
    assert cfg.replay.first_cutoff == datetime(2019, 1, 2, tzinfo=UTC)
    assert cfg.replay.period_end == boundary
    assert len(cfg.replay.boundary_cutoffs) == 11
    assert cfg.replay.boundary_cutoffs == tuple(sorted(cfg.replay.boundary_cutoffs))
    micro = timedelta(microseconds=1)
    slot_end = datetime(2023, 10, 18, 11, 15, tzinfo=UTC)
    outage_edge = datetime(2019, 1, 4, tzinfo=UTC)
    assert {slot_end - micro, slot_end, slot_end + micro} <= set(cfg.replay.boundary_cutoffs)
    assert {outage_edge - micro, outage_edge, outage_edge + micro} <= set(
        cfg.replay.boundary_cutoffs
    )
    assert boundary - micro in cfg.replay.boundary_cutoffs

    cal = cfg.calendar
    assert (cal.first_year, cal.last_year) == (2016, 2026)
    assert cal.timezone == "America/New_York"
    assert cal.tiers == {"fomc_statement": 4, "cpi": 3, "employment_situation": 2, "ppi": 1}
    assert set(cal.bls_titles) == {"cpi", "ppi", "employment_situation"}
    assert cal.fomc_statement_time == time(14, 0)
    assert cal.fomc_rolling_from_year == 2021
    assert cal.pages.fomc_rolling == "fomc/calendars.9cb9e5ff5b19.html"


def test_walk_forward_folds_follow_the_rule():
    folds = load_config().split_boundaries.folds()

    assert len(folds) == 18
    assert sum(f.test_start is not None for f in folds) == 17

    first, last_with_test, refit = folds[0], folds[16], folds[17]
    start = datetime(2019, 1, 1, tzinfo=UTC)
    assert (first.train_start, first.train_end) == (start, datetime(2021, 1, 1, tzinfo=UTC))
    assert (first.test_start, first.test_end) == (
        datetime(2021, 4, 1, tzinfo=UTC),
        datetime(2021, 7, 1, tzinfo=UTC),
    )
    assert last_with_test.train_end == datetime(2025, 1, 1, tzinfo=UTC)
    assert last_with_test.test_end == datetime(2025, 7, 1, tzinfo=UTC)
    assert refit.train_end == datetime(2025, 8, 1, tzinfo=UTC)
    assert refit.test_start is None and refit.test_end is None


def test_unknown_config_key_is_rejected(tmp_path: Path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["news"]["join_window_hour"] = 24
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="join_window_hour"):
        load_config(path)


def test_missing_config_key_is_rejected(tmp_path: Path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    del raw["targets"]["boundary"]
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="boundary"):
        load_config(path)
