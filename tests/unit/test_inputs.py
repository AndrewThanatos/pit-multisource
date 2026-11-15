from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from pit_multisource.build.news import GKG_FIELDS, build_news, stream_records
from pit_multisource.config import load_config
from pit_multisource.experiments import inputs
from pit_multisource.experiments.inputs import diff, run_inputs
from pit_multisource.experiments.inputs_prefix import expected_bars
from pit_multisource.experiments.inputs_projection import Projection

CONFIG = load_config()
T0 = datetime(2020, 12, 28, tzinfo=UTC)
H = timedelta(hours=1)
M = timedelta(minutes=1)
_US = pl.Datetime("us", "UTC")


def _record(record_id: str, slot: datetime, names: str = "", url: str = "") -> dict[str, object]:
    row: dict[str, object] = {"slot_time": slot, **{name: "" for name in GKG_FIELDS}}
    row |= {
        "gkg_record_id": record_id,
        "source_common_name": f"site{record_id[-1]}.com",
        "document_identifier": url or f"https://example.com/{record_id}",
        "v15_tone": "1,0,0",
        "v21_all_names": names,
    }
    return row


def _records() -> pl.DataFrame:
    rows = [
        _record("r1", T0, "Bitcoin;Alpha Corp;Beta Bank"),
        _record("r2", T0 + 2 * H, "Bitcoin;Alpha Corp;Beta Bank"),
        _record("r3", T0 + 5 * H, "Ethereum;Alpha Corp;Beta Bank"),
        _record("r4", T0 + 30 * H, "Bitcoin;Gamma Ltd;Delta Inc"),
        _record("r5", T0 + 400 * H, "Bitcoin;Omega Group;Zeta Fund"),
    ]
    schema = {"slot_time": _US, **{name: pl.String() for name in GKG_FIELDS}}
    return pl.DataFrame(rows, schema=schema, orient="row")


def _bars(start: datetime, hours: int) -> pl.DataFrame:
    opens = [start + i * H for i in range(hours) if i != 50]
    closes = [o + H - timedelta(milliseconds=1) for o in opens]
    closes[10] = opens[10] - 13 * M
    n = len(opens)
    return pl.DataFrame(
        {
            "open_time": opens, "open": [1.0] * n, "high": [2.0] * n, "low": [0.5] * n,
            "close": [float(i) for i in range(n)], "base_volume": [3.0] * n, "close_time": closes,
            "quote_volume": [4.0] * n, "trade_count": list(range(n)),
            "taker_buy_base_volume": [1.5] * n, "taker_buy_quote_volume": [2.5] * n,
        },
        schema_overrides={"open_time": _US, "close_time": _US},
    )  # fmt: skip


def _calendar() -> pl.DataFrame:
    def row(rid: str, family: str, at: datetime, known: datetime) -> dict[str, object]:
        return {"release_id": rid, "family": family, "released_at": at, "scheduled": True,
                "schedule_available_at": known}  # fmt: skip

    year = datetime(2020, 1, 1, tzinfo=UTC)
    next_year = datetime(2021, 1, 1, tzinfo=UTC)
    rows = [
        row("cpi:a", "cpi", datetime(2020, 12, 10, 13, 30, tzinfo=UTC), year),
        row("cpi:b", "cpi", datetime(2021, 1, 13, 13, 30, tzinfo=UTC), next_year),
        row("fomc:a", "fomc_statement", datetime(2020, 12, 16, 19, tzinfo=UTC), year),
        row("fomc:x", "fomc_statement", T0 + 20 * H, T0 + 20 * H),
        row("fomc:b", "fomc_statement", datetime(2021, 1, 27, 19, tzinfo=UTC), next_year),
    ]
    return pl.DataFrame(rows, schema_overrides={"released_at": _US, "schedule_available_at": _US})


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    (tmp_path / "news" / "gkg_records").mkdir(parents=True)
    records = _records()
    records.write_parquet(tmp_path / "news" / "gkg_records" / "year=2020.parquet")
    tables = build_news(records, CONFIG)
    tables.records.write_parquet(tmp_path / "news" / "news_records.parquet")
    tables.clusters.write_parquet(tmp_path / "news" / "news_clusters.parquet")
    (tmp_path / "market").mkdir()
    for symbol in ("btcusdt", "ethusdt"):
        _bars(T0 - 100 * H, 600).write_parquet(tmp_path / "market" / f"{symbol}_1h.parquet")
    (tmp_path / "calendar").mkdir()
    _calendar().write_parquet(tmp_path / "calendar" / "macro_releases.parquet")
    return tmp_path


CUTOFFS = [
    T0 + 10 * M,
    T0 + 15 * M,
    T0 + 3 * H,
    T0 + 5 * H + 15 * M - timedelta(microseconds=1),
    T0 + 5 * H + 15 * M,
    T0 + 20 * H,
    T0 + 31 * H + 30 * M,
    T0 + 90 * H,
]


def test_routes_agree_and_every_fault_is_detected(dataset: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(inputs, "cutoffs", lambda config: CUTOFFS)

    out = run_inputs(dataset, CONFIG)

    assert out["inputs.compared"] == 2 * len(CUTOFFS)
    for component in inputs.COMPONENTS:
        assert out[f"inputs.{component}.equal"] == out["inputs.compared"], component
        assert out[f"inputs.{component}.field_mismatches"] == 0
        assert out[f"inputs.{component}.missingness_mismatches"] == 0
    for fault in inputs.FAULTS:
        assert out[f"inputs.fault.{fault}.changed"] > 0, fault
        assert out[f"inputs.fault.{fault}.detected"] == out[f"inputs.fault.{fault}.changed"]


def test_projection_dates_the_flag_and_hides_later_records(dataset: Path):
    projection = Projection(dataset)
    membership, clusters, counts = projection.news(T0 + 6 * H)

    assert clusters.rows() == [("r1", 3, 3, True, True)]
    assert counts == {"btc.1h": 1, "btc.24h": 1, "eth.1h": 1, "eth.24h": 1}
    assert membership["record_id"].to_list() == ["r1", "r2", "r3"]
    assert projection.news(T0 + 5 * H + 14 * M)[1]["eth"].to_list() == [False]


def test_stream_records_assigns_like_build_news():
    records = _records()
    streamed = [(r.record_id, r.cluster_id) for r in stream_records(records, CONFIG)]
    built = build_news(records, CONFIG).records.select("record_id", "cluster_id").rows()
    assert streamed == built


@pytest.mark.parametrize(
    ("cutoff", "count"),
    [
        (datetime(2021, 1, 8, 12, tzinfo=UTC), 168),
        (datetime(2021, 1, 8, 12, 30, tzinfo=UTC), 167),
        (datetime(2021, 1, 8, 12, 0, 0, 1, tzinfo=UTC), 167),
    ],
)
def test_expected_bars_counts_whole_hours_that_ended(cutoff: datetime, count: int):
    assert expected_bars(cutoff) == count


def test_diff_separates_values_from_missingness():
    base = pl.DataFrame({"k": [1, 2], "v": [1.0, None]})
    assert diff(base, base, "k") == (False, False)
    assert diff(base, pl.DataFrame({"k": [1, 2], "v": [9.0, None]}), "k") == (True, False)
    assert diff(base, pl.DataFrame({"k": [1, 2], "v": [1.0, 5.0]}), "k") == (False, True)
    assert diff(base, pl.DataFrame({"k": [1], "v": [1.0]}), "k") == (False, True)
