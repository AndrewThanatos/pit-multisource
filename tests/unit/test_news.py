from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from pit_multisource import cli
from pit_multisource.build.news import (
    GKG_FIELDS,
    ClusterParams,
    build_news,
    read_gkg_records,
    resolve_source_domain,
)
from pit_multisource.config import load_config
from pit_multisource.pit.availability import flags_state_as_of

CONFIG = load_config()
T0 = datetime(2020, 1, 1, tzinfo=UTC)
H = timedelta(hours=1)
SLOT = timedelta(minutes=15)


def _record(record_id: str, slot: datetime, **fields: str) -> dict[str, object]:
    row: dict[str, object] = {"slot_time": slot, **{name: "" for name in GKG_FIELDS}}
    row["gkg_record_id"] = record_id
    row["date"] = f"{slot:%Y%m%d%H%M%S}"
    row["source_common_name"] = fields.pop("source", "example.com")
    row["document_identifier"] = fields.pop("url", f"https://example.com/{record_id}")
    row["v1_themes"] = fields.pop("themes", "")
    row["v1_organizations"] = fields.pop("orgs", "")
    row["v15_tone"] = fields.pop("tone", "1.5,2,0.5")
    row["v21_all_names"] = fields.pop("names", "")
    assert not fields, fields
    return row


def _frame(*rows: dict[str, object]) -> pl.DataFrame:
    schema = {"slot_time": pl.Datetime("us", "UTC"), **{name: pl.String() for name in GKG_FIELDS}}
    return pl.DataFrame(list(rows), schema=schema, orient="row")


def _cluster_of(tables, record_id: str) -> str:
    return tables.records.filter(pl.col("record_id") == record_id)["cluster_id"].item()


def test_records_sorted_by_slot_then_record_id_string(tmp_path: Path):
    first = T0.replace(year=2019)
    root = tmp_path / "news" / "gkg_records"
    root.mkdir(parents=True)
    _frame(
        _record("20200101000000-134", T0, names="Xcorp,1;Ycorp,2"),
        _record("20200101000000-1146", T0, names="Xcorp,1;Ycorp,2"),
    ).write_parquet(root / "year=2020.parquet")
    _frame(_record("20190101000000-9", first)).write_parquet(root / "year=2019.parquet")

    records = read_gkg_records(tmp_path)
    tables = build_news(records.reverse(), CONFIG)

    expected = ["20190101000000-9", "20200101000000-1146", "20200101000000-134"]
    assert records["gkg_record_id"].to_list() == expected
    assert records.columns == ["slot_time", *GKG_FIELDS]
    assert tables.records["record_id"].to_list() == expected
    assert _cluster_of(tables, "20200101000000-134") == "20200101000000-1146"


def test_url_join_ignores_window_but_respects_age_cap():
    rows = (
        _record("a", T0, url="u", names="Xcorp;Ycorp"),
        _record("b", T0 + 48 * H, url="u"),
        _record("d", T0 + 60 * H, names="Xcorp;Ycorp"),
        _record("c", T0 + 72 * H + SLOT, url="u"),
    )

    capped = build_news(_frame(*rows), CONFIG)
    uncapped = build_news(_frame(*rows), CONFIG, ClusterParams(url_age_capped=False))

    assert [_cluster_of(capped, r) for r in "abdc"] == ["a", "a", "d", "c"]
    assert (capped.url_joins, capped.url_joins_refused_by_age_cap) == (1, 1)
    assert capped.records["is_first_observation"].to_list() == [True, False, True, False]
    assert [_cluster_of(uncapped, r) for r in "abdc"] == ["a", "a", "d", "a"]
    assert (uncapped.url_joins, uncapped.url_joins_refused_by_age_cap) == (2, 0)


def test_age_cap_is_inclusive_at_exactly_72h():
    names = "Xcorp;Ycorp"
    tables = build_news(
        _frame(
            _record("a", T0, url="u", names=names),
            _record("b", T0 + 24 * H, names=names),
            _record("c", T0 + 48 * H, names=names),
            _record("d", T0 + 72 * H, names=names),
            _record("e", T0 + 72 * H, url="u"),
            _record("f", T0 + 72 * H + SLOT, names=names),
            _record("g", T0 + 72 * H + SLOT, url="u", names=names),
        ),
        CONFIG,
    )

    assert [_cluster_of(tables, r) for r in "abcdefg"] == ["a"] * 5 + ["f", "f"]
    assert tables.url_joins_refused_by_age_cap == 1
    assert tables.clusters["cluster_id"].to_list() == ["a", "f"]


def test_tie_break_latest_active_then_earliest_created():
    # "-1146" sorts before "-134", so it is created first within the slot.
    first, second, third = "20200101000000-1146", "20200101000000-134", "20200101000000-2"
    tables = build_news(
        _frame(
            _record(first, T0, names="A;B"),
            _record(second, T0, url="v", names="C;D"),
            _record(third, T0, names="E;F"),
            _record("p0", T0 + 2 * SLOT, url="v"),
            _record("p1", T0 + H, names="A;B;C;D;E;F"),
            _record("p2", T0 + 2 * H, names="A;B;E;F"),
            _record("p3", T0 + 3 * H, names="A;B;C;E;F"),
        ),
        CONFIG,
    )

    assert _cluster_of(tables, "p0") == second
    assert _cluster_of(tables, "p1") == second
    assert _cluster_of(tables, "p2") == first
    assert _cluster_of(tables, "p3") == first


def test_tie_break_prefers_larger_overlap_over_later_activity():
    tables = build_news(
        _frame(
            _record("a", T0, names="A;B;C"),
            _record("b", T0 + H, names="D;E"),
            _record("p", T0 + 2 * H, names="A;B;C;D;E"),
        ),
        CONFIG,
        ClusterParams(min_shared=2),
    )

    assert _cluster_of(tables, "p") == "a"


def test_generic_fallback_tags_both():
    tables = build_news(
        _frame(
            _record("1", T0, names="Coinbase,1"),
            _record("2", T0, names="Bitcoins,3"),
            _record("3", T0, url="https://x.com/ethereum-news"),
            _record("4", T0, themes="ECON_BITCOIN,12;TAX_FNCACT"),
            _record("5", T0, orgs="bitcoin's fund"),
            _record("6", T0, names="Bitcoin,1;ETHEREUM,9"),
        ),
        CONFIG,
    )

    flags = tables.records.select("btc_relevant", "eth_relevant", "relevance_basis").rows()
    assert flags == [
        (True, True, "tag_both_assets"),
        (True, True, "tag_both_assets"),
        (False, True, "named"),
        (True, False, "named"),
        (True, False, "named"),
        (True, True, "named"),
    ]


def test_flag_earned_by_first_member():
    tables = build_news(
        _frame(
            _record("a", T0, names="Bitcoin,1;Xcorp,5;Ycorp,9", tone="-2.25,1"),
            _record("b", T0 + H, names="Ethereum;Xcorp;Ycorp", tone=""),
            _record("c", T0 + 2 * H, names="Ethereum;Xcorp;Ycorp", tone="nope,1"),
        ),
        CONFIG,
    )

    assert tables.clusters.rows(named=True) == [
        {
            "cluster_id": "a",
            "opened_at": T0,
            "btc_relevant_earned_at": T0,
            "btc_relevant_earned_by": "a",
            "btc_relevant_basis": "earned_at_record",
            "eth_relevant_earned_at": T0 + H,
            "eth_relevant_earned_by": "b",
            "eth_relevant_basis": "earned_at_record",
        }
    ]
    assert tables.records["n_entities"].to_list() == [2, 2, 2]
    assert tables.records["n_entities"].dtype == pl.Int32
    assert tables.records["tone"].to_list() == [-2.25, None, None]


def test_source_domain_four_outcomes():
    cases = [
        ("Example.COM", "https://other.org/a", ("Example.COM", "observed", 1.0)),
        (" x.com", "https://www.example.com/a", ("example.com", "reconstructed_from_url", 1.0)),
        ("", "http://192.168.0.1/a", (None, "unknown_not_a_domain", 0.0)),
        ("", "https://localhost/a", (None, "unknown_not_a_domain", 0.0)),
        ("", "https://news.example.com/a", (None, "unknown_reconstruction_ambiguous", 0.0)),
        ("  ", "not a url", (None, "unknown_not_attempted", 0.0)),
    ]

    for observed, url, expected in cases:
        assert resolve_source_domain(observed, url) == expected, (observed, url)

    tables = build_news(_frame(_record("r", T0, source="", url="https://www.example.com/")), CONFIG)
    assert tables.records.select(
        "source_domain", "source_domain_status", "source_domain_present"
    ).row(0) == ("example.com", "reconstructed_from_url", 1.0)


def test_flags_state_as_of_counts_slot_end():
    clusters = pl.DataFrame(
        {
            "cluster_id": ["x", "y"],
            "opened_at": [T0, T0 + H],
            "btc_relevant_earned_at": [T0, None],
            "eth_relevant_earned_at": [T0 + 2 * SLOT, T0 + H],
        }
    )
    micro = timedelta(microseconds=1)

    before = flags_state_as_of(clusters, T0 + SLOT - micro)
    at_end = flags_state_as_of(clusters, T0 + SLOT)

    assert before.columns == ["cluster_id", "opened_at", "btc_relevant", "eth_relevant"]
    assert before.rows() == [("x", T0, False, False), ("y", T0 + H, False, False)]
    assert at_end.rows() == [("x", T0, True, False), ("y", T0 + H, False, False)]
    assert flags_state_as_of(clusters, T0 + H + SLOT).rows() == [
        ("x", T0, True, True),
        ("y", T0 + H, False, True),
    ]
    with pytest.raises(ValueError):
        flags_state_as_of(clusters, datetime(2020, 1, 2))


def test_cli_builds_news_tables(tmp_path: Path):
    root = tmp_path / "data" / "news" / "gkg_records"
    root.mkdir(parents=True)
    _frame(_record("a", T0, names="Bitcoin")).write_parquet(root / "year=2020.parquet")

    dataset = str(tmp_path / "data")
    status = cli.main(["build", "news", "--dataset", dataset, "--out", str(tmp_path)])

    assert status == 0
    assert pl.read_parquet(tmp_path / "news_records.parquet")["cluster_id"].to_list() == ["a"]
    assert pl.read_parquet(tmp_path / "news_clusters.parquet").height == 1
