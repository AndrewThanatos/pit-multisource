"""Builds the news-record table and the chronological cluster table from the GKG records."""

from __future__ import annotations

import argparse
import ipaddress
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import polars as pl

from pit_multisource.build.selection import entity_pattern, theme_tokens
from pit_multisource.commands import command_group, register
from pit_multisource.config import Config, load_config

GKG_FIELDS = (
    "gkg_record_id", "date", "source_collection_id", "source_common_name",
    "document_identifier", "v1_counts", "v21_counts", "v1_themes", "v2_enhanced_themes",
    "v1_locations", "v2_enhanced_locations", "v1_persons", "v2_enhanced_persons",
    "v1_organizations", "v2_enhanced_organizations", "v15_tone", "v21_enhanced_dates",
    "v2_gcam", "v21_sharing_image", "v21_related_images", "v21_social_image_embeds",
    "v21_social_video_embeds", "v21_quotations", "v21_all_names", "v21_amounts",
    "v21_translation_info", "v2_extras_xml",
)  # fmt: skip
# The record columns the news build reads.
NEWS_COLUMNS = (
    "slot_time", "gkg_record_id", "source_common_name", "document_identifier", "v1_themes",
    "v2_enhanced_themes", "v1_organizations", "v15_tone", "v21_all_names",
)  # fmt: skip
ASSETS = ("btc", "eth")
EARNED_BASIS = "earned_at_record"

_US = pl.Datetime("us", "UTC")
_US_PER_HOUR = 3_600_000_000
_CHUNK_ROWS = 100_000
_NAME_OFFSET = re.compile(r",\d+$")

RECORDS_SCHEMA: dict[str, pl.DataType] = {
    "record_id": pl.String(),
    "slot_time": _US,
    "url": pl.String(),
    "source_domain": pl.String(),
    "source_domain_status": pl.String(),
    "source_domain_present": pl.Float64(),
    "cluster_id": pl.String(),
    "is_first_observation": pl.Boolean(),
    "btc_relevant": pl.Boolean(),
    "eth_relevant": pl.Boolean(),
    "relevance_basis": pl.String(),
    "tone": pl.Float64(),
    "n_entities": pl.Int32(),
}
CLUSTERS_SCHEMA: dict[str, pl.DataType] = {
    "cluster_id": pl.String(),
    "opened_at": _US,
    **{
        f"{asset}_relevant_{part}": dtype
        for asset in ASSETS
        for part, dtype in (("earned_at", _US), ("earned_by", pl.String()), ("basis", pl.String()))
    },
}


@dataclass(frozen=True)
class ClusterParams:
    """Grouping policy; the defaults are the published one."""

    join_window_h: int = 24
    age_cap_h: int = 72
    min_shared: int = 2
    url_age_capped: bool = True

    def __post_init__(self) -> None:
        if self.min_shared < 1:
            raise ValueError(f"min_shared must be at least 1, got {self.min_shared}")

    @classmethod
    def from_config(cls, config: Config) -> ClusterParams:
        news = config.news
        return cls(news.join_window_hours, news.age_cap_hours, news.min_shared_entities)


@dataclass(frozen=True)
class NewsTables:
    """The news-record and cluster tables with the address-route counters."""

    records: pl.DataFrame
    clusters: pl.DataFrame
    url_joins: int
    url_joins_refused_by_age_cap: int


def read_gkg_records(dataset_root: Path, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """GKG records of every year, ordered by slot time, then record id as a string."""
    names = ["slot_time", *GKG_FIELDS] if columns is None else list(columns)
    files = sorted((dataset_root / "news" / "gkg_records").glob("year=*.parquet"))
    if not files:
        raise FileNotFoundError(f"no year=*.parquet under {dataset_root / 'news' / 'gkg_records'}")
    years = [pl.read_parquet(f, columns=names).sort("slot_time", "gkg_record_id") for f in files]
    for earlier, later in zip(years, years[1:], strict=False):
        if earlier.height and later.height and earlier["slot_time"][-1] > later["slot_time"][0]:
            raise ValueError("year files overlap in slot time")
    return pl.concat(years)


def normalize_entity(raw: str) -> str:
    """Entity token without its character offset, lower-cased, whitespace collapsed."""
    return " ".join(_NAME_OFFSET.sub("", raw).lower().split())


def parse_tone(text: str) -> float | None:
    """The average tone, the first comma-separated value, or None when it is not a number."""
    head = text.split(",")[0].strip()
    try:
        return float(head) if head else None
    except ValueError:
        return None


def resolve_source_domain(observed: str, url: str) -> tuple[str | None, str, float]:
    """(domain, status, presence mask) from the vendor's source name, else the URL host."""
    if observed and observed.strip() == observed:
        return observed, "observed", 1.0
    try:
        host = urlsplit(url).hostname or None
    except ValueError:
        host = None
    if host is None:
        return None, "unknown_not_attempted", 0.0
    host = host.strip().lower()
    if not host or "." not in host or _is_ip_literal(host):
        return None, "unknown_not_a_domain", 0.0
    labels = host.split(".")
    candidates = [".".join(labels[i:]) for i in range(len(labels) - 1)]
    candidates = [c for c in candidates if not c.startswith("www.")]
    if len(candidates) == 1:
        return candidates[0], "reconstructed_from_url", 1.0
    return None, "unknown_reconstruction_ambiguous", 0.0


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def build_news(
    records: pl.DataFrame, config: Config, params: ClusterParams | None = None
) -> NewsTables:
    """Assigns each record, in (slot time, record id) order, to a cluster and dates its flags."""
    params = ClusterParams.from_config(config) if params is None else params
    frame = _prepared(records, config)
    state = _Clusters(params)
    stop = config.news.stop_entities
    patterns = {a: entity_pattern(config.news.entity_terms[a]) for a in ASSETS}
    asset_tokens = {a: frozenset(config.news.asset_theme_tokens[a]) for a in ASSETS}

    cluster_index: list[int] = []
    first_seen: list[bool] = []
    flags: dict[str, list[bool]] = {a: [] for a in ASSETS}
    basis: list[str] = []
    tones: list[float | None] = []
    n_entities: list[int] = []
    domains: list[str | None] = []
    statuses: list[str] = []
    present: list[float] = []

    columns = ("record_id", "slot_us", "url", "source", "names", "orgs", "tone", "themes")
    for chunk in frame.select(columns).iter_slices(_CHUNK_ROWS):
        for record_id, now, url, source, names, orgs, tone, themes in zip(
            *(chunk.get_column(c).to_list() for c in columns), strict=True
        ):
            entities = {normalize_entity(t) for t in f"{names};{orgs}".split(";")}
            entities.discard("")
            entities -= stop
            first_seen.append(url not in state.url_cluster)
            k = state.assign(record_id, now, url, frozenset(entities))

            tokens = theme_tokens(themes) if themes is not None else set()
            haystack = f"{names};{orgs};{url}"
            named = {
                a: bool(asset_tokens[a] & tokens) or patterns[a].search(haystack) is not None
                for a in ASSETS
            }
            relevant = named if any(named.values()) else dict.fromkeys(ASSETS, True)
            for a in ASSETS:
                flags[a].append(relevant[a])
                if relevant[a] and state.earned_at[a][k] is None:
                    state.earned_at[a][k] = now
                    state.earned_by[a][k] = record_id
            basis.append("named" if any(named.values()) else "tag_both_assets")

            cluster_index.append(k)
            tones.append(parse_tone(tone))
            n_entities.append(len(entities))
            domain, status, mask = resolve_source_domain(source, url)
            domains.append(domain)
            statuses.append(status)
            present.append(mask)

    cluster_ids = pl.Series(state.ids, dtype=pl.String())
    out = frame.select("record_id", "slot_time", "url").with_columns(
        pl.Series("source_domain", domains, dtype=pl.String()),
        pl.Series("source_domain_status", statuses, dtype=pl.String()),
        pl.Series("source_domain_present", present, dtype=pl.Float64()),
        cluster_ids.gather(cluster_index).alias("cluster_id"),
        pl.Series("is_first_observation", first_seen, dtype=pl.Boolean()),
        *(pl.Series(f"{a}_relevant", flags[a], dtype=pl.Boolean()) for a in ASSETS),
        pl.Series("relevance_basis", basis, dtype=pl.String()),
        pl.Series("tone", tones, dtype=pl.Float64()),
        pl.Series("n_entities", n_entities, dtype=pl.Int32()),
    )
    return NewsTables(
        out.select(list(RECORDS_SCHEMA)),
        state.frame(),
        state.url_joins,
        state.url_joins_refused,
    )


def _prepared(records: pl.DataFrame, config: Config) -> pl.DataFrame:
    """The columns the build reads, sorted, first occurrence of each record id."""
    tokens = sorted({t for a in ASSETS for t in config.news.asset_theme_tokens[a]})
    themes = pl.concat_str("v1_themes", "v2_enhanced_themes", separator=";")
    if tokens:
        # Only rows that can hold a token are tokenised: an ASCII match ignoring case, or any
        # non-ASCII character, which upper-casing could turn into one.
        pattern = "(?i)(" + "|".join(re.escape(t) for t in tokens) + r")|[^\x00-\x7F]"
        maybe = pl.any_horizontal(
            [pl.col(c).str.contains(pattern) for c in ("v1_themes", "v2_enhanced_themes")]
        )
        themes = pl.when(maybe).then(themes)
    else:
        themes = pl.lit(None, dtype=pl.String())
    frame = (
        records.lazy()
        .select(
            pl.col("gkg_record_id").alias("record_id"),
            "slot_time",
            pl.col("slot_time").dt.epoch("us").alias("slot_us"),
            pl.col("document_identifier").alias("url"),
            pl.col("source_common_name").alias("source"),
            pl.col("v21_all_names").alias("names"),
            pl.col("v1_organizations").alias("orgs"),
            pl.col("v15_tone").alias("tone"),
            themes.alias("themes"),
        )
        .sort("slot_time", "record_id")
        .collect()
    )
    if frame["record_id"].n_unique() < frame.height:
        frame = frame.filter(pl.col("record_id").is_first_distinct())
    return frame


class _Clusters:
    """Open clusters and the address map, updated record by record in time order."""

    def __init__(self, params: ClusterParams) -> None:
        self.window = params.join_window_h * _US_PER_HOUR
        self.cap = params.age_cap_h * _US_PER_HOUR
        self.min_shared = params.min_shared
        self.url_age_capped = params.url_age_capped
        self.ids: list[str] = []
        self.opened: list[int] = []
        self.last: list[int] = []
        self.earned_at: dict[str, list[int | None]] = {a: [] for a in ASSETS}
        self.earned_by: dict[str, list[str | None]] = {a: [] for a in ASSETS}
        self.url_cluster: dict[str, int] = {}
        self.url_joins = 0
        self.url_joins_refused = 0
        # Seed entity -> clusters within the age cap; clusters leave in creation order.
        self._postings: dict[str, set[int]] = {}
        self._seeds: dict[int, frozenset[str]] = {}
        self._by_age: deque[int] = deque()

    def assign(self, record_id: str, now: int, url: str, entities: frozenset[str]) -> int:
        """The cluster index for a record, opening a new cluster when no route joins one."""
        self._drop_aged(now)
        k = self.url_cluster.get(url)
        if k is not None:
            if self.url_age_capped and now - self.opened[k] > self.cap:
                self.url_joins_refused += 1
                k = None
            else:
                self.url_joins += 1
        if k is None and len(entities) >= self.min_shared:
            k = self._best_entity_match(now, entities)
        if k is None:
            k = self._open(record_id, now, entities)
        elif now - self.last[k] <= self.window and now - self.opened[k] <= self.cap:
            self.last[k] = now
        self.url_cluster[url] = k
        return k

    def _drop_aged(self, now: int) -> None:
        while self._by_age and now - self.opened[self._by_age[0]] > self.cap:
            k = self._by_age.popleft()
            for entity in self._seeds.pop(k):
                members = self._postings[entity]
                members.discard(k)
                if not members:
                    del self._postings[entity]

    def _best_entity_match(self, now: int, entities: frozenset[str]) -> int | None:
        """Largest seed overlap, then latest activity, then earliest creation."""
        overlap: dict[int, int] = {}
        for entity in entities:
            for k in self._postings.get(entity, ()):
                overlap[k] = overlap.get(k, 0) + 1
        best: tuple[int, int, int] | None = None
        for k, shared in overlap.items():
            if shared >= self.min_shared and now - self.last[k] <= self.window:
                key = (shared, self.last[k], -k)
                if best is None or key > best:
                    best = key
        return None if best is None else -best[2]

    def _open(self, record_id: str, now: int, entities: frozenset[str]) -> int:
        k = len(self.ids)
        self.ids.append(record_id)
        self.opened.append(now)
        self.last.append(now)
        for a in ASSETS:
            self.earned_at[a].append(None)
            self.earned_by[a].append(None)
        self._seeds[k] = entities
        self._by_age.append(k)
        for entity in entities:
            self._postings.setdefault(entity, set()).add(k)
        return k

    def frame(self) -> pl.DataFrame:
        """One row per cluster, sorted by opening time, then cluster id."""
        columns: dict[str, pl.Series] = {
            "cluster_id": pl.Series(self.ids, dtype=pl.String()),
            "opened_at": pl.Series(self.opened, dtype=pl.Int64()).cast(_US),
        }
        for a in ASSETS:
            earned_at = pl.Series(self.earned_at[a], dtype=pl.Int64()).cast(_US)
            columns[f"{a}_relevant_earned_at"] = earned_at
            columns[f"{a}_relevant_earned_by"] = pl.Series(self.earned_by[a], dtype=pl.String())
            columns[f"{a}_relevant_basis"] = pl.Series(
                [EARNED_BASIS if by is not None else None for by in self.earned_by[a]],
                dtype=pl.String(),
            )
        return pl.DataFrame(columns).select(list(CLUSTERS_SCHEMA)).sort("opened_at", "cluster_id")


def _run(args: argparse.Namespace) -> int:
    tables = build_news(read_gkg_records(args.dataset, NEWS_COLUMNS), load_config())
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in (("news_records", tables.records), ("news_clusters", tables.clusters)):
        table.write_parquet(args.out / f"{name}.parquet")
        print(f"{name}.parquet: {table.height} rows")
    print(f"address joins refused by the age cap: {tables.url_joins_refused_by_age_cap}")
    return 0


@register
def _register_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = command_group(subparsers, "build").add_parser(
        "news", help="news records and chronological clusters from the GKG records"
    )
    parser.add_argument("--dataset", type=Path, required=True, help="released dataset directory")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.set_defaults(func=_run)
