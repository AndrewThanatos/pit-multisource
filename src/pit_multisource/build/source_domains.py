"""Source-domain diagnostics: singleton families, co-appearance edges and per-cluster counts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from pit_multisource.commands import command_group, register

UNKNOWN_STATUSES = (
    "unknown_not_a_domain",
    "unknown_not_attempted",
    "unknown_reconstruction_ambiguous",
)
RECORD_COLUMNS = ("cluster_id", "source_domain", "source_domain_status", "tone", "n_entities")
# Two domains in one cluster with equal values here count as one source unit.
FINGERPRINT = ("tone", "n_entities")
FAMILY_PROVENANCE = "singleton_by_default"


@dataclass(frozen=True)
class DomainTables:
    """The family, edge and cluster-count tables."""

    families: pl.DataFrame
    edges: pl.DataFrame
    cluster_counts: pl.DataFrame


def _is_unknown() -> pl.Expr:
    return pl.col("source_domain_status").is_in(UNKNOWN_STATUSES)


def _known(records: pl.DataFrame) -> pl.DataFrame:
    return records.filter(~_is_unknown() & pl.col("source_domain").is_not_null())


def _families(known: pl.DataFrame) -> pl.DataFrame:
    domains = known["source_domain"].unique().sort()
    return pl.DataFrame(
        {
            "family_id": domains,
            "member_domain": domains,
            "provenance": pl.Series([FAMILY_PROVENANCE] * domains.len(), dtype=pl.String),
        }
    )


def _edges(known: pl.DataFrame) -> pl.DataFrame:
    pairs = known.select("cluster_id", "source_domain").unique()
    return (
        pairs.join(pairs, on="cluster_id", suffix="_right")
        .filter(pl.col("source_domain") < pl.col("source_domain_right"))
        .group_by("source_domain", "source_domain_right")
        .agg(pl.len().cast(pl.Int64).alias("n_shared_clusters"))
        .rename({"source_domain": "domain_a", "source_domain_right": "domain_b"})
        .sort("domain_a", "domain_b")
    )


def _find(parent: dict[str, str], node: str) -> str:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _syndication_merges(known: pl.DataFrame) -> pl.DataFrame:
    """Per cluster: successful unions over the groups of domains sharing a fingerprint."""
    groups = (
        known.select("cluster_id", "source_domain", *FINGERPRINT)
        .unique()
        .group_by("cluster_id", *FINGERPRINT)
        .agg(pl.col("source_domain").unique().sort().alias("domains"))
        .filter(pl.col("domains").list.len() >= 2)
        .sort("cluster_id", *FINGERPRINT)
    )
    parents: dict[str, dict[str, str]] = {}
    merges: dict[str, int] = {}
    for cluster_id, *_fingerprint, domains in groups.iter_rows():
        parent = parents.setdefault(cluster_id, {})
        for domain in domains:
            parent.setdefault(domain, domain)
        head = domains[0]
        for other in domains[1:]:
            root_head, root_other = _find(parent, head), _find(parent, other)
            if root_head != root_other:
                parent[max(root_head, root_other)] = min(root_head, root_other)
                merges[cluster_id] = merges.get(cluster_id, 0) + 1
    return pl.DataFrame(
        {"cluster_id": list(merges), "syndication_merges": list(merges.values())},
        schema={"cluster_id": pl.String, "syndication_merges": pl.Int64},
    )


def _cluster_counts(records: pl.DataFrame, known: pl.DataFrame) -> pl.DataFrame:
    families = "deduplicated_source_family_count"
    return (
        records.group_by("cluster_id")
        .agg(
            pl.len().cast(pl.Int64).alias("raw_article_count"),
            pl.col("source_domain")
            .filter(~_is_unknown())
            .drop_nulls()
            .n_unique()
            .cast(pl.Int64)
            .alias(families),
            _is_unknown().sum().cast(pl.Int64).alias("unknown_domain_rows"),
        )
        .join(_syndication_merges(known), on="cluster_id", how="left")
        .with_columns(
            (pl.col(families) - pl.col("syndication_merges").fill_null(0)).alias(
                "independent_source_units"
            )
        )
        .sort("cluster_id")
        .select(
            "cluster_id",
            "raw_article_count",
            families,
            "independent_source_units",
            "unknown_domain_rows",
        )
    )


def build_source_domains(news_records: pl.DataFrame) -> DomainTables:
    """Builds the three tables from the full article table (no period filter)."""
    known = _known(news_records)
    return DomainTables(
        families=_families(known),
        edges=_edges(known),
        cluster_counts=_cluster_counts(news_records, known),
    )


def _run(args: argparse.Namespace) -> int:
    records = pl.read_parquet(
        args.dataset / "news" / "news_records.parquet", columns=RECORD_COLUMNS
    )
    tables = build_source_domains(records)
    args.out.mkdir(parents=True, exist_ok=True)
    named = (
        ("families", tables.families),
        ("edges", tables.edges),
        ("cluster_counts", tables.cluster_counts),
    )
    for name, table in named:
        table.write_parquet(args.out / f"{name}.parquet")
        print(f"{name}.parquet: {table.height} rows")
    return 0


@register
def _register_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = command_group(subparsers, "build").add_parser(
        "source-domains", help="source-domain families, edges and cluster counts"
    )
    parser.add_argument("--dataset", type=Path, required=True, help="released dataset directory")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.set_defaults(func=_run)
