"""Builds the typed entity dictionary, surface forms and alias candidates from GKG records."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import polars as pl

from pit_multisource.build.news import normalize_entity, read_gkg_records
from pit_multisource.commands import command_group, register
from pit_multisource.config import Config, load_config

# Record column -> (name written to the tables, entity type), in the order the fields are visited.
ENTITY_COLUMNS: dict[str, tuple[str, str]] = {
    "v1_persons": ("v1_persons", "person"),
    "v2_enhanced_persons": ("v2_persons", "person"),
    "v1_organizations": ("v1_organisations", "organisation"),
    "v2_enhanced_organizations": ("v2_organisations", "organisation"),
    "v21_all_names": ("all_names", "untyped_name"),
}
# The source columns GDELT writes without character offsets.
_OFFSETLESS = frozenset({"v1_persons", "v1_organisations"})
ENTITY_RECORD_COLUMNS = ("slot_time", "gkg_record_id", *ENTITY_COLUMNS)
ALIAS_DISPOSITION = "AMBIGUOUS_NO_MERGE"

_US = pl.Datetime("us", "UTC")
# Only for the raw name, which keeps its case; the normalised name uses normalize_entity.
_OFFSET = re.compile(r",\d+$")
_ASCII_PUNCTUATION = re.compile(r"[!-/:-@\[-`{-~]")

ENTITIES_SCHEMA: dict[str, pl.DataType] = {
    "entity_type": pl.String(),
    "normalized_name": pl.String(),
    "alias_normalized_key": pl.String(),
    "n_surface_forms": pl.Int32(),
    "gkg_columns": pl.String(),
    "first_knowable_at": _US,
    "first_knowable_earned_by": pl.String(),
    "n_records_development": pl.Int64(),
    "n_records_reserve": pl.Int64(),
    "corpus_cutoff": _US,
}
SURFACE_FORMS_SCHEMA: dict[str, pl.DataType] = {
    "entity_type": pl.String(),
    "normalized_name": pl.String(),
    "raw_name": pl.String(),
    "gkg_column": pl.String(),
    "first_knowable_at": _US,
    "first_knowable_earned_by": pl.String(),
    "n_records_development": pl.Int64(),
    "n_records_reserve": pl.Int64(),
}
ALIAS_CANDIDATES_SCHEMA: dict[str, pl.DataType] = {
    "entity_type": pl.String(),
    "alias_normalized_key": pl.String(),
    "n_names": pl.Int32(),
    "normalized_names": pl.String(),
    "disposition": pl.String(),
}


@dataclass(frozen=True)
class EntityTables:
    """The entity, surface-form and alias-candidate tables."""

    entities: pl.DataFrame
    surface_forms: pl.DataFrame
    alias_candidates: pl.DataFrame


def alias_key(name: str) -> str:
    """The name with ASCII punctuation turned into spaces and whitespace runs collapsed."""
    return " ".join(_ASCII_PUNCTUATION.sub(" ", name).split())


def build_entities(records: pl.DataFrame, config: Config) -> EntityTables:
    """Entity tables over the records before the boundary; counts are token occurrences."""
    boundary = config.targets.boundary
    ordered = (
        records.filter(pl.col("slot_time") < boundary)
        .select(["slot_time", "gkg_record_id", *ENTITY_COLUMNS])
        .sort("slot_time", "gkg_record_id")
        .with_row_index("pos")
    )
    tokens = pl.concat(
        [
            ordered.select(
                pl.lit(column).alias("column"),
                pl.col(column).fill_null("").str.split(";").alias("token"),
                "pos",
            ).explode("token")
            for column in ENTITY_COLUMNS
        ]
    )
    # One row per distinct field token: its occurrence count and earliest record position.
    unique = tokens.group_by("column", "token").agg(
        pl.len().alias("n"), pl.col("pos").min().alias("pos")
    )
    del tokens
    pieces = _split_tokens(unique)
    first = ordered.select("pos", pl.col("slot_time").alias("first_knowable_at"),
                           pl.col("gkg_record_id").alias("first_knowable_earned_by"))  # fmt: skip

    forms = (
        pieces.group_by("entity_type", "normalized_name", "raw_name", "gkg_column")
        .agg(pl.col("n").sum().alias("n_records_development"), pl.col("pos").min())
        .join(first, on="pos", how="left")
        .with_columns(pl.lit(None, dtype=pl.Int64).alias("n_records_reserve"))
        .select(list(SURFACE_FORMS_SCHEMA))
        .sort("entity_type", "normalized_name", "raw_name", "gkg_column")
    )
    cutoff = boundary - timedelta(microseconds=1)
    entities = (
        pieces.group_by("entity_type", "normalized_name")
        .agg(
            pl.col("raw_name").n_unique().cast(pl.Int32).alias("n_surface_forms"),
            pl.col("gkg_column").unique().sort().str.join(",").alias("gkg_columns"),
            pl.col("n").sum().alias("n_records_development"),
            pl.col("pos").min(),
        )
        .join(first, on="pos", how="left")
        .with_columns(
            pl.col("normalized_name")
            .map_elements(alias_key, return_dtype=pl.String())
            .alias("alias_normalized_key"),
            pl.lit(None, dtype=pl.Int64).alias("n_records_reserve"),
            pl.lit(cutoff, dtype=_US).alias("corpus_cutoff"),
        )
        .select(list(ENTITIES_SCHEMA))
        .sort("entity_type", "normalized_name")
    )
    return EntityTables(entities, forms, _alias_candidates(entities))


def _split_tokens(unique: pl.DataFrame) -> pl.DataFrame:
    """Distinct tokens stripped, offset-removed and normalised; empty and invalid ones handled."""
    columns, raw_tokens, counts, positions = (
        unique.get_column(c).to_list() for c in ("column", "token", "n", "pos")
    )
    kept: list[int] = []
    raw_names: list[str] = []
    normalized: list[str] = []
    for i, (column, token) in enumerate(zip(columns, raw_tokens, strict=True)):
        raw = token.strip()
        if not raw:
            continue
        raw_name = _OFFSET.sub("", raw)
        if raw_name != raw and ENTITY_COLUMNS[column][0] in _OFFSETLESS:
            raise ValueError(f"character offset in the offsetless column {column}: {raw!r}")
        name = normalize_entity(raw)
        if not name:
            continue
        if normalize_entity(name) != name:
            raise ValueError(f"entity name {name!r} is not its own normalisation")
        kept.append(i)
        raw_names.append(raw_name)
        normalized.append(name)
    return pl.DataFrame(
        {
            "entity_type": [ENTITY_COLUMNS[columns[i]][1] for i in kept],
            "normalized_name": normalized,
            "raw_name": raw_names,
            "gkg_column": [ENTITY_COLUMNS[columns[i]][0] for i in kept],
            "n": [counts[i] for i in kept],
            "pos": [positions[i] for i in kept],
        },
        schema={
            "entity_type": pl.String(),
            "normalized_name": pl.String(),
            "raw_name": pl.String(),
            "gkg_column": pl.String(),
            "n": pl.Int64(),
            "pos": pl.UInt32(),
        },
    )


def _alias_candidates(entities: pl.DataFrame) -> pl.DataFrame:
    """Groups of names sharing an alias key; recorded as ambiguous and never merged."""
    return (
        entities.group_by("entity_type", "alias_normalized_key")
        .agg(
            pl.len().cast(pl.Int32).alias("n_names"),
            pl.col("normalized_name").sort().str.join("|").alias("normalized_names"),
        )
        .filter(pl.col("n_names") > 1)
        .with_columns(pl.lit(ALIAS_DISPOSITION).alias("disposition"))
        .select(list(ALIAS_CANDIDATES_SCHEMA))
        .sort("entity_type", "alias_normalized_key")
    )


def _run(args: argparse.Namespace) -> int:
    tables = build_entities(read_gkg_records(args.dataset, ENTITY_RECORD_COLUMNS), load_config())
    args.out.mkdir(parents=True, exist_ok=True)
    named = (
        ("entities", tables.entities),
        ("entity_surface_forms", tables.surface_forms),
        ("entity_alias_candidates", tables.alias_candidates),
    )
    for name, table in named:
        table.write_parquet(args.out / f"{name}.parquet")
        print(f"{name}.parquet: {table.height} rows")
    return 0


@register
def _register_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = command_group(subparsers, "build").add_parser(
        "entities", help="typed entity dictionary from the GKG records"
    )
    parser.add_argument("--dataset", type=Path, required=True, help="released dataset directory")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.set_defaults(func=_run)
