"""Agreement between the theme annotation and the entity-name test, per asset.

Annotator A is GDELT's theme list (fields 7 and 8): the asset marker is in some theme. Annotator B
is a text test on the extracted names plus organizations (field 13): the asset token is in some
normalised name. The deployed mapping reads names from field 24 (amounts, one column off); the
corrected mapping reads field 23. Every record that parses (a valid date and a non-empty URL)
counts, development and later months alike. Keys of the returned dict are
`agreement.<asset>.<mapping>.<field>` with asset `btc` or `eth`, mapping `deployed` or
`corrected`, and field `both`, `a_only`, `b_only`, `neither`, `rows`, `kappa` or `psa`. An
undefined kappa or positive specific agreement is None.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import polars as pl

from pit_multisource.build.news import normalize_entity, read_gkg_records

ASSETS = {"btc": ("BITCOIN", "bitcoin"), "eth": ("ETHEREUM", "ethereum")}
MAPPINGS = {"deployed": "v21_amounts", "corrected": "v21_all_names"}
COLUMNS = (
    "date",
    "document_identifier",
    "v1_themes",
    "v2_enhanced_themes",
    "v1_organizations",
    "v21_all_names",
    "v21_amounts",
)
CELLS = ("both", "a_only", "b_only", "neither")
_DATE_FORMAT = "%Y%m%d%H%M%S"
_CHUNK_ROWS = 100_000


def cohen_kappa(both: int, a_only: int, b_only: int, neither: int) -> float | None:
    """Cohen's kappa of a 2x2 table; None when a marginal is 0 or N or chance agreement is 1."""
    total = both + a_only + b_only + neither
    positives_a = both + a_only
    positives_b = both + b_only
    if total == 0 or positives_a in (0, total) or positives_b in (0, total):
        return None
    p_a = positives_a / total
    p_b = positives_b / total
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected == 1.0:
        return None
    return ((both + neither) / total - expected) / (1 - expected)


def positive_specific_agreement(both: int, a_only: int, b_only: int) -> float | None:
    """2 * both / (positives_a + positives_b); None when either annotator never fires."""
    positives_a = both + a_only
    positives_b = both + b_only
    if positives_a == 0 or positives_b == 0:
        return None
    return 2 * both / (positives_a + positives_b)


def _parses(date: str | None, url: str | None) -> bool:
    """The raw-line parser's keep rule: a valid date in field 1 and a non-empty URL in field 4."""
    if not url:
        return False
    try:
        datetime.strptime(date or "", _DATE_FORMAT)
    except ValueError:
        return False
    return True


def _theme_flag(v1: str | None, v2: str | None, marker: str) -> bool:
    """Marker in some theme, each cut at its first comma, upper-cased."""
    return any(
        marker in token.split(",")[0].upper() for token in f"{v1 or ''};{v2 or ''}".split(";")
    )


def _entity_flag(names: str | None, orgs: str | None, token: str) -> bool:
    """Token in some normalised name of the names field and the organizations field."""
    return any(token in normalize_entity(raw) for raw in f"{names or ''};{orgs or ''}".split(";"))


def _count(
    frame: pl.DataFrame, variants: Mapping[str, tuple[str, str, str]]
) -> dict[str, dict[str, int]]:
    cells = {name: dict.fromkeys(CELLS, 0) for name in variants}
    for start in range(0, frame.height, _CHUNK_ROWS):
        chunk = frame.slice(start, _CHUNK_ROWS).select(COLUMNS)
        for date, url, themes_1, themes_2, orgs, all_names, amounts in chunk.iter_rows():
            if not _parses(date, url):
                continue
            names = {"v21_all_names": all_names, "v21_amounts": amounts}
            for name, (marker, token, column) in variants.items():
                a = _theme_flag(themes_1, themes_2, marker)
                b = _entity_flag(names[column], orgs, token)
                cells[name][
                    "both" if a and b else "a_only" if a else "b_only" if b else "neither"
                ] += 1
    return cells


def agreement_cells(
    frame: pl.DataFrame, marker: str, token: str, entity_column: str
) -> dict[str, int]:
    """The four cells for one marker, token and names column over `COLUMNS` of a record frame."""
    return _count(frame, {"cells": (marker, token, entity_column)})["cells"]


def run_agreement(dataset_root: Path) -> dict[str, int | float | None]:
    """Cells, kappa and positive specific agreement per asset and mapping over all records."""
    frame = read_gkg_records(Path(dataset_root), columns=["slot_time", "gkg_record_id", *COLUMNS])
    variants = {
        f"{asset}.{mapping}": (marker, token, column)
        for asset, (marker, token) in ASSETS.items()
        for mapping, column in MAPPINGS.items()
    }
    result: dict[str, int | float | None] = {}
    for name, cells in _count(frame, variants).items():
        key = f"agreement.{name}"
        for cell in CELLS:
            result[f"{key}.{cell}"] = cells[cell]
        result[f"{key}.rows"] = sum(cells.values())
        result[f"{key}.kappa"] = cohen_kappa(*(cells[c] for c in CELLS))
        result[f"{key}.psa"] = positive_specific_agreement(
            cells["both"], cells["a_only"], cells["b_only"]
        )
    return result
