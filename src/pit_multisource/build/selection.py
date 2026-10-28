"""Selects the crypto-related lines of a GDELT GKG slot archive and describes the result."""

from __future__ import annotations

import gzip
import hashlib
import io
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import cache

from pit_multisource.config import Config

FIELD_COUNT = 27
_RECORD_ID = 0
_URL = 4
_THEME_FIELDS = (7, 8)
_ORGANISATIONS = 13
_ALL_NAMES = 23


@dataclass(frozen=True)
class SlotSelection:
    """The lines kept from one slot archive, in archive order, and the row counters."""

    retained: list[bytes]
    rows_total: int
    rows_matched: int
    rows_malformed: int
    rows_duplicate_in_slot: int


@cache
def entity_pattern(terms: tuple[str, ...], suffix_terms: tuple[str, ...] = ()) -> re.Pattern[str]:
    """Whole-word, case-insensitive match of any term; suffix terms also match longer words."""
    alternatives = "|".join(t + (r"\w*" if t in suffix_terms else "") for t in terms)
    return re.compile(rf"\b({alternatives})\b", re.IGNORECASE)


def theme_tokens(themes: str) -> set[str]:
    """Theme codes of `;`-joined theme fields, without character offsets, upper-cased."""
    tokens = {element.split(",", 1)[0].strip().upper() for element in themes.split(";")}
    tokens.discard("")
    return tokens


def select_lines(raw_tsv: bytes, config: Config) -> SlotSelection:
    """Keeps the first line of each record id that has a theme token or an entity term."""
    wanted_tokens = set(config.news.theme_tokens)
    pattern = entity_pattern(config.news.selection_terms, config.news.suffix_terms)
    retained: list[bytes] = []
    seen: set[str] = set()
    total = malformed = matched = duplicates = 0
    for line in raw_tsv.decode("utf-8", errors="replace").split("\n"):
        if not line.strip():
            continue
        total += 1
        fields = line.split("\t")
        if len(fields) != FIELD_COUNT:
            malformed += 1
            continue
        # Amounts (field 24) are left out of the entity haystack on purpose.
        haystack = f"{fields[_ALL_NAMES]};{fields[_ORGANISATIONS]};{fields[_URL]}"
        themes = ";".join(fields[i] for i in _THEME_FIELDS)
        if not (wanted_tokens & theme_tokens(themes)) and pattern.search(haystack) is None:
            continue
        matched += 1
        if fields[_RECORD_ID] in seen:
            duplicates += 1
            continue
        seen.add(fields[_RECORD_ID])
        retained.append(line.encode("utf-8"))
    return SlotSelection(retained, total, matched, malformed, duplicates)


def shard_bytes(lines: Sequence[bytes]) -> bytes:
    """Gzip (level 9, no name, mtime 0) of the lines, each followed by a newline."""
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        for line in lines:
            handle.write(line + b"\n")
    return buffer.getvalue()


def ledger_line(
    slot: datetime,
    *,
    zip_sha256: str | None,
    selection: SlotSelection | None,
    shard: bytes | None,
    attempts: int,
    error: str | None,
    fetched_at: datetime,
    ingest_version: str,
) -> dict[str, object]:
    """One slot ledger record; `selection` is None when no archive was served."""
    if error is not None:
        status = "unresolved"
    elif selection is None:
        status = "missing_source"
    else:
        status = "retained" if selection.retained else "served_no_match"
    counts = selection or SlotSelection([], 0, 0, 0, 0)
    return {
        "slot": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,
        "zip_sha256": zip_sha256,
        "vendor_md5": None,
        "vendor_digest_state": None if selection is None else "NOT_PUBLISHED",
        "rows_total": counts.rows_total,
        "rows_matched": counts.rows_matched,
        "rows_malformed": counts.rows_malformed,
        "rows_duplicate_in_slot": counts.rows_duplicate_in_slot,
        "records_retained": len(counts.retained) or None,
        "shard_sha256": None if shard is None else hashlib.sha256(shard).hexdigest(),
        "shard_bytes": None if shard is None else len(shard),
        "attempts": attempts,
        "error": error,
        "fetched_at": fetched_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "ingest_version": ingest_version,
    }
