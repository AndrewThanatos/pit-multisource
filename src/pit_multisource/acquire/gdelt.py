"""Downloads GDELT GKG 2.1 slot archives and writes the selected lines with a slot ledger."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pit_multisource.build.selection import ledger_line, select_lines, shard_bytes
from pit_multisource.commands import command_group, register
from pit_multisource.config import Config, load_config

SLOT = timedelta(minutes=15)
LEDGER_NAME = "slot_ledger.jsonl"
_STAMP = "%Y%m%d%H%M%S"
_TERMINAL = {"retained", "served_no_match", "missing_source"}
_NOTICE = (
    "GDELT serves live content: the archives fetched now may differ from the ones the released "
    "dataset was built from in 2026."
)

Fetch = Callable[[str], bytes]
Sleep = Callable[[float], None]
Now = Callable[[], datetime]


class GdeltRunAborted(RuntimeError):
    """Too many slots in a row could not be fetched."""


@dataclass(frozen=True)
class SlotFetch:
    """The settled answer for one slot: an archive, a 404 (no archive, no error) or an error."""

    archive: bytes | None
    tsv: bytes | None
    attempts: int
    error: str | None


def slot_url(slot: datetime, config: Config | None = None) -> str:
    config = load_config() if config is None else config
    return f"{config.gdelt.base_url}/{slot.strftime(_STAMP)}.gkg.csv.zip"


def urllib_fetch(config: Config) -> Fetch:
    """A fetch function that sends the configured user agent and timeout."""

    def fetch(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": config.gdelt.user_agent})
        with urllib.request.urlopen(request, timeout=config.gdelt.timeout_seconds) as response:
            body: bytes = response.read()
        return body

    return fetch


def read_member(archive: bytes) -> bytes:
    """The single member of a slot archive."""
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        [name] = zf.namelist()
        return zf.read(name)


def fetch_slot(
    slot: datetime, config: Config, fetch: Fetch, sleep: Sleep = time.sleep
) -> SlotFetch:
    """Fetches one slot, retrying throttling, server errors, broken transfers and bad archives."""
    gdelt = config.gdelt
    error = "unresolved"
    for attempt in range(gdelt.max_retries + 1):
        if attempt:
            sleep(min(gdelt.backoff_base_seconds * 2 ** (attempt - 1), gdelt.backoff_cap_seconds))
        sleep(gdelt.request_delay_seconds)
        try:
            archive = fetch(slot_url(slot, config))
            return SlotFetch(archive, read_member(archive), attempt + 1, None)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return SlotFetch(None, None, attempt + 1, None)
            if exc.code != 429 and exc.code < 500:
                raise
            error = f"HTTP {exc.code}"
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
            zipfile.BadZipFile,
            zlib.error,  # a corrupt deflate stream
            EOFError,  # a truncated deflate stream
            ValueError,  # an archive with other than one member
        ) as exc:
            error = f"{type(exc).__name__}: {exc}"
    return SlotFetch(None, None, gdelt.max_retries + 1, error)


def _settled_slots(ledger: Path) -> set[str]:
    """Slots whose latest ledger line has a terminal status."""
    latest: dict[str, str] = {}
    if ledger.exists():
        for text in ledger.read_text(encoding="utf-8").splitlines():
            record = json.loads(text)
            latest[record["slot"]] = record["status"]
    return {slot for slot, status in latest.items() if status in _TERMINAL}


def _grid(start: datetime, end: datetime) -> list[datetime]:
    if start.minute % 15 or start.second or start.microsecond:
        raise ValueError(f"start must sit on the 15-minute grid, got {start.isoformat()}")
    count = max(0, -((start - end) // SLOT))
    return [start + i * SLOT for i in range(count)]


def run_slots(
    config: Config,
    start: datetime,
    end: datetime,
    out: Path,
    fetch: Fetch,
    sleep: Sleep = time.sleep,
    now: Now = lambda: datetime.now(UTC),
) -> Counter[str]:
    """Processes the slots in [start, end) in order; settled slots are skipped on a re-run."""
    ledger = out / LEDGER_NAME
    done = _settled_slots(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    statuses: Counter[str] = Counter()
    unresolved_in_a_row = 0
    for slot in _grid(start, end):
        if slot.strftime("%Y-%m-%dT%H:%M:%SZ") in done:
            continue
        fetched = fetch_slot(slot, config, fetch, sleep)
        selection = None if fetched.tsv is None else select_lines(fetched.tsv, config)
        shard = None
        if selection is not None and selection.retained:
            shard = shard_bytes(selection.retained)
            path = out / "shards" / slot.strftime("%Y-%m-%d") / f"{slot.strftime(_STAMP)}.tsv.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".tmp")
            partial.write_bytes(shard)
            partial.replace(path)
        record = ledger_line(
            slot,
            zip_sha256=None
            if fetched.archive is None
            else hashlib.sha256(fetched.archive).hexdigest(),
            selection=selection,
            shard=shard,
            attempts=fetched.attempts,
            error=fetched.error,
            fetched_at=now(),
            ingest_version=config.gdelt.ingest_version,
        )
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
        statuses[str(record["status"])] += 1
        unresolved_in_a_row = unresolved_in_a_row + 1 if fetched.error else 0
        if unresolved_in_a_row > config.gdelt.max_consecutive_unresolved:
            raise GdeltRunAborted(f"{unresolved_in_a_row} slots in a row could not be fetched")
    return statuses


def _instant(text: str) -> datetime:
    parsed = datetime.fromisoformat(text)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _run(args: argparse.Namespace) -> int:
    config = load_config()
    print(_NOTICE)
    statuses = run_slots(config, args.start, args.end, args.out, urllib_fetch(config))
    print(dict(statuses))
    return 0


@register
def _register_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = command_group(subparsers, "acquire").add_parser(
        "gdelt",
        help="download GDELT GKG slot archives and keep the crypto-related lines",
        description=_NOTICE,
    )
    parser.add_argument("--start", type=_instant, required=True, help="first slot, UTC (ISO-8601)")
    parser.add_argument("--end", type=_instant, required=True, help="end of the range, exclusive")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.set_defaults(func=_run)
