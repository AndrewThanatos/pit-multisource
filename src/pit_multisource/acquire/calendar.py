"""Downloads the FOMC and BLS pages the release calendar is built from."""

from __future__ import annotations

import argparse
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from pit_multisource.build.calendar import build_calendar
from pit_multisource.commands import command_group, register
from pit_multisource.config import Config, load_config

NOTICE = (
    "The Fed and BLS serve live pages: the pages fetched now may differ from the 2026 archive "
    "the released calendar was built from."
)

Fetch = Callable[[str], bytes]
Sleep = Callable[[float], None]


def urllib_fetch(user_agent: str, timeout_seconds: float) -> Fetch:
    """A fetch function that sends the given user agent; bls.gov refuses one without a contact."""

    def fetch(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body: bytes = response.read()
        return body

    return fetch


def fetch_pages(dest: Path, config: Config, fetch: Fetch, sleep: Sleep = time.sleep) -> list[Path]:
    """Fetches every page the calendar reads and writes each to its mapped path under `dest`."""
    print(NOTICE)
    written: list[Path] = []

    def read(url: str, relative: str) -> bytes:
        payload = fetch(url)
        path = dest / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        written.append(path)
        sleep(config.calendar.request_delay_seconds)
        return payload

    build_calendar(read, config)
    return written


def _run(args: argparse.Namespace) -> int:
    config = load_config()
    fetch = urllib_fetch(args.user_agent, config.calendar.timeout_seconds)
    for path in fetch_pages(args.out, config, fetch):
        print(path)
    return 0


@register
def _register_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = command_group(subparsers, "acquire").add_parser(
        "calendar", help="download the FOMC and BLS calendar pages", description=NOTICE
    )
    parser.add_argument("--out", type=Path, required=True, help="pages directory")
    parser.add_argument(
        "--user-agent", required=True, help="user agent with a contact address (bls.gov needs one)"
    )
    parser.set_defaults(func=_run)
