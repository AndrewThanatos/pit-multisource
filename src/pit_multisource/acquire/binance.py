"""Downloads Binance monthly kline archives and verifies them against the published checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections.abc import Callable
from pathlib import Path

from pit_multisource.build.bars import INTERVALS, months
from pit_multisource.commands import command_group, register
from pit_multisource.config import load_config

_BASE = "https://data.binance.vision/data/spot/monthly/klines"
_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

Fetch = Callable[[str], bytes]


class ChecksumMismatchError(ValueError):
    """The downloaded archive does not match its published sha256."""


def urllib_fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        data: bytes = response.read()
    return data


def monthly_zip_url(symbol: str, interval: str, month: str) -> str:
    if not _MONTH.match(month):
        raise ValueError(f"month must look like YYYY-MM, got {month!r}")
    return f"{_BASE}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"


def _published_sha256(text: str) -> str:
    token = text.split()[0] if text.split() else ""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", token):
        raise ValueError(f"malformed checksum file: {text!r}")
    return token.lower()


def download_month(
    symbol: str, interval: str, month: str, dest: Path, fetch: Fetch = urllib_fetch
) -> Path:
    """Fetches one archive into `dest` unless a verified copy is there; refuses a bad checksum."""
    url = monthly_zip_url(symbol, interval, month)
    expected = _published_sha256(fetch(url + ".CHECKSUM").decode("utf-8"))
    path = dest / url.rsplit("/", 1)[1]
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return path
    payload = fetch(url)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ChecksumMismatchError(
            f"checksum mismatch for {url}: expected {expected}, got {actual}"
        )
    dest.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".zip.part")
    partial.write_bytes(payload)
    partial.replace(path)
    record = {"url": url, "sha256": actual, "size_bytes": len(payload)}
    with (dest / "download_manifest.jsonl").open("a", encoding="utf-8") as manifest:
        manifest.write(json.dumps(record) + "\n")
    return path


def _run(args: argparse.Namespace) -> int:
    config = load_config()
    for symbol in config.bars.symbols:
        for interval in INTERVALS:
            for month in months(config):
                dest = args.out / "binance" / symbol / interval
                print(download_month(symbol, interval, month, dest))
    return 0


@register
def _register_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = command_group(subparsers, "acquire").add_parser(
        "binance", help="download the monthly Binance spot kline archives"
    )
    parser.add_argument("--out", type=Path, required=True, help="raw sources directory")
    parser.set_defaults(func=_run)
