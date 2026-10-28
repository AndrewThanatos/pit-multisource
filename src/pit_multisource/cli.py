"""Command-line entry point; subcommands are registered by the modules that implement them."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence

from pit_multisource.commands import _REGISTRARS

COMMAND_MODULES = (
    "pit_multisource.acquire.binance",
    "pit_multisource.acquire.calendar",
    "pit_multisource.acquire.gdelt",
    "pit_multisource.build.bars",
    "pit_multisource.build.calendar",
    "pit_multisource.build.entities",
    "pit_multisource.build.news",
    "pit_multisource.build.source_domains",
    "pit_multisource.build.targets",
    "pit_multisource.example",
    "pit_multisource.reproduce",
)


def build_parser() -> argparse.ArgumentParser:
    for name in COMMAND_MODULES:
        importlib.import_module(name)  # importing a module registers its commands
    parser = argparse.ArgumentParser(
        prog="pit-multisource",
        description="Rebuild the point-in-time dataset and reproduce the article's numbers.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for registrar in _REGISTRARS:
        registrar(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 2
    return int(args.func(args))
