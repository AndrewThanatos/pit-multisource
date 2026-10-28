"""The subcommand registry; command modules attach their parsers here."""

from __future__ import annotations

import argparse
from collections.abc import Callable

Registrar = Callable[[argparse._SubParsersAction], None]

_REGISTRARS: list[Registrar] = []


def register(registrar: Registrar) -> Registrar:
    """Adds a function that attaches its subparsers (with set_defaults(func=...)) to the CLI."""
    _REGISTRARS.append(registrar)
    return registrar


def command_group(subparsers: argparse._SubParsersAction, name: str) -> argparse._SubParsersAction:
    """Returns the subparsers of a command group such as `build`, creating the group once."""
    group = subparsers.choices.get(name) or subparsers.add_parser(name, help=f"{name} commands")
    if not hasattr(group, "_members"):
        group._members = group.add_subparsers(dest=f"{name}_command", metavar="<command>")  # type: ignore[attr-defined]
    members: argparse._SubParsersAction = group._members  # type: ignore[attr-defined]
    return members
