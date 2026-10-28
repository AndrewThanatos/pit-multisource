import argparse

import pytest

from pit_multisource import cli


def test_registered_command_is_dispatched(monkeypatch: pytest.MonkeyPatch):
    seen: list[str] = []

    def setup(sub: argparse._SubParsersAction) -> None:
        parser = sub.add_parser("hello", help="test command")
        parser.add_argument("--name", default="x")
        parser.set_defaults(func=lambda args: seen.append(args.name) or 0)

    monkeypatch.setattr(cli, "_REGISTRARS", [setup])

    assert cli.main(["hello", "--name", "y"]) == 0
    assert seen == ["y"]


def test_no_command_prints_help_and_fails(capsys: pytest.CaptureFixture[str]):
    assert cli.main([]) == 2
    assert "usage: pit-multisource" in capsys.readouterr().out
