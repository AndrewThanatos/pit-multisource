"""Checks on the published face of the repository: README, citation files and file contents."""

import argparse
import codecs
import ipaddress
import json
import re
import shlex
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

from pit_multisource.cli import build_parser

ROOT = Path(__file__).resolve().parents[2]
THIS_FILE = Path(__file__).resolve().relative_to(ROOT).as_posix()

ARTICLE_TITLE = (
    "Point-in-Time Consistency for Multisource Time-Series Data: A Formal Model and Construction"
    " Method for Evolving News Metadata"
)
ARTICLE_AUTHORS = ["Shymkovych", "Berezhniuk", "Telenyk", "Nowakowski"]
ORCID = "0009-0003-4581-2520"
AFFILIATION = 'National Technical University of Ukraine "Igor Sikorsky Kyiv Polytechnic Institute"'

CODE_DOI = "10.5281/zenodo.23886257"
DATASET_DOI = "10.5281/zenodo.23886186"
MATERIALS_DOI = "10.5281/zenodo.23886238"
PLACEHOLDER = re.compile(r"\b10\.\d{4,9}/(?:zenodo\.)?[A-Z]{4,}\b")

# Stored in ROT13 so that this file does not contain the terms it looks for.
_ROT13 = ("pnzcnvta", "yvarntr", "ynqqre", "nagribegn", "cbfgibegn", "nsgrevd")
_ROT13_PATHS = ("/hfref/", "/cevingr/", "/ubzr/", "qrfxgbc", "n.orerguavhx")
BANNED_SUBSTRINGS = tuple(codecs.decode(t, "rot13") for t in _ROT13 + _ROT13_PATHS)
BANNED_WORD = re.compile(r"\b" + codecs.decode("NQE", "rot13") + r"\b")
HOLDOUT_WORD = codecs.decode("erfreir", "rot13")
HOLDOUT_ALLOWED = ("n_records_" + HOLDOUT_WORD, "federal " + HOLDOUT_WORD, "federal" + HOLDOUT_WORD)
SECRET_SHAPES = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})"),
    "AWS key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "Slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "Google key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),
    "Telegram token": re.compile(r"\b\d{8,10}:AA[0-9A-Za-z_-]{30,}"),
    "assigned secret": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|passw(?:or)?d)\b[\"']?\s*[:=]\s*[\"']?[\w/+.\-]{16,}"
    ),
}
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def repo_files() -> dict[str, str]:
    """Text of every tracked or new, not ignored file, keyed by its path in the repository."""
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    texts = {}
    for name in sorted(set(listed.decode().split("\0")) - {""}):
        path = ROOT / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" not in data:
            texts[name] = data.decode("utf-8", errors="replace")
    return texts


def _is_public_address(text: str) -> bool:
    try:
        return ipaddress.ip_address(text).is_global
    except ValueError:
        return False


def test_no_internal_terms_or_private_paths():
    found = []
    for name, text in repo_files().items():
        lowered = text.lower()
        found += [(name, term) for term in BANNED_SUBSTRINGS if term in lowered]
        found += [(name, m.group()) for m in BANNED_WORD.finditer(text)]
        for allowed in HOLDOUT_ALLOWED:
            lowered = lowered.replace(allowed, "")
        found += [(name, HOLDOUT_WORD)] if HOLDOUT_WORD in lowered else []
        found += [(name, kind) for kind, shape in SECRET_SHAPES.items() if shape.search(text)]
        found += [(name, ip) for ip in IPV4.findall(text) if _is_public_address(ip)]
    assert found == []


COMMAND_LINE = re.compile(r"\s*(?:\w+=\S*\s+)*(?:uv run\s+)?pit-multisource(?:\s|$)")


def readme_commands(text: str) -> list[list[str]]:
    """Arguments of every `pit-multisource` call in the README's code blocks and inline code."""
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.S)
    lines = [line for block in blocks for line in block.replace("\\\n", " ").splitlines()]
    lines += re.findall(r"`([^`\n]+)`", re.sub(r"```.*?```", "", text, flags=re.S))
    commands = []
    for line in filter(COMMAND_LINE.match, lines):
        words = shlex.split(line, comments=True)
        while words and re.fullmatch(r"\w+=\S*", words[0]):
            words.pop(0)
        if words[:2] == ["uv", "run"]:
            words = words[2:]
        if words[:1] == ["pit-multisource"]:
            commands.append(words[1:])
    return commands


def _subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def leaf_commands(parser: argparse.ArgumentParser, path: tuple[str, ...] = ()) -> set[tuple]:
    """Every runnable command path, such as ('build', 'news')."""
    children = _subcommands(parser)
    if not children:
        return {path}
    return set().union(*(leaf_commands(p, (*path, n)) for n, p in children.items()))


def check_command(parser: argparse.ArgumentParser, words: list[str]) -> tuple[tuple, list[str]]:
    """The command or command group the words name and the flags it does not have."""
    path: list[str] = []
    while (children := _subcommands(parser)) and words and not words[0].startswith("-"):
        assert words[0] in children, f"unknown command: {' '.join([*path, *words])}"
        path.append(words[0])
        parser, words = children[words[0]], words[1:]
    flags = {s for action in parser._actions for s in action.option_strings}
    unknown = [w.split("=")[0] for w in words if w.startswith("-") and w.split("=")[0] not in flags]
    return tuple(path), unknown


def test_readme_commands_exist_in_cli():
    parser = build_parser()
    commands = readme_commands((ROOT / "README.md").read_text(encoding="utf-8"))
    shown = set()
    for words in commands:
        path, unknown = check_command(parser, words)
        assert unknown == [], f"{' '.join(path)} has no {unknown}"
        shown.add(path)
    assert leaf_commands(parser) <= shown


def test_readme_command_parser_reads_blocks_and_inline_code():
    text = (
        "Run `pit-multisource example --dataset d` first.\n"
        "```bash\nSSL_CERT_FILE=/etc/ssl/cert.pem uv run pit-multisource build news \\\n"
        "    --dataset d --out o   # comment\n```\n"
    )
    assert readme_commands(text) == [
        ["build", "news", "--dataset", "d", "--out", "o"],
        ["example", "--dataset", "d"],
    ]
    parser = build_parser()
    assert check_command(parser, ["build", "news", "--outdir", "o"]) == (
        ("build", "news"),
        ["--outdir"],
    )
    assert check_command(parser, ["build", "--help"]) == (("build",), [])
    with pytest.raises(AssertionError, match="unknown command: build tables"):
        check_command(parser, ["build", "tables"])


def test_citation_cff_fields():
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert cff["cff-version"] == "1.2.0"
    assert cff["type"] == "software"
    assert cff["license"] == "MIT"
    assert cff["version"] == project["version"]
    assert cff["doi"] == CODE_DOI
    assert cff["authors"] == [
        {
            "family-names": "Berezhniuk",
            "given-names": "Andrii",
            "orcid": f"https://orcid.org/{ORCID}",
            "affiliation": AFFILIATION,
        }
    ]
    article, dataset = cff["references"]
    assert (article["type"], article["title"]) == ("article", ARTICLE_TITLE)
    assert article["journal"] == "Journal of Big Data"
    assert [a["family-names"] for a in article["authors"]] == ARTICLE_AUTHORS
    assert (dataset["type"], dataset["doi"]) == ("dataset", DATASET_DOI)
    assert "doi" not in article
    assert [a["family-names"] for a in dataset["authors"]] == ["Berezhniuk"]


def test_zenodo_json_fields():
    meta = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))

    assert meta["upload_type"] == "software"
    assert meta["license"] == "MIT"
    assert meta["version"] == cff["version"]
    assert meta["creators"] == [
        {"name": "Berezhniuk, Andrii", "orcid": ORCID, "affiliation": AFFILIATION}
    ]
    assert meta["keywords"] == cff["keywords"]
    relations = {(r["relation"], r["identifier"]) for r in meta["related_identifiers"]}
    assert relations == {
        ("references", DATASET_DOI),
        ("references", MATERIALS_DOI),
    }


def test_no_doi_placeholders_remain_and_real_dois_are_used():
    files = {name: text for name, text in repo_files().items() if name != THIS_FILE}
    assert not [name for name, text in files.items() if PLACEHOLDER.search(text)]
    assert CODE_DOI in files["README.md"] and DATASET_DOI in files["README.md"]
    assert MATERIALS_DOI in files["README.md"]
