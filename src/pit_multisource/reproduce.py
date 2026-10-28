"""`pit-multisource reproduce`: runs every experiment and compares its numbers with the article."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from pit_multisource.commands import register
from pit_multisource.config import load_config
from pit_multisource.experiments import EXPERIMENTS

_PACKAGED = Path(__file__).parent / "expected" / "paper_values.json"
_SOURCE_TREE = Path(__file__).resolve().parents[2] / "expected" / "paper_values.json"
VALUES_PATH = _SOURCE_TREE if _SOURCE_TREE.exists() else _PACKAGED
SKIP_REASONS = {"shared_ids": "needs the shared-id file from the Materials record"}
MATCH, MISMATCH = "yes", "NO"


@dataclass(frozen=True)
class Comparison:
    """One article number against its reproduced value; status is yes, NO or `skipped: …`."""

    id: str
    section: str
    paper: str
    reproduced: str
    status: str
    supplementary: bool = False


def load_values(path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """The article values, the supplementary values and the results not reproducible here."""
    return json.loads((VALUES_PATH if path is None else path).read_text(encoding="utf-8"))


def _fixed(number: float, decimals: int) -> str:
    text = f"{number:.{decimals}f}"
    return text.lstrip("-") if float(text) == 0 else text


def matches(entry: Mapping[str, Any], reproduced: Any) -> bool:
    """Numbers agree at the article's printed decimals; flags, None and text must be equal."""
    value = entry["value"]
    if value is None or isinstance(value, bool):
        return reproduced is value
    if isinstance(value, str):
        return isinstance(reproduced, str) and reproduced == value
    if isinstance(reproduced, bool) or not isinstance(reproduced, int | float):
        return False
    decimals = entry.get("decimals", 0)
    return _fixed(value, decimals) == _fixed(reproduced * entry.get("scale", 1), decimals)


def _show(entry: Mapping[str, Any], value: Any, *, scaled: bool) -> str:
    if value is None or isinstance(value, bool | str):
        return json.dumps(value)
    if not isinstance(value, int | float):
        return repr(value)
    number = value * entry.get("scale", 1) if scaled else value
    return format(Decimal(_fixed(number, entry.get("decimals", 0))), ",")


def _compare(
    entry: Mapping[str, Any], results: Mapping[str, Any], supplementary: bool
) -> Comparison:
    paper = _show(entry, entry["value"], scaled=False)
    if entry["key"] not in results:
        return Comparison(entry["id"], entry["section"], paper, "absent", MISMATCH, supplementary)
    reproduced = results[entry["key"]]
    status = MATCH if matches(entry, reproduced) else MISMATCH
    shown = _show(entry, reproduced, scaled=True)
    return Comparison(entry["id"], entry["section"], paper, shown, status, supplementary)


def reproduce(
    dataset_root: Path,
    out: Path,
    extra: Mapping[str, Path],
    *,
    only: Collection[str] | None = None,
) -> list[Comparison]:
    """Runs the experiments, writes `out/<experiment>.json` and compares every listed value."""
    values = load_values()
    config = load_config()
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for experiment in EXPERIMENTS:
        if only is not None and experiment.name not in only:
            continue
        absent = [need for need in experiment.needs if need not in extra]
        if absent:
            skipped[experiment.name] = "skipped: " + "; ".join(SKIP_REASONS[a] for a in absent)
            print(f"{experiment.name}: {skipped[experiment.name]}", flush=True)
            continue
        start = time.perf_counter()
        results[experiment.name] = experiment.results(dataset_root, config, extra)
        (out / f"{experiment.name}.json").write_text(
            json.dumps(results[experiment.name], indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"{experiment.name}: {time.perf_counter() - start:.1f} s wall time", flush=True)
    comparisons = []
    for group, supplementary in (("paper", False), ("supplementary", True)):
        for entry in values[group]:
            name = entry["experiment"]
            if name in skipped:
                paper = _show(entry, entry["value"], scaled=False)
                comparisons.append(
                    Comparison(
                        entry["id"], entry["section"], paper, "", skipped[name], supplementary
                    )
                )
            elif name in results:
                comparisons.append(_compare(entry, results[name], supplementary))
    return comparisons


def _table(rows: list[Comparison]) -> list[str]:
    header = ("id", "section", "paper", "reproduced", "match")
    cells = [header, *((r.id, r.section, r.paper, r.reproduced, r.status) for r in rows)]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header) - 1)]
    return [
        " | ".join([*(c.ljust(w) for c, w in zip(row, widths, strict=False)), row[-1]])
        for row in cells
    ]


def _run(args: argparse.Namespace) -> int:
    extra = {"shared_ids": args.shared_ids} if args.shared_ids else {}
    started = time.perf_counter()
    comparisons = reproduce(args.dataset, args.out, extra, only=args.only)
    article = [c for c in comparisons if not c.supplementary]
    supplementary = [c for c in comparisons if c.supplementary]
    print("\nArticle values")
    print("\n".join(_table(article)))
    if supplementary:
        print("\nSupplementary values (reproduced by the code, not printed in the article)")
        print("\n".join(_table(supplementary)))
    print("\nNot reproducible here — see the Materials record:")
    for item in load_values()["not_reproducible"]:
        print(f"- {item['section']}: {item['result']}")
    counts = {s: sum(c.status == s for c in comparisons) for s in (MATCH, MISMATCH)}
    others = len(comparisons) - counts[MATCH] - counts[MISMATCH]
    print(
        f"\n{counts[MATCH]} match, {counts[MISMATCH]} mismatch, {others} skipped; "
        f"total wall time {time.perf_counter() - started:.1f} s"
    )
    return 1 if counts[MISMATCH] else 0


@register
def _register_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "reproduce", help="run every experiment and compare its numbers with the article's"
    )
    parser.add_argument("--dataset", type=Path, required=True, help="released dataset directory")
    parser.add_argument("--out", type=Path, default=Path("results"), help="results directory")
    parser.add_argument(
        "--shared-ids",
        type=Path,
        help="shared record ids from the Materials record (Table 4, Table 6 rows 4-5)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[experiment.name for experiment in EXPERIMENTS],
        metavar="EXPERIMENT",
        help="run only these experiments",
    )
    parser.set_defaults(func=_run)
