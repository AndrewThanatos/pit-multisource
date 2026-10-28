"""Builds the US macro release calendar from archived FOMC and BLS pages."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from pit_multisource.commands import command_group, register
from pit_multisource.config import CalendarConfig, Config, load_config

RELEASE_SCHEMA = pl.Schema(
    {
        "release_id": pl.String(),
        "family": pl.String(),
        "tier": pl.Int8(),
        "released_at": pl.Datetime("us", "UTC"),
        "scheduled": pl.Boolean(),
        "schedule_available_at": pl.Datetime("us", "UTC"),
        "release_time_source": pl.String(),
        "label": pl.String(),
        "source_url": pl.String(),
        "source_sha256": pl.String(),
    }
)
FOMC = "fomc_statement"

# Reads one page given its URL and its path relative to the pages directory.
ReadPage = Callable[[str, str], bytes]
Row = dict[str, object]

_TAGS = re.compile(r"<[^>]*>")
_SPACES = re.compile(r"\s+")
_RELEASE_LINE = re.compile(
    r"[Ff]or release at\s+(\d{1,2}):(\d{2})\s*([ap])\.?m\.?\s*([A-Z]{3,4})", re.I
)
_YEAR_PANEL = re.compile(
    r'<h5[^>]*class="[^"]*panel-heading[^"]*"[^>]*>(.*?)</h5>(.*?)(?=<h5|\Z)', re.S
)
_QUALIFIER = re.compile(r"\(([^)]+)\)")
_STATEMENT_LINK = re.compile(r"/newsevents/pressreleases/(monetary(\d{8})[a-z]*)\.htm")
_ANY_PRESS_LINK = re.compile(r"/newsevents/pressreleases/monetary\d{8}")
_ROLLING_YEAR = re.compile(
    r'<div class="panel panel-default"><div class="panel-heading">'
    r'<h4><a id="\d+">(\d{4}) FOMC Meetings</a></h4></div>'
)
_MEETING_BLOCK = re.compile(r'<div[^>]*class="[^"]*\brow fomc-meeting\b[^"]*"')
_STATEMENT_SECTION = re.compile(r"<strong>\s*Statement:\s*</strong>(.*?)(?=<strong>|\Z)", re.S)
_MONTH_CELL = re.compile(r'class="[^"]*fomc-meeting__month[^"]*"[^>]*>(.*?)</div>', re.S)
_DAYS_CELL = re.compile(r'class="[^"]*fomc-meeting__date[^"]*"[^>]*>(.*?)</div>', re.S)
_DAY_RANGE = re.compile(r"(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?")
_MONTHS = (
    "January February March April May June July August September October November December".split()
)
_TABLE_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_TABLE_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)


@dataclass(frozen=True)
class Meeting:
    """One FOMC meeting; a meeting that has not happened yet has no statement name."""

    label: str
    statement_name: str | None
    released_on: date
    scheduled: bool


def text_of(html: str) -> str:
    """The text of an HTML fragment: tags become spaces, whitespace is collapsed and stripped."""
    return _SPACES.sub(" ", _TAGS.sub(" ", html)).strip()


def to_utc(day: date, wall: time, zone: ZoneInfo, stated_zone: str | None = None) -> datetime:
    """Converts a wall-clock time to UTC; a stated EST/EDT must match the date's own zone name."""
    local = datetime(day.year, day.month, day.day, wall.hour, wall.minute, tzinfo=zone)
    resolved = local.tzname()
    if stated_zone is not None and resolved is not None and stated_zone.upper() != resolved:
        raise ValueError(f"{day} is {resolved} but the page states {stated_zone}")
    return local.astimezone(UTC)


def parse_release_time(html: str) -> tuple[time, str] | None:
    """The time and zone of a line like `For release at 2:00 p.m. EST`, or None."""
    found = _RELEASE_LINE.search(html)
    if found is None:
        return None
    hour, minute, meridiem, zone = found.groups()
    clock = int(hour) % 12 + (12 if meridiem.lower() == "p" else 0)
    return time(clock, int(minute)), zone.upper()


def parse_fomc_year(html: str) -> list[Meeting]:
    """Meetings on a historical year page that produced a statement."""
    meetings = []
    for heading_html, body in _YEAR_PANEL.findall(html):
        heading = text_of(heading_html)
        qualifiers = {q.strip().lower() for q in _QUALIFIER.findall(heading)}
        if qualifiers & {"cancelled", "canceled"} or "Meeting" not in heading:
            continue
        link = _STATEMENT_LINK.search(body)
        if link is None:
            continue
        released_on = datetime.strptime(link.group(2), "%Y%m%d").date()
        scheduled = "unscheduled" not in qualifiers
        meetings.append(Meeting(heading, link.group(1), released_on, scheduled))
    return meetings


def _block_label(block: str) -> str:
    month, days = _MONTH_CELL.search(block), _DAYS_CELL.search(block)
    parts = (text_of(month.group(1)) if month else "", text_of(days.group(1)) if days else "")
    return " ".join([*(part for part in parts if part), "Meeting"])


def _future_meeting(block: str, year: int) -> Meeting:
    """A meeting without a statement yet, dated on the last day of its day range."""
    month, days = _MONTH_CELL.search(block), _DAYS_CELL.search(block)
    day_range = _DAY_RANGE.search(text_of(days.group(1))) if days else None
    name = text_of(month.group(1)).lower() if month else ""
    numbers = {m.lower(): i for i, m in enumerate(_MONTHS, start=1)}
    if day_range is None or name not in numbers:
        raise ValueError(f"unreadable meeting block in the {year} panel of the rolling page")
    released_on = date(year, numbers[name], int(day_range.group(2) or day_range.group(1)))
    return Meeting(_block_label(block), None, released_on, True)


def parse_fomc_rolling(html: str) -> list[Meeting]:
    """Past and future meetings on the rolling calendar page, which groups them by year."""
    sections = _ROLLING_YEAR.split(html)
    years, bodies = sections[1::2], sections[2::2]
    in_panels = sum(len(_MEETING_BLOCK.split(body)) - 1 for body in bodies)
    if in_panels != len(_MEETING_BLOCK.split(html)) - 1:
        raise ValueError("the rolling page has meeting blocks outside a year panel")
    meetings = []
    for year_text, body in zip(years, bodies, strict=True):
        for block in _MEETING_BLOCK.split(body)[1:]:
            text = text_of(block).lower()
            if "cancelled" in text or "canceled" in text:
                continue
            section = _STATEMENT_SECTION.search(block)
            link = _STATEMENT_LINK.search(section.group(1)) if section else None
            if link is not None:
                released_on = datetime.strptime(link.group(2), "%Y%m%d").date()
                scheduled = "unscheduled" not in text
                meetings.append(Meeting(_block_label(block), link.group(1), released_on, scheduled))
            elif _ANY_PRESS_LINK.search(block) or "notation vote" in text:
                continue
            elif section is not None:
                raise ValueError(f"unreadable statement link in the {year_text} rolling panel")
            else:
                meetings.append(_future_meeting(block, int(year_text)))
    return meetings


def parse_bls_year(html: str, titles: Mapping[str, str]) -> list[tuple[str, date, time, str]]:
    """`(family, day, wall clock, label)` for rows whose title is `<title> for <Month> <Year>`."""
    months = "|".join(_MONTHS)
    patterns = [
        (re.compile(rf"^{re.escape(title)} for (?:{months}) \d{{4}}$"), family)
        for family, title in titles.items()
    ]
    rows = []
    for row_html in _TABLE_ROW.findall(html):
        cells = [text_of(cell) for cell in _TABLE_CELL.findall(row_html)]
        if len(cells) < 3:
            continue
        label = cells[2]
        family = next((name for pattern, name in patterns if pattern.match(label)), None)
        if family is None:
            continue
        try:
            day = datetime.strptime(cells[0].split(", ", 1)[1], "%B %d, %Y").date()
            wall = datetime.strptime(cells[1].replace("&nbsp;", "").strip(), "%I:%M %p").time()
        except (ValueError, IndexError):
            continue
        rows.append((family, day, wall, label))
    return rows


def _row(
    family: str,
    release_id: str,
    released_at: datetime,
    scheduled: bool,
    source: str,
    label: str,
    url: str,
    sha: str,
) -> Row:
    """One output row; a scheduled release is knowable from 1 January of its UTC year."""
    available = datetime(released_at.year, 1, 1, tzinfo=UTC) if scheduled else released_at
    return {
        "release_id": release_id,
        "family": family,
        "released_at": released_at,
        "scheduled": scheduled,
        "schedule_available_at": available,
        "release_time_source": source,
        "label": label,
        "source_url": url,
        "source_sha256": sha,
    }


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decode(payload: bytes) -> str:
    return payload.decode("utf-8", errors="replace")


def _fomc_rows(read: ReadPage, cal: CalendarConfig) -> list[Row]:
    """FOMC statements per year; each row carries the digest of the year or rolling page."""
    zone = ZoneInfo(cal.timezone)
    rows: list[Row] = []
    rolling: tuple[list[Meeting], str] | None = None
    for year in range(cal.first_year, cal.last_year + 1):
        meetings: list[Meeting] = []
        sha = ""
        if year < cal.fomc_rolling_from_year:
            page = read(cal.urls.fomc_year.format(year=year), cal.pages.fomc_year.format(year=year))
            meetings, sha = parse_fomc_year(_decode(page)), _sha(page)
        if not meetings:
            if rolling is None:
                page = read(cal.urls.fomc_rolling, cal.pages.fomc_rolling)
                rolling = parse_fomc_rolling(_decode(page)), _sha(page)
            meetings, sha = rolling
        year_rows = []
        for meeting in meetings:
            if meeting.released_on.year != year:
                continue
            if meeting.statement_name is None:
                stated = None
                release_id = f"{FOMC}:scheduled-{meeting.released_on:%Y%m%d}"
                source, url = "schedule_page", cal.urls.fomc_rolling
            else:
                name = meeting.statement_name
                url = cal.urls.fomc_press.format(name=name)
                stated = parse_release_time(
                    _decode(read(url, cal.pages.fomc_press.format(name=name)))
                )
                release_id, source = f"{FOMC}:{name}", "stated" if stated else "convention"
            wall, stated_zone = stated if stated else (cal.fomc_statement_time, None)
            released_at = to_utc(meeting.released_on, wall, zone, stated_zone)
            year_rows.append(
                _row(
                    FOMC,
                    release_id,
                    released_at,
                    meeting.scheduled,
                    source,
                    meeting.label,
                    url,
                    sha,
                )
            )
        if not year_rows:
            raise ValueError(f"no FOMC statement found for {year}")
        rows += year_rows
    return rows


def _bls_rows(read: ReadPage, cal: CalendarConfig) -> list[Row]:
    """BLS releases of the configured titles from each year's schedule page."""
    zone = ZoneInfo(cal.timezone)
    rows = []
    for year in range(cal.first_year, cal.last_year + 1):
        url = cal.urls.bls_year.format(year=year)
        page = read(url, cal.pages.bls_year.format(year=year))
        sha = _sha(page)
        for family, day, wall, label in parse_bls_year(_decode(page), cal.bls_titles):
            released_at = to_utc(day, wall, zone)
            release_id = f"{family}:{released_at:%Y%m%dT%H%M}"
            rows.append(
                _row(family, release_id, released_at, True, "schedule_page", label, url, sha)
            )
    return rows


def build_calendar(read: ReadPage, config: Config) -> pl.DataFrame:
    """The release calendar from pages supplied by `read`, sorted by release time and id."""
    cal = config.calendar
    rows = _fomc_rows(read, cal) + _bls_rows(read, cal)
    rows.sort(key=lambda row: (row["released_at"], row["release_id"]))
    for row in rows:
        row["tier"] = cal.tiers[str(row["family"])]
    return pl.DataFrame(
        {name: [row[name] for row in rows] for name in RELEASE_SCHEMA}, schema=RELEASE_SCHEMA
    )


def parse_calendar(pages_root: Path, config: Config) -> pl.DataFrame:
    """The release calendar from the archived pages under `pages_root`, found by the page map."""
    return build_calendar(lambda url, relative: (pages_root / relative).read_bytes(), config)


def _run(args: argparse.Namespace) -> int:
    calendar = parse_calendar(args.pages, load_config())
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "macro_releases.parquet"
    calendar.write_parquet(target)
    print(f"{target.name}: {calendar.height} rows")
    return 0


@register
def _register_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = command_group(subparsers, "build").add_parser(
        "calendar", help="US macro release calendar from archived FOMC and BLS pages"
    )
    parser.add_argument("--pages", type=Path, required=True, help="archived calendar pages")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.set_defaults(func=_run)
