import dataclasses
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from pit_multisource.acquire.calendar import fetch_pages
from pit_multisource.build.calendar import RELEASE_SCHEMA, parse_calendar
from pit_multisource.config import Config, load_config

BLS_2020 = "https://www.bls.gov/schedule/2020/home.htm"
PRESS = "https://www.federalreserve.gov/newsevents/pressreleases/{}.htm"
ROLLING = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"


def _config(year: int) -> Config:
    base = load_config()
    calendar = dataclasses.replace(base.calendar, first_year=year, last_year=year)
    return dataclasses.replace(base, calendar=calendar)


def _panel(heading: str, name: str | None = None) -> str:
    link = f'<p><a href="/newsevents/pressreleases/{name}.htm">Statement</a></p>' if name else ""
    return (
        '<div class="panel panel-default panel-padded">'
        f'<h5 class="panel-heading panel-heading--shaded">{heading}</h5>'
        f'<div class="row">{link}<p>Minutes</p></div></div>\n'
    )


def _press(stated: str) -> str:
    return f"<html><body><p>For release at {stated}</p><p>Text.</p></body></html>"


def _bls_row(day: str, wall: str, name: str, period: str) -> str:
    return (
        f'<tr><td class="date-cell"><p>{day}</p></td>'
        f'<td class="time-cell"><p>{wall}</p></td>'
        f'<td class="desc-cell"><p><strong>{name}</strong> for {period}</p></td></tr>\n'
    )


def _bls_page(*rows: str) -> str:
    header = "<tr><th><p>Date</p></th><th><p>Time</p></th><th><p>Release</p></th></tr>\n"
    return f"<html><table>{header}{''.join(rows)}</table></html>"


def _block(month: str, days: str, body: str = "") -> str:
    return (
        '<div class="row fomc-meeting" ">'
        f'<div class="fomc-meeting__month col-xs-5"><strong>{month}</strong></div>'
        f'<div class="fomc-meeting__date col-xs-4">{days}</div>'
        f'<div class="col-xs-12">{body}</div></div>\n'
    )


def _year_panel(year: int, *blocks: str) -> str:
    return (
        '<div class="panel panel-default"><div class="panel-heading">'
        f'<h4><a id="{year}0">{year} FOMC Meetings</a></h4></div>\n' + "".join(blocks) + "</div>\n"
    )


def _pages_2020() -> dict[str, str]:
    historical = "<html><body>" + "".join(
        [
            _panel("January 28-29 Meeting - 2020", "monetary20200129a"),
            _panel("March 2 (unscheduled) Meeting - 2020", "monetary20200303a"),
            _panel("March 17-18 (cancelled) Meeting - 2020"),
            _panel("March 19 (notation vote) - 2020", "monetary20200319a"),
            _panel("July 28-29 Meeting - 2020", "monetary20200729a"),
        ]
    )
    return {
        "fomc/historical-2020.html": historical + "</body></html>",
        "fomc/monetary20200129a.html": _press("2:00 p.m. EST"),
        "fomc/monetary20200303a.html": _press("10:00 a.m. EST"),
        "fomc/monetary20200729a.html": _press("2:00 p.m. EDT"),
        "bls/year-2020.html": _bls_page(
            _bls_row(
                "Tuesday, January 14, 2020", "08:30 AM", "Consumer Price Index", "December 2019"
            ),
            _bls_row("Tuesday, January 14, 2020", "08:30 AM", "Real Earnings", "December 2019"),
            _bls_row("Friday, July 10, 2020", "08:30 AM", "Producer Price Index", "June 2020"),
        ),
    }


def _pages_2026() -> dict[str, str]:
    statement = (
        '<strong>Statement:</strong><br><a href="/monetarypolicy/files/monetary20260128a1.pdf">PDF'
        '</a> | <a href="/newsevents/pressreleases/monetary20260128a.htm">HTML</a><br>'
        '<a href="/newsevents/pressreleases/monetary20260128a1.htm">Implementation Note</a>'
        '<strong>Minutes:</strong><a href="/monetarypolicy/fomcminutes20260128.htm">HTML</a>'
    )
    notation = '<a href="/newsevents/pressreleases/monetary20260822a.htm">Statement on Goals</a>'
    rolling = "<html><body>" + "".join(
        [
            _year_panel(
                2027,
                _block("January", "26-27"),
            ),
            _year_panel(
                2026,
                _block("January", "27-28", statement),
                _block("August", "22 (notation vote)", notation),
                _block("September", "15-16*"),
            ),
        ]
    )
    return {
        "fomc/calendars.9cb9e5ff5b19.html": rolling + "</body></html>",
        "fomc/monetary20260128a.html": _press("2:00 p.m. EST"),
        "bls/year-2026.html": _bls_page(
            _bls_row("Friday, March 6, 2026", "08:30 AM", "Employment Situation", "February 2026"),
        ),
    }


def _write(root: Path, pages: dict[str, str]) -> Path:
    for relative, text in pages.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _row(frame: pl.DataFrame, release_id: str) -> dict:
    rows = frame.filter(pl.col("release_id") == release_id).to_dicts()
    assert len(rows) == 1, release_id
    return rows[0]


def _utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def test_unscheduled_meeting_available_at_release(tmp_path: Path):
    frame = parse_calendar(_write(tmp_path, _pages_2020()), _config(2020))

    row = _row(frame, "fomc_statement:monetary20200303a")
    assert row["scheduled"] is False
    assert row["released_at"] == _utc(2020, 3, 3, 15)
    assert row["schedule_available_at"] == row["released_at"]
    assert row["label"] == "March 2 (unscheduled) Meeting - 2020"
    assert frame.filter(pl.col("family") == "fomc_statement")["release_id"].to_list() == [
        "fomc_statement:monetary20200129a",
        "fomc_statement:monetary20200303a",
        "fomc_statement:monetary20200729a",
    ]


def test_scheduled_available_from_first_january(tmp_path: Path):
    pages = _pages_2020()
    frame = parse_calendar(_write(tmp_path, pages), _config(2020))

    assert frame.schema == RELEASE_SCHEMA
    assert frame["release_id"].to_list() == [
        "cpi:20200114T1330",
        "fomc_statement:monetary20200129a",
        "fomc_statement:monetary20200303a",
        "ppi:20200710T1230",
        "fomc_statement:monetary20200729a",
    ]
    scheduled = frame.filter(pl.col("scheduled"))
    assert scheduled.height == 4
    assert set(scheduled["schedule_available_at"].to_list()) == {_utc(2020, 1, 1)}
    cpi = _row(frame, "cpi:20200114T1330")
    assert (cpi["family"], cpi["tier"]) == ("cpi", 3)
    assert cpi["released_at"] == _utc(2020, 1, 14, 13, 30)
    assert cpi["release_time_source"] == "schedule_page"
    assert cpi["label"] == "Consumer Price Index for December 2019"
    assert cpi["source_url"] == BLS_2020
    assert cpi["source_sha256"] == hashlib.sha256(pages["bls/year-2020.html"].encode()).hexdigest()


def test_stated_release_time_used(tmp_path: Path):
    pages = _pages_2020()
    pages["fomc/monetary20200129a.html"] = "<html><body><p>No release line.</p></body></html>"
    frame = parse_calendar(_write(tmp_path, pages), _config(2020))

    stated = _row(frame, "fomc_statement:monetary20200303a")
    assert (stated["released_at"], stated["release_time_source"]) == (
        _utc(2020, 3, 3, 15),
        "stated",
    )
    summer = _row(frame, "fomc_statement:monetary20200729a")
    assert summer["released_at"] == _utc(2020, 7, 29, 18)
    fallback = _row(frame, "fomc_statement:monetary20200129a")
    assert (fallback["released_at"], fallback["release_time_source"]) == (
        _utc(2020, 1, 29, 19),
        "convention",
    )


def test_dst_abbreviation_checked(tmp_path: Path):
    pages = _pages_2020()
    pages["fomc/monetary20200129a.html"] = _press("2:00 p.m. EDT")

    with pytest.raises(ValueError, match="EST"):
        parse_calendar(_write(tmp_path, pages), _config(2020))


def test_fomc_digest_is_year_page(tmp_path: Path):
    pages = _pages_2020()
    frame = parse_calendar(_write(tmp_path, pages), _config(2020))

    year_sha = hashlib.sha256(pages["fomc/historical-2020.html"].encode()).hexdigest()
    fomc = frame.filter(pl.col("family") == "fomc_statement")
    assert set(fomc["source_sha256"].to_list()) == {year_sha}
    assert fomc["source_url"].to_list() == [
        PRESS.format(name)
        for name in ("monetary20200129a", "monetary20200303a", "monetary20200729a")
    ]


def test_rolling_page_gives_past_and_future_meetings(tmp_path: Path):
    pages = _pages_2026()
    frame = parse_calendar(_write(tmp_path, pages), _config(2026))

    rolling_sha = hashlib.sha256(pages["fomc/calendars.9cb9e5ff5b19.html"].encode()).hexdigest()
    fomc = frame.filter(pl.col("family") == "fomc_statement")
    assert fomc["release_id"].to_list() == [
        "fomc_statement:monetary20260128a",
        "fomc_statement:scheduled-20260916",
    ]
    assert set(fomc["source_sha256"].to_list()) == {rolling_sha}
    past = _row(frame, "fomc_statement:monetary20260128a")
    assert (past["label"], past["source_url"]) == (
        "January 27-28 Meeting",
        PRESS.format("monetary20260128a"),
    )
    future = _row(frame, "fomc_statement:scheduled-20260916")
    assert future["released_at"] == _utc(2026, 9, 16, 18)
    assert future["label"] == "September 15-16* Meeting"
    assert (future["release_time_source"], future["source_url"]) == ("schedule_page", ROLLING)


def test_fetch_pages_writes_the_mapped_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config = _config(2026)
    pages = _pages_2026()
    archive = _write(tmp_path / "archive", pages)
    by_url = {
        ROLLING: pages["fomc/calendars.9cb9e5ff5b19.html"],
        PRESS.format("monetary20260128a"): pages["fomc/monetary20260128a.html"],
        "https://www.bls.gov/schedule/2026/home.htm": pages["bls/year-2026.html"],
    }
    requested: list[str] = []

    def fetch(url: str) -> bytes:
        requested.append(url)
        return by_url[url].encode()

    written = fetch_pages(tmp_path / "live", config, fetch, sleep=lambda seconds: None)

    assert "may differ" in capsys.readouterr().out
    assert sorted(requested) == sorted(by_url)
    assert sorted(p.relative_to(tmp_path / "live").as_posix() for p in written) == sorted(pages)
    assert parse_calendar(tmp_path / "live", config).equals(parse_calendar(archive, config))
