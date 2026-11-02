# pit-multisource

[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.CODE.svg)](https://doi.org/10.5281/zenodo.CODE)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.DATASET.svg)](https://doi.org/10.5281/zenodo.DATASET)
[![tests](https://github.com/AndrewThanatos/pit-multisource/actions/workflows/tests.yml/badge.svg)](https://github.com/AndrewThanatos/pit-multisource/actions/workflows/tests.yml)

Builds a point-in-time dataset of Bitcoin and Ether prices, GDELT news metadata and US release
calendars, and reproduces the numbers of the article *Point-in-Time Consistency for Multisource
Time-Series Data: A Formal Model and Construction Method for Evolving News Metadata*
(Journal of Big Data, 2026).

"Point in time" means that every row keeps the timestamps needed to decide what was knowable at a
given moment, so historical inputs can be assembled without information that arrived later. The
released dataset is on Zenodo ([DOI 10.5281/zenodo.DATASET](https://doi.org/10.5281/zenodo.DATASET));
this repository is the code that built it and the code that recomputes the article's tables from it.

Rewritten from the research codebase used for the article. Every released table except the raw GKG
records, the slot ledger and `news/gdelt_slots.parquet` is rebuilt from sources and compared row for
row; GDELT selection is checked on a 30-slot sample of the original archives; the news and entity
builds start from the released raw GKG records and the source-domain build from the released news
records.

## What is inside

```
SOURCE                 DOWNLOAD            BUILD                                    TABLES
Binance spot klines ─▶ acquire binance ──▶ build bars ────────────────────────────▶ market/
                                           build targets ─────────────────────────▶ 4 h / 24 h targets
GDELT GKG 2.1 slots ─▶ acquire gdelt ────▶ build news, entities, source-domains ──▶ news/, diagnostics/
FOMC and BLS pages ──▶ acquire calendar ─▶ build calendar ────────────────────────▶ calendar/

released tables ─▶ reproduce ─▶ one experiment per article table ─▶ paper | reproduced | match
```

| Part | Modules | What it does |
|---|---|---|
| Acquire | `acquire/binance.py`, `acquire/gdelt.py`, `acquire/calendar.py` | Downloads the raw sources: monthly Binance zips (checksum-verified), 15-minute GDELT GKG archives (with a slot ledger), FOMC and BLS calendar pages |
| Build | `build/bars.py`, `build/selection.py`, `build/news.py`, `build/entities.py`, `build/source_domains.py`, `build/calendar.py`, `build/targets.py` | Turns the sources into the released tables: bars, selected news records, chronological clusters with dated asset flags, entities, source-domain diagnostics, the release calendar, and diagnostic-period targets |
| Point in time | `pit/availability.py`, `pit/windows.py` | When a cluster flag becomes knowable; the coverage state of the news window before each decision time |
| Experiments | `experiments/*.py`, `reproduce.py`, `example.py` | One module per article table, the comparison with the article's values, and the article's worked example |
| Parameters | `configs/pipeline.yaml` | The article's time windows, thresholds, selection terms and diagnostic boundary |

[`docs/pipeline.md`](docs/pipeline.md) shows the whole flow on one page, with the availability
rule of each source and what each module reads.

## Install

Python 3.12 or newer.

```bash
git clone https://github.com/AndrewThanatos/pit-multisource
cd pit-multisource
uv sync                      # or: python -m pip install .
```

With `uv`, either activate `.venv` or prefix each command below with `uv run`. The measured runs
in this README used Python 3.12.3 and polars 2.1.0 on an Apple M2 Pro with 16 GB of
memory; the unit tests also run on Python 3.12 and 3.14 in CI. Exact equality with the released
tables is verified with the locked versions, which `uv sync` installs; `pyproject.toml` allows only
polars 2.1 and later within major version 2.

## Quick start

Download the dataset from Zenodo ([DOI 10.5281/zenodo.DATASET](https://doi.org/10.5281/zenodo.DATASET),
about 6.4 GB, of which 6.12 GB are the full 27-field GKG records) and unpack it, for example into
`pit-multisource-v1.0.0/`. Then ask what was knowable at 2023-10-18 12:00 UTC, the article's worked
example (Section 4.5):

```bash
pit-multisource example --dataset pit-multisource-v1.0.0
```

```
decision_time: 2023-10-18T12:00:00+00:00
bar.open_time: 2023-10-18T11:00:00+00:00
bar.close_time: 2023-10-18T11:59:59.999+00:00
bar.close: 28340.64
news.records: 49
news.btc_relevant: 49
news.clusters: 30
last_release.id: cpi:20231012T1230
last_release.released_at: 2023-10-12T12:30:00+00:00
next_release.id: fomc_statement:monetary20231101a
next_release.family: fomc_statement
next_release.released_at: 2023-11-01T18:00:00+00:00
next_release.date: 2023-11-01
```

The last closed hourly bar, the 49 news records in 30 clusters that became available in the hour
before, and the next scheduled release: everything here was knowable at that moment, and nothing
later. The same example as a plain script is [`examples/point_in_time.py`](examples/point_in_time.py).

## Reproduce the paper

One command recomputes the article's numbers from the released dataset and compares each with the
value printed in the article:

```bash
export PYTHONHASHSEED=0 POLARS_MAX_THREADS=1 TZ=UTC
pit-multisource reproduce --dataset pit-multisource-v1.0.0 \
  --shared-ids pit-multisource-materials-v1.0.0/inputs/shared_record_ids.parquet
```

`shared_record_ids.parquet` lists the 2,003,579 records shared by the released news corpus and the
earlier corpus that Table 4, Table 6 (rows 4–5), §3.6 and §4.2 compare on. It is in `inputs/` of the
Materials record ([DOI 10.5281/zenodo.MATERIALS](https://doi.org/10.5281/zenodo.MATERIALS)). Without `--shared-ids`
the whole `grouping` experiment (those 42 entries) is reported as skipped and the rest still runs.

The command writes `results/<experiment>.json` (change with `--out`), prints the comparison, and
exits with 1 if any number differs. A number matches when it agrees at the precision the article
prints. Expected output, shortened:

```
sizes: 0.3 s wall time
...
sensitivity: 1333.2 s wall time
example: 0.1 s wall time

Article values
id                                 | section         | paper                           | reproduced                      | match
t2.hourly_bars.btc                 | Table 2         | 66,397                          | 66,397                          | yes
t3.dated.agree                     | Table 3         | 207                             | 207                             | yes
t4.bounded.clusters                | Table 4         | 668,519                         | 668,519                         | yes
t5.min1_age24.clusters             | Table 5         | 361,365                         | 361,365                         | yes
t6.row1.changed_hours              | Table 6 row 1   | 47,059                          | 47,059                          | yes
...

Supplementary values (reproduced by the code, not printed in the article)
...

Not reproducible here — see the Materials record:
...

219 match, 0 mismatch, 0 skipped; total wall time 2038.2 s
```

What each experiment covers and how long it took on an Apple M2 Pro with 16 GB. The summary line
above is the time of everything `reproduce` does, config loading and printing included (2,038.2 s);
the whole command, interpreter start-up included, took 2,040.8 s (about 34 minutes) with a peak
memory of 6.1 GB.

| Experiment | Article result | Numbers | Time |
|---|---|---:|---:|
| `sizes` | Table 2 (table sizes) | 10 | 0.3 s |
| `replay` | Table 3 (historical replay against reconstruction), §4.1 | 23 | 157.5 s |
| `grouping` | Table 4 (bounded and unbounded grouping), Table 6 rows 4–5, §3.6, §4.2 | 42 | 326.4 s |
| `sensitivity` | Table 5 (nine clustering settings), §4.3 | 55 | 1,333.2 s |
| `features` | Table 6 rows 1–3 (clock and flag conventions), §4.4 | 14 | 4.3 s |
| `coverage` | Table 7 (news coverage of the observation windows), §4.5 | 22 | 1.3 s |
| `targets` | Table 8 (target eligibility), §4.5 | 21 | 1.1 s |
| `descriptive` | slot ledger and cluster shape, §3.6, §4.5 | 17 + 4 supplementary | 2.2 s |
| `agreement` | agreement of the GDELT theme annotation with the entity-name test, §4.5 | 6 | 211.6 s |
| `example` | worked example, §4.5 | 5 | 0.1 s |

The four supplementary values are reproduced by the code and reported in Supplementary
Material S1, not in the article text. `--only` runs a subset, for example
`pit-multisource reproduce --dataset pit-multisource-v1.0.0 --only sizes coverage targets`.

## Rebuild the dataset from its sources

Every released table except the raw GKG records (`news/gkg_records/`), the slot ledger and
`news/gdelt_slots.parquet` is rebuilt in full and compared with the released file row for row by the
equivalence tests. `news/gdelt_slots.parquet` is the last ledger line of each slot; the coverage
experiment reads it and no command builds it. The GDELT record selection is checked on a 30-slot
sample of the original archives (below). Sizes and times are for this release on the machine above.

| Tables | Commands | Input | Time |
|---|---|---|---|
| `market/*` bars | `acquire binance`, then `build bars` | 364 monthly zips from data.binance.vision, 372 MB | 9 s build |
| `news/news_records`, `news/news_clusters` | `build news` | `news/gkg_records/` of the dataset | about 3.5 min, 3 GB memory |
| `news/entities`, surface forms, alias candidates | `build entities` | `news/gkg_records/` of the dataset | about 70 s, 6 GB memory |
| `diagnostics/source_domains/*` | `build source-domains` | `news/news_records` of the dataset | 6 s |
| `calendar/macro_releases` | `acquire calendar`, then `build calendar` | 132 archived FOMC and BLS pages, 12 MB | 1 s build |
| GDELT record selection and slot ledger | `acquire gdelt` | GDELT GKG 2.1 archives | see below |
| 4 h / 24 h targets (not in the dataset) | `build targets` | `market/*_1m` of the dataset | 4 s |

```bash
pit-multisource acquire binance --out sources
pit-multisource build bars --sources sources --out rebuilt/market

pit-multisource build news --dataset pit-multisource-v1.0.0 --out rebuilt/news
pit-multisource build entities --dataset pit-multisource-v1.0.0 --out rebuilt/news
pit-multisource build source-domains --dataset pit-multisource-v1.0.0 --out rebuilt/source_domains

pit-multisource build calendar --pages pit-multisource-materials-v1.0.0/inputs/calendar_pages --out rebuilt/calendar
pit-multisource acquire calendar --out live_calendar --user-agent "your-name your@email.example"

pit-multisource acquire gdelt --start 2023-10-18T00:00:00Z --end 2023-10-19T00:00:00Z --out gdelt
pit-multisource build targets --dataset pit-multisource-v1.0.0 --out rebuilt/targets
```

Things to know before rebuilding:

- **Network and certificates.** `data.gdeltproject.org` redirects `http` to `https`. A python.org
  build of Python on macOS may then fail with `CERTIFICATE_VERIFY_FAILED`; point `SSL_CERT_FILE`
  at a CA bundle, for example `SSL_CERT_FILE=/etc/ssl/cert.pem`, or run that Python's
  `Install Certificates.command`.
- **Live sources can differ from the 2026 archive.** GDELT, the Federal Reserve and BLS serve
  today's content, which may not be what the dataset was built from in 2026. The archived calendar
  pages are in the Materials record; build the released calendar from them (`--pages`).
  `acquire calendar` fetches today's pages and writes today's rolling FOMC calendar under the
  archived file name, so never point its `--out` at your copy of the archive.
- **GDELT at full scale.** The dataset requested 265,824 fifteen-minute slots. A full
  `acquire gdelt` run downloads all of them, roughly 1.5 TB of compressed archives (estimated from a
  test sample of 30 slots, 5.9 MB per served archive). The selection was checked on that sample,
  ten slots of each status (records kept, no match, missing at the source): it picks exactly the
  released records and writes the same ledger lines. `build news` and `build entities` read the
  same records from `news/gkg_records/` of the dataset.
- **Targets cover the article's diagnostic period only.** `build targets` cuts the minute bars to
  `open_time < 2025-08-01` before it computes anything (`targets.boundary` in
  `configs/pipeline.yaml`). Targets are not part of the released dataset; their equality with the
  article's targets was checked by the author and is recorded in
  [`equivalence/targets_report.json`](equivalence/targets_report.json).

## What is not reproducible here

Some results in the article are records of what happened in the original runs, not something code
can recompute from the released data. They are documented in the Materials record
([DOI 10.5281/zenodo.MATERIALS](https://doi.org/10.5281/zenodo.MATERIALS)), and `reproduce` lists
them at the end of every run:

- run times and memory of the article's experiments (Table 4, §4.5); the times above are this
  package on one machine;
- the two diagnostic access incidents (56 boundary-crossing target rows and 35,096 out-of-scope
  outcome rows) and the seven synthetic access-control cases of the diagnostic target reader;
- the repeatability check of the diagnostic outputs and the freeze-manifest check of the inputs;
- the backup and recovery checks;
- the model-assisted second annotation pass and the page audit;
- the reconciliation with the earlier news corpus and its cluster example (§4.1), because the
  earlier corpus is not part of the released dataset.

## Layout and tests

```
configs/pipeline.yaml          parameters of the article
src/pit_multisource/
  acquire/                     downloads from the original sources
  build/                       the released tables
  pit/                         availability rules and coverage windows
  experiments/                 one module per article table
  reproduce.py, example.py     the reproduce and example commands
expected/paper_values.json     every compared number, with its table or section in the article
equivalence/targets_report.json
examples/point_in_time.py
tools/fetch_gdelt_sample.py    downloads the 30-slot GDELT test sample
tests/unit/                    fast tests on synthetic data (run in CI)
tests/equivalence/             rebuild each table and compare it with the released file
docs/pipeline.md
```

```bash
uv run pytest -m "not equivalence"                 # unit tests, seconds, no data needed
uv run ruff check . && uv run ruff format --check .
```

The equivalence tests need the dataset and the raw sources, given by environment variables, and are
skipped when a variable is not set:

```bash
export PIT_DATASET=pit-multisource-v1.0.0          # the unpacked dataset
export PIT_SOURCES=sources                         # binance/ from `acquire binance`, calendar/ = a copy of
                                                   # inputs/calendar_pages from the Materials record,
                                                   # gdelt_sample/ from the tools script
export PIT_SHARED_IDS=pit-multisource-materials-v1.0.0/inputs/shared_record_ids.parquet
export PIT_FROZEN_TARGETS=frozen_targets           # author only: the article's frozen targets (not published)
uv run pytest -m equivalence
```

The frozen targets are not published, so `test_targets_equivalence.py` is skipped for public users;
its recorded result is in `equivalence/targets_report.json`.

Equality means equal data frames: the same columns, types, rows and row order, with floats equal
exactly. The grouping and sensitivity test takes about 32 minutes; the others take seconds to a
few minutes each.

## Citation

If you use this code, please cite the article and the software; if you use the data, cite the
dataset as well. [`CITATION.cff`](CITATION.cff) holds the same information.

```bibtex
@article{shymkovych2026pit,
  author  = {Shymkovych, Volodymyr and Berezhniuk, Andrii and Telenyk, Sergii and Nowakowski, Grzegorz},
  title   = {Point-in-Time Consistency for Multisource Time-Series Data: A Formal Model and Construction
             Method for Evolving News Metadata},
  journal = {Journal of Big Data},
  doi     = {10.1186/ARTICLE},
  year    = {2026}
}

@misc{berezhniuk2026pitcode,
  author       = {Berezhniuk, Andrii},
  title        = {pit-multisource: point-in-time construction of a multisource dataset and reproduction
                  of the article's results},
  howpublished = {Zenodo, software},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.CODE},
  year         = {2026}
}

@misc{berezhniuk2026pitdata,
  author       = {Berezhniuk, Andrii},
  title        = {Point-in-time multisource dataset: Binance BTC/ETH, GDELT GKG news metadata and US release
                  calendars, 2019--2026},
  howpublished = {Zenodo, dataset},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.DATASET},
  year         = {2026}
}
```

## Licences

The code is under the MIT licence ([`LICENSE`](LICENSE)). The dataset is distributed separately
under CC BY-NC-SA 4.0, as the Binance Vision terms require for redistributed derivative works; the
raw sources remain under the terms of Binance, the GDELT Project, the Federal Reserve Board and the
U.S. Bureau of Labor Statistics.
