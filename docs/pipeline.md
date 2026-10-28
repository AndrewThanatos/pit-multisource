# The pipeline on one page

Three public sources become the tables of the released dataset; the experiments read those tables
and print the article's numbers. The article's parameters (time windows, thresholds, selection
terms, the diagnostic boundary) live in [`configs/pipeline.yaml`](../configs/pipeline.yaml).

```
Binance spot klines ──acquire binance──▶ monthly zips ──build bars──▶ market/{btc,eth}usdt_{1h,1m}
                                                                          │ build targets
                                                                          ▼
                                                              4 h / 24 h targets (before 2025-08-01)

GDELT GKG 2.1 slots ──acquire gdelt──▶ news/gkg_records/ (27 fields) + slot ledger
                                          ├─build news──────────▶ news/news_records, news/news_clusters
                                          │                          └─build source-domains─▶ diagnostics/source_domains/
                                          └─build entities──────▶ news/entities, surface forms, alias candidates

FOMC and BLS pages ──acquire calendar──▶ HTML pages ──build calendar──▶ calendar/macro_releases

released tables ──reproduce──▶ results/<experiment>.json and the table "paper | reproduced | match"
```

`acquire gdelt` writes the selected lines of each slot and the ledger; the released dataset keeps
the same lines as Parquet in `news/gkg_records/`, which is what `build news`, `build entities` and
the experiments read.

## Experiments

| Experiment | Reads | Article result |
|---|---|---|
| `sizes` | every released table | Table 2 |
| `replay` | `gkg_records`, `news_records`, `news_clusters` | Table 3, §4.1 |
| `grouping` | `gkg_records` and the shared record ids from the Materials record | Table 4, Table 6 rows 4–5, §3.6, §4.2 |
| `sensitivity` | `gkg_records` | Table 5, §4.3 |
| `features` | `news_records`, `news_clusters` | Table 6 rows 1–3, §4.4 |
| `coverage` | `gdelt_slots`, `news_records` | Table 7, §4.5 |
| `targets` | `market/*_1m` (targets rebuilt for the diagnostic period) | Table 8, §4.5 |
| `descriptive` | slot ledger, `news_records` | §3.6, §4.5 |
| `agreement` | `gkg_records` | §4.5 (theme annotation against the entity-name test) |
| `example` | market, news, calendar | §4.5 worked example |

## When a row may be used

Each source has its own clock. A row may be used at decision time `H` when:

| Source | Rule | Where in the code |
|---|---|---|
| Market bars | the bar has closed: `close_time <= H` (robust form: `open_time + interval <= H`) | `example.py` |
| News records | their 15-minute GDELT slot has ended: `slot_time + 15 min <= H`. `slot_time` is the archive slot, not the publication time | `pit/availability.py`, `example.py` |
| Clusters | the opening record's slot has ended: `opened_at + 15 min <= H` | `experiments/replay.py` |
| Cluster asset flags | the record that earned the flag became available: `*_relevant_earned_at + 15 min <= H` | `pit/availability.py` |
| Calendar releases | past: `released_at <= H`; future: only if the schedule was known, `schedule_available_at <= H` (scheduled releases from 1 January of their year, unscheduled ones at the release instant) | `build/calendar.py`, `example.py` |
| Targets | as a label once the exit bar has closed: `label_availability_time <= H`; a target belongs to the diagnostic period only when its exposure ends before 2025-08-01 | `build/targets.py` |

Source-domain diagnostics and the whole-period counts in the entity table have no clock: they are
for retrospective analysis, never historical inputs.

## How the news tables are built

1. **Selection** (`build/selection.py`). A GKG line is kept when its themes carry `ECON_BITCOIN` or
   a crypto term (`bitcoin`, `ethereum`, `stablecoin`, exchange names, …) appears as a whole word
   in its names, organisations or URL. Of several kept lines with one record id in a slot, the
   first wins.
2. **Clustering** (`build/news.py`). Records are processed once, in `(slot_time, record_id)` order.
   A record joins the cluster that already holds its URL, unless that cluster is older than
   72 hours; otherwise the open cluster whose seed shares at least two entities with it and was
   active in the last 24 hours (largest overlap, then latest activity, then earliest creation);
   otherwise it opens a new cluster. Generic entities (`bitcoin`, `crypto`, exchange names) never
   count towards the overlap.
3. **Dated flags.** A record is relevant to an asset when it names it (theme or whole-word term);
   a record that names neither asset counts for both. A cluster becomes Bitcoin- or
   Ether-relevant at the slot of its first relevant member and is unflagged before that, so an
   earlier state never changes when later members arrive.

## Modules

| Module | What it does |
|---|---|
| `acquire/binance.py` | Downloads the monthly kline zips and checks each against the published checksum |
| `acquire/gdelt.py` | Downloads GKG slot archives with retries, keeps the selected lines, writes the append-only slot ledger |
| `acquire/calendar.py` | Downloads exactly the FOMC and BLS pages the calendar build reads |
| `build/bars.py` | Hourly and one-minute bar tables from the zips |
| `build/selection.py` | The record-selection rule for one slot archive and its ledger counters |
| `build/news.py` | News records, chronological clusters and dated asset flags |
| `build/entities.py` | Typed entity dictionary, surface forms and alias candidates (recorded, never merged) |
| `build/source_domains.py` | Source-domain families, co-appearance edges and per-cluster counts |
| `build/calendar.py` | US release calendar (CPI, PPI, Employment Situation, FOMC statements) from the pages |
| `build/targets.py` | Hourly 4 h / 24 h log-return targets from the one-minute bars before the boundary |
| `pit/availability.py` | When a cluster flag becomes knowable |
| `pit/windows.py` | Coverage state of the news window before each decision time |
| `experiments/*.py` | One module per article table (above); each returns its numbers as a flat dict |
| `reproduce.py` | Runs the experiments and compares every number with `expected/paper_values.json` |
| `example.py` | What was knowable at 2023-10-18 12:00 UTC |
| `config.py`, `cli.py`, `commands.py` | Configuration loading and the `pit-multisource` command line |
