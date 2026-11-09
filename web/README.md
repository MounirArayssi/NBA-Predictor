# The Hoop Ledger — website

A static, multipage website for the NBA prediction model in this repository. It
shows daily forecasts, game pages, team profiles, a searchable forecast archive,
model evaluation, a research notebook, and methodology.

The site is completely separate from the model. It never connects to the database
from the browser, never runs the model, and nothing in `web/` modifies the
prediction pipeline.

```
PostgreSQL (local)
   │  SELECT only, on a read-only connection
   ▼
web/scripts/export_data.py ──► web/src/data/*.json         (read at build time)
   │                        └► web/public/data/archive-*.json (fetched by the archive page)
   ▼
Astro build (npm run build) ──► web/dist/   static HTML/CSS/JS, deploy anywhere
```

## Why Astro

Static output, file-based routes for 259 pages (games, teams, notebook), Markdown
content for the notebook, and almost no client JavaScript: pages render as HTML at
build time and only the interactive controls (date navigation, filters, sorting,
the archive) ship small scripts. Charts are drawn as static SVG at build time, so no
chart library is needed. Fonts (IBM Plex Sans, IBM Plex Mono, Source Serif 4) are
self-hosted through Fontsource; there are no third-party requests.

## Directory layout

```
web/
  scripts/
    export_data.py     read-only exporter (Python, uses the project's existing DB config)
    notebook_figures.py read-only analyses behind the notebook charts -> src/data/notebook_figures.json
    check_build.mjs    link, secret and data-consistency checks on dist/
    test_ui.mjs        browser interaction tests (Playwright + local Chrome)
    shots.mjs          screenshot / horizontal-overflow check at several widths
  src/
    data/              exported JSON (generated; do not edit by hand)
    content/notebook/  notebook entries, one Markdown file each
    components/        GameList, ProbabilityBar, SourceLabel, header/footer, charts/
    layouts/Base.astro page shell: metadata, header, footer
    lib/               data access + formatting (all times shown in ET), table sorting
    pages/             routes (see below)
    styles/global.css  design tokens and base styles
  public/
    data/              archive JSON partitions (generated)
    favicon.svg
  .cache/              cached NBA schedule (tip-off times), created by the exporter
```

## Routes

| Route | Content |
|---|---|
| `/` | Today's forecasts as scoreboards (chosen in the browser by ET date), biggest surprise of the week, headline record, home court, notebook |
| `/forecasts/` | Day/week browsing with date picker, 7-day strip, team and status filters; state in the URL (`?date=&view=week&team=&status=`) |
| `/forecasts/archive/` | Every graded forecast by season, filters, paging; state in the URL |
| `/games/[id]/` | Forecast court, **Inside the Forecast** (saved inputs + exact explanation where verifiable), **result against the forecast** (surprise ranks, box score), quarters, injury report, form, head-to-head |
| `/teams/` | Team tiles, Elo ladder, sortable table |
| `/teams/[slug]/` | Team profile: Elo, data basketball, **Predictability** (per season and all seasons, four charts, intervals), **Surprise games** |
| `/teams/predictability/` | **Team Predictability** explorer: season / games / minimum-sample filters, league chart, sortable table |
| `/surprises/` | **Surprise Games** explorer: unexpected winners, margin misses, most recent; season / team / type / date filters |
| `/performance/` | Record by season, every game, this season so far, links to the explorers, what moves the forecasts, calibration, errors, contexts, market benchmark |
| `/notebook/`, `/notebook/[slug]/` | Research notes |
| `/methodology/`, `/about/` | How it works (incl. how explanations are verified); about and attribution |

Game pages exist for every game of the current season, every game with an issued
forecast, and every game the surprise features link to (top 40 per season in each
ranking, top 5 per team per list). Other games appear in the archive and explorers
without their own page.

## Forecast intelligence features

Three features built on an independent, read-only analytics layer (`web/analytics/`);
none of them changes the model or the prediction pipeline.

```
predictions + games (PostgreSQL, read-only)
  -> scripts/export_data.py      one forecast per game; analytics/metrics.py
       -> public/data/archive-<season>.json   (+ surprise fields w, pw, su, se)
       -> src/data/predictability.json        (team metrics per season x game type)
       -> src/data/surprises.json             (week's biggest, per-team lists)
       -> src/data/team_analytics.json        (per-team game series for charts)
  -> scripts/explain_forecasts.py  analytics/explain.py (SHAP)
       -> src/data/explanations.json          (game pages only)
```

**Datasets.** Every metric uses the same combined record: one forecast per completed
game, the issued forecast where one exists, otherwise the walk-forward forecast
(see "Data rules"). Each record keeps its origin (`fc.origin`) and timestamp.

**Inside the Forecast** (`analytics/explain.py`, `scripts/explain_forecasts.py`).
Inputs come from the feature vector saved with the forecast. Attributions are exact
TreeSHAP values for the home and away score models; the margin contribution of an
input is home minus away, and baseline + contributions = projected margin (tested).
Attribution is published only for forecasts whose model can be verified: the
current production artifacts for forecasts saved after they were written, or each
season's walk-forward models rebuilt in memory with `model.train.train_model`
(never saved). A group is verified when at least 90% of its forecasts reproduce
both stored whole-point scores; each forecast in a verified group is explained if
its margin is within the 1-point rounding bound. Status at the last run: the
production model (saved 2026-11-07) and walk-forward 2026-27 verify (111 forecasts
explained); walk-forward 2022-23 to 2025-26 do not (11-18% exact matches, about
chance), and issued forecasts before 2026-11-07 came from artifacts that no longer
exist. Those games show their saved inputs and say why there is no attribution.
Re-running the walk-forward backfill with the current code would make older
seasons explainable; that writes to the database and needs separate approval.

**Team Predictability** (`analytics/metrics.py`). Per team, from the team's side:
margin error = actual margin - forecast margin; MAE (headline), mean error (bias),
sample SD, winner accuracy, expected wins (sum of pregame probabilities), gap,
Brier, log loss; 90% percentile-bootstrap intervals (1,000 resamples, fixed seed)
for MAE and bias. Sample tiers: under 20 games "small" (no ranking or summary
sentence), 20-59 "moderate", 60+ "stable". The summary sentence is generated only
when the bias interval excludes zero.

**Surprise Games** (`analytics/metrics.py`). Outcome surprise = -ln(p of the actual
winner), with p bounded at 1e-4 inside the log only; scoreline miss = |actual home
margin - forecast home margin|. Separate rankings; only final games.

### Regenerating

From the repository root, after the daily pipeline:

```bash
python web/scripts/export_data.py         # data + predictability + surprises (~1 min)
python web/scripts/explain_forecasts.py   # explanations (~8 min: rebuilds features and walk-forward models)
cd web && npm run build
```

`explain_forecasts.py` needs `shap` (`pip install shap`; analysis only, not a
pipeline dependency). Set `EXPLAIN_FEATURE_CACHE=<path>.pkl` during development
to reuse the feature dataset between runs.

### Verifying

```bash
python -m unittest discover -s web/analytics/tests -t web   # 27 tests: formulas, orientation, SHAP additivity, verification
node web/scripts/check_build.mjs                             # independent JS recomputation of every metric + links
node web/scripts/test_ui.mjs                                 # with `npm run preview` running
```

## Commands

From `web/`:

```bash
npm install                          # once
python scripts/export_data.py        # export data (needs the local database)
python scripts/notebook_figures.py   # refresh notebook chart data (read-only, ~1 min)
npm run build                        # build static site into dist/
npm run preview                      # serve dist/ at http://localhost:4321
npm run dev                          # development server with live reload
node scripts/check_build.mjs         # after a build: links, secrets, data consistency
node scripts/test_ui.mjs             # with preview running: interaction tests
```

The exporter uses the repository's Python environment (`.venv`) and the database
settings already in `config/settings.py` / `.env`. Add `--refresh-schedule` to
re-download tip-off times (cached in `.cache/`).

## Updating after the daily pipeline runs

1. Run the prediction pipeline as usual (nothing here changes it).
2. `python scripts/export_data.py`
3. `npm run build`, then optionally `node scripts/check_build.mjs`
4. Deploy `dist/`.

The site does not rebuild itself. Every page shows when its data was exported, and
the homepage says so explicitly when that edition is more than about 30 hours old.

## Deploying

`dist/` is plain static files. Any static host works (Cloudflare Pages, Netlify,
GitHub Pages, S3). The production domain is https://thehoopledger.com (set in
`astro.config.mjs`); canonical URLs and the sitemap use it. Override it for a
preview deployment with `SITE_URL`:

```bash
SITE_URL=https://preview.example.com npm run build
```

Nothing in `dist/` contains credentials; `check_build.mjs` scans for connection
strings and API-key patterns.

## Data rules the exporter enforces

- **One forecast per game**: the issued forecast (`is_official`, not a backfill) if the
  game has one, otherwise the walk-forward forecast (`model_version = 'wf-v2'`). Issued
  rows keep their timing check (`verified` / `unverified` / `after_tip`) as metadata.
- **Backfills** (`prediction_type = 'backfill'`) are not exported; only their count is shown.
- Games whose stored home and away team are the same (10 rows in the source table) are
  excluded everywhere and reported as a warning.
- One prediction per game per population; the export stops if that is ambiguous.
- The connection is set read-only before any query; the export aborts if it is not.

## Adding a notebook entry

Create `src/content/notebook/YYYY-MM-DD-short-name.mdx` (plain `.md` also works):

```markdown
---
title: "Question or finding"
date: 2026-11-08
topic: Experiments        # Model Development | Forecast Evaluation | Team Trends | Data Quality | Experiments | Research Notes
summary: "One or two sentences."
outcome: Open             # Kept | Rejected | Open | Note
---

## Question
...
```

The file name becomes the URL. Rebuild to publish.

Entries are MDX, so they can embed components from `src/components/notebook/`:

- `<CodeFile file="model/train.py" def="train_model" />` shows real code from the repo, read at
  build time (also `section="name"` for `# --- name` markers, or `from="..." to="..."`). Snippets
  therefore always match the code that runs; the build fails if the excerpt is not found.
- `BarRows`, `Dumbbell`, `EloPaths`, `EloCalibration`, `BlendWeight` charts, fed from
  `src/data/notebook_figures.json` or `performance.json`.

The homepage graphics (`src/components/home/`) are drawn from the exported data too: the
data basketball has one dot per backtest game, and the court figures are computed from the
backtest archive.

## Not built (future work)

- Live in-game scores. Game status is limited to scheduled, final and unknown. A future
  live-score source should update only the `score`/`status` of a game and never touch the
  saved forecast; `GameList.astro` and the game page are where it would surface.
- Per-game explanations of how each input moved a forecast.
- Player projections.
- A fuller market comparison (needs historical odds).
- Dark mode.
