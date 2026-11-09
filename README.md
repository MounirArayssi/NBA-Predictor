# The Hoop Ledger: NBA game forecasting

A personal basketball research project. It forecasts the final score of every NBA game from
publicly available data, turns each projection into a win probability, and grades every
forecast against the result. The research is published at **[thehoopledger.com](https://thehoopledger.com)**.

The project is independent and noncommercial. It is not affiliated with the NBA, any team,
or any other organization, and it uses only public data.

---

## Contents

1. [What it does](#what-it-does)
2. [How it works](#how-it-works)
3. [The model](#the-model)
4. [Evaluation](#evaluation)
5. [Repository layout](#repository-layout)
6. [Setup](#setup)
7. [Daily use](#daily-use)
8. [The website](#the-website)
9. [Limitations](#limitations)
10. [Disclaimer](#disclaimer)

---

## What it does

For each NBA game:

- a projected score for both teams, the projected winner, and the home team's win probability;
- a saved record of exactly what the model knew when it made the forecast;
- a grade once the game is final (winner, margin error, total error, Brier score).

Forecasts are saved before tip-off. The first saved forecast for a game becomes the official
one; a rerun can replace it only within the last hour before tip-off (to pick up late injury
news), and never after the game starts.

---

## How it works

```
NBA Stats (nba_api) ─┐
Official NBA         ├──► PostgreSQL ──► features ──► score models ──► forecast ──► graded
injury report ───────┘                   (21 inputs)   (home, away)     saved        after final
                                                                        pre-tip
                                    │
                                    └──► web/ exporter (read-only) ──► static site
```

1. **Ingest**: schedules, results and box scores from NBA Stats; player availability from
   the official NBA injury report (archived per day, so past dates use that day's report).
2. **Compute**: rolling team stats, Elo ratings, season margins, rest, playoff series state,
   and missing rotation scoring.
3. **Train**: two gradient-boosted models, one for the home score and one for the away score.
4. **Predict**: score both sides, derive margin and total, and convert the margin to a win
   probability with a calibration curve.
5. **Evaluate**: grade official forecasts once games are final.

---

## The model

### Inputs (21)

All inputs are known before tip-off. Box-score stats from the game itself are never inputs.

| Group | Inputs |
|---|---|
| Team strength | Elo rating (each team), Elo gap including home advantage, season scoring margin (each team, blended with last season for the first 20 games), games played (each team) |
| Recent form, last 10 games | Offensive rating and defensive rating (each team), combined pace |
| Availability | Points per game of rotation players listed Out or Doubtful on the official injury report (each team) |
| Rest | Days since last game, capped at 5 (each team); back-to-back flag (each team) |
| Playoff series | Home team's series wins, elimination game, series pressure |

The full list is `FEATURE_COLS` in `model/train.py`.

**Elo** (`data/ingestion/team_strength.py`): margin-of-victory Elo starting October 2021,
K = 20, 70 Elo points of home advantage (about 2.5 points of margin), 75% of each rating's
distance from 1500 carried into the next season, 28 Elo points per point of expected margin.
Training and the daily forecast replay the same games, so the ratings match.

**Rotation players**: played in at least 2 of the team's last 10 games averaging 15+ minutes.
Questionable players are treated as available.

### Score models (`model/train.py`)

| Setting | Value |
|---|---|
| Algorithm | scikit-learn `GradientBoostingRegressor`, one model per side |
| Trees / learning rate / depth | 150 / 0.04 / 3 |
| Min samples per leaf / subsample | 15 / 0.7 |
| Recency weighting | `exp(-days_ago / 550)`, so a game three seasons old counts about 13% |
| Training data | every game since 2021-22 |

The forecast is the team model alone: margin = home score − away score. A per-player model
(`model/train_players.py`, `model/predict_players.py`) still runs and its player lines are
printed in the pipeline output, but it is not blended into the game score; in testing the
blend cost correct winners in every slice.

**Win probability** (`model/calibrate.py`): Platt scaling (logistic regression on the
projected margin), fit on earlier graded forecasts.

### Ideas tested and dropped

Each was tested with paired walk-forward cross-validation (`model/experiment.py`) and did
not help: a 42-feature set (shooting splits, matchup products, scoring variance), schedule
load and travel distance, per-team home-court strength, and an NBA Cup flag. The research
notebook on the website explains each one.

---

## Evaluation

**Walk-forward backtest** (`misc/walkforward_backfill.py`): every game from 2022-23 on is
predicted by a model trained only on the seasons before it, through the same code path as
the daily forecast, with that day's archived injury report. These forecasts are stored as
`model_version = 'wf-v2'` and are clearly labeled as produced after the fact.

Record as of November 9, 2026 (5,390 graded games, 2022-23 to 2026-27):

| Measure | Result |
|---|---|
| Winners picked | 65.1% |
| Home team won (baseline) | 55.6% |
| Higher-Elo team won (baseline) | 64.0% |
| Margin error (mean absolute) | 11.0 pts |
| Total points error (mean absolute) | 15.0 pts |
| Brier score | 0.217 (always saying 50% scores 0.250) |

Up-to-date numbers, calibration charts and season-by-season results are on the website's
Performance page. Daily forecasts are graded with `model/evaluate_predictions.py`
(official forecasts only).

---

## Repository layout

```
NBA-Predictor/
├── config/settings.py              # season settings, API settings, rate limits (reads .env)
├── data/
│   ├── ingestion/
│   │   ├── fetch_games.py                    # schedule and results
│   │   ├── fetch_box_scores.py               # player and team box scores
│   │   ├── fetch_official_injuries.py        # official NBA injury report (archived daily)
│   │   ├── fetch_injuries.py                 # ESPN injuries (live fallback only)
│   │   ├── fetch_teams.py, fetch_players.py  # metadata (initial setup)
│   │   ├── compute_rolling_stats.py          # team rolling windows
│   │   ├── compute_player_rolling_stats.py   # player rolling windows (player model)
│   │   ├── compute_playoff_factors.py
│   │   ├── compute_team_similarity.py        # legacy; not a model input
│   │   ├── team_strength.py                  # Elo and season-margin blend
│   │   ├── update_series.py                  # playoff series state
│   │   └── build_features.py                 # assembles the feature matrix
│   └── storage/                    # SQLAlchemy engine and ORM models
├── model/
│   ├── train.py                    # team score models (production fit)
│   ├── predict.py                  # daily forecasts; official-forecast locking
│   ├── calibrate.py                # margin -> win probability
│   ├── evaluate_predictions.py     # grade official forecasts
│   ├── experiment.py               # walk-forward CV for feature tests
│   ├── train_players.py, predict_players.py   # player model (display only)
│   ├── feature_analysis.py         # legacy analysis script
│   └── *.pkl                       # trained models
├── pipeline/run_daily.py           # daily orchestration
├── misc/
│   ├── walkforward_backfill.py     # out-of-sample backtest, resumable
│   ├── backfill_predictions.py     # predictions for past dates
│   ├── backfill_missing_players.py # refetch incomplete player box scores
│   └── backfill_rebounds.py        # fill missing offensive/defensive rebounds
├── web/                            # The Hoop Ledger website (see web/README.md)
├── dashboard/, site/               # earlier Streamlit dashboard and static page (superseded by web/)
└── requirements.txt
```

---

## Setup

Requirements: Python 3.10+, PostgreSQL 14+, Node 18+ (for the website only).

```bash
git clone https://github.com/MounirArayssi/NBA-Predictor.git
cd NBA-Predictor
python -m venv .venv
.venv\Scripts\activate          # Windows; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the project root:

```dotenv
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nba_predictor
DB_USER=postgres
DB_PASSWORD=your_password
```

Create the tables and load history (the NBA Stats API is rate-limited, so a full history
pull takes hours):

```bash
python -c "from data.storage.db import engine; from data.storage.models import Base; Base.metadata.create_all(engine)"
python data/ingestion/fetch_teams.py
python data/ingestion/fetch_players.py
python data/ingestion/fetch_games.py
python data/ingestion/fetch_box_scores.py
python data/ingestion/compute_rolling_stats.py
python data/ingestion/compute_player_rolling_stats.py
python model/train.py
```

---

## Daily use

Run from the repository root with `PYTHONPATH=.` and `PYTHONIOENCODING=utf-8` (the output
contains non-ASCII characters).

```bash
python pipeline/run_daily.py                  # full: ingest, recompute, retrain, predict
python pipeline/run_daily.py --today-only     # only what today's games need; no retrain
python pipeline/run_daily.py --predict-only   # predict only (e.g. a rerun in the last hour before tip)
python pipeline/run_daily.py --skip-retrain   # full ingest, keep the current models
```

Before predicting, the pipeline checks that every team playing today has its previous game
final, and re-fetches if not, so forecasts never miss a just-finished game.

Grading a finished slate:

```python
from model.evaluate_predictions import evaluate_predictions
evaluate_predictions(start_date=d, end_date=d)
```

Extending the walk-forward backtest:

```bash
python misc/walkforward_backfill.py --seasons 2026-27   # resumable; skips games already done
```

---

## The website

`web/` is an Astro static site generated from a **read-only** export of the database. It
never connects to the database from the browser and never runs the model. It is deployed
to Cloudflare Pages. Refresh order after the pipeline runs (from `web/`):

```bash
python scripts/export_data.py
python scripts/explain_forecasts.py
python scripts/breakdown_inputs.py
python scripts/what_wins.py
npm run build
node scripts/check_build.mjs
npx wrangler pages deploy dist --project-name thehoopledger --branch main
```

Details, routes and data rules: [web/README.md](web/README.md).

---

## Limitations

- Most graded forecasts come from the walk-forward backtest, produced after the games by
  models that had not seen them; only recent forecasts were published before tip-off.
- Forecast margins are much narrower than real margins, so blowouts are rarely forecast as
  blowouts.
- A missing player is measured by points per game, not total on-court impact; Questionable
  players are treated as available.
- `model/train.py` sorts training rows by date only, so games on the same date can be
  ordered differently between database states, and a retrain is not exactly reproducible.
- The pipeline is run by hand; it is not yet scheduled.

---

## Disclaimer

This is an independent, noncommercial research and educational project. It uses only
publicly available data, is not affiliated with, endorsed by, or connected to the NBA, any
team, or any other company or organization, and represents only the author's own work.
Forecasts are uncertain and are not advice of any kind.

---

**Author:** Mounir Arayssi
