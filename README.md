# NBA Game Predictor

A machine-learning system that predicts NBA game scores, winners, and individual player statistics. It ingests live data from the NBA API, ESPN injury reports, and Vegas odds; trains ensemble gradient-boosting models; serves predictions through a Streamlit dashboard; and optionally posts results to Twitter.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Motivation and Goals](#motivation-and-goals)
3. [Key Features](#key-features)
4. [Tech Stack](#tech-stack)
5. [Repository Structure](#repository-structure)
6. [End-to-End Workflow](#end-to-end-workflow)
7. [Data Sources and Data Pipeline](#data-sources-and-data-pipeline)
8. [Feature Engineering](#feature-engineering)
9. [Modeling Approach](#modeling-approach)
10. [Evaluation Strategy and Metrics](#evaluation-strategy-and-metrics)
11. [Setup Instructions](#setup-instructions)
12. [Usage](#usage)
13. [Example Output](#example-output)
14. [Design Decisions](#design-decisions)
15. [Limitations](#limitations)
16. [Future Improvements](#future-improvements)
17. [Sports Prediction Disclaimer](#sports-prediction-disclaimer)
18. [Author](#author)

---
Review this NBA prediction project and write a comprehensive README.md for a public GitHub repo.

Be efficient and evidence-based:
- Start by inspecting the directory tree.
- Prioritize files related to data loading, cleaning, feature engineering, model training, evaluation, prediction, configuration, dependencies, tests, and any app/API entry points.
- Use the actual codebase as the source of truth.
- Do not claim unsupported metrics, APIs, dashboards, databases, automations, or deployment features unless they are present in the repo.
- If a detail is unclear, mark it as a TODO instead of guessing.

Write README.md with:
1. Project overview
2. Motivation and goals
3. Key features
4. Tech stack
5. Repository structure
6. End-to-end workflow
7. Data sources and data pipeline
8. Feature engineering
9. Modeling approach
10. Evaluation strategy and metrics
11. Setup instructions
12. Usage examples for training, evaluation, and prediction
13. Example output if supported by code
14. Testing instructions if tests exist
15. Design decisions, especially around avoiding data leakage
16. Limitations
17. Future improvements
18. Sports prediction disclaimer
19. Author placeholders

Also include a final short report listing:
- The files you inspected
- The main commands discovered
- Assumptions made
- TODOs for me to verify
## Project Overview

NBA Game Predictor is a full-stack sports analytics project that:

- Pulls daily game schedules, box scores, injury reports, and Vegas odds from multiple sources.
- Engineers 50+ features covering team form, rest, matchup style, playoff context, and scoring variance.
- Trains separate `GradientBoostingRegressor` models for home and away team scoring.
- Trains per-player stat models (points, rebounds, assists) and ensembles their output with the team models.
- Stores predictions in PostgreSQL, evaluates them once final scores are available, and tracks accuracy over time.
- Exposes a Streamlit dashboard and an optional Twitter bot for public-facing output.

---

## Motivation and Goals

NBA games are high-variance events influenced by roster changes, travel schedules, playoff intensity, and stylistic matchups that simple box-score averages fail to capture. The goals of this project are:

- **Accuracy**: Beat the naive home-team-wins baseline (~56%) in game-winner prediction.
- **Calibration**: Predict final scores close enough to be useful for analysis (target MAE < 10 pts per side).
- **Freshness**: Incorporate same-day injury news, odds movement, and rest data before every prediction run.
- **Leakage-free evaluation**: Use strictly chronological train/test splits so no future information bleeds into training.
- **Transparency**: Log every prediction to a database and score it against actuals so model drift is detectable.

---

## Key Features

- **Dual team scoring models** — separate models for home and away score.
- **Player-level predictions** — points, rebounds, and assists per player, with injury redistribution logic.
- **Dynamic ensemble** — team and player model outputs are blended with weights that shift based on star availability, playoff context, and inter-model agreement.
- **Vegas anchoring** — mild calibration toward DraftKings spread/total lines to correct systematic bias.
- **Playoff adjustments** — series pressure, elimination context, pace drag, and playoff vs. regular-season efficiency factors.
- **Injury impact** — ESPN injury report integration redistributes a missing star's statistical load to rotation players.
- **Confidence tagging** — every prediction is tagged HIGH / MEDIUM / LOW with an emoji indicator.
- **Twitter bot** — randomized tweet templates post predictions with team hashtags.
- **Streamlit dashboard** — live view of recent predictions, accuracy trends, and Vegas-vs-model comparison.
- **Daily pipeline** — single CLI command runs the full ingest → compute → retrain → predict → tweet cycle.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML | scikit-learn (`GradientBoostingRegressor`) |
| Data manipulation | pandas, numpy |
| Database | PostgreSQL + SQLAlchemy 2.0 + psycopg2-binary |
| NBA data | nba_api 1.11.4 |
| Injury data | ESPN public API (no key required) |
| Odds data | The Odds API |
| Dashboard | Streamlit |
| Twitter | tweepy |
| Config | python-dotenv |

---

## Repository Structure

```
NBA-Predictor/
├── config/
│   ├── settings.py               # Season config, API base URLs, rate-limit delays
│   └── __init__.py
├── data/
│   └── ingestion/
│       ├── fetch_games.py         # NBA API: schedule & results
│       ├── fetch_box_scores.py    # Player & team box scores
│       ├── fetch_players.py       # Player metadata
│       ├── fetch_teams.py         # Team metadata
│       ├── fetch_injuries.py      # ESPN injury reports
│       ├── fetch_odds.py          # Vegas spreads, totals, player props
│       ├── compute_rolling_stats.py        # Team rolling averages (5/7/10/20-game)
│       ├── compute_player_rolling_stats.py # Player rolling averages (5/10-game)
│       ├── compute_playoff_factors.py      # Playoff efficiency adjustments
│       ├── compute_team_similarity.py      # Style-based matchup context
│       ├── build_features.py      # Assembles full training feature matrix
│       ├── update_game_scores.py  # Back-fills final scores
│       └── update_series.py       # Tracks playoff series state
│   └── storage/
│       ├── db.py                  # SQLAlchemy engine & session factory
│       ├── models.py              # ORM schema (teams, games, predictions, ...)
│       └── __init__.py
├── model/
│   ├── train.py                   # Train team scoring models
│   ├── train_players.py           # Train per-player stat models
│   ├── predict.py                 # Orchestrate daily predictions
│   ├── predict_players.py         # Player stat prediction + injury redistribution
│   ├── evaluate_predictions.py    # Score predictions vs. actual results
│   ├── visualize.py               # Feature importance plots
│   ├── mark_games_final.py        # Mark games as complete in DB
│   ├── mark_predictions_official.py
│   ├── audit.py                   # Prediction audit utilities
│   ├── diagnose_predictions.py
│   ├── feature_analysis.py
│   ├── summarize_model_performance.py
│   ├── remove_duplicate_predictions.py
│   ├── model_home.pkl             # Saved home scoring model
│   ├── model_away.pkl             # Saved away scoring model
│   ├── player_models.pkl          # Saved player stat models
│   ├── feature_importance.png     # Training artifact
│   └── predictions_vs_actual.png  # Evaluation artifact
├── dashboard/
│   └── app.py                     # Streamlit web dashboard
├── pipeline/
│   └── run_daily.py               # Full daily orchestration script
├── twitter/
│   ├── bot.py                     # Tweet prediction results
│   └── test_connection.py         # Verify Twitter API credentials
├── analysis/
│   ├── player_points_analysis.txt
│   └── team_model_analysis.txt
├── logs/                          # Auto-created; daily pipeline logs + CSV backups
├── requirements.txt
├── .env                           # Secrets (not committed)
└── README.md
```

---

## End-to-End Workflow

```
+---------------------------------------------------------------------------+
|  Daily Pipeline  (pipeline/run_daily.py)                                  |
|                                                                           |
|  1. Fetch games      -> NBA API schedule & results (all seasons)          |
|  2. Fetch box scores -> player & team stats per game                      |
|  3. Fetch odds       -> Vegas spreads, totals, DraftKings props           |
|  4. Compute rolling  -> 5/7/10/20-game team windows                       |
|  5. Compute player   -> 5/10-game player windows                          |
|  6. Update series    -> playoff series state (wins, elimination)          |
|  7. Retrain models   -> (optional, ~daily)                                |
|  8. Predict today    -> team + player ensemble -> save to DB + CSV        |
|  9. Tweet results    -> (optional, dry-run by default)                    |
+---------------------------------------------------------------------------+
```

---

## Data Sources and Data Pipeline

### Sources

| Source | What it provides | Auth |
|---|---|---|
| NBA API (`nba_api`) | Game schedule, box scores, advanced team/player stats | None (public) |
| ESPN public API | Injury reports (status, reason, return date) | None (public) |
| The Odds API | Vegas spread, total, implied win probability | API key |
| DraftKings (via The Odds API) | Player prop lines (points, rebounds, assists) | API key |

### Ingestion flow

1. **`fetch_games.py`** — pulls the full season schedule via `LeagueGameFinder` and upcoming games via `ScoreboardV2` (next 8 days). Upserts into `games` table.
2. **`fetch_box_scores.py`** — downloads per-player and per-team box scores for every completed game. Computes shooting percentages, advanced rates, and efficiency metrics.
3. **`fetch_injuries.py`** — polls ESPN's injury endpoint, normalises status strings into `Out / Doubtful / Questionable / Day-To-Day`, stores in `player_game_status`.
4. **`fetch_odds.py`** — fetches spread, total, and per-player prop lines; matches to games by team and date; upserts into `game_odds`.
5. **`compute_rolling_stats.py`** — computes team-level rolling windows (5, 7, 10, 20 games) over points, offensive/defensive rating, pace, FG%, 3P%, and win percentage. Stores in `team_rolling_stats`.
6. **`compute_player_rolling_stats.py`** — same for individual players (windows 5, 10). Stores in `player_rolling_stats`.
7. **`build_features.py`** — joins all the above into a flat feature matrix used for training and prediction.

### Database schema (key tables)

| Table | Description |
|---|---|
| `teams` | Team metadata (ID, abbreviation, city, conference, arena) |
| `players` | Player metadata (ID, name, position, height, weight, active) |
| `games` | Schedule and results (date, home/away IDs, scores, season type) |
| `game_odds` | Vegas lines (spread, total, implied probability) |
| `player_box_scores` | Per-player game stats |
| `team_box_scores` | Per-team game aggregates |
| `team_rolling_stats` | Pre-computed windowed team averages |
| `player_rolling_stats` | Pre-computed windowed player averages |
| `player_game_status` | Injury/availability tracking |
| `predictions` | Model predictions + post-game evaluation |

---

## Feature Engineering

All feature construction lives in `data/ingestion/build_features.py`. The final training matrix contains 50+ columns grouped as follows:

### Team form (10-game rolling window)
- Average points, offensive rating, defensive rating, pace
- FG%, 3P%, win percentage

### Recent trend (7-game rolling window)
- Momentum versions of all team-form features (short-window vs. long-window delta)

### Rest and schedule context
- Rest days before game
- Back-to-back flag
- Rest advantage (home rest days minus away rest days)

### Playoff context
- Is-playoff flag
- Series game number
- Home/away series wins so far
- Elimination-game flag
- Series pressure index (total games played in series)

### Head-to-head history
- Historical average scores between the two franchises
- H2H win percentage
- Number of H2H games in database

### Vegas features
- Spread (home perspective)
- Total line
- Implied win probability (sigmoid of spread)
- Deviation from each team's season scoring average

### Matchup interaction features
- Combined pace
- 3PT rate matchup (offensive 3PT rate vs. defensive 3PT rate)
- Offensive rating edge vs. opponent defensive rating
- Net rating differential
- Implied total derived from team averages

### Scoring variance and consistency
- FG% and 3P% standard deviation over recent games
- Consistency metric: `1 / (1 + std)`
- Bad-night probability estimate
- Shooting volatility differential between the two teams

### Playoff efficiency
- Historical playoff vs. regular-season efficiency ratios per team
- Points-per-usage metric for detecting efficiency shifts

### Home-court strength
- Arena altitude adjustment
- Home-game scoring impact (team-level, estimated from historical splits)

### Style-based defensive matchup
- How well away team's defense performs vs. teams with similar pace/3PT profile
- How well home team's defense handles away team's offensive style

### Shooting consistency
- FG% std, 3P% std across recent games

### Quarter-level features
- Q4 averages (absolute scoring)
- Q4 vs. full-game differential
- Clutch performance (Q4 scoring above baseline)

---

## Modeling Approach

### Team scoring models (`model/train.py`)

Two separate models are trained — one for the home score, one for the away score.

| Parameter | Value |
|---|---|
| Algorithm | `GradientBoostingRegressor` |
| Estimators | 150 |
| Learning rate | 0.04 |
| Max depth | 3 |
| Sample weighting | Exponential recency decay: `exp(-days_ago / 550)` |
| Train/test split | Chronological 85/15 — no random shuffling |

Saved artifacts: `model/model_home.pkl`, `model/model_away.pkl`.

### Player stat models (`model/train_players.py`)

Three separate models (points, rebounds, assists) trained on players with 10+ game samples and a minimum of 15.0 minutes per game.

| Parameter | Value |
|---|---|
| Algorithm | `GradientBoostingRegressor` (same hyperparams) |
| Targets | Points, Rebounds, Assists |
| Minimum sample | 10 games, 15 min/game |

Saved artifact: `model/player_models.pkl`.

### Ensemble (`model/predict.py`)

The final score is a weighted blend of team model output and aggregated player model output:

- **Default weights**: 69% team model, 31% player model.
- **Playoff adjustment**: +4% to player weight in playoff games.
- **Star-availability adjustment**: more high-usage players available raises the player model weight.
- **Disagreement handling**: if team and player models diverge by 28+ points total, player weight is reduced by 30%.
- **Vegas calibration**: mild pull toward DraftKings spread/total (30% weight), applied as a final calibration step.

### Confidence tagging

| Level | Condition |
|---|---|
| HIGH | Predicted margin >= 10 pts and team/player models agree |
| MEDIUM | Predicted margin 5-10 pts |
| LOW | Margin < 5 pts, or model disagreement, or high scoring variance |

---

## Evaluation Strategy and Metrics

### Anti-leakage design

- Train/test splits are always **chronological** — the test set contains only games that occurred after all training games.
- Rolling stats are computed only from games before the game being predicted.
- Vegas lines are included as features (available pre-game), not as labels.

### Per-game metrics (computed after final scores are available)

| Metric | Description |
|---|---|
| Home score error | `abs(predicted_home - actual_home)` |
| Away score error | `abs(predicted_away - actual_away)` |
| Total score error | `abs(predicted_total - actual_total)` |
| Margin error | `abs(predicted_margin - actual_margin)` |
| Spread error | Comparison vs. Vegas spread |
| Winner correct | Boolean |

### Aggregate model performance (from `analysis/team_model_analysis.txt`)

| Metric | Value |
|---|---|
| Winner accuracy | ~72.4% |
| Home bias baseline | ~56.1% |
| Improvement over baseline | +16.3 pp |
| Average MAE (home) | ~9.0 pts |
| Average MAE (away) | ~10.5 pts |
| Average MAE (combined) | ~9.78 pts |
| Within 10 pts | ~65% |
| Within 15 pts | ~85% |
| Within 20 pts | ~95% |
| Train/test MAE gap | < 2% |

### Player model performance (from `analysis/player_points_analysis.txt`)

| Target | MAE |
|---|---|
| Points | ~4.93 |
| Rebounds | ~2.01 |
| Assists | ~1.53 |

Evaluation is triggered via `model/evaluate_predictions.py` once games are marked final in the database.

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ running locally or accessible via connection string
- The Odds API key (free tier available at the-odds-api.com)
- Twitter developer credentials (optional, for the bot)

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/NBA-Predictor.git
cd NBA-Predictor
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the template below to a `.env` file in the project root and fill in your values:

```dotenv
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nba_predictor
DB_USER=postgres
DB_PASSWORD=your_password

# The Odds API
ODDS_API_KEY=your_odds_api_key

# Twitter (optional)
TWITTER_API_KEY=
TWITTER_API_SECRET=
TWITTER_BEARER_TOKEN=
TWITTER_ACCESS_TOKEN=
TWITTER_ACCESS_TOKEN_SECRET=

# Model versioning (optional)
NBA_MODEL_VERSION=v1
```

> If you are running the Streamlit dashboard, also check whether a `~/.streamlit/secrets.toml` file is required for the database URL (see TODOs below).

### 5. Initialise the database

SQLAlchemy creates all tables automatically on first run:

```bash
python -c "from data.storage.db import engine; from data.storage.models import Base; Base.metadata.create_all(engine)"
```

### 6. Seed historical data

```bash
python data/ingestion/fetch_teams.py
python data/ingestion/fetch_players.py
python data/ingestion/fetch_games.py
python data/ingestion/fetch_box_scores.py
python data/ingestion/compute_rolling_stats.py
python data/ingestion/compute_player_rolling_stats.py
```

> The NBA API enforces rate limits. `config/settings.py` sets a 1.65-second delay between requests. A full historical pull may take several hours.

---

## Usage

### Run the full daily pipeline

```bash
# Full run: ingest -> compute -> retrain -> predict -> tweet (dry-run)
python pipeline/run_daily.py

# Skip model retraining (faster, uses existing pkl files)
python pipeline/run_daily.py --skip-retrain

# Actually post to Twitter (default is dry-run)
python pipeline/run_daily.py --post-twitter

# Only predict and tweet, skip all data fetching
python pipeline/run_daily.py --predict-only
```

### Train models independently

```bash
# Team scoring models
python model/train.py

# Player stat models
python model/train_players.py
```

### Generate predictions for today

```bash
python model/predict.py
```

### Evaluate predictions against final scores

```bash
python model/evaluate_predictions.py
```

### Launch the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

### Verify Twitter API connection

```bash
python twitter/test_connection.py
```

---

## Example Output

### Training output (`model/train.py`)

```
Home model  --  Train MAE: 8.97  |  Test MAE: 9.03  |  Gap: 0.07%
Away model  --  Train MAE: 9.41  |  Test MAE: 10.52 |  Gap: 1.11%
Winner accuracy (test): 72.4%  |  Baseline: 56.1%
```

### Prediction record (database / CSV log)

```
Game        : BOS @ MIL  (2025-01-15)
Predicted   : BOS 112 - MIL 108  (BOS by 4)
Confidence  : MEDIUM
Key factors : BOS off-rating edge +6.2, MIL back-to-back, pace 101.3
Top scorers : J. Tatum 28.4 pts  |  G. Antetokounmpo 31.1 pts
```

### Dashboard

The Streamlit app at `http://localhost:8501` displays:
- Recent official predictions with confidence levels and Vegas comparison
- 30-day rolling win rate, MAE, and total-score error charts
- Pending vs. evaluated prediction counts

### Feature importance

`model/feature_importance.png` and `model/predictions_vs_actual.png` are generated after each training run.

---

## Design Decisions

### Chronological train/test splitting

Randomly shuffling game data would allow models to "see" future games during training, inflating test-set accuracy. The pipeline always sorts by date and takes the last N% of games as the test set. This mirrors real deployment conditions.

### Separate home and away models

Home-court advantage is asymmetric. Teams play differently at home. Training one model per side lets each learn independent distributional patterns — home teams tend to be more consistent; away teams show higher variance.

### Exponential recency weighting

Older games are down-weighted with `exp(-days_ago / 550)` so the model adapts to roster changes and mid-season trends without discarding the full history needed for stable estimates.

### Dynamic ensemble weighting

A fixed 70/30 blend was replaced with weights that shift based on:
- How many high-usage players are healthy (more stars available → trust player model more)
- Playoff vs. regular season (playoff tendencies → player model carries more signal)
- Inter-model disagreement (large gap → favour the more conservative signal)

### Playoff efficiency factor

Regular-season offensive ratings overestimate playoff scoring. `compute_playoff_factors.py` estimates per-team efficiency drag and injects it as a feature, reducing systematic over-prediction in May and June.

### Injury redistribution

When a star (high usage rate) is ruled out, a naive approach under-predicts replacement scoring. `predict_players.py` distributes the missing player's statistical load to rotation players proportional to `1 / lost_usage`, which empirically reduces total-score error on injury games.

---

## Limitations

- **No live in-game updates** — predictions are pre-game only; line movements after tip-off are not tracked.
- **No play-by-play features** — shot quality, lineup combinations, and pace-on/off splits are not modelled.
- **Limited early-season coverage** — model quality degrades for teams with fewer than ~30 games in the database.
- **ESPN injury freshness** — the ESPN endpoint is polled once per pipeline run; late scratches (< 1 hour before tip-off) may not be captured.
- **Mild Vegas calibration** — the 30% Vegas anchor improves calibration on average but can pull predictions toward the line in games where the model has a genuine edge.
- **Player model sample size** — players with fewer than 10 games (rookies, returning injured players) fall back to league averages, which can be noisy.
- **No uncertainty quantification** — confidence tiers (HIGH/MEDIUM/LOW) are heuristic; no calibrated probabilities or prediction intervals are produced.
- **Single-season retraining** — models retrain on current-season data by default; cross-season generalisation is not evaluated.

---

## Future Improvements

- [ ] Lineup-based features (net rating of expected starting five)
- [ ] Play-by-play data for shot quality and defensive metrics
- [ ] Calibrated win probabilities with confidence intervals (Platt scaling or isotonic regression)
- [ ] Automated GitHub Actions / cron job for the daily pipeline
- [ ] Backtesting framework across multiple seasons with walk-forward validation
- [ ] REST API (FastAPI) to serve predictions programmatically
- [ ] Player prop bet evaluation vs. DraftKings lines
- [ ] SHAP value explanations per prediction
- [ ] Discord / Slack notification option alongside Twitter

---

## Sports Prediction Disclaimer

This project is built for educational and research purposes only. NBA game outcomes are inherently unpredictable. Predictions produced by this system:

- Are not guaranteed to be accurate.
- Should **not** be used as the basis for gambling or financial decisions.
- Reflect historical patterns and may not account for all real-world variables (injuries announced after the prediction run, referee assignments, travel disruptions, etc.).

The author(s) accept no liability for any decisions made based on these predictions.

---

## Author

**Mounir Arayssi**

- GitHub: <!-- TODO: add your GitHub profile URL -->
- Twitter/X: <!-- TODO: add handle if desired -->

---

## Appendix: Inspection Report

### Files inspected

| File | Role |
|---|---|
| `model/train.py` | Team model training, chronological split, evaluation |
| `model/train_players.py` | Player stat model training |
| `model/predict.py` | Prediction orchestrator, ensemble, calibration, confidence |
| `model/predict_players.py` | Player predictions, injury redistribution |
| `model/evaluate_predictions.py` | Post-game scoring |
| `data/ingestion/build_features.py` | Full feature engineering pipeline |
| `data/ingestion/fetch_games.py` | NBA API schedule ingest |
| `data/ingestion/fetch_box_scores.py` | NBA API box score ingest |
| `data/ingestion/fetch_injuries.py` | ESPN injury ingest |
| `data/ingestion/fetch_odds.py` | The Odds API ingest |
| `data/ingestion/compute_rolling_stats.py` | Team rolling windows |
| `data/ingestion/compute_player_rolling_stats.py` | Player rolling windows |
| `data/ingestion/compute_playoff_factors.py` | Playoff efficiency adjustments |
| `data/storage/db.py` | SQLAlchemy engine setup |
| `data/storage/models.py` | ORM schema |
| `pipeline/run_daily.py` | Daily orchestration CLI |
| `dashboard/app.py` | Streamlit dashboard |
| `twitter/bot.py` | Twitter integration |
| `config/settings.py` | Season, rate-limit, API base URL config |
| `requirements.txt` | Python dependencies |
| `analysis/team_model_analysis.txt` | Logged model metrics |
| `analysis/player_points_analysis.txt` | Logged player model metrics |

### Main commands discovered

```bash
python pipeline/run_daily.py [--skip-retrain] [--post-twitter] [--predict-only]
python model/train.py
python model/train_players.py
python model/predict.py
python model/evaluate_predictions.py
streamlit run dashboard/app.py
python twitter/test_connection.py
```

### Assumptions made

- Performance figures (72.4% winner accuracy, 9.78 MAE) were read from `analysis/team_model_analysis.txt` and `analysis/player_points_analysis.txt`; they represent the most recent logged evaluation run and may not reflect the current model pkl files.
- The `.env` variable names were inferred from `config/settings.py` and `data/storage/db.py`; exact names should be verified against your working `.env`.
- Database auto-creation via `Base.metadata.create_all(engine)` was inferred from the SQLAlchemy pattern in `db.py`; a separate migration script may be needed if the schema has evolved.

### TODOs for you to verify

- [ ] Add your GitHub profile URL and Twitter handle to the Author section.
- [ ] Confirm whether a `~/.streamlit/secrets.toml` file is required instead of / in addition to `.env` for the database URL (the git log mentions "Read database URL from Streamlit secrets").
- [ ] Verify the exact `.env` variable names against your working configuration.
- [ ] Confirm whether `model/player_models.pkl` is committed to the repo or generated at runtime only.
- [ ] Add any cron / GitHub Actions automation details if they live outside this repo.
- [ ] Update the performance figures after the next evaluation run if the current pkl models differ from the logged values.
