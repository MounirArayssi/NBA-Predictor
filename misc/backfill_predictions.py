"""
backfill_predictions.py
=======================
Backfills game data and/or predictions for a date range.

Usage
-----
# Step 1 – refresh all game/score/rolling data (run once):
    python backfill_predictions.py --data-only

# Step 2 – generate predictions for every date from start to yesterday:
    python backfill_predictions.py --start 2026-05-02 --end 2026-05-25

# Both steps together:
    python backfill_predictions.py --start 2026-05-02 --end 2026-05-25 --with-data

# Step 3 – evaluate predictions against final scores (run after backfill):
    python backfill_predictions.py --evaluate --start 2026-05-02 --end 2026-05-25

Notes
-----
- Injury report and player props are fetched LIVE (today's data) because historical
  snapshots are not stored. Team rolling stats and Vegas odds in the DB ARE from the
  correct historical dates, so team-level predictions are still meaningful.
- Games with no DB entry for the target date are skipped silently.
- Backfill predictions are saved with prediction_type='backfill' and is_official=True
  (unless an official prediction already exists for that game).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
from datetime import date, timedelta

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(f'logs/backfill_{date.today().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def daterange(start: date, end: date):
    """Yield each date from start through end (inclusive)."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ---------------------------------------------------------------------------
# Data refresh steps (mirrors run_daily.py steps 1-6)
# ---------------------------------------------------------------------------

def refresh_data():
    log.info("=" * 55)
    log.info("REFRESHING GAME DATA")
    log.info("=" * 55)

    log.info("▶  Fetching games (current season)...")
    from data.ingestion.fetch_games import fetch_and_store_games, fetch_upcoming_games
    from config.settings import CURRENT_SEASON
    fetch_and_store_games(season=CURRENT_SEASON, season_type='Playoffs')
    fetch_and_store_games(season=CURRENT_SEASON, season_type='Regular Season')
    fetch_upcoming_games()
    log.info("✅ Games fetched")

    log.info("▶  Fetching box scores (new games only)...")
    from data.ingestion.fetch_box_scores import fetch_all_box_scores
    fetch_all_box_scores()
    log.info("✅ Box scores fetched")

    log.info("▶  Computing team rolling stats (windows 5/7/10/20)...")
    from data.ingestion.compute_rolling_stats import (
        compute_and_store_rolling_stats, compute_todays_stats
    )
    for w in [5, 7, 10, 20]:
        log.info(f"   window={w}")
        compute_and_store_rolling_stats(window=w)
        compute_todays_stats(window=w)
    log.info("✅ Team rolling stats done")

    log.info("▶  Computing player rolling stats (windows 5/10)...")
    from data.ingestion.compute_player_rolling_stats import compute_player_rolling_stats
    for w in [5, 10]:
        log.info(f"   window={w}")
        compute_player_rolling_stats(window=w)
    log.info("✅ Player rolling stats done")

    log.info("▶  Updating playoff series context...")
    _update_series()
    log.info("✅ Series context updated")


def _update_series():
    from sqlalchemy import text
    from sqlalchemy.orm import Session
    from data.storage.db import engine
    query = text("""
        WITH ordered_series AS (
            SELECT
                g.game_id,
                g.game_date,
                g.home_team_id,
                g.away_team_id,
                g.home_score,
                g.away_score,
                g.is_final,
                LEAST(g.home_team_id, g.away_team_id)    AS team_a,
                GREATEST(g.home_team_id, g.away_team_id) AS team_b,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        LEAST(g.home_team_id, g.away_team_id),
                        GREATEST(g.home_team_id, g.away_team_id)
                    ORDER BY g.game_date, g.game_id
                ) AS game_num
            FROM games g
            WHERE g.season_type = 'Playoffs'
            AND g.season = '2025-26'
        ),
        series_wins AS (
            SELECT
                cur.game_id,
                cur.game_num,
                cur.home_team_id,
                cur.away_team_id,
                COALESCE((
                    SELECT COUNT(*)
                    FROM ordered_series prev
                    WHERE prev.team_a = cur.team_a AND prev.team_b = cur.team_b
                    AND prev.game_num < cur.game_num AND prev.is_final = TRUE
                    AND (
                        (prev.home_team_id = cur.home_team_id AND prev.home_score > prev.away_score)
                        OR (prev.away_team_id = cur.home_team_id AND prev.away_score > prev.home_score)
                    )
                ), 0) AS home_wins,
                COALESCE((
                    SELECT COUNT(*)
                    FROM ordered_series prev
                    WHERE prev.team_a = cur.team_a AND prev.team_b = cur.team_b
                    AND prev.game_num < cur.game_num AND prev.is_final = TRUE
                    AND (
                        (prev.home_team_id = cur.away_team_id AND prev.home_score > prev.away_score)
                        OR (prev.away_team_id = cur.away_team_id AND prev.away_score > prev.home_score)
                    )
                ), 0) AS away_wins
            FROM ordered_series cur
        )
        UPDATE games
        SET
            series_game_num  = sw.game_num,
            home_series_wins = sw.home_wins,
            away_series_wins = sw.away_wins,
            is_elimination   = CASE WHEN sw.home_wins = 3 OR sw.away_wins = 3 THEN TRUE ELSE FALSE END
        FROM series_wins sw
        WHERE games.game_id = sw.game_id
    """)
    with Session(engine) as session:
        session.execute(query)
        session.commit()


# ---------------------------------------------------------------------------
# Check how many games exist in DB for a date range
# ---------------------------------------------------------------------------

def report_db_coverage(start: date, end: date):
    from sqlalchemy import text
    from data.storage.db import engine
    q = text("""
        SELECT game_date, COUNT(*) AS cnt, SUM(CASE WHEN is_final THEN 1 ELSE 0 END) AS final
        FROM games
        WHERE game_date BETWEEN :s AND :e
        GROUP BY game_date
        ORDER BY game_date
    """)
    with engine.connect() as conn:
        rows = conn.execute(q, {'s': str(start), 'e': str(end)}).fetchall()
    if not rows:
        log.warning(f"No games in DB for {start} – {end}. Run --data-only first.")
        return
    log.info(f"DB coverage for {start} – {end}:")
    for r in rows:
        log.info(f"  {r[0]}  total={r[1]}  final={r[2]}")


# ---------------------------------------------------------------------------
# Per-date prediction
# ---------------------------------------------------------------------------

def predict_for_date(target: date, injury_report, all_player_props):
    """Generate and save predictions for a single past date."""
    from model.predict import (
        load_models, load_calibration_model, get_todays_games, predict_single_game,
        print_prediction, print_footer, save_predictions, print_header
    )

    print_header(target)
    model_home, model_away, model_margin = load_models()
    calibration_model = load_calibration_model()
    games = get_todays_games(game_date=target, include_final=True)

    if not games:
        log.info(f"  No games in DB for {target} — skipping")
        return 0

    log.info(f"  {len(games)} games found for {target}")
    predictions = []
    for game in games:
        pred = predict_single_game(
            game, model_home, model_away, injury_report, all_player_props,
            model_margin=model_margin, calibration_model=calibration_model,
        )
        if pred:
            predictions.append(pred)
            print_prediction(pred, game, injury_report)

    print_footer(len(predictions))
    save_predictions(predictions, is_backfill=True)
    return len(predictions)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(start: date, end: date):
    log.info(f"▶  Evaluating predictions {start} – {end}...")
    from model.evaluate_predictions import evaluate_predictions
    evaluate_predictions(
        start_date=start,
        end_date=end,
        official_only=True,
        include_already_evaluated=False,
    )
    log.info("✅ Evaluation complete")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='NBA Prediction Backfill')
    parser.add_argument('--start',      default='2026-05-02',
                        help='First date to predict (YYYY-MM-DD)')
    parser.add_argument('--end',        default=str(date.today() - timedelta(days=1)),
                        help='Last date to predict (YYYY-MM-DD), default: yesterday')
    parser.add_argument('--data-only',  action='store_true',
                        help='Only refresh game/score/rolling data, do not predict')
    parser.add_argument('--with-data',  action='store_true',
                        help='Refresh data then predict')
    parser.add_argument('--evaluate',   action='store_true',
                        help='Score predictions against final results after backfill')
    parser.add_argument('--check-db',   action='store_true',
                        help='Print DB coverage for the date range and exit')
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    if end >= date.today():
        log.warning(f"End date {end} is today or in the future — capping at yesterday")
        end = date.today() - timedelta(days=1)

    if args.check_db:
        report_db_coverage(start, end)
        return

    if args.data_only or args.with_data:
        refresh_data()

    if args.data_only:
        log.info("--data-only: skipping predictions")
        return

    # Report what's in the DB before predicting
    report_db_coverage(start, end)

    # Fetch live injury/odds once for all dates (historical snapshots not stored)
    log.info("Fetching live injury report and player props (used for all dates)...")
    try:
        from data.ingestion.fetch_injuries import fetch_injury_report
        injury_report = fetch_injury_report()
        log.info(f"  Injury report: {sum(len(v) for v in injury_report.values())} entries")
    except Exception as e:
        log.warning(f"  Could not fetch injury report: {e} — using empty dict")
        injury_report = {}

    try:
        from data.ingestion.fetch_odds import fetch_todays_player_props
        all_player_props = fetch_todays_player_props()
        log.info(f"  Player props fetched")
    except Exception as e:
        log.warning(f"  Could not fetch player props: {e} — using empty dict")
        all_player_props = {}

    # Loop through every date
    total_predictions = 0
    skipped_dates = []
    log.info("=" * 55)
    log.info(f"BACKFILL PREDICTIONS  {start} → {end}")
    log.info("=" * 55)

    for target in daterange(start, end):
        log.info(f"\n--- {target} ---")
        try:
            n = predict_for_date(target, injury_report, all_player_props)
            total_predictions += n
            if n == 0:
                skipped_dates.append(target)
        except Exception as e:
            log.error(f"  ERROR predicting {target}: {e}")
            import traceback
            traceback.print_exc()

    log.info("=" * 55)
    log.info(f"BACKFILL COMPLETE — {total_predictions} predictions saved")
    if skipped_dates:
        log.info(f"  Dates with no games: {[str(d) for d in skipped_dates]}")
    log.info("=" * 55)

    if args.evaluate:
        run_evaluation(start, end)


if __name__ == '__main__':
    main()
