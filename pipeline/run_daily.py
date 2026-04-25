import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import subprocess
from datetime import date, datetime
from sqlalchemy import text
from sqlalchemy.orm import Session
from data.storage.db import engine

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s | %(message)s',
    handlers = [
        logging.FileHandler(
            f'logs/pipeline_{date.today().strftime("%Y%m%d")}.log'
        ),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def run_step(name, func, skip=False):
    if skip:
        log.info(f"⏭️  Skipping: {name}")
        return True
    log.info(f"▶️  {name}...")
    try:
        func()
        log.info(f"✅ {name} done")
        return True
    except Exception as e:
        log.error(f"❌ {name} failed: {e}")
        return False


def step_fetch_games():
    from data.ingestion.fetch_games import (
        fetch_and_store_games,
        fetch_upcoming_games
    )
    from config.settings import CURRENT_SEASON
    fetch_and_store_games(season=CURRENT_SEASON, season_type='Playoffs')
    fetch_and_store_games(season=CURRENT_SEASON, season_type='Regular Season')
    fetch_upcoming_games()


def step_fetch_box_scores():
    from data.ingestion.fetch_box_scores import fetch_all_box_scores
    fetch_all_box_scores()


def step_fetch_odds():
    from data.ingestion.fetch_odds import fetch_todays_odds, parse_and_store_odds
    odds = fetch_todays_odds()
    if odds:
        parse_and_store_odds(odds)


def step_rolling_stats():
    from data.ingestion.compute_rolling_stats import (
        compute_and_store_rolling_stats,
        compute_todays_stats
    )
    for w in [5, 7, 10, 20]:
        compute_and_store_rolling_stats(window=w)
        compute_todays_stats(window=w)


def step_player_rolling_stats():
    from data.ingestion.compute_player_rolling_stats import (
        compute_player_rolling_stats
    )
    for w in [5, 10]:
        compute_player_rolling_stats(window=w)


def step_update_series():
    """Update playoff series context for all games."""
    log.info("  Updating series context...")
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
                    WHERE prev.team_a = cur.team_a
                    AND prev.team_b = cur.team_b
                    AND prev.game_num < cur.game_num
                    AND prev.is_final = TRUE
                    AND (
                        (prev.home_team_id = cur.home_team_id
                         AND prev.home_score > prev.away_score)
                        OR
                        (prev.away_team_id = cur.home_team_id
                         AND prev.away_score > prev.home_score)
                    )
                ), 0) AS home_wins,
                COALESCE((
                    SELECT COUNT(*)
                    FROM ordered_series prev
                    WHERE prev.team_a = cur.team_a
                    AND prev.team_b = cur.team_b
                    AND prev.game_num < cur.game_num
                    AND prev.is_final = TRUE
                    AND (
                        (prev.home_team_id = cur.away_team_id
                         AND prev.home_score > prev.away_score)
                        OR
                        (prev.away_team_id = cur.away_team_id
                         AND prev.away_score > prev.home_score)
                    )
                ), 0) AS away_wins
            FROM ordered_series cur
        )
        UPDATE games
        SET
            series_game_num  = sw.game_num,
            home_series_wins = sw.home_wins,
            away_series_wins = sw.away_wins,
            is_elimination   = CASE
                                 WHEN sw.home_wins = 3 OR sw.away_wins = 3
                                 THEN TRUE ELSE FALSE
                               END
        FROM series_wins sw
        WHERE games.game_id = sw.game_id
    """)
    with Session(engine) as session:
        session.execute(query)
        session.commit()
    log.info("  Series context updated")


def step_predict():
    from model.predict import predict_todays_games
    predictions = predict_todays_games()
    return predictions


def step_tweet(predictions, dry_run=False):
    from twitter.bot import post_predictions
    if predictions:
        post_predictions(predictions, dry_run=dry_run)
    else:
        log.info("  No predictions to tweet")


def run_pipeline(skip_retrain=False, dry_run_twitter=True,
                 skip_twitter=False):
    start = datetime.now()
    log.info("=" * 55)
    log.info(f"DAILY PIPELINE — {date.today()}")
    log.info("=" * 55)

    run_step("Fetch games",              step_fetch_games)
    run_step("Fetch box scores",         step_fetch_box_scores)
    run_step("Fetch odds",               step_fetch_odds)
    run_step("Rolling stats",            step_rolling_stats)
    run_step("Player rolling stats",     step_player_rolling_stats)
    run_step("Update series context",    step_update_series)

    if not skip_retrain:
        def retrain():
            from data.ingestion.build_features import build_feature_dataset
            from model.train import (
                chronological_split, train_model,
                save_models, FEATURE_COLS,
                TARGET_HOME, TARGET_AWAY
            )
            df = build_feature_dataset(window=10)
            df = df.dropna(
                subset=FEATURE_COLS + [TARGET_HOME, TARGET_AWAY]
            )
            df = df.sort_values('game_date').reset_index(drop=True)
            train, test = chronological_split(df, test_ratio=0.15)
            mh = train_model(train, TARGET_HOME)
            ma = train_model(train, TARGET_AWAY)
            save_models(mh, ma)
        run_step("Retrain model", retrain)
    else:
        log.info("⏭️  Skipping retrain")

    predictions = step_predict()

    if not skip_twitter:
        run_step(
            "Post to Twitter",
            lambda: step_tweet(predictions, dry_run=dry_run_twitter)
        )

    elapsed = (datetime.now() - start).seconds
    log.info("=" * 55)
    log.info(f"PIPELINE COMPLETE — {elapsed}s")
    log.info("=" * 55)

    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NBA Daily Pipeline')
    parser.add_argument('--skip-retrain',  action='store_true')
    parser.add_argument('--skip-twitter',  action='store_true')
    parser.add_argument('--post-twitter',  action='store_true',
                        help='Actually post tweets (default is dry run)')
    parser.add_argument('--predict-only',  action='store_true',
                        help='Just run predictions, skip data fetching')
    args = parser.parse_args()

    if args.predict_only:
        predictions = step_predict()
        step_tweet(predictions, dry_run=not args.post_twitter)
    else:
        run_pipeline(
            skip_retrain   = args.skip_retrain,
            dry_run_twitter = not args.post_twitter,
            skip_twitter   = args.skip_twitter
        )