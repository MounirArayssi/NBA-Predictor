import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Steps print emoji; on a non-UTF-8 console (e.g. output piped on Windows) that
# raised UnicodeEncodeError mid-step and silently skipped the rest of the step.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

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


def _todays_team_ids():
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT home_team_id, away_team_id FROM games WHERE game_date = :d
        """), {"d": date.today()}).fetchall()
    return {t for r in rows for t in r}


def step_fetch_games_today():
    """Only results since the last un-finalized game, plus today's schedule."""
    from datetime import timedelta
    from data.ingestion.fetch_games import (
        fetch_and_store_games,
        fetch_upcoming_games
    )
    from config.settings import CURRENT_SEASON
    with engine.connect() as conn:
        oldest_pending = conn.execute(text("""
            SELECT MIN(game_date) FROM games
            WHERE is_final = FALSE AND game_date < :today AND season = :season
              AND season_type IN ('Regular Season', 'Playoffs')
        """), {"today": date.today(), "season": CURRENT_SEASON}).scalar()
    date_from = oldest_pending or (date.today() - timedelta(days=1))
    fetch_and_store_games(CURRENT_SEASON, 'Playoffs', date_from=date_from)
    fetch_and_store_games(CURRENT_SEASON, 'Regular Season', date_from=date_from)
    fetch_upcoming_games(days_ahead=1)


def step_rolling_stats_today():
    from data.ingestion.compute_rolling_stats import compute_todays_stats
    for w in [5, 7, 10, 20]:
        compute_todays_stats(window=w)


def step_player_rolling_stats_today():
    from data.ingestion.compute_player_rolling_stats import (
        compute_player_rolling_stats
    )
    team_ids = _todays_team_ids()
    if not team_ids:
        log.info("  No games today")
        return
    for w in [5, 10]:
        compute_player_rolling_stats(window=w, today_only=True, team_ids=team_ids)


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
    from config.settings import CURRENT_SEASON
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
            AND g.season = :season
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
        session.execute(query, {"season": CURRENT_SEASON})
        session.commit()
    log.info("  Series context updated")


def _stale_teams():
    """Teams playing today whose previous game this season isn't final yet.
    Predicting them would use stale Elo, rest days and rolling stats."""
    from config.settings import CURRENT_SEASON
    with engine.connect() as conn:
        rows = conn.execute(text("""
            WITH today AS (
                SELECT home_team_id AS team_id FROM games WHERE game_date = :d
                UNION SELECT away_team_id FROM games WHERE game_date = :d
            ),
            prev AS (
                SELECT DISTINCT ON (t.team_id) t.team_id, g.game_date, g.is_final
                FROM today t
                JOIN games g ON t.team_id IN (g.home_team_id, g.away_team_id)
                WHERE g.game_date < :d AND g.season = :season
                  AND g.season_type IN ('Regular Season', 'Playoffs')
                  AND g.home_team_id <> g.away_team_id
                ORDER BY t.team_id, g.game_date DESC
            )
            SELECT tm.abbreviation, p.game_date FROM prev p
            JOIN teams tm ON tm.team_id = p.team_id
            WHERE NOT p.is_final
            ORDER BY tm.abbreviation
        """), {"d": date.today(), "season": CURRENT_SEASON}).fetchall()
    return [(r[0], r[1]) for r in rows]


def step_check_previous_games_final(retries=2):
    """Before predicting: every team playing today must have its previous game
    final. If not (e.g. the results fetch failed quietly), re-fetch results, box
    scores and today's rolling stats, then check again; fail loudly if still stale."""
    import time
    stale = _stale_teams()
    for attempt in range(retries):
        if not stale:
            return
        log.info(f"  Previous game not final for {', '.join(f'{t} ({d})' for t, d in stale)}"
                 f" — re-fetching (attempt {attempt + 1}/{retries})")
        if attempt:
            time.sleep(30)
        step_fetch_games_today()
        step_fetch_box_scores()
        step_rolling_stats_today()
        step_player_rolling_stats_today()
        stale = _stale_teams()
    if stale:
        raise RuntimeError(
            "stale inputs: previous game not final for "
            + ", ".join(f"{t} ({d})" for t, d in stale)
            + ". Their forecasts would miss that game (Elo, rest, rolling stats).")


def step_predict():
    from model.predict import predict_todays_games
    predictions = predict_todays_games()
    return predictions


def step_tweet(predictions, dry_run=False):
    try:
        from twitter.bot import post_predictions
    except ImportError:
        log.info("  Twitter module not available — skipping")
        return
    if predictions:
        post_predictions(predictions, dry_run=dry_run)
    else:
        log.info("  No predictions to tweet")


def run_pipeline(skip_retrain=False, dry_run_twitter=True,
                 skip_twitter=False, today_only=False):
    start = datetime.now()
    log.info("=" * 55)
    log.info(f"DAILY PIPELINE — {date.today()}"
             f"{' (today only)' if today_only else ''}")
    log.info("=" * 55)

    if today_only:
        # Only what today's predictions read. Historical rolling rows are left
        # for the next full run, so retraining is skipped too.
        run_step("Fetch games (recent + today)", step_fetch_games_today)
        run_step("Fetch box scores",             step_fetch_box_scores)
        run_step("Fetch odds",                   step_fetch_odds)
        run_step("Rolling stats (today)",        step_rolling_stats_today)
        run_step("Player rolling stats (today)", step_player_rolling_stats_today)
        run_step("Update series context",        step_update_series)
        skip_retrain = True
    else:
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
                train_production_models,
                save_models, FEATURE_COLS,
                TARGET_HOME, TARGET_AWAY
            )
            df = build_feature_dataset(window=10)
            df = df.dropna(
                subset=FEATURE_COLS + [TARGET_HOME, TARGET_AWAY]
            )
            df = df.sort_values('game_date').reset_index(drop=True)
            mh, ma, mm = train_production_models(df)
            save_models(mh, ma, mm)
        run_step("Retrain model", retrain)
    else:
        log.info("⏭️  Skipping retrain")

    run_step("Check previous games are final", step_check_previous_games_final)
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
    parser.add_argument('--today-only',    action='store_true',
                        help="Only fetch/compute what today's games need; "
                             "skips history backfill and retraining")
    args = parser.parse_args()

    if args.predict_only:
        run_step("Check previous games are final", step_check_previous_games_final)
        predictions = step_predict()
        step_tweet(predictions, dry_run=not args.post_twitter)
    else:
        run_pipeline(
            skip_retrain   = args.skip_retrain,
            dry_run_twitter = not args.post_twitter,
            skip_twitter   = args.skip_twitter,
            today_only     = args.today_only
        )