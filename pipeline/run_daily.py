import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import date, datetime
from sqlalchemy.orm import Session
from data.storage.db import engine
from data.storage.models import Game, Team
from config.settings import CURRENT_SEASON

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(
            f'logs/pipeline_{date.today().strftime("%Y%m%d")}.log'
        ),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def step_fetch_new_games():
    log.info("STEP 1: Fetching new completed games...")
    try:
        from data.ingestion.fetch_games import (
            fetch_and_store_games,
            fetch_upcoming_games
        )
        fetch_and_store_games(
            season=CURRENT_SEASON,
            season_type='Playoffs'
        )
        fetch_and_store_games(
            season=CURRENT_SEASON,
            season_type='Regular Season'
        )
        fetch_upcoming_games()
        log.info("  ✅ Games fetched")
    except Exception as e:
        log.error(f"  ❌ Failed: {e}")
        raise


def step_fetch_box_scores():
    log.info("STEP 2: Fetching missing box scores...")
    try:
        from data.ingestion.fetch_box_scores import fetch_all_box_scores
        fetch_all_box_scores()
        log.info("  ✅ Box scores fetched")
    except Exception as e:
        log.error(f"  ❌ Failed: {e}")
        raise


def step_fetch_odds():
    log.info("STEP 3: Fetching today's odds...")
    try:
        from data.ingestion.fetch_odds import (
            fetch_todays_odds,
            parse_and_store_odds
        )
        odds_data = fetch_todays_odds()
        if odds_data:
            parse_and_store_odds(odds_data)
            log.info(f"  ✅ Odds fetched for {len(odds_data)} games")
        else:
            log.info("  ⚠️  No odds available today")
    except Exception as e:
        log.error(f"  ❌ Failed: {e}")
        raise


def step_compute_rolling_stats():
    log.info("STEP 4: Recomputing rolling stats...")
    try:
        from data.ingestion.compute_rolling_stats import (
            compute_and_store_rolling_stats
        )
        for window in [5, 10, 20]:
            compute_and_store_rolling_stats(window=window)
        log.info("  ✅ Rolling stats updated")
    except Exception as e:
        log.error(f"  ❌ Failed: {e}")
        raise


def step_compute_similarity():
    log.info("STEP 5: Recomputing team similarity...")
    try:
        from data.ingestion.compute_team_similarity import (
            compute_and_store_similarity
        )
        compute_and_store_similarity()
        log.info("  ✅ Team similarity updated")
    except Exception as e:
        log.error(f"  ❌ Failed: {e}")
        raise


def step_retrain_model():
    log.info("STEP 6: Retraining model...")
    try:
        from data.ingestion.build_features import build_feature_dataset
        from model.train import (
            chronological_split,
            train_model,
            save_models,
            FEATURE_COLS,
            TARGET_HOME,
            TARGET_AWAY
        )
        import pandas as pd

        df = build_feature_dataset(window=10)
        df = df.dropna(
            subset=FEATURE_COLS + [TARGET_HOME, TARGET_AWAY]
        )
        df = df.sort_values('game_date').reset_index(drop=True)

        train, test = chronological_split(df, test_ratio=0.2)

        model_home = train_model(train, TARGET_HOME)
        model_away = train_model(train, TARGET_AWAY)

        save_models(model_home, model_away)
        log.info("  ✅ Model retrained and saved")

        return model_home, model_away

    except Exception as e:
        log.error(f"  ❌ Failed: {e}")
        raise


def step_generate_predictions(model_home, model_away):
    log.info("STEP 7: Generating predictions for today's games...")
    try:
        from data.ingestion.build_features import build_feature_dataset
        from model.train import FEATURE_COLS
        import numpy as np
        import pickle

        # Load feature cols
        with open('model/feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)

        # Get today's scheduled games
        today = date.today()
        with Session(engine) as session:
            games = session.query(Game).filter(
                Game.game_date == today,
                Game.status == 'scheduled'
            ).all()

            if not games:
                log.info("  No games scheduled today")
                return []

            log.info(f"  Found {len(games)} games today")

            # Build features for today's games
            df = build_feature_dataset(window=10)
            today_df = df[
                df['game_date'] == pd.Timestamp(today)
            ].copy()

            if today_df.empty:
                log.info("  No feature data for today's games yet")
                return []

            predictions = []
            for _, row in today_df.iterrows():
                # Check all features available
                missing = [
                    f for f in FEATURE_COLS
                    if f not in row.index or pd.isna(row[f])
                ]
                if missing:
                    log.warning(
                        f"  Missing features for "
                        f"{row['away_team']} @ {row['home_team']}: "
                        f"{missing[:3]}..."
                    )
                    continue

                features = row[FEATURE_COLS].values.reshape(1, -1)
                home_pred = float(model_home.predict(features)[0])
                away_pred = float(model_away.predict(features)[0])

                home_wins = home_pred > away_pred
                margin = abs(home_pred - away_pred)

                # Confidence based on predicted margin
                if margin >= 12:
                    confidence = "High"
                elif margin >= 6:
                    confidence = "Medium"
                else:
                    confidence = "Low"

                pred = {
                    'home_team':    row['home_team'],
                    'away_team':    row['away_team'],
                    'home_pred':    round(home_pred),
                    'away_pred':    round(away_pred),
                    'predicted_winner': row['home_team'] if home_wins
                                        else row['away_team'],
                    'margin':       round(margin, 1),
                    'confidence':   confidence,
                    'is_playoff':   row['season_type'] == 'Playoffs',
                    'series_game':  int(row.get('series_game_num', 0)),
                }
                predictions.append(pred)

                log.info(
                    f"  {row['away_team']} @ {row['home_team']} | "
                    f"Pred: {row['home_team']} {round(home_pred)} — "
                    f"{row['away_team']} {round(away_pred)} | "
                    f"Winner: {pred['predicted_winner']} "
                    f"({confidence} confidence)"
                )

        return predictions

    except Exception as e:
        log.error(f"  ❌ Failed: {e}")
        raise


def run_pipeline(skip_retrain=False):
    """
    Run the full daily pipeline.
    skip_retrain=True for faster runs when you just want fresh data.
    """
    start = datetime.now()
    log.info("=" * 55)
    log.info(f"DAILY PIPELINE STARTING — {date.today()}")
    log.info("=" * 55)

    try:
        step_fetch_new_games()
        step_fetch_box_scores()
        step_fetch_odds()
        step_compute_rolling_stats()
        step_compute_similarity()

        if not skip_retrain:
            model_home, model_away = step_retrain_model()
        else:
            import pickle
            with open('model/model_home.pkl', 'rb') as f:
                model_home = pickle.load(f)
            with open('model/model_away.pkl', 'rb') as f:
                model_away = pickle.load(f)
            log.info("STEP 6: Skipped retraining — using saved model")

        predictions = step_generate_predictions(model_home, model_away)

        elapsed = (datetime.now() - start).seconds
        log.info("=" * 55)
        log.info(
            f"PIPELINE COMPLETE — {elapsed}s | "
            f"{len(predictions)} predictions generated"
        )
        log.info("=" * 55)

        return predictions

    except Exception as e:
        log.error(f"PIPELINE FAILED: {e}")
        raise


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(
        description='NBA Prediction Daily Pipeline'
    )
    parser.add_argument(
        '--skip-retrain',
        action='store_true',
        help='Skip model retraining (faster, uses saved model)'
    )
    parser.add_argument(
        '--odds-only',
        action='store_true',
        help='Only fetch odds and generate predictions'
    )
    args = parser.parse_args()

    if args.odds_only:
        # Quick run — just odds + predictions
        log.info("Quick run: odds + predictions only")
        step_fetch_odds()
        import pickle
        with open('model/model_home.pkl', 'rb') as f:
            model_home = pickle.load(f)
        with open('model/model_away.pkl', 'rb') as f:
            model_away = pickle.load(f)
        predictions = step_generate_predictions(model_home, model_away)
    else:
        predictions = run_pipeline(skip_retrain=args.skip_retrain)