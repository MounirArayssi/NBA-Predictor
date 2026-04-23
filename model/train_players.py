import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sqlalchemy import text
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from data.storage.db import engine


# Stats we're predicting per player
TARGET_STATS = ['points', 'rebounds', 'assists']

# Minimum games to qualify for prediction
MIN_GAMES_QUALIFIER = 10
MIN_MINUTES         = 15.0


def load_player_training_data():
    """
    Load historical player game data with all features.
    One row per player per game.
    """
    print("Loading player training data...")

    query = text("""
        SELECT
            pbs.player_id,
            pbs.team_id,
            pbs.game_id,
            p.full_name,
            t.abbreviation AS team,

            -- Target stats
            pbs.points,
            pbs.rebounds,
            pbs.assists,
            pbs.steals,
            pbs.blocks,
            pbs.turnovers,
            pbs.minutes_played,

            -- Player rolling stats (10 game window)
            prs.avg_points       AS player_avg_points_l10,
            prs.avg_rebounds     AS player_avg_reb_l10,
            prs.avg_assists      AS player_avg_ast_l10,
            prs.avg_minutes      AS player_avg_min_l10,
            prs.avg_usage_rate   AS player_avg_usage_l10,
            prs.avg_true_shooting AS player_avg_ts_l10,
            prs.avg_fg_pct       AS player_avg_fg_l10,
            prs.avg_fg3_pct      AS player_avg_fg3_l10,
            prs.avg_plus_minus   AS player_avg_pm_l10,
            prs.avg_turnovers    AS player_avg_tov_l10,

            -- Player rolling stats (5 game window — recent form)
            prs5.avg_points      AS player_avg_points_l5,
            prs5.avg_rebounds    AS player_avg_reb_l5,
            prs5.avg_assists     AS player_avg_ast_l5,
            prs5.avg_usage_rate  AS player_avg_usage_l5,
            prs5.avg_true_shooting AS player_avg_ts_l5,

            -- Opponent team stats
            opp_trs.avg_defensive_rating AS opp_def_rating,
            opp_trs.avg_pace             AS opp_pace,
            opp_trs.avg_points_allowed   AS opp_points_allowed,

            -- Team context
            team_trs.avg_pace            AS team_pace,
            team_trs.avg_offensive_rating AS team_off_rating,

            -- Game context
            g.game_date,
            g.season,
            g.season_type,
            tbs.is_home,
            tbs.offensive_rating  AS game_team_off_rating,
            tbs.pace              AS game_pace,

            -- Opponent team ID
            CASE
                WHEN tbs.team_id = g.home_team_id THEN g.away_team_id
                ELSE g.home_team_id
            END AS opponent_team_id

        FROM player_box_scores pbs
        JOIN players p ON pbs.player_id = p.player_id
        JOIN teams t ON pbs.team_id = t.team_id
        JOIN games g ON pbs.game_id = g.game_id
        JOIN team_box_scores tbs
            ON tbs.game_id = g.game_id
            AND tbs.team_id = pbs.team_id

        -- Player 10-game rolling stats
        JOIN player_rolling_stats prs
            ON prs.player_id = pbs.player_id
            AND prs.team_id  = pbs.team_id
            AND prs."window" = 10
            AND prs.as_of_date = (
                SELECT MAX(as_of_date)
                FROM player_rolling_stats
                WHERE player_id = pbs.player_id
                AND team_id     = pbs.team_id
                AND "window"    = 10
                AND as_of_date  <= g.game_date
            )

        -- Player 5-game rolling stats
        JOIN player_rolling_stats prs5
            ON prs5.player_id = pbs.player_id
            AND prs5.team_id  = pbs.team_id
            AND prs5."window" = 5
            AND prs5.as_of_date = (
                SELECT MAX(as_of_date)
                FROM player_rolling_stats
                WHERE player_id = pbs.player_id
                AND team_id     = pbs.team_id
                AND "window"    = 5
                AND as_of_date  <= g.game_date
            )

        -- Opponent defensive stats
        JOIN team_rolling_stats opp_trs
            ON opp_trs."window" = 10
            AND opp_trs.team_id = CASE
                WHEN pbs.team_id = g.home_team_id THEN g.away_team_id
                ELSE g.home_team_id
            END
            AND opp_trs.as_of_date = (
                SELECT MAX(as_of_date)
                FROM team_rolling_stats
                WHERE team_id = CASE
                    WHEN pbs.team_id = g.home_team_id THEN g.away_team_id
                    ELSE g.home_team_id
                END
                AND "window"   = 10
                AND as_of_date <= g.game_date
            )

        -- Own team rolling stats
        JOIN team_rolling_stats team_trs
            ON team_trs."window"  = 10
            AND team_trs.team_id  = pbs.team_id
            AND team_trs.as_of_date = (
                SELECT MAX(as_of_date)
                FROM team_rolling_stats
                WHERE team_id  = pbs.team_id
                AND "window"   = 10
                AND as_of_date <= g.game_date
            )

        WHERE g.is_final = TRUE
        AND pbs.minutes_played >= :min_minutes
        AND pbs.points IS NOT NULL
        ORDER BY g.game_date, pbs.player_id
    """)

    df = pd.read_sql(
        query, engine,
        params={'min_minutes': MIN_MINUTES}
    )
    print(f"  Loaded {len(df)} player game records")
    return df


def add_player_features(df):
    """Add computed features to player dataset."""
    print("  Adding player features...")

    df['game_date'] = pd.to_datetime(df['game_date'])

    # Momentum — recent form vs 10-game average
    df['points_momentum'] = (
        df['player_avg_points_l5'] - df['player_avg_points_l10']
    )
    df['usage_momentum'] = (
        df['player_avg_usage_l5'] - df['player_avg_usage_l10']
    )
    df['ts_momentum'] = (
        df['player_avg_ts_l5'] - df['player_avg_ts_l10']
    )

    # Home/away flag
    df['is_home'] = df['is_home'].astype(int)

    # Playoff flag
    df['is_playoff'] = (
        df['season_type'] == 'Playoffs'
    ).astype(int)

    # Pace context — high pace = more possessions = more stats
    df['pace_factor'] = df['opp_pace'] / 98.0  # normalize around 98

    # Offensive environment
    df['off_environment'] = (
        df['team_off_rating'] - df['opp_def_rating']
    )

    # Usage share proxy
    df['usage_rate_clean'] = df['player_avg_usage_l10'].fillna(0.20)

    return df


# Feature columns for player models
PLAYER_FEATURE_COLS = [
    # Player recent form (10 game)
    'player_avg_points_l10',
    'player_avg_reb_l10',
    'player_avg_ast_l10',
    'player_avg_min_l10',
    'player_avg_usage_l10',
    'player_avg_ts_l10',
    'player_avg_fg_l10',
    'player_avg_fg3_l10',
    'player_avg_pm_l10',
    'player_avg_tov_l10',

    # Recent form (5 game)
    'player_avg_points_l5',
    'player_avg_reb_l5',
    'player_avg_ast_l5',
    'player_avg_usage_l5',
    'player_avg_ts_l5',

    # Momentum
    'points_momentum',
    'usage_momentum',
    'ts_momentum',

    # Game context
    'is_home',
    'is_playoff',
    'pace_factor',
    'off_environment',
    'usage_rate_clean',

    # Opponent context
    'opp_def_rating',
    'opp_pace',
    'team_off_rating',
]


def train_player_model(df, target_col):
    """Train a model for one stat (points, rebounds, or assists)."""

    # Filter to players with enough history
    valid = df.dropna(subset=PLAYER_FEATURE_COLS + [target_col])

    # Chronological split — 85/15
    split_idx = int(len(valid) * 0.85)
    train = valid.iloc[:split_idx]
    test  = valid.iloc[split_idx:]

    X_train = train[PLAYER_FEATURE_COLS]
    y_train = train[target_col]
    X_test  = test[PLAYER_FEATURE_COLS]
    y_test  = test[target_col]

    # Recency weighting
    max_date = pd.to_datetime(train['game_date']).max()
    days_ago = (
        max_date - pd.to_datetime(train['game_date'])
    ).dt.days
    sample_weights = np.exp(-days_ago / 540)

    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.04,
        max_depth=3,
        min_samples_leaf=15,
        subsample=0.7,
        max_features=0.8,
        random_state=42
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluate
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae  = mean_absolute_error(y_test, test_preds)

    print(f"  {target_col:10} → "
          f"Train MAE: {train_mae:.2f} | "
          f"Test MAE: {test_mae:.2f} | "
          f"Train size: {len(train)}")

    return model, test_mae


def evaluate_player_predictions(models, df):
    """Show sample player predictions vs actuals."""
    print("\nSample player predictions vs actuals:")

    # Use test set — last 15%
    valid = df.dropna(subset=PLAYER_FEATURE_COLS + TARGET_STATS)
    split_idx = int(len(valid) * 0.85)
    test = valid.iloc[split_idx:].copy()

    X_test = test[PLAYER_FEATURE_COLS]

    for stat in TARGET_STATS:
        test[f'pred_{stat}'] = np.round(
            models[stat].predict(X_test), 1
        )

    # Show recent high-usage players
    recent = test[
        test['player_avg_usage_l10'] >= 0.25
    ].tail(15)

    print(f"\n{'Player':<25} {'Team':>4} "
          f"{'Pred Pts':>8} {'Act Pts':>7} "
          f"{'Pred Reb':>8} {'Act Reb':>7} "
          f"{'Pred Ast':>8} {'Act Ast':>7}")
    print("-" * 75)

    for _, row in recent.iterrows():
        print(
            f"{row['full_name']:<25} {row['team']:>4} "
            f"{row['pred_points']:>8.1f} {row['points']:>7.0f} "
            f"{row['pred_rebounds']:>8.1f} {row['rebounds']:>7.0f} "
            f"{row['pred_assists']:>8.1f} {row['assists']:>7.0f}"
        )


def save_player_models(models):
    """Save trained player models."""
    os.makedirs('model', exist_ok=True)
    with open('model/player_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    with open('model/player_feature_cols.pkl', 'wb') as f:
        pickle.dump(PLAYER_FEATURE_COLS, f)
    print("\n✅ Player models saved to model/")


if __name__ == "__main__":
    # Load data
    df = load_player_training_data()
    df = add_player_features(df)

    print(f"\n  Total records: {len(df)}")
    print(f"  Unique players: {df['player_id'].nunique()}")
    print(f"  Date range: {df['game_date'].min().date()} → "
          f"{df['game_date'].max().date()}")

    # Train one model per stat
    print("\nTraining player models...")
    print("-" * 55)
    models = {}
    for stat in TARGET_STATS:
        model, test_mae = train_player_model(df, stat)
        models[stat] = model

    # Show sample predictions
    evaluate_player_predictions(models, df)

    # Save
    save_player_models(models)

    print("\n✅ Player model training complete")