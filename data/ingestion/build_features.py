import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

"""
Assembles the game-level feature matrix: the base rolling-stats query below, then
the category builders in data/features/ (A strength, B recent form, C availability,
D rest, E playoffs). Call order matters: H2H, similarity and in-series form fill
from rolling columns; playoff elevation reads series state from add_playoff_features.
"""
import pandas as pd
from sqlalchemy import text
from data.storage.db import engine

from data.features.strength import (
    add_team_strength_features, add_home_court_strength, add_matchup_interactions,
    add_similarity_features, add_style_defensive_matchup, add_h2h_features)
from data.features.recent_form import (
    add_momentum_features, add_scoring_variance, add_shooting_consistency,
    add_quarter_features, add_advanced_rolling_features)
from data.features.availability import add_availability_features
from data.features.rest import add_rest_days
from data.features.playoffs import (
    add_playoff_features, add_playoff_elevation_features, add_series_form_features)

# Re-exported for model/predict.py and web/scripts/breakdown_inputs.py
from data.features.rest import REST_CAP  # noqa: F401
from data.features.playoffs import SERIES_SHRINK_K, shrink_series_avg  # noqa: F401
from data.features.availability import (  # noqa: F401
    ROT_WINDOW, ROT_MIN_GAMES, ROT_MIN_MINUTES, normalize_player_name,
    compute_availability, compute_live_availability,
    _load_played_box_scores, _season_rotation)


def build_feature_dataset(window=10):
    print(f"Building feature dataset (window={window})...")

    query = text("""
        SELECT
            g.game_id,
            g.nba_game_id,
            g.game_date,
            g.season,
            g.season_type,
            g.home_score,
            g.away_score,
            g.home_team_id,
            g.away_team_id,
            ht.abbreviation AS home_team,
            at.abbreviation AS away_team,

            -- Home team rolling stats
            h.avg_points              AS home_avg_points,
            h.avg_offensive_rating    AS home_off_rating,
            h.avg_defensive_rating    AS home_def_rating,
            h.avg_pace                AS home_pace,
            h.avg_fg_pct              AS home_fg_pct,
            h.avg_fg3_pct             AS home_fg3_pct,
            h.three_point_rate        AS home_3pt_rate,
            h.win_pct                 AS home_win_pct,
            h.home_avg_points         AS home_home_avg_pts,
            h.home_avg_points_allowed AS home_home_avg_pts_allowed,
            h.avg_ft_rate             AS home_ft_rate,

            -- Away team rolling stats
            a.avg_points              AS away_avg_points,
            a.avg_offensive_rating    AS away_off_rating,
            a.avg_defensive_rating    AS away_def_rating,
            a.avg_pace                AS away_pace,
            a.avg_fg_pct              AS away_fg_pct,
            a.avg_fg3_pct             AS away_fg3_pct,
            a.three_point_rate        AS away_3pt_rate,
            a.win_pct                 AS away_win_pct,
            a.away_avg_points         AS away_away_avg_pts,
            a.away_avg_points_allowed AS away_away_avg_pts_allowed,
            a.avg_ft_rate             AS away_ft_rate,

            -- Matchup features
            ABS(h.avg_pace - a.avg_pace) AS pace_differential,

            h7.avg_points             AS home_avg_points_l7,
            h7.avg_offensive_rating   AS home_off_rating_l7,
            h7.avg_defensive_rating   AS home_def_rating_l7,
            h7.win_pct                AS home_win_pct_l7,
            a7.avg_points             AS away_avg_points_l7,
            a7.avg_offensive_rating   AS away_off_rating_l7,
            a7.avg_defensive_rating   AS away_def_rating_l7,
            a7.win_pct                AS away_win_pct_l7

        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id

        -- Home team rolling stats
        JOIN team_rolling_stats h ON h.team_id = g.home_team_id
            AND h."window" = :window
            AND h.as_of_date = (
                SELECT MAX(as_of_date)
                FROM team_rolling_stats
                WHERE team_id = g.home_team_id
                AND "window" = :window
                AND as_of_date <= g.game_date
            )

        -- Away team rolling stats
        JOIN team_rolling_stats a ON a.team_id = g.away_team_id
            AND a."window" = :window
            AND a.as_of_date = (
                SELECT MAX(as_of_date)
                FROM team_rolling_stats
                WHERE team_id = g.away_team_id
                AND "window" = :window
                AND as_of_date <= g.game_date
            )
                 
        JOIN team_rolling_stats h7 ON h7.team_id = g.home_team_id
    AND h7."window" = 7
    AND h7.as_of_date = (
        SELECT MAX(as_of_date)
        FROM team_rolling_stats
        WHERE team_id = g.home_team_id
        AND "window" = 7
        AND as_of_date <= g.game_date
    )

        JOIN team_rolling_stats a7 ON a7.team_id = g.away_team_id
        AND a7."window" = 7
        AND a7.as_of_date = (
        SELECT MAX(as_of_date)
        FROM team_rolling_stats
        WHERE team_id = g.away_team_id
        AND "window" = 7
        AND as_of_date <= g.game_date
    )

        WHERE g.is_final = TRUE
        AND g.home_score IS NOT NULL
        AND g.away_score IS NOT NULL
        ORDER BY g.game_date
    """)

    df = pd.read_sql(query, engine, params={"window": window})
    print(f"Loaded {len(df)} game records")

    # Order preserved from the original single-file version (column order too).
    df = add_rest_days(df)                    # D
    df = add_h2h_features(df)                 # A
    df = add_playoff_features(df)             # E
    df = add_similarity_features(df)          # A
    df = add_playoff_elevation_features(df)   # E  (needs series state)
    df = add_home_court_strength(df)          # A
    df = add_scoring_variance(df)             # B
    df = add_matchup_interactions(df)         # A
    df = add_momentum_features(df)            # B
    df = add_style_defensive_matchup(df)      # A
    df = add_shooting_consistency(df)         # B
    df = add_quarter_features(df)             # B
    df = add_series_form_features(df)         # E
    df = add_advanced_rolling_features(df)    # B
    df = add_availability_features(df)        # C
    df = add_team_strength_features(df)       # A  (Elo)
    return df


if __name__ == "__main__":
    df = build_feature_dataset(window=10)

    print("\nFeature dataset sample:")
    print(df[[
        'game_date', 'home_team', 'away_team',
        'home_score', 'away_score',
        'home_avg_points', 'away_avg_points',
        'home_rest_days', 'away_rest_days',
        'is_playoff', 'series_game_num',
        'series_momentum'
    ]].tail(10))

    print(f"\nTotal features: {len(df.columns)}")
    print(f"Total games:    {len(df)}")
    print(f"\nMissing values:\n"
          f"{df.isnull().sum()[df.isnull().sum() > 0]}")

    df.to_csv('data/features.csv', index=False)
    print("\n✅ Features saved to data/features.csv")