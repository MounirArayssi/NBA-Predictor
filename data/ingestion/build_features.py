import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from sqlalchemy import text
from data.storage.db import engine


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
            h.avg_points             AS home_avg_points,
            h.avg_offensive_rating   AS home_off_rating,
            h.avg_defensive_rating   AS home_def_rating,
            h.avg_pace               AS home_pace,
            h.avg_fg_pct             AS home_fg_pct,
            h.avg_fg3_pct            AS home_fg3_pct,
            h.three_point_rate       AS home_3pt_rate,
            h.win_pct                AS home_win_pct,
            h.home_avg_points        AS home_home_avg_pts,
            h.home_avg_points_allowed AS home_home_avg_pts_allowed,

            -- Away team rolling stats
            a.avg_points             AS away_avg_points,
            a.avg_offensive_rating   AS away_off_rating,
            a.avg_defensive_rating   AS away_def_rating,
            a.avg_pace               AS away_pace,
            a.avg_fg_pct             AS away_fg_pct,
            a.avg_fg3_pct            AS away_fg3_pct,
            a.three_point_rate       AS away_3pt_rate,
            a.win_pct                AS away_win_pct,
            a.away_avg_points        AS away_away_avg_pts,
            a.away_avg_points_allowed AS away_away_avg_pts_allowed,

            -- Matchup features
            ABS(h.avg_pace - a.avg_pace) AS pace_differential

        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id

        -- Join home team rolling stats (most recent before game date)
        JOIN team_rolling_stats h ON h.team_id = g.home_team_id
            AND h."window" = :window
            AND h.as_of_date = (
                SELECT MAX(as_of_date)
                FROM team_rolling_stats
                WHERE team_id = g.home_team_id
                AND "window" = :window
                AND as_of_date <= g.game_date
            )

        -- Join away team rolling stats (most recent before game date)
        JOIN team_rolling_stats a ON a.team_id = g.away_team_id
            AND a."window" = :window
            AND a.as_of_date = (
                SELECT MAX(as_of_date)
                FROM team_rolling_stats
                WHERE team_id = g.away_team_id
                AND "window" = :window
                AND as_of_date <= g.game_date
            )

        WHERE g.is_final = TRUE
        AND g.home_score IS NOT NULL
        AND g.away_score IS NOT NULL
        ORDER BY g.game_date
    """)

    df = pd.read_sql(query, engine, params={"window": window})
    print(f"Built {len(df)} game feature rows")

    # Add rest days
    df = add_rest_days(df)

    # Add head to head features
    df = add_h2h_features(df)

    return df


def add_rest_days(df):
    """Calculate rest days for each team before each game."""
    print("  Computing rest days...")

    df = df.sort_values('game_date').copy()
    df['game_date'] = pd.to_datetime(df['game_date'])

    home_last_game = {}
    away_last_game = {}

    home_rest = []
    away_rest = []

    for _, row in df.iterrows():
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        game_date = row['game_date']

        # Home rest days
        if home_id in home_last_game:
            rest = (game_date - home_last_game[home_id]).days
        else:
            rest = 3  # default for first game
        home_rest.append(rest)

        # Away rest days
        if away_id in away_last_game:
            rest = (game_date - away_last_game[away_id]).days
        else:
            rest = 3
        away_rest.append(rest)

        # Update last game date for both teams
        home_last_game[home_id] = game_date
        away_last_game[away_id] = game_date

    df['home_rest_days'] = home_rest
    df['away_rest_days'] = away_rest
    df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']
    df['home_back_to_back'] = (df['home_rest_days'] == 1).astype(int)
    df['away_back_to_back'] = (df['away_rest_days'] == 1).astype(int)

    return df


def add_h2h_features(df):
    """Add head to head historical features."""
    print("  Computing head-to-head features...")

    df = df.sort_values('game_date').copy()

    h2h_home_avg = []
    h2h_away_avg = []
    h2h_home_win_pct = []
    h2h_games_count = []

    # Build H2H history as we go (no future leakage)
    h2h_history = {}

    for _, row in df.iterrows():
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        key = tuple(sorted([home_id, away_id]))

        if key in h2h_history and len(h2h_history[key]) > 0:
            past = h2h_history[key]
            past_df = pd.DataFrame(past)

            # Filter to home team perspective
            home_games = past_df[past_df['home_team_id'] == home_id]
            away_games = past_df[past_df['home_team_id'] == away_id]

            all_home_scores = list(home_games['home_score']) + \
                              list(away_games['away_score'])
            all_away_scores = list(home_games['away_score']) + \
                              list(away_games['home_score'])

            h2h_home_avg.append(
                sum(all_home_scores) / len(all_home_scores)
                if all_home_scores else None
            )
            h2h_away_avg.append(
                sum(all_away_scores) / len(all_away_scores)
                if all_away_scores else None
            )

            home_wins = sum(1 for h, a in zip(all_home_scores, all_away_scores)
                           if h > a)
            h2h_home_win_pct.append(
                home_wins / len(all_home_scores)
                if all_home_scores else None
            )
            h2h_games_count.append(len(all_home_scores))
        else:
            h2h_home_avg.append(None)
            h2h_away_avg.append(None)
            h2h_home_win_pct.append(None)
            h2h_games_count.append(0)

        # Add this game to history
        if key not in h2h_history:
            h2h_history[key] = []
        h2h_history[key].append({
            'home_team_id': home_id,
            'away_team_id': away_id,
            'home_score': row['home_score'],
            'away_score': row['away_score'],
            'game_date': row['game_date']
        })

    df['h2h_home_avg_score'] = h2h_home_avg
    df['h2h_away_avg_score'] = h2h_away_avg
    df['h2h_home_win_pct'] = h2h_home_win_pct
    df['h2h_games_count'] = h2h_games_count

    # Fill missing H2H with rolling averages
    df['h2h_home_avg_score'] = df['h2h_home_avg_score'].fillna(
        df['home_avg_points']
    )
    df['h2h_away_avg_score'] = df['h2h_away_avg_score'].fillna(
        df['away_avg_points']
    )
    df['h2h_home_win_pct'] = df['h2h_home_win_pct'].fillna(0.5)

    return df


if __name__ == "__main__":
    df = build_feature_dataset(window=10)

    print("\nFeature dataset sample:")
    print(df[['game_date', 'home_team', 'away_team',
              'home_score', 'away_score',
              'home_avg_points', 'away_avg_points',
              'home_rest_days', 'away_rest_days',
              'h2h_games_count']].tail(10))

    print(f"\nTotal features: {len(df.columns)}")
    print(f"Total games: {len(df)}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Save to CSV for inspection
    df.to_csv('data/features.csv', index=False)
    print("\n✅ Features saved to data/features.csv")