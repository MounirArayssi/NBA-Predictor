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

            -- Matchup features
            ABS(h.avg_pace - a.avg_pace) AS pace_differential,

            -- Vegas lines (NULL for historical games without odds)
            go.home_spread        AS vegas_spread,
            go.total_line         AS vegas_total,
            go.vegas_home_implied AS vegas_home_implied,
            go.vegas_away_implied AS vegas_away_implied

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

        -- LEFT JOIN so games without odds still appear
        LEFT JOIN game_odds go ON go.game_id = g.game_id

        WHERE g.is_final = TRUE
        AND g.home_score IS NOT NULL
        AND g.away_score IS NOT NULL
        ORDER BY g.game_date
    """)

    df = pd.read_sql(query, engine, params={"window": window})
    print(f"Loaded {len(df)} game records")

    # Add features
    df = add_rest_days(df)
    df = add_h2h_features(df)
    df = add_playoff_features(df)
    df = add_vegas_features(df)
    df = add_similarity_features(df)
    df = add_playoff_elevation_features(df) 
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
        home_id   = row['home_team_id']
        away_id   = row['away_team_id']
        game_date = row['game_date']

        if home_id in home_last_game:
            rest = (game_date - home_last_game[home_id]).days
        else:
            rest = 3
        home_rest.append(rest)

        if away_id in away_last_game:
            rest = (game_date - away_last_game[away_id]).days
        else:
            rest = 3
        away_rest.append(rest)

        home_last_game[home_id] = game_date
        away_last_game[away_id] = game_date

    df['home_rest_days']    = home_rest
    df['away_rest_days']    = away_rest
    df['rest_advantage']    = df['home_rest_days'] - df['away_rest_days']
    df['home_back_to_back'] = (df['home_rest_days'] == 1).astype(int)
    df['away_back_to_back'] = (df['away_rest_days'] == 1).astype(int)

    return df


def add_h2h_features(df):
    """Add head to head historical features."""
    print("  Computing head-to-head features...")

    df = df.sort_values('game_date').copy()

    h2h_home_avg     = []
    h2h_away_avg     = []
    h2h_home_win_pct = []
    h2h_games_count  = []

    h2h_history = {}

    for _, row in df.iterrows():
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        key     = tuple(sorted([home_id, away_id]))

        if key in h2h_history and len(h2h_history[key]) > 0:
            past    = h2h_history[key]
            past_df = pd.DataFrame(past)

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

            home_wins = sum(
                1 for h, a in zip(all_home_scores, all_away_scores)
                if h > a
            )
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

        if key not in h2h_history:
            h2h_history[key] = []
        h2h_history[key].append({
            'home_team_id': home_id,
            'away_team_id': away_id,
            'home_score':   row['home_score'],
            'away_score':   row['away_score'],
            'game_date':    row['game_date']
        })

    df['h2h_home_avg_score'] = h2h_home_avg
    df['h2h_away_avg_score'] = h2h_away_avg
    df['h2h_home_win_pct']   = h2h_home_win_pct
    df['h2h_games_count']    = h2h_games_count

    # Fill missing H2H with rolling averages
    df['h2h_home_avg_score'] = df['h2h_home_avg_score'].fillna(
        df['home_avg_points']
    )
    df['h2h_away_avg_score'] = df['h2h_away_avg_score'].fillna(
        df['away_avg_points']
    )
    df['h2h_home_win_pct'] = df['h2h_home_win_pct'].fillna(0.5)

    return df


def add_playoff_features(df):
    """Add playoff-specific context features."""
    print("  Computing playoff features...")

    df = df.sort_values('game_date').copy()

    df['is_playoff'] = (df['season_type'] == 'Playoffs').astype(int)

    series_history   = {}
    series_game_num  = []
    home_series_wins = []
    away_series_wins = []
    is_elimination   = []
    series_momentum  = []

    for _, row in df.iterrows():
        if row['season_type'] != 'Playoffs':
            series_game_num.append(0)
            home_series_wins.append(0)
            away_series_wins.append(0)
            is_elimination.append(0)
            series_momentum.append(0)
            continue

        key        = tuple(sorted([row['home_team_id'], row['away_team_id']]))
        season_key = (row['season'], key)

        if season_key not in series_history:
            series_history[season_key] = {
                'games':     [],
                'home_wins': 0,
                'away_wins': 0
            }

        history  = series_history[season_key]
        game_num = len(history['games']) + 1
        h_wins   = history['home_wins']
        a_wins   = history['away_wins']

        series_game_num.append(game_num)
        home_series_wins.append(h_wins)
        away_series_wins.append(a_wins)
        is_elimination.append(int(h_wins == 3 or a_wins == 3))

        if len(history['games']) == 0:
            series_momentum.append(0)
        else:
            last_game = history['games'][-1]
            series_momentum.append(1 if last_game['home_won'] else -1)

        home_won = row['home_score'] > row['away_score']
        history['games'].append({'home_won': home_won})
        if home_won:
            history['home_wins'] += 1
        else:
            history['away_wins'] += 1

    df['series_game_num']  = series_game_num
    df['home_series_wins'] = home_series_wins
    df['away_series_wins'] = away_series_wins
    df['is_elimination']   = is_elimination
    df['series_momentum']  = series_momentum
    df['series_pressure']  = df['home_series_wins'] + df['away_series_wins']

    return df


def add_vegas_features(df):
    """Handle missing Vegas lines for historical games."""
    print("  Processing Vegas features...")

    # Fill missing odds with rolling average estimates
    df['vegas_total'] = df['vegas_total'].fillna(
        df['home_avg_points'] + df['away_avg_points']
    )
    df['vegas_spread'] = df['vegas_spread'].fillna(
        df['home_avg_points'] - df['away_avg_points']
    )
    df['vegas_home_implied'] = df['vegas_home_implied'].fillna(
        df['home_avg_points']
    )
    df['vegas_away_implied'] = df['vegas_away_implied'].fillna(
        df['away_avg_points']
    )

    # How much Vegas differs from our rolling average expectation
    df['vegas_vs_home_avg'] = \
        df['vegas_home_implied'] - df['home_avg_points']
    df['vegas_vs_away_avg'] = \
        df['vegas_away_implied'] - df['away_avg_points']

    return df

def add_similarity_features(df):
    """
    For each game, find the 3 most similar teams to each opponent
    and compute how the home/away team performed against those proxy teams.
    
    Two directions:
    1. Teams similar to the OPPONENT — how did home team do vs those teams?
    2. Teams similar to the HOME TEAM — how did similar teams do vs this opponent?
    """
    print("  Computing team similarity features...")

    # Load similarity matrix
    sim_query = text("""
        SELECT team_a_id, team_b_id, similarity_score
        FROM team_similarity
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM team_similarity)
    """)
    sim_df = pd.read_sql(sim_query, engine)

    # Build similarity lookup — both directions
    # {team_id: [(similar_team_id, score), ...]}
    similarity_map = {}
    for _, row in sim_df.iterrows():
        a = int(row['team_a_id'])
        b = int(row['team_b_id'])
        score = float(row['similarity_score'])

        if a not in similarity_map:
            similarity_map[a] = []
        if b not in similarity_map:
            similarity_map[b] = []

        similarity_map[a].append((b, score))
        similarity_map[b].append((a, score))

    # Sort by similarity descending
    for team_id in similarity_map:
        similarity_map[team_id] = sorted(
            similarity_map[team_id],
            key=lambda x: x[1],
            reverse=True
        )

    # Load all team box scores for proxy lookups
    bs_query = text("""
        SELECT
            tbs.team_id,
            tbs.points,
            tbs.offensive_rating,
            tbs.defensive_rating,
            g.game_date,
            CASE
                WHEN tbs.team_id = g.home_team_id THEN g.away_team_id
                ELSE g.home_team_id
            END AS opponent_team_id,
            CASE
                WHEN tbs.team_id = g.home_team_id THEN g.away_score
                ELSE g.home_score
            END AS opponent_points
        FROM team_box_scores tbs
        JOIN games g ON tbs.game_id = g.game_id
        WHERE g.is_final = TRUE
        ORDER BY g.game_date
    """)
    bs_df = pd.read_sql(bs_query, engine)
    bs_df['game_date'] = pd.to_datetime(bs_df['game_date'])

    # Features to compute
    home_proxy_off   = []  # home team off rating vs similar opponents
    home_proxy_pts   = []  # home team avg points vs similar opponents
    away_proxy_off   = []
    away_proxy_pts   = []
    home_sim_off     = []  # similar teams' off rating vs this opponent
    away_sim_off     = []

    df = df.sort_values('game_date').copy()
    df['game_date'] = pd.to_datetime(df['game_date'])

    for _, row in df.iterrows():
        home_id   = int(row['home_team_id'])
        away_id   = int(row['away_team_id'])
        game_date = row['game_date']

        # Get top 3 similar teams to the opponent
        away_similars = [
            t for t, _ in similarity_map.get(away_id, [])[:3]
            if t != home_id
        ]
        home_similars = [
            t for t, _ in similarity_map.get(home_id, [])[:3]
            if t != away_id
        ]

        # --- Direction 1: How did HOME team perform vs teams similar to AWAY ---
        proxy_games_home = bs_df[
            (bs_df['team_id'] == home_id) &
            (bs_df['opponent_team_id'].isin(away_similars)) &
            (bs_df['game_date'] < game_date)
        ]

        if len(proxy_games_home) >= 3:
            home_proxy_off.append(
                float(proxy_games_home['offensive_rating'].mean())
            )
            home_proxy_pts.append(
                float(proxy_games_home['points'].mean())
            )
        else:
            home_proxy_off.append(None)
            home_proxy_pts.append(None)

        # --- Direction 1: How did AWAY team perform vs teams similar to HOME ---
        proxy_games_away = bs_df[
            (bs_df['team_id'] == away_id) &
            (bs_df['opponent_team_id'].isin(home_similars)) &
            (bs_df['game_date'] < game_date)
        ]

        if len(proxy_games_away) >= 3:
            away_proxy_off.append(
                float(proxy_games_away['offensive_rating'].mean())
            )
            away_proxy_pts.append(
                float(proxy_games_away['points'].mean())
            )
        else:
            away_proxy_off.append(None)
            away_proxy_pts.append(None)

        # --- Direction 2: How did teams similar to HOME perform vs AWAY ---
        sim_home_vs_away = bs_df[
            (bs_df['team_id'].isin(home_similars)) &
            (bs_df['opponent_team_id'] == away_id) &
            (bs_df['game_date'] < game_date)
        ]

        home_sim_off.append(
            float(sim_home_vs_away['offensive_rating'].mean())
            if len(sim_home_vs_away) >= 2 else None
        )

        # --- Direction 2: How did teams similar to AWAY perform vs HOME ---
        sim_away_vs_home = bs_df[
            (bs_df['team_id'].isin(away_similars)) &
            (bs_df['opponent_team_id'] == home_id) &
            (bs_df['game_date'] < game_date)
        ]

        away_sim_off.append(
            float(sim_away_vs_home['offensive_rating'].mean())
            if len(sim_away_vs_home) >= 2 else None
        )

    df['home_proxy_off_rating'] = home_proxy_off
    df['home_proxy_avg_pts']    = home_proxy_pts
    df['away_proxy_off_rating'] = away_proxy_off
    df['away_proxy_avg_pts']    = away_proxy_pts
    df['home_sim_off_rating']   = home_sim_off
    df['away_sim_off_rating']   = away_sim_off

    # Fill missing with rolling averages
    df['home_proxy_off_rating'] = df['home_proxy_off_rating'].fillna(
        df['home_off_rating']
    )
    df['home_proxy_avg_pts'] = df['home_proxy_avg_pts'].fillna(
        df['home_avg_points']
    )
    df['away_proxy_off_rating'] = df['away_proxy_off_rating'].fillna(
        df['away_off_rating']
    )
    df['away_proxy_avg_pts'] = df['away_proxy_avg_pts'].fillna(
        df['away_avg_points']
    )
    df['home_sim_off_rating'] = df['home_sim_off_rating'].fillna(
        df['home_off_rating']
    )
    df['away_sim_off_rating'] = df['away_sim_off_rating'].fillna(
        df['away_off_rating']
    )

    return df

def add_playoff_elevation_features(df):
    """Add team playoff elevation scores as model features."""
    print("  Computing playoff elevation features...")

    from data.ingestion.compute_playoff_factors import (
        compute_player_playoff_factors,
        compute_team_playoff_elevation
    )

    # Compute factors
    factor_dict, _ = compute_player_playoff_factors()
    if not factor_dict:
        df['home_playoff_elevation'] = 0.0
        df['away_playoff_elevation'] = 0.0
        df['playoff_elevation_diff'] = 0.0
        return df

    team_elevations = compute_team_playoff_elevation(factor_dict)

    # Map to games
    df['home_playoff_elevation'] = df['home_team_id'].map(
        team_elevations
    ).fillna(0.0)
    df['away_playoff_elevation'] = df['away_team_id'].map(
        team_elevations
    ).fillna(0.0)

    # Differential — positive means home team elevates more
    df['playoff_elevation_diff'] = (
        df['home_playoff_elevation'] - df['away_playoff_elevation']
    )

    # Only apply during playoffs — zero out for regular season
    df.loc[df['season_type'] != 'Playoffs', 'home_playoff_elevation'] = 0.0
    df.loc[df['season_type'] != 'Playoffs', 'away_playoff_elevation'] = 0.0
    df.loc[df['season_type'] != 'Playoffs', 'playoff_elevation_diff'] = 0.0

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
        'series_momentum',
        'vegas_total', 'vegas_spread',
        'vegas_home_implied', 'vegas_away_implied'
    ]].tail(10))

    print(f"\nTotal features: {len(df.columns)}")
    print(f"Total games:    {len(df)}")
    print(f"\nMissing values:\n"
          f"{df.isnull().sum()[df.isnull().sum() > 0]}")

    df.to_csv('data/features.csv', index=False)
    print("\n✅ Features saved to data/features.csv")