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

            h7.avg_points             AS home_avg_points_l7,
            h7.avg_offensive_rating   AS home_off_rating_l7,
            h7.avg_defensive_rating   AS home_def_rating_l7,
            h7.win_pct                AS home_win_pct_l7,
            a7.avg_points             AS away_avg_points_l7,
            a7.avg_offensive_rating   AS away_off_rating_l7,
            a7.avg_defensive_rating   AS away_def_rating_l7,
            a7.win_pct                AS away_win_pct_l7,      
                  
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
    df = add_home_court_strength(df)      # add
    df = add_scoring_variance(df)         # add
    df = add_matchup_interactions(df)
    df = add_style_defensive_matchup(df)
    df = add_shooting_consistency(df)
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
    print("  Processing Vegas features...")

    # For historical games — don't fill with rolling averages
    # Use 0 as a flag meaning "no real odds available"
    # Model will learn to ignore when 0, trust when real
    df['has_vegas_odds'] = df['vegas_total'].notna().astype(int)

    df['vegas_total'] = df['vegas_total'].fillna(0)
    df['vegas_spread'] = df['vegas_spread'].fillna(0)
    df['vegas_home_implied'] = df['vegas_home_implied'].fillna(0)
    df['vegas_away_implied'] = df['vegas_away_implied'].fillna(0)

    # Only compute these when real odds exist
    df['vegas_vs_home_avg'] = df.apply(
        lambda r: r['vegas_home_implied'] - r['home_avg_points']
        if r['has_vegas_odds'] else 0,
        axis=1
    )
    df['vegas_vs_away_avg'] = df.apply(
        lambda r: r['vegas_away_implied'] - r['away_avg_points']
        if r['has_vegas_odds'] else 0,
        axis=1
    )

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
    """
    Add team playoff elevation scores as model features.
    Applies series context adjustments for playoff games.
    """
    print("  Computing playoff elevation features...")

    from data.ingestion.compute_playoff_factors import (
        compute_player_playoff_factors,
        compute_team_playoff_elevation,
        get_series_adjusted_elevations
    )

    # Compute base factors
    factor_dict, _ = compute_player_playoff_factors()
    if not factor_dict:
        df['home_playoff_elevation'] = 0.0
        df['away_playoff_elevation'] = 0.0
        df['playoff_elevation_diff'] = 0.0
        return df

    team_elevations = compute_team_playoff_elevation(factor_dict)

    home_elevations = []
    away_elevations = []

    for _, row in df.iterrows():
        if row['season_type'] != 'Playoffs':
            home_elevations.append(0.0)
            away_elevations.append(0.0)
            continue

        home_adj, away_adj = get_series_adjusted_elevations(
            team_elevations  = team_elevations,
            home_team_id     = row['home_team_id'],
            away_team_id     = row['away_team_id'],
            series_game_num  = int(row['series_game_num']) 
                               if row['series_game_num'] > 0 else 1,
            home_series_wins = int(row['home_series_wins']),
            away_series_wins = int(row['away_series_wins'])
        )

        home_elevations.append(home_adj)
        away_elevations.append(away_adj)

    df['home_playoff_elevation'] = home_elevations
    df['away_playoff_elevation'] = away_elevations
    df['playoff_elevation_diff'] = (
        df['home_playoff_elevation'] - df['away_playoff_elevation']
    )

    return df

def add_home_court_strength(df):
    """
    Compute how much stronger each team's home court is
    vs league average home court advantage.
    Some arenas (OKC, BOS, DEN) have measurably stronger impact.
    """
    print("  Computing home court strength...")

    from sqlalchemy import text

    query = text("""
        SELECT
            g.home_team_id AS team_id,
            AVG(tbs_home.points - tbs_away.points) AS avg_home_margin,
            COUNT(*) AS home_games
        FROM games g
        JOIN team_box_scores tbs_home
            ON tbs_home.game_id = g.game_id
            AND tbs_home.team_id = g.home_team_id
        JOIN team_box_scores tbs_away
            ON tbs_away.game_id = g.game_id
            AND tbs_away.team_id = g.away_team_id
        WHERE g.is_final = TRUE
        AND g.season IN ('2024-25', '2025-26')
        GROUP BY g.home_team_id
        HAVING COUNT(*) >= 20
    """)

    home_margins = pd.read_sql(query, engine)

    # League average home margin
    league_avg_margin = float(home_margins['avg_home_margin'].mean())

    # Home court strength = how much better than league average
    home_margins['home_court_strength'] = (
        home_margins['avg_home_margin'] - league_avg_margin
    )

    strength_map = dict(zip(
        home_margins['team_id'],
        home_margins['home_court_strength']
    ))

    df['home_court_strength'] = df['home_team_id'].map(
        strength_map
    ).fillna(0.0)

    return df


def add_scoring_variance(df):
    """
    Add scoring consistency features.
    High variance teams are harder to predict.
    Low variance teams are more consistent and predictable.
    """
    print("  Computing scoring variance...")

    query = text("""
        SELECT
            tbs.team_id,
            g.game_date,
            tbs.points,
            tbs.is_home
        FROM team_box_scores tbs
        JOIN games g ON tbs.game_id = g.game_id
        WHERE g.is_final = TRUE
        ORDER BY tbs.team_id, g.game_date
    """)

    scores_df = pd.read_sql(query, engine)
    scores_df['game_date'] = pd.to_datetime(scores_df['game_date'])
    df['game_date'] = pd.to_datetime(df['game_date'])

    home_var = []
    away_var = []
    home_consistency = []
    away_consistency = []

    for _, row in df.iterrows():
        game_date = row['game_date']
        home_id   = row['home_team_id']
        away_id   = row['away_team_id']

        # Last 10 games before this game
        home_recent = scores_df[
            (scores_df['team_id'] == home_id) &
            (scores_df['game_date'] < game_date)
        ].tail(10)

        away_recent = scores_df[
            (scores_df['team_id'] == away_id) &
            (scores_df['game_date'] < game_date)
        ].tail(10)

        # Scoring standard deviation
        h_std = float(home_recent['points'].std()) \
                if len(home_recent) >= 5 else 12.0
        a_std = float(away_recent['points'].std()) \
                if len(away_recent) >= 5 else 12.0

        # Consistency score — lower std = more consistent
        h_consistency = 1.0 / (1.0 + h_std)
        a_consistency = 1.0 / (1.0 + a_std)

        home_var.append(h_std)
        away_var.append(a_std)
        home_consistency.append(h_consistency)
        away_consistency.append(a_consistency)

    df['home_scoring_std']     = home_var
    df['away_scoring_std']     = away_var
    df['home_consistency']     = home_consistency
    df['away_consistency']     = away_consistency
    df['variance_differential'] = (
        df['home_scoring_std'] - df['away_scoring_std']
    )

    return df


def add_matchup_interactions(df):
    """
    Add interaction features between offensive and defensive styles.
    These directly affect predicted totals:
    - High 3pt rate offense vs good 3pt defense = lower scoring
    - High TOV forcing defense vs turnover prone offense = lower scoring
    - High FT rate offense vs foul prone defense = higher scoring
    """
    print("  Computing matchup interactions...")

    # 3pt matchup — home offense vs away defense
    # Positive = home team 3pt rate exceeds away team's 3pt defense
    df['home_3pt_matchup'] = (
        df['home_3pt_rate'] - df['away_3pt_rate']
    )
    df['away_3pt_matchup'] = (
        df['away_3pt_rate'] - df['home_3pt_rate']
    )

    # Pace matchup — combined pace determines total possessions
    df['combined_pace'] = (
        df['home_pace'] + df['away_pace']
    ) / 2

    # Offensive vs defensive rating matchup
    # How much does home offense exceed away defense?
    df['home_off_vs_away_def'] = (
        df['home_off_rating'] - df['away_def_rating']
    )
    df['away_off_vs_home_def'] = (
        df['away_off_rating'] - df['home_def_rating']
    )

    # Net rating differential — overall team quality gap
    df['net_rating_diff'] = (
        (df['home_off_rating'] - df['home_def_rating']) -
        (df['away_off_rating'] - df['away_def_rating'])
    )

    # Predicted game total based on ratings
    # Rough estimate: (home_off + away_off) / 2 * pace factor
    df['implied_total'] = (
        df['home_avg_points'] + df['away_avg_points']
    )

    # Momentum — are they trending up or down recently?
    # Positive = scoring more in last 7 than last 10
    df['home_momentum'] = (
        df['home_avg_points_l7'] - df['home_avg_points']
    )
    df['away_momentum'] = (
        df['away_avg_points_l7'] - df['away_avg_points']
    )
    df['home_def_momentum'] = (
        df['home_def_rating'] - df['home_def_rating_l7']
    )
    df['away_def_momentum'] = (
        df['away_def_rating'] - df['away_def_rating_l7']
    )

    return df

def add_style_defensive_matchup(df):
    """
    For each game compute how each defense performs against
    teams with a similar offensive style to tonight's opponent.
    
    This goes beyond raw defensive rating to capture:
    - Does this defense struggle against pace?
    - Does this defense struggle against 3pt heavy teams?
    - Does this defense struggle against paint-heavy teams?
    """
    print("  Computing style-based defensive matchups...")

    # Pull all team box scores with opponent style info
    query = text("""
        SELECT
            tbs_def.team_id        AS def_team_id,
            tbs_off.team_id        AS off_team_id,
            tbs_def.defensive_rating,
            tbs_off.offensive_rating,
            tbs_off.pace,
            tbs_off.fg3_pct,
            tbs_off.fg3a,
            tbs_off.fga,
            tbs_off.oreb_pct,
            tbs_off.tov_pct,
            tbs_def.points         AS points_allowed,
            g.game_date
        FROM team_box_scores tbs_def
        JOIN games g ON tbs_def.game_id = g.game_id
        JOIN team_box_scores tbs_off
            ON tbs_off.game_id = g.game_id
            AND tbs_off.team_id != tbs_def.team_id
        WHERE g.is_final = TRUE
        AND tbs_def.defensive_rating IS NOT NULL
        AND tbs_off.pace IS NOT NULL
        ORDER BY g.game_date
    """)

    matchup_df = pd.read_sql(query, engine)
    matchup_df['game_date'] = pd.to_datetime(matchup_df['game_date'])
    matchup_df['three_point_rate'] = (
        matchup_df['fg3a'] / matchup_df['fga'].replace(0, 1)
    )
    df['game_date'] = pd.to_datetime(df['game_date'])

    home_def_vs_style = []
    away_def_vs_style = []

    for _, row in df.iterrows():
        game_date = row['game_date']
        home_id   = int(row['home_team_id'])
        away_id   = int(row['away_team_id'])

        # Home team's offensive style
        home_style_pace = row.get('home_pace', 98.0)
        home_style_3pt  = row.get('home_3pt_rate', 0.35)

        # Away team's offensive style
        away_style_pace = row.get('away_pace', 98.0)
        away_style_3pt  = row.get('away_3pt_rate', 0.35)

        # How does AWAY defense perform vs teams similar to HOME offense?
        # Find games where away team defended against teams
        # with similar pace and 3pt rate to home team
        away_def_history = matchup_df[
            (matchup_df['def_team_id'] == away_id) &
            (matchup_df['game_date'] < game_date) &
            (abs(matchup_df['pace'] - home_style_pace) < 5) &
            (abs(matchup_df['three_point_rate'] - home_style_3pt) < 0.07)
        ]

        if len(away_def_history) >= 3:
            away_def_vs_style.append(
                float(away_def_history['defensive_rating'].mean())
            )
        else:
            # Fall back to overall defensive rating
            away_def_vs_style.append(
                float(row.get('away_def_rating', 112.0))
            )

        # How does HOME defense perform vs teams similar to AWAY offense?
        home_def_history = matchup_df[
            (matchup_df['def_team_id'] == home_id) &
            (matchup_df['game_date'] < game_date) &
            (abs(matchup_df['pace'] - away_style_pace) < 5) &
            (abs(matchup_df['three_point_rate'] - away_style_3pt) < 0.07)
        ]

        if len(home_def_history) >= 3:
            home_def_vs_style.append(
                float(home_def_history['defensive_rating'].mean())
            )
        else:
            home_def_vs_style.append(
                float(row.get('home_def_rating', 112.0))
            )

    df['home_def_vs_away_style'] = home_def_vs_style
    df['away_def_vs_home_style'] = away_def_vs_style

    # How much better/worse is this defense vs this specific style
    # compared to their overall defensive rating
    df['home_def_style_edge'] = (
        df['home_def_rating'] - df['home_def_vs_away_style']
    )
    df['away_def_style_edge'] = (
        df['away_def_rating'] - df['away_def_vs_home_style']
    )

    return df

def add_shooting_consistency(df):
    """
    For each team compute:
    - FG% standard deviation (how consistent is their shooting?)
    - % of games below their average FG% by >5%
    - 3pt% variance (3pt teams are more boom/bust)
    - Free throw rate consistency
    """
    print("  Computing shooting consistency features...")

    query = text("""
        SELECT
            tbs.team_id,
            tbs.fg_pct,
            tbs.fg3_pct,
            tbs.ft_pct,
            tbs.ft_rate,
            tbs.points,
            g.game_date
        FROM team_box_scores tbs
        JOIN games g ON tbs.game_id = g.game_id
        WHERE g.is_final = TRUE
        ORDER BY tbs.team_id, g.game_date
    """)

    shoot_df = pd.read_sql(query, engine)
    shoot_df['game_date'] = pd.to_datetime(shoot_df['game_date'])
    df['game_date'] = pd.to_datetime(df['game_date'])

    home_fg_std       = []
    away_fg_std       = []
    home_3pt_std      = []
    away_3pt_std      = []
    home_bad_night_pct = []
    away_bad_night_pct = []
    home_ft_rate_std  = []
    away_ft_rate_std  = []

    for _, row in df.iterrows():
        game_date = row['game_date']
        home_id   = row['home_team_id']
        away_id   = row['away_team_id']

        for team_id, lists in [
            (home_id, (home_fg_std, home_3pt_std,
                       home_bad_night_pct, home_ft_rate_std)),
            (away_id, (away_fg_std, away_3pt_std,
                       away_bad_night_pct, away_ft_rate_std))
        ]:
            recent = shoot_df[
                (shoot_df['team_id'] == team_id) &
                (shoot_df['game_date'] < game_date)
            ].tail(50)  # last 20 games

            if len(recent) < 5:
                lists[0].append(0.05)   # default fg std
                lists[1].append(0.06)   # default 3pt std
                lists[2].append(0.25)   # default bad night %
                lists[3].append(0.02)   # default ft rate std
                continue

            fg_vals  = recent['fg_pct'].dropna()
            fg3_vals = recent['fg3_pct'].dropna()
            ft_vals  = recent['ft_rate'].dropna()
            pts_vals = recent['points'].dropna()

            fg_std  = float(fg_vals.std())  if len(fg_vals)  > 2 else 0.05
            fg3_std = float(fg3_vals.std()) if len(fg3_vals) > 2 else 0.06
            ft_std  = float(ft_vals.std())  if len(ft_vals)  > 2 else 0.02

            # Bad shooting night = FG% more than 1 std below average
            fg_mean = float(fg_vals.mean())
            bad_nights = (fg_vals < fg_mean - fg_std).sum()
            bad_night_pct = float(bad_nights / len(fg_vals))

            lists[0].append(fg_std)
            lists[1].append(fg3_std)
            lists[2].append(bad_night_pct)
            lists[3].append(ft_std)

    df['home_fg_consistency']    = home_fg_std
    df['away_fg_consistency']    = away_fg_std
    df['home_3pt_consistency']   = home_3pt_std
    df['away_3pt_consistency']   = away_3pt_std
    df['home_bad_night_pct']     = home_bad_night_pct
    df['away_bad_night_pct']     = away_bad_night_pct
    df['home_ft_rate_std']       = home_ft_rate_std
    df['away_ft_rate_std']       = away_ft_rate_std

    # Boom/bust differential
    # High = one team is much more volatile than the other
    df['shooting_volatility_diff'] = (
        df['home_fg_consistency'] - df['away_fg_consistency']
    )

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