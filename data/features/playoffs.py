"""
E — Playoffs: series state, playoff elevation, and in-series scoring form.
Regular-season rows get neutral values (0s, or the rolling average for form).
"""
import pandas as pd

# Weight (in games) the rolling average gets when blended with in-series form.
# A 1-2 game series average is mostly noise, so it's shrunk toward the rolling avg.
SERIES_SHRINK_K = 3


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

        # Wins are tracked per TEAM: home court alternates within a series, so
        # "wins by whoever was home" is not tonight's home team's series wins.
        if season_key not in series_history:
            series_history[season_key] = {'games': [], 'wins': {}}

        history  = series_history[season_key]
        game_num = len(history['games']) + 1
        h_wins   = history['wins'].get(row['home_team_id'], 0)
        a_wins   = history['wins'].get(row['away_team_id'], 0)

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
        winner = row['home_team_id'] if home_won else row['away_team_id']
        history['wins'][winner] = history['wins'].get(winner, 0) + 1

    df['series_game_num']  = series_game_num
    df['home_series_wins'] = home_series_wins
    df['away_series_wins'] = away_series_wins
    df['is_elimination']   = is_elimination
    df['series_momentum']  = series_momentum
    df['series_pressure']  = df['home_series_wins'] + df['away_series_wins']
    df['series_wins_diff'] = df['home_series_wins'] - df['away_series_wins']
    return df


def add_playoff_elevation_features(df):
    """
    Add team playoff elevation scores as model features.
    Applies series context adjustments for playoff games.
    Needs series_game_num / *_series_wins from add_playoff_features.
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


def shrink_series_avg(series_avg, n_games, rolling_avg):
    """Blend an n-game in-series average with the team's rolling average."""
    return (n_games * series_avg + SERIES_SHRINK_K * rolling_avg) / (n_games + SERIES_SHRINK_K)


def add_series_form_features(df):
    """
    For each playoff game, compute the avg scores from games already
    played in this series, shrunk toward the rolling avg by series length.
    Game 1 falls back to rolling avg.
    Regular-season rows get their rolling avg as a neutral fill.
    """
    print("  Computing in-series form features...")

    df = df.sort_values('game_date').copy()
    series_history = {}
    in_series_home = []
    in_series_away = []

    for _, row in df.iterrows():
        home_id = int(row['home_team_id'])
        away_id = int(row['away_team_id'])

        if row['season_type'] != 'Playoffs':
            in_series_home.append(float(row['home_avg_points']))
            in_series_away.append(float(row['away_avg_points']))
            continue

        season = row['season']
        key = (season, tuple(sorted([home_id, away_id])))
        history = series_history.get(key, [])

        if history:
            home_scores = []
            away_scores = []
            for g in history:
                if g['home_id'] == home_id:
                    home_scores.append(g['home_score'])
                    away_scores.append(g['away_score'])
                else:
                    home_scores.append(g['away_score'])
                    away_scores.append(g['home_score'])
            n = len(home_scores)
            in_series_home.append(shrink_series_avg(
                sum(home_scores) / n, n, float(row['home_avg_points'])))
            in_series_away.append(shrink_series_avg(
                sum(away_scores) / n, n, float(row['away_avg_points'])))
        else:
            in_series_home.append(float(row['home_avg_points']))
            in_series_away.append(float(row['away_avg_points']))

        if key not in series_history:
            series_history[key] = []
        series_history[key].append({
            'home_id': home_id,
            'away_id': away_id,
            'home_score': float(row['home_score']),
            'away_score': float(row['away_score']),
        })

    df['in_series_home_avg_pts'] = in_series_home
    df['in_series_away_avg_pts'] = in_series_away
    df['in_series_total_avg'] = (
        df['in_series_home_avg_pts'] + df['in_series_away_avg_pts']
    )
    df['in_series_margin'] = (
        df['in_series_home_avg_pts'] - df['in_series_away_avg_pts']
    )

    return df
