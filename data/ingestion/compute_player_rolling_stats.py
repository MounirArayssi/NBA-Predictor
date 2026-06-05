import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from datetime import date
from sqlalchemy import text, UniqueConstraint
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, Numeric, Date, DateTime, String, ForeignKey
from sqlalchemy.sql import func
from data.storage.db import engine
from data.storage.models import Base, PlayerRollingStats

def compute_player_rolling_stats(window=10):
    """
    Compute rolling stats for every player.
    Similar to team rolling stats but at individual level.
    """
    print(f"Computing player rolling stats (window={window})...")
    
    query = text("""
    SELECT
        pbs.player_id,
        pbs.team_id,
        pbs.points,
        pbs.minutes_played,
        pbs.usage_rate,
        pbs.fg_pct,
        pbs.fg3_pct,
        pbs.true_shooting,
        pbs.rebounds,
        pbs.assists,
        pbs.steals,
        pbs.blocks,
        pbs.turnovers,
        pbs.plus_minus,
        pbs.efg_pct,
        pbs.oreb_pct,
        pbs.dreb_pct,
        pbs.ast_pct,
        pbs.off_rating,
        pbs.def_rating,
        pbs.net_rating,
        pbs.blk_pct,
        pbs.stl_pct,
        g.game_date,
        g.season
    FROM player_box_scores pbs
    JOIN games g ON pbs.game_id = g.game_id
    WHERE g.is_final = TRUE
    AND pbs.minutes_played >= 5
    AND pbs.points IS NOT NULL
    ORDER BY pbs.player_id, g.game_date
""")

    df = pd.read_sql(query, engine)
    print(f"  Loaded {len(df)} player game records")

    # Group by player+team combination
    player_teams = df.groupby(
        ['player_id', 'team_id']
    ).size().reset_index()[['player_id', 'team_id']]

    records_added = 0

    with Session(engine) as session:
        for _, pt in player_teams.iterrows():
            player_id = int(pt['player_id'])
            team_id   = int(pt['team_id'])

            player_df = df[
                (df['player_id'] == player_id) &
                (df['team_id'] == team_id)
            ].copy()
            player_df = player_df.sort_values(
                'game_date'
            ).reset_index(drop=True)

            # Need at least window games
            if len(player_df) < window:
                continue

            # Compute rolling stats for each game
            for i in range(window, len(player_df)):
                current_game = player_df.iloc[i]
                window_df    = player_df.iloc[i-window:i]
                as_of_date   = current_game['game_date']

                existing = session.query(PlayerRollingStats).filter_by(
                    player_id  = player_id,
                    team_id    = team_id,
                    as_of_date = as_of_date,
                    window     = window
                ).first()

                if existing:
                    continue

                def safe_mean(col):
                    vals = window_df[col].dropna()
                    return float(vals.mean()) if len(vals) > 0 else None

                stats = PlayerRollingStats(
                    player_id        = player_id,
                    team_id          = team_id,
                    as_of_date       = as_of_date,
                    window           = window,
                    season           = current_game['season'],
                    avg_points       = safe_mean('points'),
                    avg_minutes      = safe_mean('minutes_played'),
                    avg_usage_rate   = safe_mean('usage_rate'),
                    avg_fg_pct       = safe_mean('fg_pct'),
                    avg_fg3_pct      = safe_mean('fg3_pct'),
                    avg_true_shooting = safe_mean('true_shooting'),
                    avg_rebounds     = safe_mean('rebounds'),
                    avg_assists      = safe_mean('assists'),
                    avg_steals       = safe_mean('steals'),
                    avg_blocks       = safe_mean('blocks'),
                    avg_turnovers    = safe_mean('turnovers'),
                    avg_plus_minus   = safe_mean('plus_minus'),
                    avg_efg_pct    = safe_mean('efg_pct'),
                    avg_oreb_pct   = safe_mean('oreb_pct'),
                    avg_dreb_pct   = safe_mean('dreb_pct'),
                    avg_ast_pct    = safe_mean('ast_pct'),
                    avg_off_rating = safe_mean('off_rating'),
                    avg_def_rating = safe_mean('def_rating'),
                    avg_net_rating = safe_mean('net_rating'),
                    avg_blk_pct    = safe_mean('blk_pct'),
                    avg_stl_pct    = safe_mean('stl_pct'),
                    games_counted    = window
                )
                session.add(stats)
                records_added += 1

            session.commit()

        # Also compute today's stats for each player
        today = date.today()
        for _, pt in player_teams.iterrows():
            player_id = int(pt['player_id'])
            team_id   = int(pt['team_id'])

            player_df = df[
                (df['player_id'] == player_id) &
                (df['team_id'] == team_id)
            ].copy().sort_values('game_date')

            if len(player_df) < window:
                continue

            window_df = player_df.tail(window)

            existing = session.query(PlayerRollingStats).filter_by(
                player_id  = player_id,
                team_id    = team_id,
                as_of_date = today,
                window     = window
            ).first()

            def safe_mean(col):
                vals = window_df[col].dropna()
                return float(vals.mean()) if len(vals) > 0 else None

            stats_data = dict(
                player_id        = player_id,
                team_id          = team_id,
                as_of_date       = today,
                window           = window,
                season           = player_df.iloc[-1]['season'],
                avg_points       = safe_mean('points'),
                avg_minutes      = safe_mean('minutes_played'),
                avg_usage_rate   = safe_mean('usage_rate'),
                avg_fg_pct       = safe_mean('fg_pct'),
                avg_fg3_pct      = safe_mean('fg3_pct'),
                avg_true_shooting = safe_mean('true_shooting'),
                avg_rebounds     = safe_mean('rebounds'),
                avg_assists      = safe_mean('assists'),
                avg_steals       = safe_mean('steals'),
                avg_blocks       = safe_mean('blocks'),
                avg_turnovers    = safe_mean('turnovers'),
                avg_plus_minus   = safe_mean('plus_minus'),    
                avg_efg_pct    = safe_mean('efg_pct'),
                avg_oreb_pct   = safe_mean('oreb_pct'),
                avg_dreb_pct   = safe_mean('dreb_pct'),
                avg_ast_pct    = safe_mean('ast_pct'),
                avg_off_rating = safe_mean('off_rating'),
                avg_def_rating = safe_mean('def_rating'),
                avg_net_rating = safe_mean('net_rating'),
                avg_blk_pct    = safe_mean('blk_pct'),
                avg_stl_pct    = safe_mean('stl_pct'),
                games_counted    = window
            )

            if existing:
                for k, v in stats_data.items():
                    setattr(existing, k, v)
            else:
                session.add(PlayerRollingStats(**stats_data))
                records_added += 1

        session.commit()

    print(f"✅ Player rolling stats complete — {records_added} records added")


def compute_playoff_rolling_stats(window=5, season='2025-26'):
    """
    Compute rolling stats using ONLY playoff games.
    This prevents regular season data from diluting playoff breakouts.
    """
    print(f"Computing PLAYOFF-ONLY rolling stats (window={window})...")
    
    query = text("""
    SELECT
        pbs.player_id,
        pbs.team_id,
        pbs.points,
        pbs.minutes_played,
        pbs.usage_rate,
        pbs.fg_pct,
        pbs.fg3_pct,
        pbs.true_shooting,
        pbs.rebounds,
        pbs.assists,
        pbs.steals,
        pbs.blocks,
        pbs.turnovers,
        pbs.plus_minus,
        pbs.efg_pct,
        pbs.oreb_pct,
        pbs.dreb_pct,
        pbs.ast_pct,
        pbs.off_rating,
        pbs.def_rating,
        pbs.net_rating,
        pbs.blk_pct,
        pbs.stl_pct,
        g.game_date,
        g.season
    FROM player_box_scores pbs
    JOIN games g ON pbs.game_id = g.game_id
    WHERE g.is_final = TRUE
    AND g.season_type = 'Playoffs'  -- PLAYOFF ONLY
    AND g.season = :season
    AND pbs.minutes_played >= 5
    AND pbs.points IS NOT NULL
    ORDER BY pbs.player_id, g.game_date
    """)

    df = pd.read_sql(query, engine, params={'season': season})
    print(f"  Loaded {len(df)} playoff game records")

    # Rest is same as regular rolling stats...
    player_teams = df.groupby(['player_id', 'team_id']).size().reset_index()[['player_id', 'team_id']]
    records_added = 0

    with Session(engine) as session:
        for _, pt in player_teams.iterrows():
            player_id = int(pt['player_id'])
            team_id   = int(pt['team_id'])

            player_df = df[
                (df['player_id'] == player_id) &
                (df['team_id'] == team_id)
            ].copy().sort_values('game_date').reset_index(drop=True)

            if len(player_df) < window:
                continue

            # Compute rolling stats for each game
            for i in range(window, len(player_df)):
                current_game = player_df.iloc[i]
                window_df    = player_df.iloc[i-window:i]
                as_of_date   = current_game['game_date']

                existing = session.query(PlayerRollingStats).filter_by(
                    player_id  = player_id,
                    team_id    = team_id,
                    as_of_date = as_of_date,
                    window     = window,
                    season_type = 'Playoffs'  # Mark as playoff-specific
                ).first()

                if existing:
                    continue

                def safe_mean(col):
                    vals = window_df[col].dropna()
                    return float(vals.mean()) if len(vals) > 0 else None

                stats = PlayerRollingStats(
                    player_id        = player_id,
                    team_id          = team_id,
                    as_of_date       = as_of_date,
                    window           = window,
                    season           = current_game['season'],
                    season_type      = 'Playoffs',  # NEW FIELD
                    avg_points       = safe_mean('points'),
                    avg_minutes      = safe_mean('minutes_played'),
                    avg_usage_rate   = safe_mean('usage_rate'),
                    avg_fg_pct       = safe_mean('fg_pct'),
                    avg_fg3_pct      = safe_mean('fg3_pct'),
                    avg_true_shooting = safe_mean('true_shooting'),
                    avg_rebounds     = safe_mean('rebounds'),
                    avg_assists      = safe_mean('assists'),
                    avg_steals       = safe_mean('steals'),
                    avg_blocks       = safe_mean('blocks'),
                    avg_turnovers    = safe_mean('turnovers'),
                    avg_plus_minus   = safe_mean('plus_minus'),
                    avg_efg_pct      = safe_mean('efg_pct'),
                    avg_oreb_pct     = safe_mean('oreb_pct'),
                    avg_dreb_pct     = safe_mean('dreb_pct'),
                    avg_ast_pct      = safe_mean('ast_pct'),
                    avg_off_rating   = safe_mean('off_rating'),
                    avg_def_rating   = safe_mean('def_rating'),
                    avg_net_rating   = safe_mean('net_rating'),
                    avg_blk_pct      = safe_mean('blk_pct'),
                    avg_stl_pct      = safe_mean('stl_pct'),
                    games_counted    = window
                )
                session.add(stats)
                records_added += 1

        # Also compute "today" playoff stats
        today = date.today()
        for _, pt in player_teams.iterrows():
            player_id = int(pt['player_id'])
            team_id   = int(pt['team_id'])

            player_df = df[
                (df['player_id'] == player_id) &
                (df['team_id'] == team_id)
            ].copy().sort_values('game_date')

            if len(player_df) < window:
                continue

            window_df = player_df.tail(window)

            existing = session.query(PlayerRollingStats).filter_by(
                player_id  = player_id,
                team_id    = team_id,
                as_of_date = today,
                window     = window,
                season_type = 'Playoffs'
            ).first()

            def safe_mean(col):
                vals = window_df[col].dropna()
                return float(vals.mean()) if len(vals) > 0 else None

            stats_data = dict(
                player_id        = player_id,
                team_id          = team_id,
                as_of_date       = today,
                window           = window,
                season           = player_df.iloc[-1]['season'],
                season_type      = 'Playoffs',
                avg_points       = safe_mean('points'),
                avg_minutes      = safe_mean('minutes_played'),
                avg_usage_rate   = safe_mean('usage_rate'),
                avg_fg_pct       = safe_mean('fg_pct'),
                avg_fg3_pct      = safe_mean('fg3_pct'),
                avg_true_shooting = safe_mean('true_shooting'),
                avg_rebounds     = safe_mean('rebounds'),
                avg_assists      = safe_mean('assists'),
                avg_steals       = safe_mean('steals'),
                avg_blocks       = safe_mean('blocks'),
                avg_turnovers    = safe_mean('turnovers'),
                avg_plus_minus   = safe_mean('plus_minus'),
                avg_efg_pct      = safe_mean('efg_pct'),
                avg_oreb_pct     = safe_mean('oreb_pct'),
                avg_dreb_pct     = safe_mean('dreb_pct'),
                avg_ast_pct      = safe_mean('ast_pct'),
                avg_off_rating   = safe_mean('off_rating'),
                avg_def_rating   = safe_mean('def_rating'),
                avg_net_rating   = safe_mean('net_rating'),
                avg_blk_pct      = safe_mean('blk_pct'),
                avg_stl_pct      = safe_mean('stl_pct'),
                games_counted    = window
            )

            if existing:
                for k, v in stats_data.items():
                    setattr(existing, k, v)
            else:
                session.add(PlayerRollingStats(**stats_data))
                records_added += 1

        session.commit()

    print(f"✅ Playoff rolling stats complete — {records_added} records added")


# Update main block
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--playoffs-only', action='store_true',
                        help='Compute playoff-only rolling stats')
    args = parser.parse_args()
    
    if args.playoffs_only:
        for window in [3, 5]:  # Smaller windows for playoffs
            compute_playoff_rolling_stats(window=window)
    else:
        for window in [5, 10]:
            compute_player_rolling_stats(window=window)