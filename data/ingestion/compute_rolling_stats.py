import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session
from data.storage.db import engine
from data.storage.models import TeamRollingStats
from config.settings import CURRENT_SEASON


def compute_and_store_rolling_stats(window=10):
    print(f"Computing rolling stats (window={window})...")

    with Session(engine) as session:
        # Pull all team box scores joined with game info
        query = text("""
            SELECT 
                tbs.team_id,
                tbs.game_id,
                tbs.is_home,
                tbs.points,
                tbs.offensive_rating,
                tbs.defensive_rating,
                tbs.net_rating,
                tbs.pace,
                tbs.fg_pct,
                tbs.fg3m,
                tbs.fg3a,
                tbs.fg3_pct,
                tbs.ft_pct,
                tbs.efg_pct,
                tbs.tov_pct,
                tbs.oreb_pct,
                tbs.ft_rate,
                tbs.rebounds,
                tbs.assists,
                tbs.steals,
                tbs.blocks,
                tbs.turnovers,
                g.game_date,
                g.season
            FROM team_box_scores tbs
            JOIN games g ON tbs.game_id = g.game_id
            WHERE g.is_final = TRUE
            ORDER BY tbs.team_id, g.game_date
        """)

        df = pd.read_sql(query, engine)
        print(f"Loaded {len(df)} team game records")

        teams = df['team_id'].unique()
        print(f"Computing for {len(teams)} teams...")

        records_added = 0

        for team_id in teams:
            team_df = df[df['team_id'] == team_id].copy()
            team_df = team_df.sort_values('game_date').reset_index(drop=True)

            # For each game, compute rolling stats from previous N games
            for i in range(window, len(team_df)):
                current_game = team_df.iloc[i]
                window_df = team_df.iloc[i-window:i]  # previous N games

                as_of_date = current_game['game_date']

                # Check if already exists
                existing = session.query(TeamRollingStats).filter_by(
                    team_id=int(team_id),
                    as_of_date=as_of_date,
                    window=window
                ).first()

                if existing:
                    continue

                # Home/away splits
                home_games = window_df[window_df['is_home'] == True]
                away_games = window_df[window_df['is_home'] == False]

                # Three point rate
                total_fga = window_df['fg3a'].sum() + 0.001  # avoid div by zero
                three_point_rate = float(window_df['fg3a'].sum() / total_fga) \
                                   if total_fga > 0 else None

                # Win/loss — compare points to opponent
                # We'll approximate wins from net_rating being positive
                wins = int((window_df['net_rating'] > 0).sum())
                losses = window - wins

                stats = TeamRollingStats(
                    team_id     = int(team_id),
                    as_of_date  = as_of_date,
                    window      = window,
                    season      = current_game['season'],

                    # Offensive
                    avg_points           = float(window_df['points'].mean()),
                    avg_offensive_rating = float(window_df['offensive_rating'].mean()) \
                                          if window_df['offensive_rating'].notna().any() else None,
                    avg_pace             = float(window_df['pace'].mean()) \
                                          if window_df['pace'].notna().any() else None,
                    avg_fg_pct           = float(window_df['fg_pct'].mean()) \
                                          if window_df['fg_pct'].notna().any() else None,
                    avg_fg3_pct          = float(window_df['fg3_pct'].mean()) \
                                          if window_df['fg3_pct'].notna().any() else None,
                    avg_ft_rate          = float(window_df['ft_rate'].mean()) \
                                          if window_df['ft_rate'].notna().any() else None,
                    three_point_rate     = three_point_rate,

                    # Defensive
                    avg_points_allowed   = None,  # computed below
                    avg_defensive_rating = float(window_df['defensive_rating'].mean()) \
                                          if window_df['defensive_rating'].notna().any() else None,

                    # Home/away splits
                    home_avg_points = float(home_games['points'].mean()) \
                                      if len(home_games) > 0 else None,
                    away_avg_points = float(away_games['points'].mean()) \
                                      if len(away_games) > 0 else None,
                    home_avg_points_allowed = None,  # computed below
                    away_avg_points_allowed = None,  # computed below

                    # Win/loss
                    wins    = wins,
                    losses  = losses,
                    win_pct = float(wins / window),

                    games_counted = window
                )
                session.add(stats)
                records_added += 1

            # Commit per team
            session.commit()
            print(f"  Team {team_id} done — {records_added} records so far")

        print(f"✅ Rolling stats complete — {records_added} total records added")


if __name__ == "__main__":
    # Compute for multiple windows
    for window in [5, 10, 20]:
        compute_and_store_rolling_stats(window=window)