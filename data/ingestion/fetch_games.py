import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from sqlalchemy.orm import Session
from data.storage.db import engine
from data.storage.models import Game, Team
from config.settings import NBA_API_DELAY, CURRENT_SEASON

def fetch_and_store_games(season=CURRENT_SEASON, season_type="Playoffs"):
    print(f"Fetching {season_type} games for {season}...")

    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable=season_type,
            league_id_nullable="00"  # NBA
        )
        games_df = gamefinder.get_data_frames()[0]
        print(f"Found {len(games_df)} game records")

    except Exception as e:
        print(f"❌ Failed to fetch games: {e}")
        return

    # LeagueGameFinder returns one row per team per game
    # We need to deduplicate into one row per game
    # Group by GAME_ID and take home/away pairs
    game_ids = games_df['GAME_ID'].unique()
    print(f"Unique games: {len(game_ids)}")

    with Session(engine) as session:
        for game_id in game_ids:
            # Check if already exists
            existing = session.query(Game).filter_by(
                nba_game_id=game_id
            ).first()

            if existing:
                print(f"  Skipping game {game_id} — already exists")
                continue

            # Get both team rows for this game
            game_rows = games_df[games_df['GAME_ID'] == game_id]

            if len(game_rows) < 2:
                print(f"  Skipping game {game_id} — incomplete data")
                continue

            # Determine home/away from MATCHUP column
            # MATCHUP format: "BOS vs. MIA" (home) or "BOS @ MIA" (away)
            home_row = game_rows[game_rows['MATCHUP'].str.contains('vs\.')].iloc[0] \
                       if any(game_rows['MATCHUP'].str.contains('vs\.')) else game_rows.iloc[0]
            away_row = game_rows[game_rows['MATCHUP'].str.contains('@')].iloc[0] \
                       if any(game_rows['MATCHUP'].str.contains('@')) else game_rows.iloc[1]

            # Look up team IDs in our database
            home_team = session.query(Team).filter_by(
                nba_team_id=int(home_row['TEAM_ID'])  # convert to native int
            ).first()
            
            away_team = session.query(Team).filter_by(
                nba_team_id=int(away_row['TEAM_ID'])  # convert to native int
            ).first()

            if not home_team or not away_team:
                print(f"  Skipping game {game_id} — team not found")
                continue

            # Parse game date
            game_date = pd.to_datetime(home_row['GAME_DATE']).date()

            # Determine if game is final
            is_final = pd.notna(home_row.get('PTS', None))

            game = Game(
                nba_game_id  = game_id,
                season       = season,
                season_type  = season_type,
                game_date    = game_date,
                home_team_id = home_team.team_id,
                away_team_id = away_team.team_id,
                home_score   = int(home_row['PTS']) if is_final else None,
                away_score   = int(away_row['PTS']) if is_final else None,
                is_final     = is_final,
                status       = 'final' if is_final else 'scheduled'
            )
            session.add(game)
            print(f"  Added {away_team.abbreviation} @ {home_team.abbreviation} on {game_date}")

            time.sleep(0.1)  # small delay between inserts

        session.commit()
        print(f"✅ Games saved to database")


if __name__ == "__main__":
    seasons = ["2025-26", "2024-25", "2023-24"]
    
    for season in seasons:
        fetch_and_store_games(season=season, season_type="Regular Season")
        fetch_and_store_games(season=season, season_type="Playoffs")