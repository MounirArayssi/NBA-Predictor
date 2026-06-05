import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import pandas as pd
from datetime import date, timedelta
from nba_api.stats.endpoints import leaguegamefinder, scoreboardv2
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
            league_id_nullable="00"
        )
        games_df = gamefinder.get_data_frames()[0]
        print(f"Found {len(games_df)} game records")

    except Exception as e:
        print(f"❌ Failed to fetch games: {e}")
        return

    game_ids = games_df['GAME_ID'].unique()
    print(f"Unique games: {len(game_ids)}")

    with Session(engine) as session:
        for game_id in game_ids:
            existing = session.query(Game).filter_by(
                nba_game_id=game_id
            ).first()

            if existing:
                print(f"  Skipping game {game_id} — already exists")
                continue

            game_rows = games_df[games_df['GAME_ID'] == game_id]

            if len(game_rows) < 2:
                print(f"  Skipping game {game_id} — incomplete data")
                continue

            home_row = game_rows[
                game_rows['MATCHUP'].str.contains('vs\.')
            ].iloc[0] if any(
                game_rows['MATCHUP'].str.contains('vs\.')
            ) else game_rows.iloc[0]

            away_row = game_rows[
                game_rows['MATCHUP'].str.contains('@')
            ].iloc[0] if any(
                game_rows['MATCHUP'].str.contains('@')
            ) else game_rows.iloc[1]

            home_team = session.query(Team).filter_by(
                nba_team_id=int(home_row['TEAM_ID'])
            ).first()
            away_team = session.query(Team).filter_by(
                nba_team_id=int(away_row['TEAM_ID'])
            ).first()

            if not home_team or not away_team:
                print(f"  Skipping game {game_id} — team not found")
                continue

            game_date = pd.to_datetime(home_row['GAME_DATE']).date()
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
            print(f"  Added {away_team.abbreviation} @ "
                  f"{home_team.abbreviation} on {game_date}")

            time.sleep(0.1)

        session.commit()
        print(f"✅ Games saved to database")


def fetch_upcoming_games(season=CURRENT_SEASON, days_ahead=8):
    """Fetch scheduled upcoming games using ScoreboardV2."""
    print(f"\nFetching upcoming games for next {days_ahead} days...")

    games_added = 0

    for i in range(0, days_ahead):
        target_date = date.today() + timedelta(days=i)
        date_str = target_date.strftime("%m/%d/%Y")
        print(f"  Checking {date_str}...")

        try:
            board = scoreboardv2.ScoreboardV2(
                game_date=date_str,
                league_id="00"
            )
            # GameHeader is dataframe 0
            games_df = board.get_data_frames()[0]

            if games_df.empty:
                print(f"    No games found")
                continue

            print(f"    Found {len(games_df)} games")

            with Session(engine) as session:
                for _, row in games_df.iterrows():
                    game_id = str(row['GAME_ID'])

                    # Skip if already exists
                    existing = session.query(Game).filter_by(
                        nba_game_id=game_id
                    ).first()

                    if existing:
                        print(f"    Skipping {game_id} — already exists")
                        continue

                    home_id = row['HOME_TEAM_ID']
                    away_id = row['VISITOR_TEAM_ID']
                    if pd.isna(home_id) or pd.isna(away_id) or home_id is None or away_id is None:
                        print(f"    ⚠️  Skipping {game_id} — team IDs not yet available (TBD)")
                        continue

                    home_team = session.query(Team).filter_by(
                        nba_team_id=int(home_id)
                    ).first()
                    away_team = session.query(Team).filter_by(
                        nba_team_id=int(away_id)
                    ).first()

                    if not home_team or not away_team:
                        print(f"    ⚠️  Team not found for game {game_id}")
                        continue

                    # Determine season type
                    game_id_prefix = game_id[:4]
                    if game_id_prefix == '0042':
                        season_type = 'Playoffs'
                    elif game_id_prefix == '0022':
                        season_type = 'Regular Season'
                    else:
                        season_type = 'Playoffs'

                    game = Game(
                        nba_game_id  = game_id,
                        season       = season,
                        season_type  = season_type,
                        game_date    = target_date,
                        home_team_id = home_team.team_id,
                        away_team_id = away_team.team_id,
                        is_final     = False,
                        status       = 'scheduled'
                    )
                    session.add(game)
                    games_added += 1
                    print(f"    ✅ Added {away_team.abbreviation} @ "
                          f"{home_team.abbreviation} on {target_date}")

                session.commit()

            time.sleep(NBA_API_DELAY)

        except Exception as e:
            print(f"    ❌ Error for {date_str}: {e}")
            continue

    print(f"\n✅ Added {games_added} upcoming games")


if __name__ == "__main__":
    # Fetch historical games
    seasons = ["2025-26", "2024-25", "2023-24"]
    for season in seasons:
        fetch_and_store_games(season=season, season_type="Regular Season")
        fetch_and_store_games(season=season, season_type="Playoffs")

    # Fetch upcoming scheduled games
    fetch_upcoming_games()