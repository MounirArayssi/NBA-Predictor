import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import requests
import pandas as pd
from datetime import timedelta
from sqlalchemy.orm import Session
from data.storage.db import engine
from data.storage.models import Base, Game, Team, GameOdds
from config.settings import ODDS_API_KEY, ODDS_API_BASE

def fetch_todays_odds():
    """Fetch NBA odds for upcoming games."""
    if not ODDS_API_KEY:
        print("❌ No ODDS_API_KEY found in .env")
        return []

    url = f"{ODDS_API_BASE}/sports/basketball_nba/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    "spreads,totals",
        "oddsFormat": "american",
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"❌ API error: {response.status_code} — {response.text}")
        return []

    games = response.json()
    print(f"Found {len(games)} games with odds")
    return games


def parse_and_store_odds(odds_data):
    """Parse odds and match to games in database."""
    with Session(engine) as session:
        for game in odds_data:
            home_team_name = game.get('home_team')
            away_team_name = game.get('away_team')
            commence_time  = pd.to_datetime(game['commence_time'])
            game_date      = commence_time.date()

            print(f"\n  Processing: {away_team_name} @ {home_team_name} "
                  f"| Date: {game_date} | UTC: {commence_time}")

            # Find teams
            home_team = session.query(Team).filter(
                Team.full_name == home_team_name
            ).first()
            away_team = session.query(Team).filter(
                Team.full_name == away_team_name
            ).first()

            if not home_team:
                print(f"    ⚠️  Home team not found: {home_team_name}")
                continue
            if not away_team:
                print(f"    ⚠️  Away team not found: {away_team_name}")
                continue

            # Try matching game with date tolerance (±1 day for timezone)
            db_game = None
            for delta in [0, 1, -1]:
                check_date = game_date + timedelta(days=delta)
                db_game = session.query(Game).filter(
                    Game.home_team_id == home_team.team_id,
                    Game.away_team_id == away_team.team_id,
                    Game.game_date == check_date
                ).first()
                if db_game:
                    print(f"    Found game on {check_date} "
                          f"(delta={delta})")
                    break

            if not db_game:
                print(f"    ⚠️  Game not found in DB — "
                      f"tried {game_date}, {game_date + timedelta(1)}, "
                      f"{game_date - timedelta(1)}")
                continue

            # Parse odds from bookmakers
            home_spread    = None
            total_line     = None
            bookmaker_name = None

            for bookmaker in game.get('bookmakers', []):
                bookmaker_name = bookmaker['key']
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'spreads':
                        for outcome in market['outcomes']:
                            if outcome['name'] == home_team_name:
                                home_spread = float(outcome['point'])
                    elif market['key'] == 'totals':
                        for outcome in market['outcomes']:
                            if outcome['name'] == 'Over':
                                total_line = float(outcome['point'])
                if home_spread is not None and total_line is not None:
                    break

            if home_spread is None or total_line is None:
                print(f"    ⚠️  Incomplete odds data")
                continue

            away_spread = -home_spread

            # Implied scores
            # total = home + away
            # spread = home - away (negative means home favored)
            vegas_home = (total_line + (-home_spread)) / 2
            vegas_away = (total_line - (-home_spread)) / 2

            # Check if odds already stored for this game
            existing = session.query(GameOdds).filter_by(
                game_id=db_game.game_id
            ).first()

            if existing:
                # Update existing
                existing.home_spread        = home_spread
                existing.away_spread        = away_spread
                existing.total_line         = total_line
                existing.vegas_home_implied = vegas_home
                existing.vegas_away_implied = vegas_away
                existing.bookmaker          = bookmaker_name
                print(f"    🔄 Updated odds")
            else:
                odds = GameOdds(
                    game_id            = db_game.game_id,
                    home_spread        = home_spread,
                    away_spread        = away_spread,
                    total_line         = total_line,
                    vegas_home_implied = vegas_home,
                    vegas_away_implied = vegas_away,
                    bookmaker          = bookmaker_name
                )
                session.add(odds)
                print(f"    ✅ Spread: {home_spread:+.1f} | "
                      f"Total: {total_line} | "
                      f"Implied: {home_team.abbreviation} "
                      f"{vegas_home:.1f} — "
                      f"{away_team.abbreviation} {vegas_away:.1f}")

        session.commit()
        print("\n✅ Odds saved to database")


if __name__ == "__main__":
    odds_data = fetch_todays_odds()
    if odds_data:
        parse_and_store_odds(odds_data)