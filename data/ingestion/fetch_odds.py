import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import requests
import pandas as pd
from datetime import date, timedelta
from sqlalchemy.orm import Session
from data.storage.db import engine
from data.storage.models import Base, Game, Team, GameOdds
from config.settings import ODDS_API_KEY, ODDS_API_BASE


def fetch_todays_odds():
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


def fetch_game_ids():
    """Get today's NBA event IDs."""
    url = f"{ODDS_API_BASE}/sports/basketball_nba/events"
    params = {"apiKey": ODDS_API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return []


def fetch_player_props(event_id):
    """Fetch player points/rebounds/assists props for one game."""
    url = (
        f"{ODDS_API_BASE}/sports/basketball_nba"
        f"/events/{event_id}/odds"
    )
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    "player_points,player_rebounds,player_assists",
        "oddsFormat": "american",
        "bookmakers": "draftkings",
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    print(f"  ⚠️  Props error {response.status_code}: {response.text[:100]}")
    return None


def parse_player_props(props_data):
    """
    Parse props into {player_name: {points: line, rebounds: line, assists: line}}
    Only Over lines used as the expected value anchor.
    """
    if not props_data:
        return {}

    player_lines = {}
    for bookmaker in props_data.get('bookmakers', []):
        for market in bookmaker.get('markets', []):
            stat = market['key'].replace('player_', '')
            for outcome in market.get('outcomes', []):
                if outcome.get('name') == 'Over':
                    player = outcome.get('description', '')
                    line   = float(outcome.get('point', 0))
                    if player not in player_lines:
                        player_lines[player] = {}
                    player_lines[player][stat] = line

    return player_lines


def fetch_todays_player_props():
    """
    Fetch player props for today's games.
    Returns {'{away}@{home}': {player_name: {stat: line}}}
    """
    events = fetch_game_ids()
    if not events:
        print("  No events found")
        return {}

    all_props = {}
    today     = date.today()

    for event in events:
        commence = pd.to_datetime(event['commence_time']).date()
        if commence != today and commence != today + timedelta(1):
            continue

        home = event.get('home_team', '')
        away = event.get('away_team', '')
        print(f"  Fetching props: {away} @ {home}")

        props    = fetch_player_props(event['id'])
        parsed   = parse_player_props(props)
        game_key = f"{away}@{home}"
        all_props[game_key] = parsed
        print(f"    Found props for {len(parsed)} players")

    return all_props


def parse_and_store_odds(odds_data):
    with Session(engine) as session:
        for game in odds_data:
            home_team_name = game.get('home_team')
            away_team_name = game.get('away_team')
            commence_time  = pd.to_datetime(game['commence_time'])
            game_date      = commence_time.date()

            print(f"\n  Processing: {away_team_name} @ {home_team_name} "
                  f"| Date: {game_date}")

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

            db_game = None
            for delta in [0, 1, -1]:
                check_date = game_date + timedelta(days=delta)
                db_game = session.query(Game).filter(
                    Game.home_team_id == home_team.team_id,
                    Game.away_team_id == away_team.team_id,
                    Game.game_date    == check_date
                ).first()
                if db_game:
                    break

            if not db_game:
                print(f"    ⚠️  Game not found in DB")
                continue

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
            vegas_home  = (total_line + (-home_spread)) / 2
            vegas_away  = (total_line - (-home_spread)) / 2

            existing = session.query(GameOdds).filter_by(
                game_id=db_game.game_id
            ).first()

            if existing:
                existing.home_spread        = home_spread
                existing.away_spread        = away_spread
                existing.total_line         = total_line
                existing.vegas_home_implied = vegas_home
                existing.vegas_away_implied = vegas_away
                existing.bookmaker          = bookmaker_name
                print(f"    🔄 Updated odds")
            else:
                session.add(GameOdds(
                    game_id            = db_game.game_id,
                    home_spread        = home_spread,
                    away_spread        = away_spread,
                    total_line         = total_line,
                    vegas_home_implied = vegas_home,
                    vegas_away_implied = vegas_away,
                    bookmaker          = bookmaker_name
                ))
                print(f"    ✅ Spread: {home_spread:+.1f} | "
                      f"Total: {total_line} | "
                      f"Implied: {home_team.abbreviation} "
                      f"{vegas_home:.1f} — "
                      f"{away_team.abbreviation} {vegas_away:.1f}")

        session.commit()
        print("\n✅ Odds saved to database")


if __name__ == "__main__":
    # Team odds
    odds_data = fetch_todays_odds()
    if odds_data:
        parse_and_store_odds(odds_data)

    # Player props
    print("\nFetching player props...")
    props = fetch_todays_player_props()
    for game, players in props.items():
        print(f"\n{game}:")
        for player, lines in list(players.items())[:5]:
            print(f"  {player}: {lines}")