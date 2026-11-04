import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import pandas as pd
from nba_api.stats.endpoints import commonteamroster
from sqlalchemy.orm import Session
from data.storage.db import engine
from data.storage.models import Player, Team
from config.settings import NBA_API_DELAY, CURRENT_SEASON


def _parse_height(height_str):
    """Parse height like '6-8' into total inches."""
    if not height_str or pd.isna(height_str):
        return None
    parts = str(height_str).split('-')
    if len(parts) == 2:
        try:
            return int(parts[0]) * 12 + int(parts[1])
        except ValueError:
            return None
    return None


def _parse_weight(weight_str):
    if not weight_str or pd.isna(weight_str):
        return None
    try:
        return int(weight_str)
    except (TypeError, ValueError):
        return None


def _split_name(full_name):
    parts = full_name.split(" ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return full_name, ""


def fetch_and_store_players(season=CURRENT_SEASON):
    """
    Pull every team's current roster straight from the NBA API for `season`
    and upsert into the players table.

    Uses commonteamroster (one live call per team) rather than the static
    nba_api.stats.static players list — that static list ships bundled with
    the installed nba_api package and does not reflect a season's rookies
    or offseason signings until the library itself gets updated. Hitting
    each team's roster directly is the only reliable way to pick up new
    players (draft picks, two-way signings, etc.) as soon as the season's
    rosters are set.
    """
    print(f"Fetching NBA team rosters for {season}...")

    with Session(engine) as session:
        teams = session.query(Team).order_by(Team.abbreviation).all()
        print(f"Found {len(teams)} teams")

        added = 0
        reactivated = 0
        seen_nba_ids = set()

        for i, team in enumerate(teams):
            try:
                roster = commonteamroster.CommonTeamRoster(
                    team_id=team.nba_team_id,
                    season=season,
                ).get_data_frames()[0]
            except Exception as e:
                print(f"  ❌ Failed to fetch roster for {team.abbreviation}: {e}")
                continue

            for _, row in roster.iterrows():
                nba_player_id = int(row['PLAYER_ID'])
                seen_nba_ids.add(nba_player_id)
                full_name = str(row['PLAYER'])

                existing = session.query(Player).filter_by(
                    nba_player_id=nba_player_id
                ).first()

                if existing:
                    if not existing.is_active:
                        existing.is_active = True
                        reactivated += 1
                    continue

                first_name, last_name = _split_name(full_name)

                player = Player(
                    nba_player_id = nba_player_id,
                    full_name     = full_name,
                    first_name    = first_name,
                    last_name     = last_name,
                    position      = row.get('POSITION') or None,
                    height_inches = _parse_height(row.get('HEIGHT')),
                    weight_lbs    = _parse_weight(row.get('WEIGHT')),
                    is_active     = True,
                )
                session.add(player)
                added += 1
                print(f"  + Added {full_name} ({team.abbreviation})")

            print(f"  [{i+1}/{len(teams)}] {team.abbreviation} roster processed")
            time.sleep(NBA_API_DELAY)

        session.commit()
        print(f"✅ Done — {added} new players added, {reactivated} reactivated")
        return added, reactivated


if __name__ == "__main__":
    fetch_and_store_players()
