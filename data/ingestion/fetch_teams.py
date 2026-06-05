import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import pandas as pd
from nba_api.stats.static import teams
from sqlalchemy.orm import Session
from data.storage.db import engine
from data.storage.models import Team
from config.settings import NBA_API_DELAY

# Teams that play at altitude — affects scoring
ALTITUDE_MAP = {
    'DEN': 5280,  # Denver
    'UTA': 4327,  # Utah
}

def fetch_and_store_teams():
    print("Fetching NBA teams...")
    
    # nba_api returns a simple list of dicts — no API call needed
    all_teams = teams.get_teams()
    print(f"Found {len(all_teams)} teams")

    with Session(engine) as session:
        for t in all_teams:
            # Check if team already exists
            existing = session.query(Team).filter_by(
                nba_team_id=t['id']
            ).first()

            if existing:
                print(f"  Skipping {t['full_name']} — already exists")
                continue

            team = Team(
                nba_team_id  = t['id'],
                abbreviation = t['abbreviation'],
                full_name    = t['full_name'],
                city         = t['city'],
                altitude_ft  = ALTITUDE_MAP.get(t['abbreviation'], 0)
            )
            session.add(team)
            print(f"  Added {t['full_name']}")

        session.commit()
        print("✅ Teams saved to database")


if __name__ == "__main__":
    fetch_and_store_teams()