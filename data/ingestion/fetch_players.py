import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo
from sqlalchemy.orm import Session
from data.storage.db import engine
from data.storage.models import Player
from config.settings import NBA_API_DELAY

def fetch_and_store_players():
    print("Fetching NBA players...")

    # Get all active players
    all_players = players.get_active_players()
    print(f"Found {len(all_players)} active players")

    with Session(engine) as session:
        for i, p in enumerate(all_players):
            # Check if already exists
            existing = session.query(Player).filter_by(
                nba_player_id=p['id']
            ).first()

            if existing:
                print(f"  Skipping {p['full_name']} — already exists")
                continue

            try:
                # Fetch detailed player info
                info = commonplayerinfo.CommonPlayerInfo(
                    player_id=p['id']
                ).get_data_frames()[0]

                row = info.iloc[0]

                # Parse height (comes as "6-8" format)
                height_inches = None
                if pd.notna(row.get('HEIGHT', None)):
                    parts = str(row['HEIGHT']).split('-')
                    if len(parts) == 2:
                        height_inches = int(parts[0]) * 12 + int(parts[1])

                # Parse weight
                weight = None
                if pd.notna(row.get('WEIGHT', None)):
                    try:
                        weight = int(row['WEIGHT'])
                    except:
                        pass

                player = Player(
                    nba_player_id = p['id'],
                    full_name     = p['full_name'],
                    first_name    = p['first_name'],
                    last_name     = p['last_name'],
                    position      = row.get('POSITION', None),
                    height_inches = height_inches,
                    weight_lbs    = weight,
                    is_active     = True
                )
                session.add(player)
                print(f"  [{i+1}/{len(all_players)}] Added {p['full_name']}")

                # Be respectful to the API
                time.sleep(NBA_API_DELAY)

            except Exception as e:
                print(f"  ❌ Failed on {p['full_name']}: {e}")
                continue

        session.commit()
        print("✅ Players saved to database")

if __name__ == "__main__":
    import pandas as pd
    fetch_and_store_players()