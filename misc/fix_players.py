import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo
from sqlalchemy.orm import Session
from data.storage.db import engine
from data.storage.models import Player

special = [p for p in players.get_active_players()
           if not p['full_name'].isascii()]

print(f'Adding {len(special)} missing players...')

with Session(engine) as session:
    for p in special:
        existing = session.query(Player).filter_by(
            nba_player_id=p['id']
        ).first()

        if existing:
            print(f"  Skipping {p['full_name']} — already exists")
            continue

        try:
            info = commonplayerinfo.CommonPlayerInfo(
                player_id=p['id']
            ).get_data_frames()[0]
            row = info.iloc[0]

            height_inches = None
            if pd.notna(row.get('HEIGHT', None)):
                parts = str(row['HEIGHT']).split('-')
                if len(parts) == 2:
                    height_inches = int(parts[0]) * 12 + int(parts[1])

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
            print(f"  Added {p['full_name']}")
            time.sleep(1)

        except Exception as e:
            print(f"  Failed {p['full_name']}: {e}")

    session.commit()
    print("✅ Done")