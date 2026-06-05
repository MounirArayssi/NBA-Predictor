# model/mark_games_final.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from data.storage.db import engine
from data.storage.models import Game

def mark_games_final():
    """Mark all games with final scores as is_final = True"""
    with Session(engine) as session:
        # Find games with scores but not marked final
        games_to_update = session.query(Game).filter(
            Game.home_score != None,
            Game.away_score != None,
            Game.is_final != True
        ).all()
        
        count = 0
        for game in games_to_update:
            game.is_final = True
            count += 1
            print(f"✓ Marked as final: {game.game_date} | Game ID: {game.game_id}")
        
        session.commit()
        print(f"\n✅ Marked {count} games as final")
        
        return count

if __name__ == "__main__":
    mark_games_final()