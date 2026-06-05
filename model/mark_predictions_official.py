# model/mark_predictions_official.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session, aliased
from data.storage.db import engine
from data.storage.models import Prediction, Game, Team

def mark_predictions_official():
    """Mark all predictions as is_official = True"""
    
    HomeTeam = aliased(Team)
    AwayTeam = aliased(Team)
    
    with Session(engine) as session:
        # Get all non-official predictions
        predictions_to_update = session.query(Prediction, Game, HomeTeam, AwayTeam).join(
            Game, Prediction.game_id == Game.game_id
        ).join(
            HomeTeam, Game.home_team_id == HomeTeam.team_id
        ).join(
            AwayTeam, Game.away_team_id == AwayTeam.team_id
        ).filter(
            Prediction.is_official != True
        ).all()
        
        count = 0
        for pred, game, home_team, away_team in predictions_to_update:
            pred.is_official = True
            count += 1
            print(f"✓ Marked as official: {game.game_date} | {home_team.abbreviation} vs {away_team.abbreviation}")
        
        session.commit()
        print(f"\n✅ Marked {count} predictions as official")
        
        return count

if __name__ == "__main__":
    mark_predictions_official()