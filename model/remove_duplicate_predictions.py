# model/remove_duplicate_predictions.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from sqlalchemy import func
from data.storage.db import engine
from data.storage.models import Prediction

with Session(engine) as session:
    # Find duplicates (keep the one with latest predicted_at)
    duplicates = session.query(
        Prediction.game_id,
        func.count(Prediction.prediction_id).label('count')
    ).group_by(
        Prediction.game_id
    ).having(
        func.count(Prediction.prediction_id) > 1
    ).all()
    
    deleted = 0
    for game_id, count in duplicates:
        # Keep the latest prediction, delete the rest
        preds = session.query(Prediction).filter_by(
            game_id=game_id
        ).order_by(
            Prediction.predicted_at.desc()
        ).all()
        
        # Delete all except the first (most recent)
        for pred in preds[1:]:
            session.delete(pred)
            deleted += 1
            print(f"Deleted duplicate prediction for game_id: {game_id}")
    
    session.commit()
    print(f"\n✅ Removed {deleted} duplicate predictions")