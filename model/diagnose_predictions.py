# model/diagnose_predictions.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session, aliased
from sqlalchemy import func
from datetime import datetime, timedelta
from data.storage.db import engine
from data.storage.models import Prediction, Game, Team

def diagnose():
    with Session(engine) as session:
        # Create aliases for home and away teams
        HomeTeam = aliased(Team)
        AwayTeam = aliased(Team)
        
        # Check total predictions
        total_preds = session.query(func.count(Prediction.prediction_id)).scalar()
        print(f"\n{'='*60}")
        print(f"DIAGNOSIS")
        print(f"{'='*60}")
        print(f"Total predictions in database: {total_preds}")
        
        if total_preds == 0:
            print("\n❌ No predictions found at all!")
            print("You need to run: python model/predict.py")
            return
        
        # Check official predictions
        official_count = session.query(func.count(Prediction.prediction_id)).filter(
            Prediction.is_official == True
        ).scalar()
        print(f"Official predictions: {official_count}")
        
        # Check recent predictions
        recent = session.query(Prediction, Game, HomeTeam, AwayTeam).join(
            Game, Prediction.game_id == Game.game_id
        ).join(
            HomeTeam, Game.home_team_id == HomeTeam.team_id
        ).join(
            AwayTeam, Game.away_team_id == AwayTeam.team_id
        ).order_by(
            Prediction.predicted_at.desc()
        ).limit(10).all()
        
        print(f"\nMost recent predictions:")
        for p, game, home_team, away_team in recent:
            print(f"  {game.game_date} | {home_team.abbreviation} vs {away_team.abbreviation}")
            print(f"    Predicted: {p.home_score_predicted}-{p.away_score_predicted} | "
                  f"Official: {p.is_official} | Game Final: {game.is_final}")
            if game.home_score:
                print(f"    Actual: {game.home_score}-{game.away_score}")
        
        # Check games with final scores
        yesterday = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        final_games = session.query(Game, HomeTeam, AwayTeam).join(
            HomeTeam, Game.home_team_id == HomeTeam.team_id
        ).join(
            AwayTeam, Game.away_team_id == AwayTeam.team_id
        ).filter(
            Game.game_date >= yesterday,
            Game.is_final == True,
            Game.home_score != None
        ).all()
        
        print(f"\nFinal games in last 7 days: {len(final_games)}")
        for game, home_team, away_team in final_games[:10]:
            print(f"  {game.game_date} | {home_team.abbreviation} {game.home_score} - "
                  f"{away_team.abbreviation} {game.away_score}")
            
            # Check if there's a prediction for this game
            pred = session.query(Prediction).filter_by(game_id=game.game_id).first()
            if pred:
                print(f"    → Prediction exists | Official: {pred.is_official} | "
                      f"Evaluated: {pred.evaluated_at is not None}")
            else:
                print(f"    → ❌ No prediction found for this game")
        
        # Check what's blocking evaluation
        print(f"\n{'='*60}")
        print("CHECKING EVALUATION BLOCKERS:")
        print(f"{'='*60}")
        
        # Check predictions ready to evaluate
        ready = session.query(Prediction, Game, HomeTeam, AwayTeam).join(
            Game, Prediction.game_id == Game.game_id
        ).join(
            HomeTeam, Game.home_team_id == HomeTeam.team_id
        ).join(
            AwayTeam, Game.away_team_id == AwayTeam.team_id
        ).filter(
            Prediction.is_official == True,
            Game.is_final == True,
            Prediction.evaluated_at == None,
            Game.home_score != None,
            Game.away_score != None,
        ).all()
        
        print(f"\n✅ Predictions READY to evaluate: {len(ready)}")
        for pred, game, home_team, away_team in ready:
            print(f"  {game.game_date} | {home_team.abbreviation} vs {away_team.abbreviation}")
            print(f"    Predicted: {pred.home_score_predicted}-{pred.away_score_predicted}")
            print(f"    Actual: {game.home_score}-{game.away_score}")
        
        # Check predictions with final scores but NOT official
        not_official = session.query(Prediction, Game, HomeTeam, AwayTeam).join(
            Game, Prediction.game_id == Game.game_id
        ).join(
            HomeTeam, Game.home_team_id == HomeTeam.team_id
        ).join(
            AwayTeam, Game.away_team_id == AwayTeam.team_id
        ).filter(
            Prediction.is_official == False,
            Game.is_final == True,
            Game.home_score != None,
            Prediction.evaluated_at == None
        ).all()
        
        print(f"\n❌ Predictions NOT OFFICIAL (blocking evaluation): {len(not_official)}")
        for pred, game, home_team, away_team in not_official:
            print(f"  {game.game_date} | {home_team.abbreviation} vs {away_team.abbreviation}")
            print(f"    is_official: {pred.is_official} ← BLOCKER")
            print(f"    Predicted: {pred.home_score_predicted}-{pred.away_score_predicted}")
            print(f"    Actual: {game.home_score}-{game.away_score}")
        
        # Check predictions with games not marked final
        not_final = session.query(Prediction, Game, HomeTeam, AwayTeam).join(
            Game, Prediction.game_id == Game.game_id
        ).join(
            HomeTeam, Game.home_team_id == HomeTeam.team_id
        ).join(
            AwayTeam, Game.away_team_id == AwayTeam.team_id
        ).filter(
            Prediction.is_official == True,
            Game.is_final == False,
            Game.home_score != None,
            Prediction.evaluated_at == None
        ).all()
        
        print(f"\n❌ Predictions with games NOT FINAL (blocking evaluation): {len(not_final)}")
        for pred, game, home_team, away_team in not_final:
            print(f"  {game.game_date} | {home_team.abbreviation} vs {away_team.abbreviation}")
            print(f"    is_final: {game.is_final} ← BLOCKER")
            print(f"    Actual score exists: {game.home_score}-{game.away_score}")
        
        print(f"\n{'='*60}")
        if len(ready) > 0:
            print("✅ GOOD NEWS: You have predictions ready to evaluate!")
            print("   Run: python evaluate_predictions.py")
        elif len(not_official) > 0:
            print("⚠️  ISSUE: Predictions exist but aren't marked official")
            print("   Fix: Mark predictions as official in your predict.py")
        elif len(not_final) > 0:
            print("⚠️  ISSUE: Games have scores but aren't marked final")
            print("   Fix: Update fetch_games.py to mark games as final")
        else:
            print("⚠️  No predictions with completed games found")
            print("   Run: python model/predict.py")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    diagnose()