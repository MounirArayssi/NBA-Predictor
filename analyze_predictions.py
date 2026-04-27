"""
Show detailed game-by-game prediction analysis.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.orm import Session, aliased
from data.storage.db import engine
from data.storage.models import Prediction, Game, Team

def analyze_predictions():
    """Show detailed breakdown of each prediction."""
    
    HomeTeam = aliased(Team)
    AwayTeam = aliased(Team)
    
    with Session(engine) as session:
        # Get all evaluated predictions
        results = session.query(Prediction, Game, HomeTeam, AwayTeam).join(
            Game, Prediction.game_id == Game.game_id
        ).join(
            HomeTeam, Game.home_team_id == HomeTeam.team_id
        ).join(
            AwayTeam, Game.away_team_id == AwayTeam.team_id
        ).filter(
            Prediction.evaluated_at != None
        ).order_by(
            Game.game_date.desc()
        ).all()
        
        if not results:
            print("No evaluated predictions found")
            return
        
        print("\n" + "="*80)
        print("DETAILED PREDICTION ANALYSIS")
        print("="*80)
        
        for pred, game, home, away in results:
            print(f"\n📅 {game.game_date} | {home.full_name} vs {away.full_name}")
            print("-" * 80)
            
            # Predicted scores
            pred_home = int(pred.home_score_predicted or 0)
            pred_away = int(pred.away_score_predicted or 0)
            pred_total = pred_home + pred_away
            pred_margin = pred_home - pred_away
            
            # Actual scores
            actual_home = int(game.home_score or 0)
            actual_away = int(game.away_score or 0)
            actual_total = actual_home + actual_away
            actual_margin = actual_home - actual_away
            
            # Errors
            home_error = abs(pred_home - actual_home)
            away_error = abs(pred_away - actual_away)
            total_error = abs(pred_total - actual_total)
            margin_error = abs(pred_margin - actual_margin)
            
            # Winner
            pred_winner = home.abbreviation if pred_home > pred_away else away.abbreviation
            actual_winner = home.abbreviation if actual_home > actual_away else away.abbreviation
            winner_correct = "✅" if pred_winner == actual_winner else "❌"
            
            print(f"Predicted: {home.abbreviation} {pred_home}, {away.abbreviation} {pred_away} "
                  f"(Total: {pred_total}, Margin: {pred_margin:+d})")
            print(f"Actual:    {home.abbreviation} {actual_home}, {away.abbreviation} {actual_away} "
                  f"(Total: {actual_total}, Margin: {actual_margin:+d})")
            print(f"\nErrors:")
            print(f"  Home:   {home_error} pts")
            print(f"  Away:   {away_error} pts")
            print(f"  Total:  {total_error} pts")
            print(f"  Margin: {margin_error} pts")
            print(f"\nWinner: {winner_correct} Predicted {pred_winner}, Actual {actual_winner}")
            
            # Analysis
            if pred_total > actual_total:
                print(f"⚠️  Over-predicted total by {pred_total - actual_total} pts")
            elif pred_total < actual_total:
                print(f"⚠️  Under-predicted total by {actual_total - pred_total} pts")
            
            if pred_margin > 0 and actual_margin < 0:
                print(f"⚠️  Wrong side: predicted {home.abbreviation} by {pred_margin}, "
                      f"but {away.abbreviation} won by {abs(actual_margin)}")
            elif pred_margin < 0 and actual_margin > 0:
                print(f"⚠️  Wrong side: predicted {away.abbreviation} by {abs(pred_margin)}, "
                      f"but {home.abbreviation} won by {actual_margin}")
            elif abs(margin_error) > 10:
                print(f"⚠️  Large margin miss: off by {margin_error} pts")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    analyze_predictions()