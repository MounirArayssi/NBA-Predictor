
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from sqlalchemy.orm import Session
from datetime import datetime, timezone

from data.storage.db import engine
from data.storage.models import Prediction, Game


def evaluate_predictions():
    """
    Evaluate official predictions against final game results.

    Only evaluates:
    - is_official = True
    - games that are final
    - predictions not already evaluated

    Updates:
    - actual scores
    - winner correctness
    - home/away score error
    - total error
    - margin/spread error
    """

    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

    evaluated_count = 0
    skipped_count = 0


    with Session(engine) as session:
        official_predictions = (
            session.query(Prediction)
            .join(Game, Prediction.game_id == Game.game_id)
            .filter(
                Prediction.is_official == True,
                Game.is_final == True,
                Prediction.evaluated_at == None,
                Game.home_score != None,
                Game.away_score != None,
            )
            .all()
        )

        if not official_predictions:
            print("No official predictions ready to evaluate.")
            return {
                "evaluated": 0,
                "skipped": 0,
            }

        for pred in official_predictions:
            game = session.query(Game).filter_by(
                game_id=pred.game_id
            ).first()

            if not game:
                skipped_count += 1
                continue

            if game.home_score is None or game.away_score is None:
                skipped_count += 1
                continue

            actual_home = float(game.home_score)
            actual_away = float(game.away_score)

            pred_home = float(pred.home_score_predicted or 0)
            pred_away = float(pred.away_score_predicted or 0)

            actual_total = actual_home + actual_away
            predicted_total = pred_home + pred_away

            actual_margin = actual_home - actual_away
            predicted_margin = pred_home - pred_away

            actual_winner_id = (
                game.home_team_id
                if actual_home > actual_away
                else game.away_team_id
            )

            pred.home_score_actual = int(actual_home)
            pred.away_score_actual = int(actual_away)
            pred.actual_winner_id = actual_winner_id

            pred.winner_correct = (
                pred.predicted_winner_id == actual_winner_id
            )

            pred.home_score_error = abs(pred_home - actual_home)
            pred.away_score_error = abs(pred_away - actual_away)

            pred.total_score_error = abs(predicted_total - actual_total)

            pred.actual_margin = actual_margin
            pred.actual_total = actual_total

            pred.margin_error = abs(predicted_margin - actual_margin)

            # Same idea as margin_error, but named for market/spread analysis.
            # Your model_margin is home score - away score.
            pred.spread_error = abs(float(pred.model_margin or predicted_margin) - actual_margin)

            pred.evaluated_at = now_utc

            evaluated_count += 1

        session.commit()

    print(f"✅ Evaluated {evaluated_count} official predictions")
    print(f"⏭️ Skipped {skipped_count} predictions")

    return {
        "evaluated": evaluated_count,
        "skipped": skipped_count,
    }

if __name__ == "__main__":
    evaluate_predictions()