import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from datetime import datetime, timezone, date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_

from data.storage.db import engine
from data.storage.models import Prediction, Game


def evaluate_prediction_row(pred, game, now_utc):
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

    pred.spread_error = abs(
        float(pred.model_margin or predicted_margin) - actual_margin
    )

    pred.evaluated_at = now_utc

    return {
        "game_id": game.game_id,
        "pred_home": pred_home,
        "pred_away": pred_away,
        "actual_home": actual_home,
        "actual_away": actual_away,
        "winner_correct": pred.winner_correct,
    }


def evaluate_predictions(
    start_date=None,
    end_date=None,
    official_only=True,
    include_already_evaluated=False,
):
    """
    Evaluate predictions across a date range.

    Defaults:
    - start_date: no lower bound
    - end_date: today
    - official_only: True
    - include_already_evaluated: False

    This evaluates all final games with scores in the date range.
    """

    if end_date is None:
        end_date = date.today()

    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

    evaluated_count = 0
    skipped_count = 0

    with Session(engine) as session:
        query = (
            session.query(Prediction, Game)
            .join(Game, Prediction.game_id == Game.game_id)
            .filter(
                Game.is_final == True,
                Game.home_score != None,
                Game.away_score != None,
            )
        )

        if official_only:
            query = query.filter(Prediction.is_official == True)

        if not include_already_evaluated:
            query = query.filter(Prediction.evaluated_at == None)

        if start_date is not None:
            query = query.filter(Game.game_date >= start_date)

        if end_date is not None:
            query = query.filter(Game.game_date <= end_date)

        rows = (
            query
            .order_by(Game.game_date.asc(), Prediction.game_id.asc())
            .all()
        )

        if not rows:
            print("No predictions ready to evaluate.")
            print(f"Filters: start_date={start_date}, end_date={end_date}, official_only={official_only}")
            return {
                "evaluated": 0,
                "skipped": 0,
            }

        for pred, game in rows:
            try:
                result = evaluate_prediction_row(pred, game, now_utc)
                evaluated_count += 1

                print(
                    f"✅ Evaluated game_id={result['game_id']}: "
                    f"pred {result['pred_home']:.0f}-{result['pred_away']:.0f}, "
                    f"actual {result['actual_home']:.0f}-{result['actual_away']:.0f}, "
                    f"winner_correct={result['winner_correct']}"
                )

            except Exception as e:
                skipped_count += 1
                print(f"⚠️ Skipped prediction_id={pred.prediction_id}: {e}")

        session.commit()

    print(f"\n✅ Evaluated {evaluated_count} predictions")
    print(f"⏭️ Skipped {skipped_count} predictions")

    return {
        "evaluated": evaluated_count,
        "skipped": skipped_count,
    }


if __name__ == "__main__":
    # Backfill/evaluate all official predictions for final games up through today.
    evaluate_predictions(
        start_date=date(2026, 4, 24),
        end_date=date.today(),
        official_only=True,
        include_already_evaluated=False,
    )