"""
Summarize model performance with playoff-appropriate margin buckets.

Playoff game classification:
- Close games: ≤5 pts (one possession games, truly competitive)
- Competitive games: 6-15 pts (standard playoff margin)
- Blowouts: 16+ pts (rare, dominant performances)
"""
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from sqlalchemy.orm import Session
from data.storage.db import engine
from data.storage.models import Prediction, Game


def pct(value):
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def avg(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(value):
    return f"{value:.2f}" if value is not None else "N/A"


def summarize_playoff_performance(model_version=None):
    """
    Summarize evaluated official prediction performance using playoff-appropriate buckets.
    """

    with Session(engine) as session:
        query = (
            session.query(Prediction, Game)
            .join(Game, Prediction.game_id == Game.game_id)
            .filter(
                Prediction.is_official == True,
                Prediction.evaluated_at != None,
            )
        )

        if model_version:
            query = query.filter(Prediction.model_version == model_version)

        rows = query.all()

    if not rows:
        print("No evaluated official predictions yet.")
        return

    total_games = len(rows)

    winner_correct = [
        p.winner_correct for p, g in rows
        if p.winner_correct is not None
    ]

    win_accuracy = (
        sum(1 for x in winner_correct if x) / len(winner_correct)
        if winner_correct else None
    )

    home_errors = [p.home_score_error for p, g in rows]
    away_errors = [p.away_score_error for p, g in rows]
    total_errors = [p.total_score_error for p, g in rows]
    margin_errors = [p.margin_error for p, g in rows]

    predicted_total_diffs = []
    predicted_margin_diffs = []

    close_games = []        # ≤5 pts
    competitive_games = []  # 6-15 pts
    blowout_games = []      # 16+ pts

    for p, g in rows:
        pred_home = float(p.home_score_predicted or 0)
        pred_away = float(p.away_score_predicted or 0)
        actual_home = float(g.home_score or 0)
        actual_away = float(g.away_score or 0)

        pred_total = pred_home + pred_away
        actual_total = actual_home + actual_away

        pred_margin = pred_home - pred_away
        actual_margin = actual_home - actual_away

        predicted_total_diffs.append(pred_total - actual_total)
        predicted_margin_diffs.append(pred_margin - actual_margin)

        abs_actual_margin = abs(actual_margin)

        if abs_actual_margin <= 5:
            close_games.append((p, g))
        elif abs_actual_margin <= 15:
            competitive_games.append((p, g))
        else:
            blowout_games.append((p, g))

    print("\n" + "=" * 55)
    print("MODEL PERFORMANCE SUMMARY (Playoff Buckets)")
    print("=" * 55)

    if model_version:
        print(f"Model version: {model_version}")

    print(f"Games evaluated: {total_games}")
    print(f"Winner accuracy: {pct(win_accuracy)}")

    print("\nAverage errors:")
    print(f"  Home score error: {fmt(avg(home_errors))}")
    print(f"  Away score error: {fmt(avg(away_errors))}")
    print(f"  Margin error:     {fmt(avg(margin_errors))}")
    print(f"  Total error:      {fmt(avg(total_errors))}")

    total_bias = avg(predicted_total_diffs)
    margin_bias = avg(predicted_margin_diffs)

    print("\nBias:")
    print(f"  Avg total bias:  {fmt(total_bias)}")
    print(f"  Avg margin bias: {fmt(margin_bias)}")

    if total_bias is not None:
        if total_bias > 0:
            print("  → Model is scoring games too high on average")
        elif total_bias < 0:
            print("  → Model is scoring games too low on average")
        else:
            print("  → Model total is perfectly neutral so far")

    print("\nBy actual margin (playoff-appropriate):")
    print_bucket("Close games ≤5", close_games)
    print_bucket("Competitive games 6–15", competitive_games)
    print_bucket("Blowouts 16+", blowout_games)

    print("\n" + "=" * 55 + "\n")


def print_bucket(label, rows):
    if not rows:
        print(f"  {label}: no games")
        return

    winner_correct = [
        p.winner_correct for p, g in rows
        if p.winner_correct is not None
    ]

    accuracy = (
        sum(1 for x in winner_correct if x) / len(winner_correct)
        if winner_correct else None
    )

    margin_errors = [p.margin_error for p, g in rows]
    total_errors = [p.total_score_error for p, g in rows]

    print(
        f"  {label}: {len(rows)} games | "
        f"winner acc {pct(accuracy)} | "
        f"margin err {fmt(avg(margin_errors))} | "
        f"total err {fmt(avg(total_errors))}"
    )


if __name__ == "__main__":
    summarize_playoff_performance()