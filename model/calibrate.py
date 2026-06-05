"""
Platt scaling calibration for win probability.

Fits a logistic regression mapping predicted_margin → actual_home_win,
replacing the fixed sigmoid used in calculate_confidence.

Run this after enough evaluated predictions exist (20+):
    python model/calibrate.py
"""
import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sqlalchemy.orm import Session

from data.storage.db import engine
from data.storage.models import Prediction, Game

CALIBRATION_PATH = 'model/model_calibration.pkl'


def fit_platt_calibration(save_path=CALIBRATION_PATH):
    with Session(engine) as session:
        rows = (
            session.query(Prediction, Game)
            .join(Game, Prediction.game_id == Game.game_id)
            .filter(
                Prediction.is_official == True,
                Prediction.evaluated_at != None,
                Prediction.winner_correct != None,
            )
            .all()
        )

    if len(rows) < 20:
        print(f"Only {len(rows)} evaluated predictions — need at least 20 for calibration.")
        return None

    X = []
    y = []
    for p, g in rows:
        pred_home = float(p.home_score_predicted or 0)
        pred_away = float(p.away_score_predicted or 0)
        margin = pred_home - pred_away
        actual_home_won = 1 if float(g.home_score) > float(g.away_score) else 0
        X.append([margin])
        y.append(actual_home_won)

    X = np.array(X)
    y = np.array(y)

    clf = LogisticRegression(C=1.0, random_state=42)
    clf.fit(X, y)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(clf, f)

    probs = clf.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)
    acc = (preds == y).mean()

    print(f"Calibration fitted on {len(X)} games")
    print(f"In-sample accuracy: {acc * 100:.1f}%")
    print(f"Logistic coef: {clf.coef_[0][0]:.4f}, intercept: {clf.intercept_[0]:.4f}")
    print(f"Saved to {save_path}")
    return clf


def predict_win_probability(home_margin, calibration_model=None):
    """
    Return home team win probability (0.0–1.0) from the predicted margin.
    Falls back to a tuned sigmoid if no calibration model is available.
    """
    if calibration_model is not None:
        try:
            return float(calibration_model.predict_proba([[home_margin]])[0][1])
        except Exception:
            pass
    return float(1 / (1 + np.exp(-home_margin / 9.5)))


if __name__ == '__main__':
    fit_platt_calibration()
