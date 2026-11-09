"""
Raw-stat breakdowns of the model's inputs.

The model never sees shooting, turnovers or rebounds directly. It sees
10-game offensive and defensive ratings (points per 100 possessions), which
are themselves produced by those box-score stats. This module splits a
rating into Dean Oliver's "four factors":

    shooting       eFG%  = (FGM + 0.5 * 3PM) / FGA
    turnovers      TOV%  = TOV / (FGA + 0.44 * FTA + TOV)
    off. rebounds  OREB% = OREB / (OREB + opponent DREB)
    free throws    FTr   = FTM / FGA

Method: one linear fit of per-game offensive rating on the four factors over
every team-game in the database. A rating is an average of per-game ratings,
and a linear fit commutes with averaging, so for any set of games

    rating - league = sum_k b_k * (factor_k - league_factor_k) + other

holds exactly, where `other` is whatever the four factors don't account for
(the fit's residual plus any league-average offset). Defense is the same split
applied to the opponents' factors in those games: the defensive rating IS the
opponents' offensive rating.

Factors are stored as fractions; coefficients are rating points per 1.0 of the
factor, so b_k * 0.01 is the effect of one percentage point.
"""
from __future__ import annotations

import numpy as np

BREAKDOWN_VERSION = "2026-11-08.1"
FACTORS = ["efg", "tov", "orb", "ftr"]
FACTOR_LABELS = {
    "efg": "Shooting (eFG%)",
    "tov": "Turnovers (TOV%)",
    "orb": "Offensive rebounding (OREB%)",
    "ftr": "Free throws (FTM per FGA)",
}
# A reconstructed input must equal the value the model received to this precision.
TOL = {"elo": 0.05, "season_margin": 0.01, "rating": 0.05, "pace": 0.05,
       "missing_pts": 0.05, "rest_days": 0.0}


def four_factors(fgm, fga, fg3m, fta, ftm, tov, oreb, opp_dreb):
    """Four factors for one team-game (or arrays of them). Fractions."""
    fgm, fga, fg3m, fta, ftm, tov, oreb, opp_dreb = (
        np.asarray(v, dtype=float) for v in (fgm, fga, fg3m, fta, ftm, tov, oreb, opp_dreb))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "efg": (fgm + 0.5 * fg3m) / fga,
            "tov": tov / (fga + 0.44 * fta + tov),
            "orb": oreb / (oreb + opp_dreb),
            "ftr": ftm / fga,
        }


def fit_rating(F: np.ndarray, rating: np.ndarray) -> dict:
    """OLS fit rating ~ 1 + factors. F is (n, 4) in FACTORS order."""
    F = np.asarray(F, dtype=float)
    y = np.asarray(rating, dtype=float)
    ok = np.isfinite(F).all(axis=1) & np.isfinite(y)
    X = np.column_stack([np.ones(ok.sum()), F[ok]])
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    resid = y[ok] - X @ beta
    r2 = 1 - resid.var() / y[ok].var()
    return {"intercept": float(beta[0]),
            "coef": {k: float(b) for k, b in zip(FACTORS, beta[1:])},
            "r2": float(r2), "rmse": float(np.sqrt(np.mean(resid ** 2))), "n": int(ok.sum())}


def split_rating(rating: float, factors: dict, league_rating: float,
                 league_factors: dict, coef: dict) -> dict:
    """Split (rating - league_rating) into one part per factor plus 'other'.
    Parts are in rating points (per 100 possessions) and sum exactly to the gap."""
    parts = {k: coef[k] * (factors[k] - league_factors[k]) for k in FACTORS}
    gap = rating - league_rating
    parts["other"] = gap - sum(parts.values())
    return parts


def matches(a, b, tol) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol + 1e-9
