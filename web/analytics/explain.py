"""
Local explanations for the two score models (home points, away points).

For a gradient-boosted tree regressor, SHAP's TreeExplainer gives exact additive
contributions: base + sum(contributions) = model output. The site explains the
projected HOME MARGIN, which in production is home_model(x) - away_model(x), so:

    margin contribution of feature j = home_shap[j] - away_shap[j]
    margin baseline                   = home_base   - away_base

These are contributions to the projected margin in points. They are NOT
contributions to the win probability, which comes from a separate calibration
curve applied to the margin.

Stored projected scores are whole numbers, so reproduction is checked at that
precision, in two steps:
  1. A model version is VERIFIED for a group of forecasts (one artifact window, or
     one walk-forward season) when at least VERIFY_SHARE of the group's forecasts
     reproduce BOTH stored scores to the whole point (|model - stored| <= 0.5).
     A per-game match alone is not enough: a different model matches some games
     by chance.
  2. Within a verified group, a forecast is explained when its margin is within
     the rounding bound (MARGIN_TOL) of the stored margin.
Anything else shows its saved inputs without an attribution.
"""
from __future__ import annotations

import numpy as np

EXPLAIN_VERSION = "2026-11-08.1"
SCORE_TOL = 0.5 + 1e-9    # stored scores are whole points
MARGIN_TOL = 1.0 + 1e-9   # two rounded scores -> margin within 1 point
VERIFY_SHARE = 0.90

# Feature -> (group, label, unit). Order is the model's FEATURE_COLS order.
FEATURES = {
    "home_elo": ("strength", "Elo rating (home)", "rating"),
    "away_elo": ("strength", "Elo rating (away)", "rating"),
    "elo_diff": ("strength", "Elo gap incl. home court", "rating"),
    "home_season_margin": ("strength", "Season scoring margin (home)", "pts"),
    "away_season_margin": ("strength", "Season scoring margin (away)", "pts"),
    "home_games_played": ("strength", "Games played (home)", "games"),
    "away_games_played": ("strength", "Games played (away)", "games"),
    "home_off_rating": ("form", "Offensive rating, last 10 (home)", "pts/100"),
    "away_off_rating": ("form", "Offensive rating, last 10 (away)", "pts/100"),
    "home_def_rating": ("form", "Defensive rating, last 10 (home)", "pts/100"),
    "away_def_rating": ("form", "Defensive rating, last 10 (away)", "pts/100"),
    "combined_pace": ("form", "Combined pace, last 10", "poss."),
    "home_missing_pts": ("availability", "Rotation points missing (home)", "pts/g"),
    "away_missing_pts": ("availability", "Rotation points missing (away)", "pts/g"),
    "home_rest_days": ("rest", "Rest days (home)", "days"),
    "away_rest_days": ("rest", "Rest days (away)", "days"),
    "home_back_to_back": ("rest", "Back-to-back (home)", "yes/no"),
    "away_back_to_back": ("rest", "Back-to-back (away)", "yes/no"),
    "home_series_wins": ("playoff", "Home series wins", "wins"),
    "is_elimination": ("playoff", "Elimination game", "yes/no"),
    "series_pressure": ("playoff", "Series pressure", "index"),
}
GROUPS = {
    "strength": "Team strength",
    "form": "Recent form",
    "availability": "Player availability",
    "rest": "Rest and schedule",
    "playoff": "Playoff context",
}


def feature_matrix(vectors: list[dict], cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Rows of stored feature vectors -> (X, ok) where ok marks complete rows."""
    X = np.full((len(vectors), len(cols)), np.nan)
    for i, fv in enumerate(vectors):
        if isinstance(fv, dict):
            for j, c in enumerate(cols):
                v = fv.get(c)
                if v is not None:
                    try:
                        X[i, j] = float(v)
                    except (TypeError, ValueError):
                        pass
    return X, ~np.isnan(X).any(axis=1)


def margin_attribution(model_home, model_away, X: np.ndarray):
    """Exact TreeSHAP for both score models on complete rows of X.

    Returns dict with home/away/margin contributions (n x k), baselines, and the
    raw model outputs; raises if additivity fails."""
    import shap
    eh, ea = shap.TreeExplainer(model_home), shap.TreeExplainer(model_away)
    sh, sa = eh.shap_values(X), ea.shap_values(X)
    bh, ba = float(np.ravel(eh.expected_value)[0]), float(np.ravel(ea.expected_value)[0])
    yh, ya = model_home.predict(X), model_away.predict(X)
    err = max(np.abs(bh + sh.sum(1) - yh).max(initial=0), np.abs(ba + sa.sum(1) - ya).max(initial=0))
    if err > 1e-4:
        raise AssertionError(f"SHAP additivity failed: max error {err:.2e}")
    return dict(home=sh, away=sa, margin=sh - sa, base_home=bh, base_away=ba,
                base_margin=bh - ba, pred_home=yh, pred_away=ya, pred_margin=yh - ya)


def score_match(pred_home, pred_away, stored_ph, stored_pa) -> np.ndarray:
    """Both projected scores round to the stored whole-point scores."""
    return (np.abs(pred_home - stored_ph) <= SCORE_TOL) & (np.abs(pred_away - stored_pa) <= SCORE_TOL)


def margin_match(pred_margin, stored_ph, stored_pa) -> np.ndarray:
    return np.abs(pred_margin - (stored_ph - stored_pa)) <= MARGIN_TOL


def verify_group(pred_home, pred_away, stored_ph, stored_pa) -> tuple[bool, float]:
    """(verified, share of forecasts reproducing both scores)."""
    m = score_match(pred_home, pred_away, stored_ph, stored_pa)
    share = float(m.mean()) if len(m) else 0.0
    return share >= VERIFY_SHARE, share


def group_contributions(contrib: np.ndarray, cols: list[str]) -> dict[str, float]:
    out = {g: 0.0 for g in GROUPS}
    for j, c in enumerate(cols):
        out[FEATURES[c][0]] += float(contrib[j])
    return out


def summary_sentence(home: str, away: str, base: float, groups: dict[str, float], margin: float) -> str:
    """Deterministic, factual summary of the largest group contributions."""
    ranked = sorted(((g, v) for g, v in groups.items() if abs(v) >= 0.5), key=lambda kv: -abs(kv[1]))
    lead = home if margin > 0 else away
    s = f"The model projected {lead} by {abs(margin):.1f} points, from a baseline of {base:+.1f} for the home side."
    if ranked:
        parts = [f"{GROUPS[g].lower()} {'toward' if v > 0 else 'away from'} {home} ({v:+.1f})" for g, v in ranked[:2]]
        s += " The largest pushes on the margin came from " + " and ".join(parts) + "."
    return s
