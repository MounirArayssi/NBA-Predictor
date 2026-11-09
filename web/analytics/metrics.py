"""
Forecast-evaluation metrics for the website: team predictability and surprise games.

Pure functions over graded game rows; no database access, no model code. Every
number the site shows for these features comes from here, so the tests in
web/analytics/tests cover them directly.

A graded game row (the archive format written by web/scripts/export_data.py):
    id   NBA game id            d   date (YYYY-MM-DD)     s  season     po  playoffs?
    h/a  home/away abbr         ph/pa  predicted home/away points
    p    pregame home win probability (0..1)
    hs/as_  final home/away points
Only final games are graded; rows without scores or a probability are ignored.
"""
from __future__ import annotations

import math
import random
from statistics import mean, stdev

CALC_VERSION = "2026-11-08.1"

# Sample-size tiers for team comparisons. A team has ~82 regular-season games a
# season, so one season is "stable"; a few weeks of a new season is "small".
TIER_SMALL = 20      # n < 20: figures shown, no ranking
TIER_STABLE = 60     # 20 <= n < 60: exploratory ranking with a warning; >= 60: ranked

# Probabilities are clamped only inside the log so a (theoretical) 0 or 1 never
# produces an infinite surprise. Displayed probabilities are never altered.
P_FLOOR = 1e-4

BOOT_N = 1000
BOOT_SEED = 20261108


def is_graded(r: dict) -> bool:
    return (r.get("hs") is not None and r.get("as_") is not None and r.get("p") is not None
            and r.get("ph") is not None and r.get("pa") is not None and r["hs"] != r["as_"])


def dedupe(rows: list[dict]) -> list[dict]:
    """One row per game id; a duplicate id is an error upstream, so fail loudly."""
    seen = {}
    for r in rows:
        if r["id"] in seen:
            raise ValueError(f"duplicate game {r['id']} in graded rows")
        seen[r["id"]] = r
    return list(seen.values())


# ------------------------------------------------------------------ team view
def team_games(rows: list[dict], abbr: str) -> list[dict]:
    """Rows involving `abbr`, oriented to that team, in date order."""
    out = []
    for r in sorted((r for r in rows if is_graded(r) and abbr in (r["h"], r["a"])), key=lambda r: (r["d"], r["id"])):
        home = r["h"] == abbr
        p = r["p"] if home else 1 - r["p"]
        tp, op = (r["ph"], r["pa"]) if home else (r["pa"], r["ph"])
        ts, os_ = (r["hs"], r["as_"]) if home else (r["as_"], r["hs"])
        won = ts > os_
        out.append(dict(
            id=r["id"], d=r["d"], s=r["s"], po=bool(r.get("po")), opp=r["a"] if home else r["h"], home=home,
            p=p, tp=tp, op=op, ts=ts, os=os_, won=won,
            ok=(tp > op) == won,                       # forecast winner (by projected score) won
            score_err=ts - tp,                         # team scoring error
            me=(ts - os_) - (tp - op),                 # team margin error
        ))
    return out


def _boot_ci(values: list[float], stat, n=BOOT_N, seed=BOOT_SEED, alpha=0.10):
    """Percentile bootstrap interval (90% by default) for stat(values)."""
    if len(values) < 5:
        return None
    rng = random.Random(seed)
    k = len(values)
    est = sorted(stat([values[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    lo, hi = est[int(n * alpha / 2)], est[int(n * (1 - alpha / 2)) - 1]
    return [round(lo, 2), round(hi, 2)]


def tier(n: int) -> str:
    return "small" if n < TIER_SMALL else "moderate" if n < TIER_STABLE else "stable"


def team_summary(games: list[dict]) -> dict:
    """Predictability metrics for one team's oriented games (see team_games)."""
    n = len(games)
    if n == 0:
        return dict(n=0, tier="small")
    me = [g["me"] for g in games]
    ame = [abs(x) for x in me]
    exp_w = sum(g["p"] for g in games)
    act_w = sum(1 for g in games if g["won"])
    brier = mean((g["p"] - (1 if g["won"] else 0)) ** 2 for g in games)
    logloss = mean(-math.log(max(P_FLOOR, g["p"] if g["won"] else 1 - g["p"])) for g in games)
    sd = stdev(me) if n > 1 else None
    return dict(
        n=n, tier=tier(n),
        acc=round(100 * sum(1 for g in games if g["ok"]) / n, 1),
        mae=round(mean(ame), 2), mae_ci=_boot_ci(ame, mean),
        bias=round(mean(me), 2), bias_ci=_boot_ci(me, mean),
        sd=round(sd, 2) if sd is not None else None,
        score_bias=round(mean(g["score_err"] for g in games), 2),
        exp_w=round(exp_w, 2), act_w=act_w, gap=round(act_w - exp_w, 2),
        brier=round(brier, 4), logloss=round(logloss, 4),
    )


def bias_statement(abbr: str, s: dict) -> str | None:
    """A factual one-line summary, only when the 90% interval for the average
    margin error excludes zero and the sample is not small."""
    ci = s.get("bias_ci")
    if not ci or s["tier"] == "small" or (ci[0] <= 0 <= ci[1]):
        return None
    direction = "better" if s["bias"] > 0 else "worse"
    return (f"Across {s['n']} graded games, {abbr} finished an average of {abs(s['bias']):.1f} points "
            f"{direction} than the forecast margin (90% interval {ci[0]:+.1f} to {ci[1]:+.1f}).")


# ------------------------------------------------------------------ surprises
def outcome(r: dict) -> dict:
    """Who won, the probability the forecast gave them, and the log surprise (nats)."""
    home_won = r["hs"] > r["as_"]
    p_w = r["p"] if home_won else 1 - r["p"]
    return dict(winner=r["h"] if home_won else r["a"], loser=r["a"] if home_won else r["h"],
                p_winner=p_w, surprise=-math.log(min(1.0, max(P_FLOOR, p_w))))


def scoreline_error(r: dict) -> float:
    """|actual home margin - predicted home margin|, in points."""
    return abs((r["hs"] - r["as_"]) - (r["ph"] - r["pa"]))


def annotate(r: dict) -> dict:
    """Archive row plus surprise fields: w (winner), pw (winner's pregame
    probability), su (outcome surprise, nats), se (scoreline error, points)."""
    o = outcome(r)
    return {**r, "w": o["winner"], "pw": round(o["p_winner"], 3), "su": round(o["surprise"], 3),
            "se": round(scoreline_error(r), 1)}


def rank_outcome(rows: list[dict], k: int | None = None) -> list[dict]:
    """Most surprising results first (lowest winner probability); ties by date, newest first."""
    g = [annotate(r) for r in rows if is_graded(r)]
    g.sort(key=lambda r: (r["d"], r["id"]), reverse=True)
    g.sort(key=lambda r: -r["su"])   # stable: equal surprise keeps newest first
    return g[:k] if k else g


def rank_scoreline(rows: list[dict], k: int | None = None) -> list[dict]:
    g = [annotate(r) for r in rows if is_graded(r)]
    g.sort(key=lambda r: -r["se"])
    return g[:k] if k else g


def team_surprises(rows: list[dict], abbr: str, k: int = 5) -> dict:
    """Biggest unexpected wins and losses (by the team's own pregame probability),
    largest scoreline misses, and the most recent results the forecast missed."""
    mine = [annotate(r) for r in rows if is_graded(r) and abbr in (r["h"], r["a"])]
    wins = [r for r in mine if r["w"] == abbr]
    losses = [r for r in mine if r["w"] != abbr]
    wins.sort(key=lambda r: -r["su"])
    losses.sort(key=lambda r: -r["su"])
    by_se = sorted(mine, key=lambda r: -r["se"])
    recent_missed = sorted((r for r in mine if r["pw"] < 0.5), key=lambda r: (r["d"], r["id"]), reverse=True)
    return dict(wins=wins[:k], losses=losses[:k], scoreline=by_se[:k], recent=recent_missed[:k])
