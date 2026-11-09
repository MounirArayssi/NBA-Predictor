"""
What wins games: how well each pregame stat gap (home minus away) lined up with
the final margin, over every graded forecast on the site.

    python web/scripts/what_wins.py     # after export_data.py and breakdown_inputs.py

Two kinds of stat:
- Model inputs, taken from the feature vector saved with each forecast (the same
  one-forecast-per-game rule as the exporter: issued forecast if any, else the
  walk-forward one; 'backfill' rows excluded).
- Four-factor stats that are NOT model inputs: each team's average over its
  previous 10 final games (same window as the ratings), from team box scores.

For each stat: Pearson correlation with the final home margin, and how often the
team with the better value won (ties skipped). Read-only; writes
web/src/data/what_wins.json only.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
ROOT = os.path.dirname(WEB)
sys.path.insert(0, ROOT)
sys.path.insert(0, WEB)
sys.path.insert(0, HERE)
os.chdir(ROOT)

import numpy as np
import pandas as pd

from data.storage.db import engine
from analytics import breakdown as B
from breakdown_inputs import load_team_games, read

OUT = os.path.join(WEB, "src", "data", "what_wins.json")
WINDOW = 10

# key -> (label, group, direction note). Gaps are home minus away; for stats where
# lower is better (defensive rating, turnovers committed, ...), the gap is flipped
# so that a positive gap always means "home is better".
STATS = {
    "elo": ("Elo rating", "Model input", 1),
    "season_margin": ("Season scoring margin", "Model input", 1),
    "net": ("Net rating, last 10", "Model input", 1),
    "off": ("Offensive rating, last 10", "Model input", 1),
    "def": ("Defensive rating, last 10", "Model input", -1),
    "missing": ("Rotation points missing", "Model input", -1),
    "rest": ("Rest days", "Model input", 1),
    "b2b": ("Back-to-back", "Model input", -1),
    "efg": ("Shooting (eFG%)", "Four factors (not inputs)", 1),
    "d_efg": ("Opponents' shooting allowed", "Four factors (not inputs)", -1),
    "tov": ("Turnover rate", "Four factors (not inputs)", -1),
    "d_tov": ("Turnovers forced", "Four factors (not inputs)", 1),
    "orb": ("Offensive rebounding", "Four factors (not inputs)", 1),
    "d_orb": ("Opponents' offensive rebounding", "Four factors (not inputs)", -1),
    "ftr": ("Free throws per shot", "Four factors (not inputs)", 1),
    "d_ftr": ("Opponents' free throws per shot", "Four factors (not inputs)", -1),
}


def load_forecasts():
    df = read("""
        SELECT p.game_id, g.game_date, g.season, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, p.model_version, p.is_official, p.prediction_type,
               p.predicted_at, p.feature_vector
        FROM predictions p JOIN games g ON g.game_id = p.game_id
        WHERE g.is_final AND g.home_score IS NOT NULL AND g.home_team_id <> g.away_team_id
          AND (p.model_version = 'wf-v2' OR (p.is_official AND p.prediction_type <> 'backfill'))
    """)
    iss = df[df.model_version != "wf-v2"].sort_values("predicted_at").drop_duplicates("game_id", keep="last")
    wf = df[(df.model_version == "wf-v2") & ~df.game_id.isin(iss.game_id)].drop_duplicates("game_id")
    return pd.concat([iss, wf], ignore_index=True)


def main():
    fc = load_forecasts()
    fc["game_date"] = pd.to_datetime(fc["game_date"])
    fc["margin"] = fc.home_score.astype(float) - fc.away_score.astype(float)
    fv = pd.json_normalize(fc.feature_vector.tolist())
    g = pd.DataFrame({"game_id": fc.game_id, "season": fc.season, "date": fc.game_date,
                      "home": fc.home_team_id, "away": fc.away_team_id, "margin": fc.margin})
    g["elo"] = fv["home_elo"].values - fv["away_elo"].values
    g["season_margin"] = fv["home_season_margin"].values - fv["away_season_margin"].values
    g["off"] = fv["home_off_rating"].values - fv["away_off_rating"].values
    g["def"] = fv["home_def_rating"].values - fv["away_def_rating"].values
    g["net"] = g["off"] - g["def"]
    g["missing"] = fv["home_missing_pts"].values - fv["away_missing_pts"].values
    g["rest"] = fv["home_rest_days"].values - fv["away_rest_days"].values
    g["b2b"] = fv["home_back_to_back"].values - fv["away_back_to_back"].values

    # Four factors: previous-10-game averages per team, before each game.
    tg = load_team_games()
    cols = B.FACTORS + [f"d_{k}" for k in B.FACTORS]
    roll = []
    for tid, d in tg.groupby("team_id"):
        d = d.sort_values(["game_date", "game_id"])
        r = d[cols].rolling(WINDOW, min_periods=WINDOW).mean().shift(1)
        r["team_id"], r["game_id"] = tid, d.game_id.values
        roll.append(r)
    roll = pd.concat(roll)
    for side in ("home", "away"):
        g = g.merge(roll.rename(columns={c: f"{side}_{c}" for c in cols}).rename(columns={"team_id": side}),
                    on=["game_id", side], how="left")
    for c in cols:
        g[c] = g[f"home_{c}"] - g[f"away_{c}"]

    stats = []
    for k, (label, group, sgn) in STATS.items():
        d = g[["margin", k]].dropna()
        x = sgn * d[k].astype(float)
        if len(d) < 200 or x.std() == 0:
            continue
        r = float(np.corrcoef(x, d.margin)[0, 1])
        nz = d[x != 0]
        xs = x[x != 0]
        better_won = float(((xs > 0) == (nz.margin > 0)).mean()) if len(nz) else None
        stats.append(dict(key=k, label=label, group=group, r=round(r, 3), n=int(len(d)),
                          better_won=round(better_won, 3) if better_won is not None else None,
                          n_better=int(len(nz))))
    stats.sort(key=lambda s: -abs(s["r"]))
    home_won = float((g.margin > 0).mean())
    payload = dict(n_games=int(len(g)), first=g.date.min().strftime("%Y-%m-%d"),
                   last=g.date.max().strftime("%Y-%m-%d"), home_won=round(home_won, 3),
                   window=WINDOW, stats=stats)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    for s in stats:
        print(f"  {s['label']:34s} r={s['r']:+.3f}  better side won {100 * (s['better_won'] or 0):.1f}%  n={s['n']}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
