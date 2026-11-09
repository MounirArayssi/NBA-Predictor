"""
build_site.py
=============
Builds the static showcase site from the database:

    python dashboard/build_site.py              # writes site/index.html
    python dashboard/build_site.py --fragment out.html
                                                # also writes the page without the
                                                # <html>/<head> wrapper (artifact preview)

Everything on the page comes from three honest sources:
- the official pre-tip-off predictions (is_official, not after-the-fact backfills)
- the walk-forward backtest (model_version='wf-v2': each season predicted by models
  trained only on earlier seasons, no Vegas input)
- the live model files in model/ (training size, feature importance)

The template is dashboard/site_template.html; the data is injected as JSON.
"""
import sys
import os
import json
import pickle
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sqlalchemy import text

from data.storage.db import engine

ET = ZoneInfo("America/New_York")
TEMPLATE = os.path.join(ROOT, "dashboard", "site_template.html")
OUT = os.path.join(ROOT, "site", "index.html")
WF = "wf-v2"

FEATURE_GROUPS = {
    "Team strength": ["home_elo", "away_elo", "elo_diff", "home_season_margin",
                      "away_season_margin", "home_games_played", "away_games_played"],
    "Recent form (last 10)": ["home_off_rating", "away_off_rating", "home_def_rating",
                              "away_def_rating", "combined_pace"],
    "Availability": ["home_missing_pts", "away_missing_pts"],
    "Rest": ["home_rest_days", "away_rest_days", "home_back_to_back", "away_back_to_back"],
    "Playoff series": ["home_series_wins", "is_elimination", "series_pressure"],
}
FEATURE_LABELS = {
    "home_elo": "Home Elo", "away_elo": "Away Elo", "elo_diff": "Elo gap (incl. home court)",
    "home_season_margin": "Home season margin", "away_season_margin": "Away season margin",
    "home_games_played": "Home games played", "away_games_played": "Away games played",
    "home_off_rating": "Home offensive rating", "away_off_rating": "Away offensive rating",
    "home_def_rating": "Home defensive rating", "away_def_rating": "Away defensive rating",
    "combined_pace": "Combined pace", "home_missing_pts": "Home points missing (injuries)",
    "away_missing_pts": "Away points missing (injuries)", "home_rest_days": "Home rest days",
    "away_rest_days": "Away rest days", "home_back_to_back": "Home back-to-back",
    "away_back_to_back": "Away back-to-back", "home_series_wins": "Home series wins",
    "is_elimination": "Elimination game", "series_pressure": "Series pressure",
}


def q(sql, **params):
    return pd.read_sql(text(sql), engine, params=params)


def f(x, nd=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def clean_factors(factors):
    # Player-model notes describe a model that no longer feeds game scores.
    return [s for s in (factors or []) if "player model" not in s.lower()][:3]


def slate():
    d = q("""
        SELECT g.game_date, g.game_id, g.season_type, g.is_final,
               ht.abbreviation home, ht.full_name home_name,
               at.abbreviation away, at.full_name away_name,
               p.home_score_predicted::float ph, p.away_score_predicted::float pa,
               p.home_win_probability::float hwp, p.vegas_spread::float vs,
               p.vegas_total::float vt, p.key_factors, p.predicted_at,
               g.home_score ah, g.away_score aa, p.winner_correct
        FROM predictions p JOIN games g USING (game_id)
        JOIN teams ht ON ht.team_id = g.home_team_id
        JOIN teams at ON at.team_id = g.away_team_id
        WHERE p.is_official AND p.prediction_type <> 'backfill'
          AND g.game_date = (SELECT MAX(g2.game_date) FROM predictions p2
                             JOIN games g2 USING (game_id)
                             WHERE p2.is_official AND p2.prediction_type <> 'backfill')
        ORDER BY g.game_id
    """)
    if d.empty:
        return None
    games = []
    for r in d.itertuples():
        final = bool(r.is_final) and r.ah is not None and not pd.isna(r.ah)
        games.append(dict(
            home=r.home, away=r.away, home_name=r.home_name, away_name=r.away_name,
            ph=int(r.ph), pa=int(r.pa), hwp=f(r.hwp, 3),
            vegas_spread=f(r.vs, 1), vegas_total=f(r.vt, 1),
            factors=clean_factors(r.key_factors),
            final=final, ah=int(r.ah) if final else None, aa=int(r.aa) if final else None,
            correct=None if not final or r.winner_correct is None else bool(r.winner_correct),
            playoff=r.season_type == "Playoffs",
        ))
    made = pd.Timestamp(d.predicted_at.max()).tz_localize("UTC").tz_convert(ET)
    return dict(date=str(d.game_date.iloc[0]), made_at=made.strftime("%-I:%M %p ET")
                if os.name != "nt" else made.strftime("%#I:%M %p ET"), games=games)


def live_record():
    d = q("""
        SELECT g.season, g.season_type, g.game_date, ht.abbreviation home, at.abbreviation away,
               p.home_score_predicted::float ph, p.away_score_predicted::float pa,
               g.home_score ah, g.away_score aa, p.winner_correct,
               p.margin_error::float me, p.home_score_error::float he,
               p.away_score_error::float ae, p.home_win_probability::float hwp
        FROM predictions p JOIN games g USING (game_id)
        JOIN teams ht ON ht.team_id = g.home_team_id
        JOIN teams at ON at.team_id = g.away_team_id
        WHERE p.is_official AND p.prediction_type <> 'backfill' AND p.evaluated_at IS NOT NULL
        ORDER BY g.game_date DESC, g.game_id DESC
    """)
    by_season = []
    for s, x in d.groupby("season", sort=False):
        by_season.append(dict(season=s, n=len(x), wins=int(x.winner_correct.sum()),
                              margin_mae=f(x.me.mean()), team_mae=f(((x.he + x.ae) / 2).mean())))
    games = [dict(date=str(r.game_date), season=r.season, home=r.home, away=r.away,
                  ph=int(r.ph), pa=int(r.pa), ah=int(r.ah), aa=int(r.aa),
                  correct=bool(r.winner_correct), playoff=r.season_type == "Playoffs",
                  hwp=f(r.hwp, 3))
             for r in d.head(40).itertuples()]
    return dict(by_season=by_season, games=games, n=len(d), wins=int(d.winner_correct.sum()))


def backtest():
    d = q(f"""
        SELECT g.season, g.season_type, p.winner_correct::int w, p.margin_error::float me,
               p.total_score_error::float te, p.home_score_error::float he,
               p.away_score_error::float ae, p.home_win_probability::float p,
               (g.home_score > g.away_score)::int hw, p.model_margin::float pm,
               p.actual_margin::float am,
               g.home_score ah, g.away_score aa, p.home_score_predicted::float ph,
               p.away_score_predicted::float pa, g.game_date,
               ht.abbreviation home, at.abbreviation away
        FROM predictions p JOIN games g USING (game_id)
        JOIN teams ht ON ht.team_id = g.home_team_id
        JOIN teams at ON at.team_id = g.away_team_id
        WHERE p.model_version = '{WF}'
    """)

    def summ(x):
        return dict(n=len(x), win=f(100 * x.w.mean(), 1), margin_mae=f(x.me.mean()),
                    total_mae=f(x.te.mean()), team_mae=f(((x.he + x.ae) / 2).mean()),
                    brier=f(((x.p - x.hw) ** 2).mean(), 3), home_win=f(100 * x.hw.mean(), 1))

    seasons = []
    for s, x in d.groupby("season"):
        seasons.append(dict(season=s, all=summ(x),
                            regular=summ(x[x.season_type == "Regular Season"]),
                            playoffs=summ(x[x.season_type == "Playoffs"])))

    # Calibration: confidence in the picked team vs how often the pick won
    pick_p = np.maximum(d.p, 1 - d.p)
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0]
    cut = pd.cut(pick_p, bins, include_lowest=True)
    calib = []
    for b, x in d.assign(pick_p=pick_p).groupby(cut, observed=True):
        calib.append(dict(lo=f(b.left, 2), hi=f(b.right, 2), n=len(x),
                          predicted=f(100 * x.pick_p.mean(), 1), actual=f(100 * x.w.mean(), 1)))

    # Team score misses, both teams pooled
    miss = np.concatenate([d.he.values, d.ae.values])
    edges = list(range(0, 32, 2))
    hist = []
    for lo in edges:
        hi = lo + 2
        n = int(((miss >= lo) & (miss < hi)).sum()) if lo < 30 else int((miss >= lo).sum())
        hist.append(dict(lo=lo, hi=hi if lo < 30 else None, n=n))

    exact = d[(d.ph == d.ah) & (d.pa == d.aa)].sort_values("game_date")
    exact_games = [dict(date=str(r.game_date), home=r.home, away=r.away, ah=int(r.ah),
                        aa=int(r.aa), playoff=r.season_type == "Playoffs")
                   for r in exact.itertuples()]

    return dict(all=summ(d), seasons=seasons, calibration=calib, miss_hist=hist,
                miss_median=f(np.median(miss), 1),
                within5=f(100 * (miss <= 5).mean(), 0), within10=f(100 * (miss <= 10).mean(), 0),
                exact=exact_games)


def model_info():
    path = os.path.join(ROOT, "model", "model_home.pkl")
    trained = datetime.fromtimestamp(os.path.getmtime(path), ET)
    with open(path, "rb") as fh:
        mh = pickle.load(fh)
    with open(os.path.join(ROOT, "model", "model_away.pkl"), "rb") as fh:
        ma = pickle.load(fh)
    with open(os.path.join(ROOT, "model", "feature_cols.pkl"), "rb") as fh:
        cols = pickle.load(fh)
    imp = (np.array(mh.feature_importances_) + np.array(ma.feature_importances_)) / 2
    imp = dict(zip(cols, imp))
    groups = []
    for g, feats in FEATURE_GROUPS.items():
        groups.append(dict(group=g, share=f(100 * sum(imp.get(c, 0) for c in feats), 1),
                           features=[dict(name=FEATURE_LABELS.get(c, c),
                                          share=f(100 * imp.get(c, 0), 1)) for c in feats]))
    from data.ingestion.team_strength import ELO_START_DATE
    span = q("""SELECT MIN(game_date) lo, MAX(game_date) hi, COUNT(*) n FROM games
                WHERE is_final AND season_type IN ('Regular Season','Playoffs')
                AND game_date >= :s""", s=ELO_START_DATE).iloc[0]
    return dict(trained=trained.strftime("%b %-d, %Y") if os.name != "nt"
                else trained.strftime("%b %#d, %Y"),
                n_features=len(cols), groups=groups, history_from=str(span.lo),
                history_to=str(span.hi), games_in_db=int(span.n))


def coverage():
    d = q("""
        SELECT g.season,
               COUNT(DISTINCT g.game_id) FILTER (WHERE g.is_final AND g.season_type IN ('Regular Season','Playoffs')) games,
               COUNT(DISTINCT g.game_date) FILTER (WHERE g.is_final) dates,
               COUNT(DISTINCT tb.game_id) box,
               COUNT(DISTINCT f.report_date) reports
        FROM games g
        LEFT JOIN team_box_scores tb ON tb.game_id = g.game_id
        LEFT JOIN injury_report_fetches f ON f.report_date = g.game_date AND f.http_status = 200
        GROUP BY g.season ORDER BY g.season
    """)
    return [dict(season=r.season, games=int(r.games), dates=int(r.dates),
                 box=int(r.box), reports=int(r.reports)) for r in d.itertuples()]


def build():
    now = datetime.now(ET)
    return dict(
        generated=now.strftime("%b %-d, %Y %-I:%M %p ET") if os.name != "nt"
        else now.strftime("%b %#d, %Y %#I:%M %p ET"),
        slate=slate(), live=live_record(), backtest=backtest(),
        model=model_info(), coverage=coverage(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", help="also write the page body without a document wrapper")
    args = ap.parse_args()

    data = build()
    with open(TEMPLATE, encoding="utf-8") as fh:
        page = fh.read().replace("/*__DATA__*/null", json.dumps(data, default=str))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           '</head>\n<body>\n' + page + '\n</body>\n</html>\n')
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"Wrote {OUT}")
    if args.fragment:
        with open(args.fragment, "w", encoding="utf-8") as fh:
            fh.write(page)
        print(f"Wrote {args.fragment}")


if __name__ == "__main__":
    main()
