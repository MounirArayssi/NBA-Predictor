"""
Read-only analyses behind the charts in the Notebook (web/src/content/notebook/).

    python web/scripts/notebook_figures.py

Same rules as export_data.py: one PostgreSQL connection set to read-only, SELECT
statements only, nothing written to the database. Model code is not run; the Elo
section reuses the TeamStrength class (pure Python, no database access) so the
ratings shown are exactly the ones the model sees.

Writes web/src/data/notebook_figures.json. Each section below is excerpted on the
notebook page it belongs to (between the "# --- name" markers), so the code shown
there is the code that produced the numbers.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
ROOT = os.path.dirname(WEB)
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sqlalchemy import text
from data.storage.db import engine  # engine only; no writes on import

OUT = os.path.join(WEB, "src", "data", "notebook_figures.json")
CACHE = os.path.join(WEB, ".cache")


def r1(x, nd=1):
    return None if x is None or pd.isna(x) else round(float(x), nd)


# --- questionable
def questionable(q):
    """Did players listed on the 5:30 PM ET official injury report play that night?"""
    rep = q("""SELECT e.game_date, e.player_id, e.team_id, e.status
               FROM injury_report_entries e JOIN injury_report_fetches f USING (fetch_id)
               WHERE e.game_date = f.report_date AND e.player_id IS NOT NULL
                 AND f.http_status = 200 AND f.report_time = '17:30'""")
    box = q("""SELECT g.game_id, g.game_date, g.season, b.player_id,
                      b.minutes_played::float mp, b.points
               FROM player_box_scores b JOIN games g USING (game_id) WHERE g.is_final""")
    gm = q("SELECT game_id, game_date, season, home_team_id, away_team_id FROM games WHERE is_final")
    for x in (rep, box, gm):
        x["game_date"] = pd.to_datetime(x.game_date)
    rep = rep.drop_duplicates(["game_date", "player_id"])

    # each listing -> that team's game that night -> did the player log minutes?
    sides = [gm.rename(columns={c: "team_id"})[["game_id", "game_date", "season", "team_id"]]
             for c in ("home_team_id", "away_team_id")]
    r = rep.merge(pd.concat(sides), on=["game_date", "team_id"])
    r = r.merge(box[["game_id", "player_id", "mp"]], on=["game_id", "player_id"], how="left")
    r["played"] = r.mp.fillna(0) > 0

    # scoring tier = the player's season points per game before that night
    pl = box[box.mp > 0].sort_values("game_date").copy()
    pl["ppg"] = pl.groupby(["player_id", "season"]).points.transform(lambda s: s.expanding().mean())
    last = pl[["player_id", "season", "game_date", "ppg"]].rename(columns={"game_date": "gd"}).sort_values("gd")
    r = pd.merge_asof(r.sort_values("game_date"), last, left_on="game_date", right_on="gd",
                      by=["player_id", "season"], allow_exact_matches=False)

    by_status = r.groupby("status").played.agg(["count", "mean"])
    qs = r[r.status == "Questionable"].copy()
    qs["tier"] = pd.cut(qs.ppg, [-1, 8, 15, 20, 99], labels=["Under 8", "8 to 15", "15 to 20", "20 or more"])
    by_tier = qs.groupby("tier", observed=True).played.agg(["count", "mean"])
    by_season = qs.groupby("season").played.agg(["count", "mean"])
    rows = lambda d: [dict(label=str(k), n=int(v["count"]), played=r1(100 * v["mean"])) for k, v in d.iterrows()]
    order = ["Probable", "Questionable", "Doubtful", "Out"]
    return dict(status=sorted(rows(by_status), key=lambda x: order.index(x["label"]) if x["label"] in order else 9),
                tier=rows(by_tier), season=rows(by_season))
# --- end questionable


# --- elo
def elo(q):
    """Replay every final game through TeamStrength and keep each team's rating path."""
    from data.ingestion.team_strength import TeamStrength, ELO_START_DATE, ELO_PER_POINT
    games = q("""SELECT game_date, season, home_team_id, away_team_id, home_score, away_score
                 FROM games WHERE is_final AND home_score IS NOT NULL
                   AND game_date >= :start AND home_team_id <> away_team_id
                 ORDER BY game_date, game_id""", start=ELO_START_DATE)
    abbr = dict(q("SELECT team_id, abbreviation FROM teams").itertuples(index=False))
    ts, path, pairs = TeamStrength(), {}, []
    for g in games.itertuples(index=False):
        f = ts.features(g.home_team_id, g.away_team_id, g.season)        # pregame state
        pairs.append((f["elo_diff"], g.home_score - g.away_score))
        ts.update(g.home_team_id, g.away_team_id, g.season, g.home_score, g.away_score)
        if g.season == "2025-26":
            for t in (g.home_team_id, g.away_team_id):
                path.setdefault(abbr[t], []).append([str(g.game_date)[:10], round(ts.elo[t], 1)])

    # does the pregame Elo gap line up with the actual margin?
    d = pd.DataFrame(pairs, columns=["diff", "margin"])
    d["bin"] = pd.cut(d["diff"], np.arange(-450, 451, 75))
    bins = d.groupby("bin", observed=True).agg(n=("margin", "size"), diff=("diff", "mean"), margin=("margin", "mean"))
    bins = bins[bins.n >= 30]
    return dict(season="2025-26", paths=path, per_point=ELO_PER_POINT,
                calibration=[dict(diff=r1(b["diff"], 0), margin=r1(b.margin), n=int(b.n)) for _, b in bins.iterrows()])
# --- end elo


# --- home_court
def home_court(q):
    """Backtest games with a road favorite: forecast margin against the actual margin."""
    d = q("""SELECT g.season, p.home_score_predicted::float - p.away_score_predicted::float pred,
                    (g.home_score - g.away_score)::float actual
             FROM predictions p JOIN games g USING (game_id)
             WHERE p.model_version = 'wf-v2' AND g.is_final AND g.season_type = 'Regular Season'
               AND g.home_team_id <> g.away_team_id""")
    road = d[d.pred < 0].copy()
    road["road_pred"], road["road_actual"] = -road.pred, -road.actual
    road["gap"] = pd.cut(road.road_pred, [0, 4, 8, 99], labels=["0 to 4", "4 to 8", "8 or more"])

    def summ(g):
        err = g.road_actual - g.road_pred
        return dict(n=len(g), pred=r1(g.road_pred.mean()), actual=r1(g.road_actual.mean()),
                    diff=r1(err.mean()), se=r1(err.std() / np.sqrt(len(g))))
    return dict(by_season=[dict(label=s, **summ(g)) for s, g in road.groupby("season")],
                by_gap=[dict(label=str(k), **summ(g)) for k, g in road.groupby("gap", observed=True)])
# --- end home_court


# --- cup
def cup(q):
    """NBA Cup group games against regular games played in the same weeks."""
    from nba_api.stats.endpoints import scheduleleaguev2
    kinds = {}
    for season in ("2023-24", "2024-25", "2025-26"):
        path = os.path.join(CACHE, f"subtype_{season}.json")
        if not os.path.exists(path):
            df = scheduleleaguev2.ScheduleLeagueV2(season=season, league_id="00", timeout=60).get_data_frames()[0]
            os.makedirs(CACHE, exist_ok=True)
            json.dump({str(r.gameId): r.gameSubtype for r in df.itertuples()}, open(path, "w"))
        kinds.update(json.load(open(path)))

    g = q("""SELECT g.nba_game_id, g.game_id, g.game_date, g.season,
                    (g.home_score + g.away_score)::float total,
                    p.home_score_predicted::float - p.away_score_predicted::float pred,
                    (g.home_score - g.away_score)::float actual,
                    h.q1_points + h.q2_points + h.q3_points - a.q1_points - a.q2_points - a.q3_points lead3,
                    h.q4_points - a.q4_points q4
             FROM games g
             JOIN team_box_scores h ON h.game_id = g.game_id AND h.team_id = g.home_team_id
             JOIN team_box_scores a ON a.game_id = g.game_id AND a.team_id = g.away_team_id
             LEFT JOIN predictions p ON p.game_id = g.game_id AND p.model_version = 'wf-v2'
             WHERE g.is_final AND g.season_type = 'Regular Season'
               AND g.season IN ('2023-24','2024-25','2025-26')""")
    g["kind"] = g.nba_game_id.astype(str).map(kinds).fillna("")
    g["cup"] = g.kind.eq("in-season")
    # regular games inside each season's Cup group-stage window
    win = g[g.cup].groupby("season").game_date.agg(["min", "max"])
    g = g.join(win, on="season")
    same_weeks = g[(g.game_date >= g["min"]) & (g.game_date <= g["max"])]

    def summ(x):
        big = x[x.lead3.abs() >= 15]
        give_back = (-np.sign(big.lead3) * big.q4)          # leader's Q4 points given back
        return dict(n=len(x), total=r1(x.total.mean()), total_se=r1(x.total.std() / np.sqrt(len(x))),
                    abs_margin=r1(x.actual.abs().mean()), n_big=len(big), give_back=r1(give_back.mean()))
    return dict(cup=summ(same_weeks[same_weeks.cup]), regular=summ(same_weeks[~same_weeks.cup]),
                by_season=[dict(label=s, cup=r1(x[x.cup].total.mean()), regular=r1(x[~x.cup].total.mean()))
                           for s, x in same_weeks.groupby("season")])
# --- end cup


def main():
    with engine.connect() as conn:
        conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        conn.exec_driver_sql("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        if conn.exec_driver_sql("SHOW transaction_read_only").scalar() != "on":
            raise SystemExit("Refusing to run: connection is not read-only")
        q = lambda sql, **p: pd.read_sql(text(sql), conn, params=p)
        out = {}
        for name, fn in (("questionable", questionable), ("elo", elo), ("home_court", home_court), ("cup", cup)):
            print(f"  {name} ...")
            out[name] = fn(q)
        conn.rollback()
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"  wrote {os.path.relpath(OUT, ROOT)} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
