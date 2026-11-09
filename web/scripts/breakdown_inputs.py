"""
Raw-stat breakdowns: rebuild every model input on a game page from the raw
results and box scores it was made from, and keep a breakdown only when it
reproduces the exact value the model received.

    python web/scripts/breakdown_inputs.py      # after web/scripts/export_data.py

What each input is rebuilt from
-------------------------------
- Elo:             every final score since 2021-10-19, replayed with the production
                   rating code (data.ingestion.team_strength.TeamStrength). Shown as
                   last season's finish -> carried start -> this season's game-by-game
                   changes. Exact.
- Season margin:   points for/against this season, blended with last season's margin
                   for the first 20 games (same replay). Exact.
- Off/def rating:  the team's last 10 final games: points, FG, 3P, FT, turnovers,
                   offensive rebounds, opponent defensive rebounds. The rating is split
                   into four factors with one linear fit (analytics/breakdown.py);
                   the split sums exactly to rating minus league average, with an
                   explicit "other" remainder.
- Pace:            the same 10 games' pace, both teams.
- Missing points:  the team's rotation (production rule, build_features._season_rotation)
                   and the official injury report that reproduces the stored value.
- Rest days:       date of the previous final game.

Database access is read-only (SELECT only; our own queries run in READ ONLY
transactions). Output: web/src/data/breakdowns.json. Nothing else is written.
"""
import contextlib
import io
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
ROOT = os.path.dirname(WEB)
sys.path.insert(0, ROOT)
sys.path.insert(0, WEB)
os.chdir(ROOT)

import numpy as np
import pandas as pd

from data.storage.db import engine
from data.ingestion import team_strength as TS
from data.ingestion import build_features as BF
from data.ingestion.fetch_official_injuries import UNAVAILABLE
from analytics import breakdown as B
from analytics import narrative as N

OUT = os.path.join(WEB, "src", "data", "breakdowns.json")
WINDOW = 10
LEAGUE_MIN_ROWS = 200      # below this, the season-to-date league average uses last season
ET = "America/New_York"


def read(sql, **params):
    with engine.connect() as conn:
        conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        df = pd.read_sql(sql, conn, params=params or None)
        conn.rollback()
    return df


def r(x, nd=2):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


# ---------------------------------------------------------------- team-games
def load_team_games():
    tg = read("""
        SELECT t.game_id, g.nba_game_id, g.game_date, g.season, t.team_id, o.team_id AS opp_id,
               t.is_home, t.points, t.fgm, t.fga, t.fg3m, t.fg3a, t.ftm, t.fta, t.turnovers,
               t.offensive_rating::float AS ortg, o.offensive_rating::float AS drtg,
               t.pace::float AS pace, o.points AS opp_points
        FROM team_box_scores t
        JOIN team_box_scores o ON o.game_id = t.game_id AND o.team_id <> t.team_id
        JOIN games g ON g.game_id = t.game_id
        WHERE g.is_final = TRUE
    """)
    reb = read("""
        SELECT game_id, team_id, SUM(oreb) AS oreb, SUM(dreb) AS dreb
        FROM player_box_scores GROUP BY game_id, team_id
    """)
    tg = tg.merge(reb, on=["game_id", "team_id"], how="left")
    tg = tg.merge(reb.rename(columns={"team_id": "opp_id", "oreb": "opp_oreb", "dreb": "opp_dreb"}),
                  on=["game_id", "opp_id"], how="left")
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    f = B.four_factors(tg.fgm, tg.fga, tg.fg3m, tg.fta, tg.ftm, tg.turnovers, tg.oreb, tg.opp_dreb)
    for k in B.FACTORS:
        tg[k] = f[k]
    # Opponent's factors in the same game = this team's defensive factors.
    opp = tg[["game_id", "team_id"] + B.FACTORS + ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                                                   "turnovers", "oreb"]]
    opp = opp.rename(columns={c: f"d_{c}" for c in opp.columns if c not in ("game_id",)})
    tg = tg.merge(opp, left_on=["game_id", "opp_id"], right_on=["game_id", "d_team_id"], how="left")
    return tg.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)


# ---------------------------------------------------------------- Elo replay
def replay_strength():
    """Per-team events from the production TeamStrength code, in order."""
    ts = TS.TeamStrength()
    ev = defaultdict(list)
    games = TS._load_final_games()
    games["game_date"] = pd.to_datetime(games["game_date"])
    for g in games.itertuples(index=False):
        for tid in (g.home_team_id, g.away_team_id):
            if ts.elo_season.get(tid) != g.season:
                prev = ts.elo.get(tid)
                ts._roll_season(tid, g.season)
                ev[tid].append(dict(kind="roll", date=g.game_date, season=g.season, prev_end=prev,
                                    start=ts.elo[tid], last_margin=ts.last_season_margin.get(tid)))
        pre_h, pre_a = ts.elo[g.home_team_id], ts.elo[g.away_team_id]
        ts.update(g.home_team_id, g.away_team_id, g.season, g.home_score, g.away_score)
        for tid, opp, home, pf, pa, pre in ((g.home_team_id, g.away_team_id, True, g.home_score, g.away_score, pre_h),
                                            (g.away_team_id, g.home_team_id, False, g.away_score, g.home_score, pre_a)):
            ev[tid].append(dict(kind="game", date=g.game_date, season=g.season, opp=opp, home=home,
                                pf=int(pf), pa=int(pa), pre=pre, post=ts.elo[tid]))
    return ev


def strength_for(events, season, date, abbr):
    prior = [e for e in events if e["date"] < date]
    games = [e for e in prior if e["kind"] == "game"]
    this = [e for e in games if e["season"] == season]
    rolls = [e for e in prior if e["kind"] == "roll" and e["season"] == season]
    if rolls:
        roll = rolls[-1]
        prev_end, start, last_margin = roll["prev_end"], roll["start"], roll["last_margin"]
    else:
        # Team hasn't played this season yet: apply the same carry-over now.
        prev_end = games[-1]["post"] if games else None
        start = TS.ELO_CARRY * (prev_end - 1500) + 1500 if prev_end is not None else 1500.0
        last = [e for e in games if games and e["season"] == games[-1]["season"]]
        last_margin = float(np.mean([e["pf"] - e["pa"] for e in last])) if last else None
    elo = this[-1]["post"] if this else start
    n = len(this)
    cur = float(np.mean([e["pf"] - e["pa"] for e in this])) if n else 0.0
    w = min(n, TS.BLEND_GAMES) / TS.BLEND_GAMES
    margin = w * cur + (1 - w) * (last_margin or 0.0)
    recent = this[-WINDOW:]
    earlier = this[:-WINDOW] if n > WINDOW else []
    return dict(
        elo=dict(v=elo, prev_end=r(prev_end, 1), start=r(start, 1), carry=TS.ELO_CARRY,
                 earlier=dict(n=len(earlier), sum=r(sum(e["post"] - e["pre"] for e in earlier), 1)),
                 games=[dict(d=e["date"].strftime("%Y-%m-%d"), o=abbr[e["opp"]], h=e["home"], pf=e["pf"],
                             pa=e["pa"], dl=r(e["post"] - e["pre"], 1)) for e in recent]),
        margin=dict(v=margin, n=n, w_n=sum(e["pf"] > e["pa"] for e in this), pf=r(np.mean([e["pf"] for e in this]) if n else None, 1),
                    pa=r(np.mean([e["pa"] for e in this]) if n else None, 1), cur=r(cur, 2),
                    last=r(last_margin, 2), w=r(w, 3), full=TS.BLEND_GAMES),
    )


# ---------------------------------------------------------------- availability
def missing_candidates(fetches, entries, date, at_et):
    """Unavailable player-id sets to try, most likely first: reports on game day at or
    before the forecast time (latest first), then later reports that day."""
    day = fetches[fetches.report_date == date.date()].sort_values("report_time", ascending=False)
    if day.empty:
        return []
    before = day[day.report_time <= at_et] if at_et else day
    order = list(before.itertuples()) + [f for f in day.itertuples() if f.fetch_id not in set(before.fetch_id)]
    out = []
    for f in order:
        e = entries.get((f.fetch_id, date.date()))
        out.append((f"{date:%Y-%m-%d} {f.report_time} ET", e if e is not None else {}))
    return out


def main():
    with open(os.path.join(WEB, "src", "data", "games.json"), encoding="utf8") as fh:
        page_games = [g for g in json.load(fh) if g.get("inputs")]
    teams = read("SELECT team_id, abbreviation FROM teams")
    abbr = dict(zip(teams.team_id, teams.abbreviation))
    tid_of = {a: t for t, a in abbr.items()}

    print("Loading team box scores...")
    tg = load_team_games()
    fit = B.fit_rating(tg[B.FACTORS].values, tg.ortg.values)
    print(f"  four-factor fit: R^2 {fit['r2']:.3f}, RMSE {fit['rmse']:.2f} over {fit['n']} team-games")
    coef = fit["coef"]
    by_team = {t: d for t, d in tg.groupby("team_id")}

    def league(season, date):
        ref = tg[(tg.season == season) & (tg.game_date < date)]
        label = f"{season} to date"
        if len(ref) < LEAGUE_MIN_ROWS:
            prev = sorted(s for s in tg.season.unique() if s < season)
            if prev:
                ref = tg[tg.season == prev[-1]]
                label = prev[-1]
        # Each average over the rows where that stat exists (OREB is missing for many 2024-26 games).
        ref = ref.dropna(subset=["ortg", "efg", "tov", "ftr"])
        f = {k: float(ref[k].mean()) for k in B.FACTORS}
        if not np.isfinite(f["orb"]):   # no OREB recorded in this stretch at all
            f["orb"] = float(tg.orb.mean())
        return dict(label=label, ortg=float(ref.ortg.mean()), f=f, pace=float(ref.pace.mean()))

    print("Replaying Elo...")
    ev = replay_strength()

    print("Loading rotations and injury reports...")
    with contextlib.redirect_stdout(io.StringIO()):
        box = BF._load_played_box_scores()
    last_team_by_date = box.sort_values("game_date")[["player_id", "game_date", "team_id"]]
    box_by_team = {t: d for t, d in box.groupby("team_id")}
    played = box.groupby(["game_id", "team_id"])["player_id"].apply(set).to_dict()
    fetches = read("""SELECT fetch_id, report_date, report_time FROM injury_report_fetches
                      WHERE http_status = 200""")
    ent = read("""SELECT fetch_id, game_date, team_id, player_id, status FROM injury_report_entries
                  WHERE player_id IS NOT NULL""")
    ent = ent[ent.status.isin(UNAVAILABLE)]
    entries = {}
    for (fid, d), grp in ent.groupby(["fetch_id", "game_date"]):
        entries[(fid, d)] = grp.groupby("team_id")["player_id"].apply(set).to_dict()
    sched = pd.concat([tg[["team_id", "game_date", "opp_id"]]]).drop_duplicates()
    finals = read("""SELECT game_id, nba_game_id, game_date, home_team_id, away_team_id FROM games
                     WHERE is_final = TRUE""")
    finals["game_date"] = pd.to_datetime(finals["game_date"])
    sched = pd.concat([
        finals.rename(columns={"home_team_id": "team_id", "away_team_id": "opp_id"})[["team_id", "opp_id", "game_date"]],
        finals.rename(columns={"away_team_id": "team_id", "home_team_id": "opp_id"})[["team_id", "opp_id", "game_date"]],
    ]).sort_values("game_date")
    sched_by_team = {t: d for t, d in sched.groupby("team_id")}
    gid_of = dict(zip(finals.nba_game_id, finals.game_id))

    out, stats = {}, defaultdict(lambda: [0, 0])
    for g in page_games:
        date = pd.Timestamp(g["date"])
        at_et = None
        if g.get("fc", {}).get("at"):
            at_et = pd.Timestamp(g["fc"]["at"]).tz_convert(ET).strftime("%H:%M")
        lg = league(g["season"], date)
        rec = {"lg": dict(label=lg["label"], ortg=r(lg["ortg"]), f={k: r(v, 4) for k, v in lg["f"].items()},
                          pace=r(lg["pace"], 1))}
        pace_sum, pace_ok = 0.0, True
        for side in ("home", "away"):
            t = tid_of[g[side]]
            inp = g["inputs"][side]
            s = {}
            # Elo + season margin
            st = strength_for(ev.get(t, []), g["season"], date, abbr)
            st["elo"]["ok"] = B.matches(st["elo"]["v"], inp.get("elo"), B.TOL["elo"])
            st["margin"]["ok"] = B.matches(st["margin"]["v"], inp.get("season_margin"), B.TOL["season_margin"])
            st["elo"]["v"], st["margin"]["v"] = r(st["elo"]["v"], 2), r(st["margin"]["v"], 2)
            s.update(st)
            # Last-10 ratings
            hist = by_team.get(t)
            win = hist[hist.game_date < date].tail(WINDOW) if hist is not None else pd.DataFrame()
            if len(win):
                o_r, d_r, p = float(win.ortg.mean()), float(win.drtg.mean()), float(win.pace.mean())
                fo = {k: float(win[k].mean()) for k in B.FACTORS}
                fd = {k: float(win[f"d_{k}"].mean()) for k in B.FACTORS}
                core = [k for k in B.FACTORS if k != "orb"]
                complete = bool(np.isfinite([fo[k] for k in core] + [fd[k] for k in core]).all())
                # Offensive rebounds are missing for many 2024-26 games. If any game in the
                # window lacks them, rebounding isn't split out: its effect stays in 'other'.
                orb_missing = int(win.orb.isna().sum()), int(win.d_orb.isna().sum())
                if orb_missing[0]:
                    fo["orb"] = lg["f"]["orb"]
                if orb_missing[1]:
                    fd["orb"] = lg["f"]["orb"]
                s["games"] = [dict(
                    d=x.game_date.strftime("%Y-%m-%d"), o=abbr[x.opp_id], h=bool(x.is_home),
                    pf=int(x.points), pa=int(x.opp_points), or_=r(x.ortg, 1), dr=r(x.drtg, 1), pc=r(x.pace, 1),
                    s=[int(x.fgm), int(x.fga), int(x.fg3m), int(x.fg3a), int(x.ftm), int(x.fta), int(x.turnovers),
                       None if pd.isna(x.oreb) else int(x.oreb)],
                    ds=[int(x.d_fgm), int(x.d_fga), int(x.d_fg3m), int(x.d_fg3a), int(x.d_ftm), int(x.d_fta),
                        int(x.d_turnovers), None if pd.isna(x.d_oreb) else int(x.d_oreb)],
                ) for x in win.itertuples()]
                s["off"] = dict(v=r(o_r), ok=B.matches(o_r, inp.get("off_rating"), B.TOL["rating"]) and complete,
                                f={k: (None if k == "orb" and orb_missing[0] else r(v, 4)) for k, v in fo.items()},
                                orb_missing=orb_missing[0],
                                parts={k: r(v, 2) for k, v in B.split_rating(o_r, fo, lg["ortg"], lg["f"], coef).items()} if complete else None)
                s["def"] = dict(v=r(d_r), ok=B.matches(d_r, inp.get("def_rating"), B.TOL["rating"]) and complete,
                                f={k: (None if k == "orb" and orb_missing[1] else r(v, 4)) for k, v in fd.items()},
                                orb_missing=orb_missing[1],
                                parts={k: r(v, 2) for k, v in B.split_rating(d_r, fd, lg["ortg"], lg["f"], coef).items()} if complete else None)
                s["pace"] = r(p, 2)
                pace_sum += p
            else:
                pace_ok = False
                s["off"] = dict(ok=False)
                s["def"] = dict(ok=False)
            # Rest
            sc = sched_by_team.get(t)
            prev = sc[sc.game_date < date].tail(1) if sc is not None else sc
            if prev is not None and len(prev):
                days = int((date - prev.game_date.iloc[0]).days)
                rest = max(0, min(BF.REST_CAP, days))
                s["rest"] = dict(v=rest, days=days, last=prev.game_date.iloc[0].strftime("%Y-%m-%d"),
                                 o=abbr[prev.opp_id.iloc[0]], cap=BF.REST_CAP,
                                 ok=B.matches(rest, inp.get("rest_days"), 0))
            else:
                s["rest"] = dict(v=3, days=None, last=None, cap=BF.REST_CAP, ok=B.matches(3, inp.get("rest_days"), 0))
            # Missing rotation points
            tb = box_by_team.get(t)
            target = inp.get("missing_pts")
            miss = dict(ok=False, v=target)
            if tb is not None and target is not None:
                rot = BF._season_rotation(tb, g["season"], date, t, last_team_by_date)
                cands = missing_candidates(fetches, entries, date, at_et)
                gid = gid_of.get(g["id"])
                if gid is not None and (gid, t) in played:
                    cands.append(("box score (players who did not play)", {t: set(rot.index) - played[(gid, t)]}))
                if not cands:
                    cands.append(("no report stored", {}))
                for src, unav in cands:
                    absent = rot[rot.index.isin(unav.get(t, set()))]
                    total = float(absent["pts"].sum())
                    if B.matches(total, target, B.TOL["missing_pts"]):
                        miss = dict(ok=True, v=r(total), src=src, rot=int(len(rot)),
                                    players=[dict(n=x.full_name, ppg=r(x.pts, 1), mpg=r(float(x.mins), 1), gp=int(x.gp))
                                             for x in absent.sort_values("pts", ascending=False).itertuples()])
                        break
            s["missing"] = miss
            for k in ("elo", "margin", "off", "def", "rest", "missing"):
                stats[k][0] += 1
                stats[k][1] += bool(s[k].get("ok"))
            s["why"] = N.team_story(g[side], s, rec["lg"])
            rec[side] = s
        # combined_pace is the average of the two teams' 10-game pace (build_features).
        rec["pace_ok"] = pace_ok and B.matches(pace_sum / 2, g["inputs"].get("combined_pace"), B.TOL["pace"])
        stats["pace"][0] += 1
        stats["pace"][1] += rec["pace_ok"]
        out[g["id"]] = rec

    # Which raw stats move ratings most: spread (standard deviation) of each factor's
    # part across every 10-game window on a game page, in rating points. Rebounding
    # uses only windows where it was recorded.
    spread = {}
    for kind in ("off", "def"):
        parts = defaultdict(list)
        for rec in out.values():
            for side in ("home", "away"):
                x = rec[side].get(kind) or {}
                if not x.get("ok") or not x.get("parts"):
                    continue
                for k in B.FACTORS + ["other"]:
                    if k == "orb" and x.get("orb_missing"):
                        continue
                    if x["parts"][k] is not None:
                        parts[k].append(x["parts"][k])
        spread[kind] = {k: dict(sd=r(np.std(v), 2), n=len(v)) for k, v in parts.items()}
    sd_f = {k: r(tg[k].std(), 4) for k in B.FACTORS}

    payload = dict(
        version=B.BREAKDOWN_VERSION, window=WINDOW, spread=spread, factor_sd=sd_f,
        fit=dict(intercept=r(fit["intercept"], 3), coef={k: r(v, 3) for k, v in coef.items()},
                 r2=r(fit["r2"], 4), rmse=r(fit["rmse"], 3), n=fit["n"]),
        labels=B.FACTOR_LABELS, factors=B.FACTORS,
        stats={k: dict(n=v[0], ok=v[1]) for k, v in stats.items()},
        games=out,
    )
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print("  reproduced exactly (ok / checked):")
    for k, v in payload["stats"].items():
        print(f"    {k:8s} {v['ok']:4d} / {v['n']}")
    print(f"  wrote {OUT} ({os.path.getsize(OUT) // 1024} KB, {len(out)} games)")


if __name__ == "__main__":
    main()
