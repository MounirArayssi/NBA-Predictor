"""
Read-only data export for The Hoop Ledger website (web/).

    python web/scripts/export_data.py                 # export from the local database
    python web/scripts/export_data.py --refresh-schedule   # also re-download tip-off times

What it does
------------
- Opens ONE PostgreSQL connection, sets it to read-only (session default and the
  open transaction), and runs SELECT statements only. Nothing is written to the
  database, and no prediction / training / ingestion code is imported or called.
- Optionally downloads the public NBA season schedule (ScheduleLeagueV2) to learn
  tip-off times, cached under web/.cache/. This is used only to check whether a
  stored forecast was saved before tip-off. It never touches the database.
- Writes public-safe JSON:
    web/src/data/*.json          read at build time by Astro
    web/public/data/archive-*.json   fetched by the archive page in the browser

One forecast per game (single combined record)
----------------------------------------------
Each game gets exactly one forecast, and every record on the site is computed
from that one set:
  1. the issued forecast (is_official = TRUE, prediction_type <> 'backfill') if
     the game has one, whatever its timing;
  2. otherwise the walk-forward forecast (model_version = 'wf-v2': each season
     predicted by models trained only on earlier seasons).
Issued rows still get a timing check (verified / unverified / after_tip), kept
only as per-game metadata. prediction_type = 'backfill' rows (an older, possibly
leaky after-the-fact fill) are not exported at all; only their count is reported.
"""
import argparse
import json
import math
import re
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # web/ for the analytics package
WEB = os.path.dirname(HERE)
ROOT = os.path.dirname(WEB)
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sqlalchemy import text

from data.storage.db import engine  # creates an engine only; no writes on import
from analytics import metrics as AM

ET = ZoneInfo("America/New_York")
SRC_DATA = os.path.join(WEB, "src", "data")
PUB_DATA = os.path.join(WEB, "public", "data")
CACHE = os.path.join(WEB, ".cache")
WF = "wf-v2"
SMALL_SAMPLE = 20

# League structure is not stored in the database (teams.conference is empty), so it
# is supplied here. Stable public fact; edit if the league realigns.
EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK",
        "ORL", "PHI", "TOR", "WAS"}

WARNINGS = []


def warn(msg):
    WARNINGS.append(msg)
    print("  WARNING:", msg)


# ----------------------------------------------------------------- helpers
def fnum(x, nd=2):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd)


def iso_utc(ts):
    """Naive-UTC database timestamp -> ISO 8601 string with Z, or None."""
    if ts is None or (isinstance(ts, float) and math.isnan(ts)) or pd.isna(ts):
        return None
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(name):
    return "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-").replace("--", "-")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"), default=str, allow_nan=False)
    print(f"  wrote {os.path.relpath(path, ROOT)} ({os.path.getsize(path)/1024:.0f} KB)")


# ----------------------------------------------------------------- schedule cache
def load_schedule(season, refresh):
    """{nba_game_id: tip-off UTC ISO} from the public NBA schedule, cached on disk."""
    path = os.path.join(CACHE, f"schedule_{season}.json")
    if os.path.exists(path) and not refresh:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    try:
        from nba_api.stats.endpoints import scheduleleaguev2
        df = scheduleleaguev2.ScheduleLeagueV2(season=season, league_id="00", timeout=60).get_data_frames()[0]
        out = {str(r.gameId): r.gameDateTimeUTC for r in df.itertuples() if r.gameDateTimeUTC}
        os.makedirs(CACHE, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
        print(f"  schedule {season}: {len(out)} tip-off times cached")
        return out
    except Exception as e:  # network trouble: fall back to whatever is cached
        warn(f"could not download {season} schedule ({e}); tip-off checks use stored times only")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        return {}


# ----------------------------------------------------------------- main export
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-schedule", action="store_true")
    args = ap.parse_args()

    from config.settings import CURRENT_SEASON
    generated = datetime.now(timezone.utc)
    print(f"Export started {generated.isoformat()}  (season {CURRENT_SEASON})")

    with engine.connect() as conn:
        # Read-only guard: session default + the transaction opened by the first statement.
        conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        conn.exec_driver_sql("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        ro = conn.exec_driver_sql("SHOW transaction_read_only").scalar()
        if ro != "on":
            raise SystemExit("Refusing to export: connection is not read-only")

        def q(sql, **params):
            return pd.read_sql(text(sql), conn, params=params)

        teams = q("SELECT team_id, abbreviation abbr, full_name name, city FROM teams ORDER BY abbreviation")
        games = q("""
            SELECT g.game_id, g.nba_game_id, g.season, g.season_type, g.game_date, g.tipoff_time_utc,
                   g.home_team_id, g.away_team_id, g.home_score, g.away_score, g.is_final, g.status
            FROM games g WHERE g.season_type IN ('Regular Season','Playoffs')
        """)
        preds = q("""
            SELECT p.prediction_id, p.game_id, p.model_version, p.prediction_type, p.is_official,
                   p.predicted_at, p.lock_reason,
                   p.home_score_predicted::float ph, p.away_score_predicted::float pa,
                   p.home_win_probability::float hwp, p.confidence_score::float conf,
                   p.winner_correct, p.feature_vector, p.vegas_spread::float vspread,
                   p.vegas_total::float vtotal
            FROM predictions p
            WHERE p.model_version = :wf OR (p.is_official AND p.prediction_type <> 'backfill')
        """, wf=WF)
        n_backfill = int(q("SELECT COUNT(*) n FROM predictions WHERE prediction_type = 'backfill'").n[0])
        quarters = q("""SELECT game_id, team_id, q1_points, q2_points, q3_points, q4_points, points,
                               fgm, fga, fg3m, fg3a, ftm, fta, rebounds, assists, turnovers, steals, blocks,
                               paint_points, fastbreak_points, second_chance_pts, pace::float pace
                        FROM team_box_scores""")
        rolling = q("""
            SELECT DISTINCT ON (team_id) team_id, as_of_date, avg_offensive_rating::float ortg,
                   avg_defensive_rating::float drtg, avg_pace::float pace, games_counted
            FROM team_rolling_stats WHERE "window" = 10 AND season = :s
            ORDER BY team_id, as_of_date DESC
        """, s=CURRENT_SEASON)
        inj_fetch = q("""SELECT fetch_id, report_date, report_time FROM injury_report_fetches
                         WHERE http_status = 200""")
        coverage = q("""
            SELECT g.season,
                   COUNT(DISTINCT g.game_id) FILTER (WHERE g.is_final AND g.season_type IN ('Regular Season','Playoffs')) games,
                   COUNT(DISTINCT tb.game_id) box,
                   COUNT(DISTINCT f.report_date) reports
            FROM games g
            LEFT JOIN team_box_scores tb ON tb.game_id = g.game_id
            LEFT JOIN injury_report_fetches f ON f.report_date = g.game_date AND f.http_status = 200
            GROUP BY g.season ORDER BY g.season
        """)
        model_span = q("""SELECT MIN(game_date) lo, MAX(game_date) hi FROM games
                          WHERE is_final AND season_type IN ('Regular Season','Playoffs')""").iloc[0]

        # ------------------------------------------------------------ teams
        tid2abbr = dict(zip(teams.team_id, teams.abbr))
        team_list = []
        for t in teams.itertuples():
            team_list.append(dict(id=int(t.team_id), abbr=t.abbr, name=t.name, city=t.city,
                                  slug=slugify(t.name), conference="East" if t.abbr in EAST else "West"))
        if len(team_list) != 30:
            warn(f"expected 30 teams, found {len(team_list)}")

        # ------------------------------------------------------------ games
        games["home"] = games.home_team_id.map(tid2abbr)
        games["away"] = games.away_team_id.map(tid2abbr)
        if games.nba_game_id.duplicated().any():
            raise SystemExit("Duplicate nba_game_id values in games table")
        bad = games[(games.home_team_id == games.away_team_id) | games.home.isna() | games.away.isna()]
        if len(bad):
            # Data-quality issue in the source table (same team stored as home and away).
            # These games cannot be oriented, so they are left out of every population.
            warn(f"{len(bad)} games have the same team as home and away and are excluded: "
                 + ", ".join(sorted(bad.nba_game_id)))
            games = games.drop(bad.index)
        n_bad_orientation = int(len(bad))

        seasons_needing_tips = sorted(set(games[games.game_id.isin(
            preds[preds.is_official].game_id)].season) | {CURRENT_SEASON})
        sched = {}
        for s in seasons_needing_tips:
            sched.update(load_schedule(s, args.refresh_schedule))

        def tip_for(row):
            if row.tipoff_time_utc is not None and not pd.isna(row.tipoff_time_utc):
                return iso_utc(row.tipoff_time_utc)
            v = sched.get(str(row.nba_game_id))
            return v.replace(".000", "") if v else None

        games["tip"] = [tip_for(r) for r in games.itertuples()]
        games["final"] = games.is_final.fillna(False).astype(bool) & games.home_score.notna() & games.away_score.notna()
        gby = games.set_index("game_id")

        # ------------------------------------------------------------ predictions
        for col in ("ph", "pa", "hwp"):
            v = preds[col]
            if v.isna().any() or (~np.isfinite(v.astype(float))).any():
                warn(f"{int(v.isna().sum())} predictions missing {col}; those rows are dropped")
        preds = preds.dropna(subset=["ph", "pa", "hwp"])
        if ((preds.hwp < 0) | (preds.hwp > 1)).any():
            raise SystemExit("Win probability outside [0, 1]")
        preds = preds[preds.game_id.isin(gby.index)]

        backtest = preds[preds.model_version == WF].copy()
        issued = preds[(preds.model_version != WF) & preds.is_official].copy()
        if backtest.game_id.duplicated().any() or issued.game_id.duplicated().any():
            raise SystemExit("More than one prediction per game in a population; selection is ambiguous")

        def verification(r):
            if str(r.model_version).startswith("csv"):
                return "unverified", "Timestamp came from a CSV import"
            tip = gby.at[r.game_id, "tip"]
            if not tip:
                return "unverified", "Tip-off time not recorded"
            at = pd.Timestamp(r.predicted_at).tz_localize("UTC")
            if at < pd.Timestamp(tip):
                return "verified", "Saved before scheduled tip-off"
            return "after_tip", "Saved at or after tip-off"

        ver = [verification(r) for r in issued.itertuples()]
        issued["verification"] = [v[0] for v in ver]
        issued["verification_note"] = [v[1] for v in ver]
        vcounts = issued.verification.value_counts().to_dict()
        print(f"  issued rows: {len(issued)} {vcounts}; backtest rows: {len(backtest)}; backfill rows excluded: {n_backfill}")

        # One forecast per game: issued where it exists, walk-forward otherwise.
        issued["origin"] = "issued"
        bt_only = backtest[~backtest.game_id.isin(issued.game_id)].copy()
        bt_only["origin"] = "walkforward"
        bt_only["verification"] = None
        bt_only["verification_note"] = None
        fc = pd.concat([issued, bt_only], ignore_index=True)
        print(f"  combined forecasts: {len(fc)} ({len(issued)} issued + {len(bt_only)} walk-forward)")

        def pred_obj(r, gid):
            g = gby.loc[gid]
            o = dict(ph=fnum(r.ph, 1), pa=fnum(r.pa, 1), hwp=fnum(r.hwp, 4), at=iso_utc(r.predicted_at),
                     origin=r.origin)
            if r.origin == "issued":
                o.update(verification=r.verification, conf=fnum(r.conf, 2))
            if g.final:
                pm, am = r.ph - r.pa, float(g.home_score - g.away_score)
                o.update(correct=bool((pm > 0) == (am > 0)), margin_err=fnum(am - pm, 1),
                         total_err=fnum((g.home_score + g.away_score) - (r.ph + r.pa), 1),
                         home_err=fnum(g.home_score - r.ph, 1), away_err=fnum(g.away_score - r.pa, 1))
            return o

        # ------------------------------------------------------------ per-game context
        finals = games[games.final].sort_values("game_date")
        team_games = {}  # team abbr -> list of (date, game row) for finals
        for r in finals.itertuples():
            for side in ("home", "away"):
                team_games.setdefault(getattr(r, side), []).append(r)

        def form(abbr, before_date, n=5):
            out = []
            for r in reversed(team_games.get(abbr, [])):
                if r.game_date >= before_date:
                    continue
                home = r.home == abbr
                ts, os_ = (r.home_score, r.away_score) if home else (r.away_score, r.home_score)
                out.append(dict(id=r.nba_game_id, date=str(r.game_date), opp=r.away if home else r.home,
                                home=home, pts=int(ts), opp_pts=int(os_), won=bool(ts > os_)))
                if len(out) == n:
                    break
            return out

        def h2h(a, b, before_date, n=5):
            out = []
            for r in reversed(team_games.get(a, [])):
                if r.game_date >= before_date or {r.home, r.away} != {a, b}:
                    continue
                out.append(dict(id=r.nba_game_id, date=str(r.game_date), season=r.season, home=r.home,
                                away=r.away, hs=int(r.home_score), as_=int(r.away_score)))
                if len(out) == n:
                    break
            return out

        qmap = {}
        BOX_KEYS = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "rebounds", "assists", "turnovers"]
        BOX_OPTIONAL = ["steals", "blocks", "paint_points", "fastbreak_points", "second_chance_pts"]
        isnum = lambda v: v is not None and not pd.isna(v)
        boxmap = {}
        for r in quarters.itertuples():
            vals = {k: getattr(r, k) for k in BOX_KEYS}
            if all(isnum(v) for v in vals.values()):
                boxmap[(r.game_id, r.team_id)] = {**{k: int(v) for k, v in vals.items()},
                                                  **{k: int(getattr(r, k)) if isnum(getattr(r, k)) else None for k in BOX_OPTIONAL}}
        for r in quarters.itertuples():
            qs = [r.q1_points, r.q2_points, r.q3_points, r.q4_points]
            if all(x is not None and not pd.isna(x) for x in qs):
                qmap[(r.game_id, r.team_id)] = dict(q=[int(x) for x in qs], total=int(r.points))

        def injuries_for(game_row, predicted_at):
            """Out/Doubtful/Questionable from the latest stored report on game day at or
            before the forecast time (ET); training-time report if none."""
            d = game_row.game_date
            day = inj_fetch[inj_fetch.report_date == d]
            if day.empty:
                return None
            pick = day
            if predicted_at is not None:
                at_et = pd.Timestamp(predicted_at).tz_localize("UTC").tz_convert(ET).strftime("%H:%M")
                earlier = day[day.report_time <= at_et]
                pick = earlier if not earlier.empty else day
            f = pick.sort_values("report_time").iloc[-1]
            rows = q("""SELECT e.team_id, e.player_name, e.status, e.reason FROM injury_report_entries e
                        WHERE e.fetch_id = :f AND e.game_date = :d
                          AND e.team_id IN (:h, :a) AND e.status IN ('Out','Doubtful','Questionable')
                        ORDER BY e.status, e.player_name""",
                     f=int(f.fetch_id), d=d, h=int(game_row.home_team_id), a=int(game_row.away_team_id))
            def side(tid):
                out = []
                for e in rows[rows.team_id == tid].itertuples():
                    reason = e.reason.strip() if isinstance(e.reason, str) else ""
                    if reason.lower().startswith("gleague"):
                        continue  # two-way / G League assignments, not injuries
                    reason = reason.replace("Injury/Illness-", "").replace("Injury/Illness - ", "")
                    # The PDF parser drops spaces ("LeftHamstringStrain"); restore them for display.
                    reason = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", reason).replace("/", " / ").strip(" /")
                    out.append(dict(name=e.player_name, status=e.status, reason=reason or None))
                return out
            return dict(report=f"{d} {f.report_time} ET", home=side(game_row.home_team_id),
                        away=side(game_row.away_team_id))

        INPUT_KEYS = ["elo", "season_margin", "off_rating", "def_rating", "missing_pts", "rest_days",
                      "back_to_back", "games_played"]

        def inputs_from(fv):
            if not isinstance(fv, dict):
                return None
            o = {"home": {}, "away": {}}
            for k in INPUT_KEYS:
                for side in ("home", "away"):
                    o[side][k] = fnum(fv.get(f"{side}_{k}"), 2)
            o["combined_pace"] = fnum(fv.get("combined_pace"), 1)
            o["elo_diff"] = fnum(fv.get("elo_diff"), 1)
            o["series"] = dict(home_wins=fnum(fv.get("home_series_wins"), 0),
                               elimination=bool(fv.get("is_elimination") or 0),
                               pressure=fnum(fv.get("series_pressure"), 0))
            return o

        # ------------------------------------------------------------ graded rows
        def archive_rows(df):
            rows = []
            for r in df.itertuples():
                g = gby.loc[r.game_id]
                if not g.final:
                    continue
                pm = r.ph - r.pa
                am = float(g.home_score - g.away_score)
                row = dict(id=str(g.nba_game_id), d=str(g.game_date), s=g.season,
                           po=g.season_type == "Playoffs", h=g.home, a=g.away,
                           ph=fnum(r.ph, 0), pa=fnum(r.pa, 0), p=fnum(r.hwp, 3),
                           hs=int(g.home_score), as_=int(g.away_score),
                           ok=bool((pm > 0) == (am > 0)), me=fnum(am - pm, 1))
                rows.append(AM.annotate(row))    # + w, pw, su, se (surprise fields)
            AM.dedupe(rows)
            return sorted(rows, key=lambda x: (x["d"], x["id"]), reverse=True)

        fc_rows = archive_rows(fc)
        nid2gid = {str(gby.at[gid, "nba_game_id"]): gid for gid in fc.game_id}

        # Game pages: the current season, any game with an issued prediction, and the
        # games the surprise features link to (top per season, top per team).
        SURPRISE_PAGES_PER_SEASON = 40
        surprise_ids = set()
        for season_ in sorted({r["s"] for r in fc_rows}):
            srows = [r for r in fc_rows if r["s"] == season_]
            surprise_ids |= {r["id"] for r in AM.rank_outcome(srows, SURPRISE_PAGES_PER_SEASON)}
            surprise_ids |= {r["id"] for r in AM.rank_scoreline(srows, SURPRISE_PAGES_PER_SEASON)}
        team_surprise = {t["abbr"]: AM.team_surprises(fc_rows, t["abbr"], k=5) for t in team_list}
        for v in team_surprise.values():
            for lst in v.values():
                surprise_ids |= {r["id"] for r in lst}
        page_ids = (set(games[games.season == CURRENT_SEASON].game_id) | set(issued.game_id)
                    | {nid2gid[i] for i in surprise_ids if i in nid2gid})
        iss_by = issued.set_index("game_id")
        fc_by = fc.set_index("game_id")
        game_pages = []
        for gid in sorted(page_ids, key=lambda i: (gby.at[i, "game_date"], gby.at[i, "nba_game_id"])):
            g = gby.loc[gid]
            today_et = datetime.now(ET).date()
            if g.final:
                status = "final"
            elif g.game_date >= today_et:
                status = "scheduled"
            else:
                status = "unknown"  # past date, no final score stored
            fp = pred_obj(fc_by.loc[gid], gid) if gid in fc_by.index else None
            fv_src = fc_by.at[gid, "feature_vector"] if gid in fc_by.index else None
            rec = dict(
                id=str(g.nba_game_id), date=str(g.game_date), tip=g.tip if isinstance(g.tip, str) and g.tip else None, season=g.season,
                stype=g.season_type, home=g.home, away=g.away, status=status,
                score=dict(h=int(g.home_score), a=int(g.away_score)) if g.final else None,
                quarters=dict(h=qmap.get((gid, g.home_team_id)), a=qmap.get((gid, g.away_team_id)))
                if g.final else None,
                box=dict(h=boxmap.get((gid, g.home_team_id)), a=boxmap.get((gid, g.away_team_id)))
                if g.final and boxmap.get((gid, g.home_team_id)) and boxmap.get((gid, g.away_team_id)) else None,
                fc=fp, inputs=inputs_from(fv_src),
                injuries=injuries_for(g, iss_by.at[gid, "predicted_at"]) if gid in iss_by.index else None,
                form=dict(home=form(g.home, g.game_date), away=form(g.away, g.game_date)),
                h2h=h2h(g.home, g.away, g.game_date),
            )
            game_pages.append(rec)
        page_set = {r["id"] for r in game_pages}

        # ------------------------------------------------------------ archive rows
        for r in fc_rows:
            r["page"] = r["id"] in page_set
        for f in os.listdir(PUB_DATA):          # drop archive files from older layouts
            if f.startswith("archive-") and f.endswith(".json"):
                os.remove(os.path.join(PUB_DATA, f))
        for s in sorted({r["s"] for r in fc_rows}):
            write_json(os.path.join(PUB_DATA, f"archive-{s}.json"), [r for r in fc_rows if r["s"] == s])

        # ------------------------------------------------------------ performance
        def frame(df):
            d = df.join(gby[["home_score", "away_score", "season", "season_type", "game_date", "final", "home", "away"]],
                        on="game_id")
            d = d[d.final].copy()
            d["pm"] = d.ph - d.pa
            d["am"] = (d.home_score - d.away_score).astype(float)
            d["pt"] = d.ph + d.pa
            d["at"] = (d.home_score + d.away_score).astype(float)
            d["hw"] = (d.am > 0).astype(int)
            d["ok"] = ((d.pm > 0) == (d.am > 0)).astype(int)
            return d

        B = frame(fc)

        def summ(d):
            if len(d) == 0:
                return dict(n=0)
            return dict(n=int(len(d)), wins=int(d.ok.sum()), win=fnum(100 * d.ok.mean(), 1),
                        margin_mae=fnum((d.am - d.pm).abs().mean(), 2),
                        total_mae=fnum((d["at"] - d.pt).abs().mean(), 2),
                        team_mae=fnum(pd.concat([(d.home_score - d.ph).abs(), (d.away_score - d.pa).abs()]).mean(), 2),
                        brier=fnum(((d.hwp - d.hw) ** 2).mean(), 3),
                        home_win=fnum(100 * d.hw.mean(), 1))

        def hist(values, lo, hi, step):
            edges = list(np.arange(lo, hi + step, step))
            v = np.clip(values, lo, hi - 1e-9)
            counts, _ = np.histogram(v, bins=edges)
            return [dict(lo=fnum(edges[i], 1), hi=fnum(edges[i + 1], 1), n=int(c)) for i, c in enumerate(counts)]

        pick_p = np.maximum(B.hwp, 1 - B.hwp)
        bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0001]
        calib = []
        for (lo, hi) in zip(bins[:-1], bins[1:]):
            m = (pick_p >= lo) & (pick_p < hi)
            if m.sum():
                calib.append(dict(lo=lo, hi=min(hi, 1.0), n=int(m.sum()),
                                  predicted=fnum(100 * pick_p[m].mean(), 1),
                                  actual=fnum(100 * B.ok[m].mean(), 1)))

        def breakdown(d, key_fn, order):
            out = []
            keys = key_fn(d)
            for k in order:
                x = d[keys == k]
                if len(x):
                    out.append(dict(label=k, **summ(x)))
            return out

        def margin_bucket(d):
            a = d.pm.abs()
            return pd.cut(a, [-0.01, 3, 6, 10, 100], labels=["0–3", "3–6", "6–10", "10+"]).astype(str)

        def fav_side(d):
            return np.where(d.pm > 0, "Home favorite", "Road favorite")

        def stype(d):
            return d.season_type.values

        bucket_cal = []
        for lo, hi, lab in [(-100, -10, "Road by 10+"), (-10, -6, "Road by 6–10"), (-6, -3, "Road by 3–6"),
                            (-3, 0, "Road by 0–3"), (0, 3, "Home by 0–3"), (3, 6, "Home by 3–6"),
                            (6, 10, "Home by 6–10"), (10, 100, "Home by 10+")]:
            x = B[(B.pm > lo) & (B.pm <= hi)]
            if len(x):
                bucket_cal.append(dict(label=lab, n=int(len(x)), pred=fnum(x.pm.mean(), 1), actual=fnum(x.am.mean(), 1)))

        team_err = np.concatenate([(B.home_score - B.ph).abs().values, (B.away_score - B.pa).abs().values])
        market = B[B.vspread.notna()]
        market_obj = None
        if len(market):
            vm = -market.vspread
            market_obj = dict(n=int(len(market)),
                              model_margin_mae=fnum((market.am - market.pm).abs().mean(), 2),
                              market_margin_mae=fnum((market.am - vm).abs().mean(), 2),
                              model_total_mae=fnum((market["at"] - market.pt).abs().mean(), 2) if market.vtotal.notna().any() else None,
                              market_total_mae=fnum((market["at"] - market.vtotal).abs().mean(), 2) if market.vtotal.notna().any() else None,
                              same_favorite=int(((market.pm > 0) == (vm > 0)).sum()),
                              first=str(market.game_date.min()), last=str(market.game_date.max()))

        performance = dict(
            record=dict(all=summ(B), first=str(B.game_date.min()), last=str(B.game_date.max()),
                          seasons=[dict(season=s, complete=s != CURRENT_SEASON, all=summ(x), regular=summ(x[x.season_type == "Regular Season"]),
                                        playoffs=summ(x[x.season_type == "Playoffs"])) for s, x in B.groupby("season")],
                          calibration=calib, margin_buckets=bucket_cal,
                          by_margin=breakdown(B, margin_bucket, ["0–3", "3–6", "6–10", "10+"]),
                          by_favorite=breakdown(B, fav_side, ["Home favorite", "Road favorite"]),
                          by_type=breakdown(B, stype, ["Regular Season", "Playoffs"]),
                          hist_margin=hist((B.am - B.pm).values, -40, 40, 4),
                          hist_total=hist((B["at"] - B.pt).values, -50, 50, 5),
                          hist_team=hist(team_err, 0, 30, 2),
                          team_median=fnum(np.median(team_err), 1),
                          team_within5=fnum(100 * (team_err <= 5).mean(), 0),
                          team_within10=fnum(100 * (team_err <= 10).mean(), 0),
                          pred_margin_sd=fnum(B.pm.std(), 1), actual_margin_sd=fnum(B.am.std(), 1)),
            counts=dict(issued=int(len(issued)), walkforward=int(len(bt_only)),
                        backfill_excluded=n_backfill),
        )

        # ------------------------------------------------------------ team analytics
        rows_by_season = {}
        for r in fc_rows:
            rows_by_season.setdefault(r["s"], []).append(r)

        def team_series(rows, abbr):
            out, ce, ca = [], 0.0, 0
            for g in AM.team_games(rows, abbr):
                ce += g["p"]
                ca += g["won"]
                out.append(dict(id=g["id"], d=g["d"], opp=g["opp"], home=g["home"], p=fnum(g["p"], 3),
                                tp=fnum(g["tp"], 0), op=fnum(g["op"], 0), ts=int(g["ts"]), os=int(g["os"]),
                                won=bool(g["won"]), ok=bool(g["ok"]), me=fnum(g["me"], 1), pe=fnum(g["score_err"], 1),
                                ce=fnum(ce, 2), ca=int(ca), po=g["po"], page=g["id"] in page_set))
            return out

        sources = []
        for s in sorted(B.season.unique(), reverse=True):
            done = s != CURRENT_SEASON
            sources.append(dict(key=s, label=s + ("" if done else " (in progress)"),
                                season=s, complete=done, rows=rows_by_season.get(s, [])))

        team_analytics = dict(sources=[{k: v for k, v in s.items() if k != "rows"} for s in sources], teams={},
                              league={}, calc_version=AM.CALC_VERSION)
        for s in sources:
            per = {}
            for t in team_list:
                games_ = team_series(s["rows"], t["abbr"])
                summ_ = AM.team_summary(AM.team_games(s["rows"], t["abbr"]))
                per[t["abbr"]] = dict(metrics=summ_, games=games_, note=AM.bias_statement(t["abbr"], summ_))
            maes = sorted(v["metrics"]["mae"] for v in per.values() if v["metrics"]["n"])
            team_analytics["league"][s["key"]] = dict(
                median_mae=fnum(np.median(maes), 2) if maes else None,
                median_sd=fnum(np.median([v["metrics"]["sd"] for v in per.values() if v["metrics"].get("sd") is not None]), 2) if maes else None,
                mae_rank={a: i + 1 for i, a in enumerate(sorted([a for a, v in per.items() if v["metrics"]["n"]],
                                                                 key=lambda a: per[a]["metrics"]["mae"]))})
            for abbr, v in per.items():
                team_analytics["teams"].setdefault(abbr, {})[s["key"]] = v

        # ------------------------------------------------------------ predictability + surprises
        def scope_rows(season_, kind):
            rows = fc_rows if season_ == "all" else rows_by_season.get(season_, [])
            return [r for r in rows if kind == "all" or (kind == "po") == bool(r["po"])]

        seasons_all = sorted(rows_by_season)
        scopes = []
        for season_ in ["all"] + seasons_all[::-1]:
            for kind in ("all", "rs", "po"):
                rows = scope_rows(season_, kind)
                if not rows:
                    continue
                teams_ = {t["abbr"]: AM.team_summary(AM.team_games(rows, t["abbr"])) for t in team_list}
                maes = [v["mae"] for v in teams_.values() if v["n"]]
                scopes.append(dict(
                    key=f"{season_}|{kind}", season=season_, kind=kind,
                    label=(f"{seasons_all[0]} to {seasons_all[-1]}" if season_ == "all" else season_)
                    + {"all": "", "rs": ", regular season", "po": ", playoffs"}[kind],
                    complete=season_ != CURRENT_SEASON, n_games=len(rows),
                    league=dict(mae=fnum(np.mean([abs(r["me"]) for r in rows]), 2),
                                median_team_mae=fnum(np.median(maes), 2) if maes else None),
                    teams=teams_))
        predictability = dict(calc_version=AM.CALC_VERSION, tiers=dict(small=AM.TIER_SMALL, stable=AM.TIER_STABLE),
                              scopes=scopes)

        latest = max(r["d"] for r in fc_rows)
        week_from = str((pd.Timestamp(latest) - pd.Timedelta(days=6)).date())
        week = [r for r in fc_rows if r["d"] >= week_from]
        surprises = dict(
            calc_version=AM.CALC_VERSION, p_floor=AM.P_FLOOR,
            week=dict(start=week_from, end=latest,
                      outcome=AM.rank_outcome(week, 1)[0] if week else None,
                      scoreline=AM.rank_scoreline(week, 1)[0] if week else None),
            teams={k: {kk: [{**r, "page": r["id"] in page_set} for r in vv] for kk, vv in v.items()}
                   for k, v in team_surprise.items()},
        )
        for k in ("outcome", "scoreline"):
            if surprises["week"][k]:
                surprises["week"][k]["page"] = surprises["week"][k]["id"] in page_set

        # ------------------------------------------------------------ team overview
        cur = games[(games.season == CURRENT_SEASON) & (games.season_type == "Regular Season")]
        rec = {}
        for r in cur[cur.final].itertuples():
            hw = r.home_score > r.away_score
            rec.setdefault(r.home, [0, 0])[0 if hw else 1] += 1
            rec.setdefault(r.away, [0, 0])[1 if hw else 0] += 1
        latest_elo = {}
        elo_hist = {}   # abbr -> {game_id: [date, pregame elo]}, for trend lines
        src_fv = fc[["game_id", "feature_vector", "predicted_at"]]
        src_fv = src_fv.join(gby[["game_date", "home", "away"]], on="game_id").sort_values("game_date")
        for r in src_fv.itertuples():
            fv = r.feature_vector if isinstance(r.feature_vector, dict) else {}
            for side in ("home", "away"):
                if fv.get(f"{side}_elo") is not None:
                    latest_elo[getattr(r, side)] = dict(elo=fnum(fv[f"{side}_elo"], 0), date=str(r.game_date))
                    elo_hist.setdefault(getattr(r, side), {})[r.game_id] = [str(r.game_date), fnum(fv[f"{side}_elo"], 0)]
        roll = {tid2abbr[r.team_id]: dict(ortg=fnum(r.ortg, 1), drtg=fnum(r.drtg, 1), pace=fnum(r.pace, 1),
                                         as_of=str(r.as_of_date), n=int(r.games_counted or 0))
                for r in rolling.itertuples() if r.team_id in tid2abbr}
        for t in team_list:
            w, l = rec.get(t["abbr"], [0, 0])
            hist = sorted(elo_hist.get(t["abbr"], {}).values())[-82:]
            t.update(wins=w, losses=l, elo=latest_elo.get(t["abbr"]), rolling=roll.get(t["abbr"]), elo_hist=hist)

        # ------------------------------------------------------------ site meta
        site = dict(
            generated_at=generated.strftime("%Y-%m-%dT%H:%M:%SZ"),
            season=CURRENT_SEASON,
            history=dict(first=str(model_span.lo), last=str(model_span.hi)),
            coverage=[dict(season=r.season, games=int(r.games), box=int(r.box), reports=int(r.reports))
                      for r in coverage.itertuples() if int(r.games) > 0],
            counts=dict(game_pages=len(game_pages), forecasts=len(fc_rows)),
            seasons=sorted({r["s"] for r in fc_rows}),
            excluded_bad_orientation=n_bad_orientation,
            warnings=WARNINGS,
        )

    write_json(os.path.join(SRC_DATA, "site.json"), site)
    write_json(os.path.join(SRC_DATA, "teams.json"), team_list)
    write_json(os.path.join(SRC_DATA, "games.json"), game_pages)
    write_json(os.path.join(SRC_DATA, "performance.json"), performance)
    write_json(os.path.join(SRC_DATA, "team_analytics.json"), team_analytics)
    write_json(os.path.join(SRC_DATA, "predictability.json"), predictability)
    write_json(os.path.join(SRC_DATA, "surprises.json"), surprises)
    print(f"Done. {len(WARNINGS)} warning(s).")


if __name__ == "__main__":
    main()
