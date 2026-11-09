"""
Inside the Forecast: exact SHAP explanations for stored forecasts, where the
model that produced them can be verified.

    python web/scripts/explain_forecasts.py        # after web/scripts/export_data.py

Which model explains which forecast
-----------------------------------
- Issued forecasts (is_official, not 'backfill'): the current production artifacts
  model/model_home.pkl + model/model_away.pkl, loaded read-only.
- Walk-forward forecasts (model_version 'wf-v2'): those models were trained in
  memory and never saved. Each season's pair is rebuilt here in memory with the
  same code and data rule (model.train.train_model on games from earlier seasons;
  random_state is fixed in train_model). Nothing is written to model/.

Verification (see analytics/explain.py): a group of forecasts — issued forecasts
saved after the current artifacts were written, or one walk-forward season — is
explained only if at least 90% of it reproduces both stored scores to the whole
point; then each forecast within the 1-point rounding bound on margin is explained.
Forecasts from older model versions keep their saved inputs without attribution.

Database access is read-only: a READ ONLY transaction for our own queries, and
model.train.load_and_prepare_data (SELECT-only feature build) for rebuilding.
Output: web/src/data/explanations.json (games that have a page) — nothing else.
"""
import contextlib
import io
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
ROOT = os.path.dirname(WEB)
sys.path.insert(0, ROOT)
sys.path.insert(0, WEB)
os.chdir(ROOT)

import numpy as np
import pandas as pd

from data.storage.db import engine
from analytics import explain as E

WF = "wf-v2"
OUT = os.path.join(WEB, "src", "data", "explanations.json")


def load_forecasts():
    with engine.connect() as conn:
        conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        df = pd.read_sql("""
            SELECT p.game_id, g.nba_game_id, g.season, g.game_date, p.model_version, p.is_official,
                   p.prediction_type, p.predicted_at,
                   p.home_score_predicted::float ph, p.away_score_predicted::float pa, p.feature_vector
            FROM predictions p JOIN games g ON g.game_id = p.game_id
            WHERE p.model_version = %(wf)s OR (p.is_official AND p.prediction_type <> 'backfill')
        """, conn, params={"wf": WF})
        conn.rollback()
    issued = df[df.model_version != WF].copy()
    issued["origin"] = "issued"
    wf = df[(df.model_version == WF) & ~df.game_id.isin(issued.game_id)].copy()
    wf["origin"] = "walkforward"
    # Same one-forecast-per-game rule as the site exporter.
    return pd.concat([issued, wf], ignore_index=True)


def explain_rows(rows, mh, ma, cols, label, out, stats):
    rows = rows.reset_index(drop=True)
    X, complete = E.feature_matrix(list(rows.feature_vector), cols)
    s = stats.setdefault(label, dict(n=len(rows), complete=int(complete.sum()), verified=False,
                                     share=0.0, explained=0))
    for i in np.where(~complete)[0]:
        out[str(rows.iloc[i].nba_game_id)] = dict(v=label, ok=False, why="incomplete inputs")
    idx = np.where(complete)[0]
    if not len(idx):
        return
    A = E.margin_attribution(mh, ma, X[idx])
    ph, pa = rows.ph.values[idx], rows.pa.values[idx]
    verified, share = E.verify_group(A["pred_home"], A["pred_away"], ph, pa)
    s.update(verified=verified, share=round(share, 3))
    mm = E.margin_match(A["pred_margin"], ph, pa)
    for k, i in enumerate(idx):
        r = rows.iloc[i]
        gid = str(r.nba_game_id)
        if not verified:
            out[gid] = dict(v=label, ok=False, why="the model version that made this forecast is not available")
            continue
        if not mm[k]:
            out[gid] = dict(v=label, ok=False, why="model output differs from the stored forecast",
                            gap=round(float(A["pred_margin"][k] - (ph[k] - pa[k])), 2))
            continue
        s["explained"] += 1
        out[gid] = dict(
            v=label, ok=True, origin=r.origin,
            base=round(A["base_margin"], 3), bh=round(A["base_home"], 3), ba=round(A["base_away"], 3),
            pm=round(float(A["pred_margin"][k]), 3),
            x={c: round(float(X[i, j]), 3) for j, c in enumerate(cols)},
            c={c: [round(float(A["home"][k, j]), 3), round(float(A["away"][k, j]), 3), round(float(A["margin"][k, j]), 3)]
               for j, c in enumerate(cols)},
        )


def main():
    with open(os.path.join(WEB, "src", "data", "games.json"), encoding="utf8") as f:
        page_games = {g["id"]: g for g in json.load(f)}
    pages = set(page_games)
    fc = load_forecasts()
    with open("model/feature_cols.pkl", "rb") as f:
        cols = list(pickle.load(f))
    assert set(cols) == set(E.FEATURES), "feature list changed; update analytics/explain.FEATURES"
    out, stats, importance = {}, {}, {}

    # Issued forecasts: current production artifacts.
    with open("model/model_home.pkl", "rb") as f:
        mh = pickle.load(f)
    with open("model/model_away.pkl", "rb") as f:
        ma = pickle.load(f)
    # Only forecasts saved after the current artifacts were written can have come
    # from them; earlier issued forecasts used artifacts that no longer exist.
    saved = pd.Timestamp(os.path.getmtime("model/model_home.pkl"), unit="s", tz="UTC")
    iss = fc[fc.origin == "issued"]
    at = pd.to_datetime(iss.predicted_at, utc=True)
    label = f"production model saved {saved.tz_convert('America/New_York'):%Y-%m-%d}"
    explain_rows(iss[at >= saved], mh, ma, cols, label, out, stats)
    print(f"  {label}: {stats.get(label)}")
    old = iss[at < saved]
    stats["earlier production versions"] = dict(n=len(old), verified=False, explained=0)
    for r in old.itertuples():
        out[str(r.nba_game_id)] = dict(v="earlier production version", ok=False,
                                       why="the model version that made this forecast is not available")

    # Walk-forward forecasts: rebuild each season's in-memory models.
    from model.train import load_and_prepare_data, train_model, TARGET_HOME, TARGET_AWAY
    cache = os.environ.get("EXPLAIN_FEATURE_CACHE")   # optional, for development only
    if cache and os.path.exists(cache):
        df = pd.read_pickle(cache)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            df = load_and_prepare_data(window=10)
        if cache:
            df.to_pickle(cache)
    for season in sorted(fc[fc.origin == "walkforward"].season.unique()):
        train = df[df["season"] < season]
        if train.empty:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            wh, wa = train_model(train, TARGET_HOME), train_model(train, TARGET_AWAY)
        label = f"walk-forward {season}"
        explain_rows(fc[(fc.origin == "walkforward") & (fc.season == season)], wh, wa, cols, label, out, stats)
        print(f"  {label}: {stats[label]}")

    # Global importance: mean |margin contribution| per feature over verified forecasts.
    ok = [v for v in out.values() if v.get("ok")]
    for c in cols:
        importance[c] = round(float(np.mean([abs(v["c"][c][2]) for v in ok])), 3) if ok else None
    groups = {g: round(sum(importance[c] for c in cols if E.FEATURES[c][0] == g and importance[c] is not None), 3)
              for g in E.GROUPS}

    # Deterministic one-line summaries (analytics.explain.summary_sentence).
    for gid, e in out.items():
        if e.get("ok") and gid in page_games:
            g = page_games[gid]
            groups_ = E.group_contributions(np.array([e["c"][c][2] for c in cols]), cols)
            e["groups"] = {k: round(v, 3) for k, v in groups_.items()}
            e["summary"] = E.summary_sentence(g["home"], g["away"], e["base"], groups_, e["pm"])

    payload = dict(
        version=E.EXPLAIN_VERSION, verify_share=E.VERIFY_SHARE, features=cols,
        meta={c: dict(group=E.FEATURES[c][0], label=E.FEATURES[c][1], unit=E.FEATURES[c][2]) for c in cols},
        groups=E.GROUPS, stats=stats, importance=importance, group_importance=groups, n_verified=len(ok),
        games={k: v for k, v in out.items() if k in pages},
    )
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote {OUT}: {len(payload['games'])} page games, {len(ok)} verified explanations overall")


if __name__ == "__main__":
    main()
