"""
walkforward_backfill.py
=======================
Honest backfill of past seasons. Each target season is predicted by team models
trained ONLY on games from earlier seasons, run through the same live prediction
path as the daily pipeline (build_game_features -> predict_single_game), with that
day's archived official injury report.

Leakage guards
- Team models: fit on df[season < target] with the production settings
  (train_production_models), saved nowhere — the live model/*.pkl files are untouched.
- Win-probability calibration: Platt fit on walk-forward rows from earlier seasons
  only (fixed sigmoid for the first season).
- Player models (player_models.pkl) were trained on every season, so they are
  switched off here. They don't feed game scores anyway (TEAM_MODEL_ONLY), only the
  confidence tag / key factors.
- Elo replays from 2021-22 (ELO_START_DATE patched in-process, production unchanged).

Rows are saved with model_version='wf-v2', prediction_type='walkforward',
is_official=False, so they never mix with the real pre-game record. Each row is
graded as it's saved. Re-running skips games that already have a wf-v2 row.

    python misc/walkforward_backfill.py                      # 2022-23 .. 2025-26
    python misc/walkforward_backfill.py --seasons 2023-24
    python misc/walkforward_backfill.py --report
"""
import sys
import os
import io
import argparse
import contextlib
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session
from sklearn.linear_model import LogisticRegression

from data.storage.db import engine
from data.storage.models import Prediction, Game

WF_VERSION = 'wf-v2'
DEFAULT_SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']
MIN_CALIBRATION_ROWS = 300
WF_ELO_START = '2021-10-19'   # first game of 2021-22, the earliest season in the DB


def no_player_model(game, row_dict, injury_report):
    return {'home_preds': [], 'home_pts': 0.0, 'away_preds': [], 'away_pts': 0.0,
            'available': False}


def fit_calibration(before_season):
    """Platt scaling on walk-forward predictions from seasons before the target."""
    q = text("""
        SELECT p.model_margin, (g.home_score > g.away_score)::int AS home_won
        FROM predictions p JOIN games g USING (game_id)
        WHERE p.model_version = :v AND g.season < :s AND g.is_final
    """)
    with engine.connect() as c:
        rows = c.execute(q, {'v': WF_VERSION, 's': before_season}).fetchall()
    if len(rows) < MIN_CALIBRATION_ROWS:
        return None, len(rows)
    X = np.array([[float(r[0])] for r in rows])
    y = np.array([r[1] for r in rows])
    return LogisticRegression().fit(X, y), len(rows)


def done_game_ids(season):
    q = text("""SELECT p.game_id FROM predictions p JOIN games g USING (game_id)
                WHERE p.model_version = :v AND g.season = :s""")
    with engine.connect() as c:
        return {r[0] for r in c.execute(q, {'v': WF_VERSION, 's': season})}


def season_dates(season):
    q = text("""SELECT DISTINCT game_date FROM games
                WHERE season = :s AND is_final
                  AND season_type IN ('Regular Season', 'Playoffs')
                ORDER BY game_date""")
    with engine.connect() as c:
        return [r[0] for r in c.execute(q, {'s': season})]


def final_game_ids(season):
    """Played games only — unplayed 'if necessary' playoff games have no score."""
    q = text("""SELECT game_id FROM games WHERE season = :s AND is_final
                AND home_score IS NOT NULL AND away_score IS NOT NULL""")
    with engine.connect() as c:
        return {r[0] for r in c.execute(q, {'s': season})}


def save_and_grade(preds, train_through):
    from model.evaluate_predictions import evaluate_prediction_row
    from data.storage.models import Team

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with Session(engine) as s:
        teams = {t.abbreviation: t.team_id for t in s.query(Team).all()}
        for p in preds:
            game = s.get(Game, p['game_id'])
            home, away = float(p['home_pred']), float(p['away_pred'])
            row = Prediction(
                game_id=game.game_id,
                model_version=WF_VERSION,
                predicted_at=now,
                is_official=False,
                prediction_type='walkforward',
                lock_reason=f'train_through_{train_through}',
                home_score_predicted=home,
                away_score_predicted=away,
                predicted_winner_id=teams.get(p['predicted_winner']),
                home_win_probability=float(p.get('home_win_prob', 50)) / 100,
                key_factors=p.get('factors', []),
                feature_vector=p.get('feature_vector'),
                model_margin=home - away,
                model_total=home + away,
            )
            evaluate_prediction_row(row, game, now)
            s.add(row)
        s.commit()


def run_season(season, df, log):
    import model.predict as P
    from model.train import train_production_models
    from data.ingestion.fetch_official_injuries import get_injury_report

    train = df[df['season'] < season]
    if train.empty:
        log(f"{season}: no earlier seasons to train on — skipped")
        return
    train_through = train['season'].max()
    log(f"{season}: training on {len(train)} games "
        f"({train['game_date'].min().date()} -> {train['game_date'].max().date()})")
    model_home, model_away, model_margin = train_production_models(train)

    calib, n_cal = fit_calibration(season)
    log(f"{season}: win-prob calibration = "
        f"{'Platt on %d earlier wf rows' % n_cal if calib else 'fixed sigmoid'}")

    P.get_player_predictions = no_player_model
    done = done_game_ids(season)
    final = final_game_ids(season)
    dates = season_dates(season)
    made = skipped = 0
    for i, d in enumerate(dates):
        games = [g for g in P.get_todays_games(game_date=d, include_final=True)
                 if g['game_id'] not in done and g['game_id'] in final
                 and g['season_type'] in ('Regular Season', 'Playoffs')]
        if not games:
            continue
        preds = []
        with contextlib.redirect_stdout(io.StringIO()):
            injury_report = get_injury_report(d)
            for g in games:
                try:
                    pred = P.predict_single_game(
                        g, model_home, model_away, injury_report, {},
                        model_margin=model_margin, calibration_model=calib)
                except Exception as e:
                    pred = None
                    err = f"{type(e).__name__}: {e}"
                else:
                    err = "no features"
                if pred:
                    preds.append(pred)
                else:
                    skipped += 1
                    log(f"  skip {d} {g['away_team']}@{g['home_team']}: {err}")
        save_and_grade(preds, train_through)
        made += len(preds)
        if i % 10 == 0:
            log(f"  {season} {d}: {made} predicted, {skipped} skipped ({i + 1}/{len(dates)} dates)")
    log(f"{season}: done — {made} predicted, {skipped} skipped")


def report():
    q = text("""
        SELECT g.season, g.season_type,
               COUNT(*) n,
               AVG(p.winner_correct::int) win_pct,
               AVG(p.margin_error) margin_mae,
               AVG(p.total_score_error) total_mae,
               AVG(POWER(p.home_win_probability - (g.home_score > g.away_score)::int, 2)) brier,
               AVG((g.home_score > g.away_score)::int) home_win_pct
        FROM predictions p JOIN games g USING (game_id)
        WHERE p.model_version = :v
        GROUP BY ROLLUP (g.season, g.season_type)
        ORDER BY g.season NULLS LAST, g.season_type NULLS FIRST
    """)
    df = pd.read_sql(q, engine, params={'v': WF_VERSION})
    df['season'] = df['season'].fillna('ALL')
    df['season_type'] = df['season_type'].fillna('all')
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(df.round(3).to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser(description='Walk-forward backfill of past seasons')
    ap.add_argument('--seasons', nargs='+', default=DEFAULT_SEASONS)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()
    if args.report:
        report()
        return

    def log(msg):
        print(f"{datetime.now():%H:%M:%S} | {msg}", flush=True)

    # Replay Elo from the first season in the DB (production starts at 2023-24 to
    # match what the live models were trained on). Patched for this process only, so
    # training and walk-forward predictions share the same Elo history.
    import data.ingestion.team_strength as ts
    ts.ELO_START_DATE = WF_ELO_START
    log(f"Elo replay starts {ts.ELO_START_DATE} (walk-forward only)")

    from model.train import load_and_prepare_data
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        df = load_and_prepare_data(window=10)
    log(buf.getvalue().strip().splitlines()[-1] if buf.getvalue().strip() else "data loaded")
    df['game_date'] = pd.to_datetime(df['game_date'])
    log("training rows by season: " + ", ".join(
        f"{s} {n}" for s, n in df.groupby('season').size().items()))

    for season in sorted(args.seasons):
        run_season(season, df, log)
    report()


if __name__ == '__main__':
    main()
