"""
Team strength features built from game results only:

- Elo: one running rating per team, updated after every final game (margin-of-victory
  scaled, like FiveThirtyEight's NBA Elo). Between seasons each rating keeps
  ELO_CARRY of its distance from 1500.
- Season blend: season-to-date average margin, blended with last season's average
  margin until BLEND_GAMES games are played. Rolling 10-game stats cross seasons, so
  this is what tells the model "new season, small sample".

Training and live predictions both use TeamStrength, so the values are identical:
training records each game's features before updating with its result; live replays
every final game before the prediction date and then reads the current state.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from sqlalchemy import text
from data.storage.db import engine

ELO_K = 20
ELO_HOME = 70          # Elo points of home advantage (~2.5 pts)
ELO_CARRY = 0.75       # share of a rating's distance from 1500 kept into the next season
ELO_PER_POINT = 28.0   # Elo points per point of expected margin
BLEND_GAMES = 20       # season-to-date margin fully trusted after this many games
# First game replayed. Training and live must use the same start, or live Elo has
# more history than the ratings the model learned from. 2021-10-19 = first game of
# 2021-22, the earliest season in the DB (moved from 2023-10-24 on 2026-11-07 with a
# retrain on the full history; out of sample it matched or beat the 2023 start on
# 2024-25, 2025-26 and 2026-27). Change it only together with a retrain.
ELO_START_DATE = '2021-10-19'

STRENGTH_COLS = ['home_elo', 'away_elo', 'elo_diff',
                 'home_season_margin', 'away_season_margin',
                 'home_games_played', 'away_games_played']


class TeamStrength:
    def __init__(self):
        self.elo = {}
        self.elo_season = {}
        self.season_stats = {}     # (team_id, season) -> [games, margin sum]
        self.last_season_margin = {}

    def _roll_season(self, tid, season):
        if self.elo_season.get(tid) == season:
            return
        if tid in self.elo_season:
            n, s = self.season_stats.get((tid, self.elo_season[tid]), (0, 0.0))
            if n:
                self.last_season_margin[tid] = s / n
            self.elo[tid] = 1500 + ELO_CARRY * (self.elo.get(tid, 1500) - 1500)
        else:
            self.elo[tid] = 1500
        self.elo_season[tid] = season

    def features(self, home_id, away_id, season):
        rec = {}
        for side, tid in (('home', home_id), ('away', away_id)):
            self._roll_season(tid, season)
            rec[f'{side}_elo'] = self.elo[tid]
            n, s = self.season_stats.get((tid, season), (0, 0.0))
            w = min(n, BLEND_GAMES) / BLEND_GAMES
            cur = s / n if n else 0.0
            rec[f'{side}_season_margin'] = w * cur + (1 - w) * self.last_season_margin.get(tid, 0.0)
            rec[f'{side}_games_played'] = n
        rec['elo_diff'] = rec['home_elo'] + ELO_HOME - rec['away_elo']
        return rec

    def update(self, home_id, away_id, season, home_score, away_score):
        self._roll_season(home_id, season)
        self._roll_season(away_id, season)
        margin = float(home_score) - float(away_score)
        dr = self.elo[home_id] + ELO_HOME - self.elo[away_id]
        exp = 1 / (1 + 10 ** (-dr / 400))
        dr_winner = dr if margin > 0 else -dr
        mov_mult = ((abs(margin) + 3) ** 0.8) / (7.5 + 0.006 * dr_winner)
        delta = ELO_K * mov_mult * ((margin > 0) - exp)
        self.elo[home_id] += delta
        self.elo[away_id] -= delta
        for tid, mg in ((home_id, margin), (away_id, -margin)):
            st = self.season_stats.setdefault((tid, season), [0, 0.0])
            st[0] += 1
            st[1] += mg


def _load_final_games(before_date=None):
    q = """
        SELECT game_id, game_date, season, home_team_id, away_team_id, home_score, away_score
        FROM games
        WHERE is_final = TRUE AND home_score IS NOT NULL AND away_score IS NOT NULL
          AND game_date >= :start
    """
    params = {'start': ELO_START_DATE}
    if before_date is not None:
        q += " AND game_date < :d"
        params['d'] = str(before_date)
    q += " ORDER BY game_date, game_id"
    return pd.read_sql(text(q), engine, params=params)


def add_team_strength_features(df):
    """Training: pre-game Elo / season-blend values for every game in df."""
    print("  Computing Elo and season blend...")
    ts = TeamStrength()
    out = {}
    for r in _load_final_games().itertuples(index=False):
        out[r.game_id] = ts.features(r.home_team_id, r.away_team_id, r.season)
        ts.update(r.home_team_id, r.away_team_id, r.season, r.home_score, r.away_score)
    feats = pd.DataFrame.from_dict(out, orient='index')[STRENGTH_COLS]
    df = df.drop(columns=[c for c in STRENGTH_COLS if c in df.columns])
    return df.join(feats, on='game_id')


_live_cache = {}


def get_live_team_strength(home_id, away_id, season, game_date):
    """Live: replay all final games before game_date, then read the current state."""
    key = str(game_date)
    if key not in _live_cache:
        ts = TeamStrength()
        for r in _load_final_games(before_date=game_date).itertuples(index=False):
            ts.update(r.home_team_id, r.away_team_id, r.season, r.home_score, r.away_score)
        _live_cache.clear()
        _live_cache[key] = ts
    # features() can roll a team into a new season, so read from a copy
    import copy
    return copy.deepcopy(_live_cache[key]).features(home_id, away_id, season)
