"""
C — Player availability: rotation players unavailable for a game.

Rotation = played >= ROT_MIN_GAMES of the team's last ROT_WINDOW games THIS SEASON
at >= ROT_MIN_MINUTES avg. (Using last season's games let departed players count
as "missing" for the first weeks of every season.)
Unavailable = listed Out/Doubtful on the official NBA injury report — the game
day's training-time report in training, the latest report live — so training and
prediction use the same information.
"""
import unicodedata

import pandas as pd
from sqlalchemy import text
from data.storage.db import engine

ROT_WINDOW = 10
ROT_MIN_GAMES = 2   # 3 dropped stars hurt in week 1 (e.g. Banchero, 2 games); 2 tested slightly better in CV (2026-11-05)
ROT_MIN_MINUTES = 15


def _load_played_box_scores(before_date=None):
    query = """
        SELECT b.game_id, b.team_id, b.player_id, b.minutes_played, b.points,
               g.game_date, g.season, p.full_name
        FROM player_box_scores b
        JOIN games g USING (game_id)
        JOIN players p USING (player_id)
        WHERE g.is_final = TRUE AND b.minutes_played > 0
    """
    params = {}
    if before_date is not None:
        query += " AND g.game_date < :before_date"
        params['before_date'] = str(before_date)
    box = pd.read_sql(text(query), engine, params=params)
    box['game_date'] = pd.to_datetime(box['game_date'])
    return box


def _rotation(window):
    agg = window.groupby('player_id').agg(
        gp=('game_id', 'nunique'), mins=('minutes_played', 'mean'),
        pts=('points', 'mean'), full_name=('full_name', 'last'))
    return agg[(agg.gp >= ROT_MIN_GAMES) & (agg.mins >= ROT_MIN_MINUTES)]


def _still_on_team(players, team_id, last_team_by_date, as_of):
    """Drop players who have since appeared for another team (traded/released)."""
    if players.empty:
        return players
    prior = last_team_by_date[
        last_team_by_date.player_id.isin(players.index) &
        (last_team_by_date.game_date < as_of)]
    latest = prior.groupby('player_id')['team_id'].last()
    return players[latest.reindex(players.index).eq(team_id).values]


def _season_rotation(tb, season, before_date, team_id, last_team_by_date):
    """Rotation from the team's last ROT_WINDOW games of `season` before before_date."""
    prior = tb[(tb.season == season) & (tb.game_date < before_date)]
    if prior.empty:
        return _rotation(prior)
    recent = (prior[['game_id', 'game_date']].drop_duplicates()
              .sort_values('game_date').tail(ROT_WINDOW)['game_id'])
    rot = _rotation(prior[prior.game_id.isin(recent)])
    return _still_on_team(rot, team_id, last_team_by_date, before_date)


def compute_availability():
    """
    Historical availability: for every team-game, minutes and points per game of
    rotation players listed Out/Doubtful on that day's official injury report.
    Dates with no stored report fall back to box-score absences (counted and
    reported, since that definition differs from what's known before tip-off).
    """
    from data.ingestion.fetch_official_injuries import load_training_unavailable

    box = _load_played_box_scores()
    last_team_by_date = box.sort_values('game_date')[['player_id', 'game_date', 'team_id']]
    unavailable = load_training_unavailable()

    rows = []
    fallback = 0
    for team_id, tb in box.groupby('team_id'):
        games = (tb[['game_id', 'game_date', 'season']].drop_duplicates()
                 .sort_values('game_date'))
        played_by_game = tb.groupby('game_id')['player_id'].apply(set).to_dict()
        for gid, gdate, season in games.itertuples(index=False):
            rot = _season_rotation(tb, season, gdate, team_id, last_team_by_date)
            report = unavailable.get(gdate.date())
            if report is not None:
                absent = rot[rot.index.isin(report.get(team_id, set()))]
            else:
                fallback += 1
                absent = rot[~rot.index.isin(played_by_game.get(gid, set()))]
            rows.append((gid, team_id, float(absent['mins'].astype(float).sum()),
                         float(absent['pts'].sum())))
    print(f"    {len(rows)} team-games, {fallback} without an injury report "
          f"(box-score absences used)")
    return pd.DataFrame(rows, columns=['game_id', 'team_id', 'missing_min', 'missing_pts'])


def add_availability_features(df):
    print("  Computing availability (unavailable rotation players)...")
    avail = compute_availability()
    for side in ('home', 'away'):
        a = avail.rename(columns={'team_id': f'{side}_team_id',
                                  'missing_min': f'{side}_missing_min',
                                  'missing_pts': f'{side}_missing_pts'})
        df = df.merge(a, on=['game_id', f'{side}_team_id'], how='left')
    return df


def normalize_player_name(name):
    name = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode()
    return ' '.join(name.lower().replace('.', '').split())


def compute_live_availability(team_id, game_date, out_names):
    """
    Availability for an unplayed game: rotation players (same definition as training)
    listed Out/Doubtful on the injury report.
    Returns (missing_min, missing_pts, missing_player_names).
    """
    if not out_names:
        return 0.0, 0.0, []
    from data.ingestion.fetch_official_injuries import season_for_date

    box = _load_played_box_scores(before_date=game_date)
    tb = box[box.team_id == team_id]
    if tb.empty:
        return 0.0, 0.0, []

    last_team_by_date = box.sort_values('game_date')[['player_id', 'game_date', 'team_id']]
    gdate = pd.Timestamp(game_date)
    rot = _season_rotation(tb, season_for_date(gdate.date()), gdate, team_id,
                           last_team_by_date)

    out = {normalize_player_name(n) for n in out_names}
    absent = rot[rot['full_name'].map(normalize_player_name).isin(out)]
    return (float(absent['mins'].astype(float).sum()), float(absent['pts'].sum()),
            absent['full_name'].tolist())
