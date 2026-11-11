"""D — Rest: days since each team's previous game, and back-to-backs."""
import numpy as np
import pandas as pd
from sqlalchemy import text
from data.storage.db import engine

# Rest days are capped here in training and in live prediction (model/predict.py)
REST_CAP = 5


def add_rest_days(df):
    """Calculate rest days for each team before each game."""
    print("  Computing rest days...")

    # Same rule as the live get_rest_days (model/predict.py): days since the
    # team's previous final game, home or away, from the full schedule (not just
    # rows in df), capped at REST_CAP; 3 when there is no previous game.
    df = df.sort_values('game_date').copy()
    df['game_date'] = pd.to_datetime(df['game_date'])

    sched = pd.read_sql(text("""
        SELECT game_date, home_team_id AS team_id FROM games WHERE is_final = TRUE
        UNION
        SELECT game_date, away_team_id AS team_id FROM games WHERE is_final = TRUE
    """), engine)
    sched['game_date'] = pd.to_datetime(sched['game_date'])
    dates_by_team = {t: np.sort(g['game_date'].unique())
                     for t, g in sched.groupby('team_id')}

    def rest_for(team_id, game_date):
        dates = dates_by_team.get(team_id)
        if dates is None:
            return 3
        i = np.searchsorted(dates, np.datetime64(game_date), side='left')
        if i == 0:
            return 3
        days = int((np.datetime64(game_date) - dates[i - 1]) / np.timedelta64(1, 'D'))
        return max(0, min(REST_CAP, days))

    df['home_rest_days'] = [rest_for(t, d) for t, d in zip(df['home_team_id'], df['game_date'])]
    df['away_rest_days'] = [rest_for(t, d) for t, d in zip(df['away_team_id'], df['game_date'])]
    df['rest_advantage']    = df['home_rest_days'] - df['away_rest_days']
    df['home_back_to_back'] = (df['home_rest_days'] == 1).astype(int)
    df['away_back_to_back'] = (df['away_rest_days'] == 1).astype(int)

    return df
