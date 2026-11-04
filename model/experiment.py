import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from model.train import FEATURE_COLS, TARGET_HOME, TARGET_AWAY, train_model
from data.ingestion.build_features import build_feature_dataset

CACHE_PATH = 'model/.feature_cache.pkl'


# Step 2 (skipped for now, CV gain ~0.03) — expected points from possessions x opponent-adjusted efficiency.
# No fitted parameters, so it can't overfit on its own.
LEAGUE_WINDOW = 300   # trailing games used for league pace / rating baselines


def add_expected_points(df):
    df = df.sort_values('game_date').copy()
    # Inputs are pre-game rolling stats, so a trailing mean over rows is leak-free
    game_pace = (df['home_pace'] + df['away_pace']) / 2
    game_rating = (df['home_off_rating'] + df['away_off_rating']) / 2
    league_pace = game_pace.rolling(LEAGUE_WINDOW, min_periods=50).mean()
    league_rating = game_rating.rolling(LEAGUE_WINDOW, min_periods=50).mean()

    df['exp_poss'] = df['home_pace'] * df['away_pace'] / league_pace
    df['exp_home_pts'] = (df['exp_poss'] * df['home_off_rating'] *
                          df['away_def_rating'] / league_rating / 100)
    df['exp_away_pts'] = (df['exp_poss'] * df['away_off_rating'] *
                          df['home_def_rating'] / league_rating / 100)
    return df.dropna(subset=['exp_poss']).reset_index(drop=True)


STEP2_NEW = ['exp_poss', 'exp_home_pts', 'exp_away_pts']
STEP2_ADD = FEATURE_COLS + STEP2_NEW
STEP2_SWAP = [c for c in FEATURE_COLS if c not in (
    'combined_pace', 'home_off_vs_away_def', 'away_off_vs_home_def')] + STEP2_NEW


# Step 4 — availability: rotation players absent from this game's box score.
# Now built in build_features.add_availability_features (part of build_feature_dataset).
STEP4_NEW = ['home_missing_min', 'away_missing_min', 'home_missing_pts', 'away_missing_pts']


def load_data(refresh=False):
    """Build the feature dataset once and cache it — the DB build is slow."""
    if os.path.exists(CACHE_PATH) and not refresh:
        return pd.read_pickle(CACHE_PATH)
    df = build_feature_dataset(window=10)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_HOME, TARGET_AWAY])
    df = df.sort_values('game_date').reset_index(drop=True)
    df.to_pickle(CACHE_PATH)
    return df


def rolling_folds(df, n_folds=5, test_frac=0.08):
    """Walk-forward splits: train on everything before each test block."""
    n = len(df)
    test_size = int(n * test_frac)
    first_test = n - n_folds * test_size
    for k in range(n_folds):
        start = first_test + k * test_size
        yield df.iloc[:start], df.iloc[start:start + test_size]


def evaluate_features(df, cols, label):
    import model.train as train_mod
    rows = []
    for k, (train, test) in enumerate(rolling_folds(df)):
        # train_model reads the module-level FEATURE_COLS
        train_mod.FEATURE_COLS = cols
        m_home = train_model(train, TARGET_HOME)
        m_away = train_model(train, TARGET_AWAY)
        train_mod.FEATURE_COLS = FEATURE_COLS

        ph = m_home.predict(test[cols])
        pa = m_away.predict(test[cols])
        mae = (mean_absolute_error(test[TARGET_HOME], ph) +
               mean_absolute_error(test[TARGET_AWAY], pa)) / 2
        winner = ((ph > pa) == (test[TARGET_HOME].values >
                                test[TARGET_AWAY].values)).mean()
        po = (test['season_type'] == 'Playoffs').values
        po_mae = (np.abs(test[TARGET_HOME].values - ph)[po].mean() +
                  np.abs(test[TARGET_AWAY].values - pa)[po].mean()) / 2                  if po.any() else np.nan
        rows.append({'fold': k, 'mae': mae, 'winner': winner,
                     'po_mae': po_mae, 'po_n': int(po.sum()),
                     'pred_std': np.concatenate([ph, pa]).std()})
    res = pd.DataFrame(rows)
    print(f"{label:<22} MAE " +
          " ".join(f"{m:5.2f}" for m in res['mae']) +
          f" | mean {res['mae'].mean():.2f} | winner {res['winner'].mean()*100:.1f}%"
          f" | pred std {res['pred_std'].mean():.1f}"
          f" | playoff MAE " + " ".join(
              f"{m:5.2f}({n})" for m, n in zip(res['po_mae'], res['po_n'])))
    return res


def evaluate_baselines(df):
    for name, fn in [
        ('league mean (train)', lambda tr, te: (
            np.full(len(te), tr[TARGET_HOME].mean()),
            np.full(len(te), tr[TARGET_AWAY].mean()))),
        ('rolling avg points', lambda tr, te: (
            te['home_avg_points'].values, te['away_avg_points'].values)),
    ]:
        maes = []
        for train, test in rolling_folds(df):
            ph, pa = fn(train, test)
            maes.append((mean_absolute_error(test[TARGET_HOME], ph) +
                         mean_absolute_error(test[TARGET_AWAY], pa)) / 2)
        print(f"{name:<22} MAE " + " ".join(f"{m:5.2f}" for m in maes) +
              f" | mean {np.mean(maes):.2f}")


if __name__ == "__main__":
    df = load_data(refresh='--refresh' in sys.argv)
    print(f"{len(df)} games, "
          f"{df['game_date'].min()} -> {df['game_date'].max()}\n")
    actual_std = pd.concat([df[TARGET_HOME], df[TARGET_AWAY]]).std()
    print(f"Actual score std: {actual_std:.1f}\n")

    print(df[STEP4_NEW].describe().round(1).T)
    print()

    evaluate_baselines(df)
    evaluate_features(df, FEATURE_COLS, f'current ({len(FEATURE_COLS)})')
    evaluate_features(df, FEATURE_COLS + STEP4_NEW, f'step4 avail ({len(FEATURE_COLS) + 4})')
