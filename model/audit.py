import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from data.ingestion.build_features import build_feature_dataset
from model.train import FEATURE_COLS, TARGET_HOME, TARGET_AWAY
from sklearn.metrics import mean_absolute_error


def run_audit():
    print("Loading models and data...")
    
    with open('model/model_home.pkl', 'rb') as f:
        model_home = pickle.load(f)
    with open('model/model_away.pkl', 'rb') as f:
        model_away = pickle.load(f)

    df = build_feature_dataset(window=10)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_HOME, TARGET_AWAY])
    df = df.sort_values('game_date').reset_index(drop=True)

    split_idx = int(len(df) * 0.85)
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()

    X_test = test[FEATURE_COLS]
    home_preds = model_home.predict(X_test)
    away_preds = model_away.predict(X_test)

    actual_home = test[TARGET_HOME].values
    actual_away = test[TARGET_AWAY].values

    # Model winner accuracy
    model_winner = (home_preds > away_preds)
    actual_winner = (actual_home > actual_away)
    model_acc = (model_winner == actual_winner).mean()

    print(f"\n{'='*55}")
    print("MODEL AUDIT RESULTS")
    print(f"{'='*55}")
    print(f"Test games: {len(test)}")
    print(f"Model winner accuracy: {model_acc*100:.1f}%")

    # --- Baseline 1: Always pick home team ---
    home_always_acc = actual_winner.mean()
    print(f"\nBaseline 1 — Always pick home team: "
          f"{home_always_acc*100:.1f}%")

    # --- Baseline 2: Always pick favorite (higher avg points) ---
    home_avg = test['home_avg_points'].values
    away_avg = test['away_avg_points'].values
    favorite_is_home = home_avg > away_avg
    favorite_acc = (favorite_is_home == actual_winner).mean()
    print(f"Baseline 2 — Always pick higher avg points team: "
          f"{favorite_acc*100:.1f}%")

    # --- Baseline 3: Random ---
    np.random.seed(42)
    random_picks = np.random.randint(0, 2, len(test)).astype(bool)
    random_acc = (random_picks == actual_winner).mean()
    print(f"Baseline 3 — Random: {random_acc*100:.1f}%")

    # --- Class imbalance check ---
    home_win_rate = actual_winner.mean()
    print(f"\nClass imbalance:")
    print(f"  Home team wins: {home_win_rate*100:.1f}%")
    print(f"  Away team wins: {(1-home_win_rate)*100:.1f}%")

    # --- Leakage check ---
    # Compare train vs test accuracy
    X_train = train[FEATURE_COLS]
    train_home = model_home.predict(X_train)
    train_away = model_away.predict(X_train)
    train_winner = (train_home > train_away)
    actual_train_winner = (
        train[TARGET_HOME].values > train[TARGET_AWAY].values
    )
    train_acc = (train_winner == actual_train_winner).mean()

    print(f"\nOverfitting check:")
    print(f"  Train winner accuracy: {train_acc*100:.1f}%")
    print(f"  Test winner accuracy:  {model_acc*100:.1f}%")
    print(f"  Gap: {abs(train_acc - model_acc)*100:.1f}%")

    if abs(train_acc - model_acc) < 0.03:
        print("  ✅ Gap < 3% — no significant overfitting")
    elif abs(train_acc - model_acc) < 0.06:
        print("  ⚠️  Gap 3-6% — mild overfitting")
    else:
        print("  ❌ Gap > 6% — significant overfitting")

    # --- Margin consistency check ---
    # If model predicts margin correctly, winner accuracy should
    # correlate with margin size
    pred_margins = abs(home_preds - away_preds)
    margin_buckets = pd.cut(
        pred_margins,
        bins=[0, 3, 6, 10, 15, 50],
        labels=['0-3', '3-6', '6-10', '10-15', '15+']
    )

    print(f"\nWinner accuracy by predicted margin:")
    for bucket in ['0-3', '3-6', '6-10', '10-15', '15+']:
        mask = margin_buckets == bucket
        if mask.sum() > 0:
            acc = (model_winner[mask] == actual_winner[mask]).mean()
            print(f"  Margin {bucket:>5}: {acc*100:.1f}% "
                  f"(n={mask.sum()})")

    # --- Feature leakage check ---
    # Check if any features correlate suspiciously with outcome
    print(f"\nTop feature correlations with actual winner:")
    test['actual_winner'] = actual_winner.astype(int)
    correlations = []
    for col in FEATURE_COLS:
        if col in test.columns:
            corr = abs(test[col].corr(test['actual_winner']))
            correlations.append((col, corr))

    correlations.sort(key=lambda x: x[1], reverse=True)
    for feat, corr in correlations[:10]:
        flag = " ⚠️ HIGH" if corr > 0.4 else ""
        print(f"  {feat:<35} {corr:.3f}{flag}")

    # --- Playoff vs Regular Season breakdown ---
    reg  = test[test['season_type'] == 'Regular Season']
    play = test[test['season_type'] == 'Playoffs']

    if len(reg) > 0:
        reg_home  = model_home.predict(reg[FEATURE_COLS])
        reg_away  = model_away.predict(reg[FEATURE_COLS])
        reg_acc   = (
            (reg_home > reg_away) ==
            (reg[TARGET_HOME].values > reg[TARGET_AWAY].values)
        ).mean()
        print(f"\nRegular season accuracy: {reg_acc*100:.1f}% "
              f"(n={len(reg)})")

    if len(play) > 0:
        play_home = model_home.predict(play[FEATURE_COLS])
        play_away = model_away.predict(play[FEATURE_COLS])
        play_acc  = (
            (play_home > play_away) ==
            (play[TARGET_HOME].values > play[TARGET_AWAY].values)
        ).mean()
        print(f"Playoff accuracy:        {play_acc*100:.1f}% "
              f"(n={len(play)})")

    print(f"\n{'='*55}")
    print("AUDIT COMPLETE")
    print(f"{'='*55}")


if __name__ == "__main__":
    run_audit()