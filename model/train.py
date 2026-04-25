import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

from data.ingestion.build_features import build_feature_dataset


# Features the model trains on
FEATURE_COLS = [
    # Home team form
    'home_avg_points',
    'home_off_rating',
    'home_def_rating',
    'home_fg_pct',
    'home_fg3_pct',
    'home_3pt_rate',
    'home_win_pct',

    # Away team form
    'away_avg_points',
    'away_off_rating',
    'away_def_rating',
    'away_fg_pct',
    'away_fg3_pct',
    'away_3pt_rate',
    'away_win_pct',

    # Matchup context
    'home_rest_days',
    'away_rest_days',
    'rest_advantage',
    'home_back_to_back',
    'away_back_to_back',

    # Playoff context
    'is_playoff',
    'series_game_num',
    'home_series_wins',
    'away_series_wins',
    'is_elimination',
    'series_pressure',


    # Scoring variance
    'home_scoring_std',
    'away_scoring_std',
    'home_consistency',
    'away_consistency',
    'variance_differential',

    # Matchup interactions
    'home_3pt_matchup',
    'away_3pt_matchup',
    'combined_pace',
    'home_off_vs_away_def',
    'away_off_vs_home_def',
    'net_rating_diff',

    # Shooting consistency
    'home_bad_night_pct',
    'away_bad_night_pct',

    'series_wins_diff',

    'home_q4_avg',
    'away_q4_avg',
    'home_q4_diff',
    'away_q4_diff',
    'home_clutch',
    'away_clutch',
    'clutch_diff',
    'q4_diff_spread',

]

TARGET_HOME = 'home_score'
TARGET_AWAY = 'away_score'


def load_and_prepare_data(window=10):
    print("Loading feature dataset...")
    df = build_feature_dataset(window=window)

    # Drop rows with missing features
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_HOME, TARGET_AWAY])
    after = len(df)
    print(f"Dropped {before - after} rows with missing values "
          f"({after} remaining)")

    # Sort chronologically
    df = df.sort_values('game_date').reset_index(drop=True)

    return df


def chronological_split(df, test_ratio=0.2):
    """Split data chronologically — never randomly for time series."""
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()

    print(f"Train: {len(train)} games "
          f"({train['game_date'].min()} → {train['game_date'].max()})")
    print(f"Test:  {len(test)} games "
          f"({test['game_date'].min()} → {test['game_date'].max()})")

    return train, test


def train_model(train, target_col):
    """Train a GradientBoosting model for one score target."""
    X_train = train[FEATURE_COLS]
    y_train = train[target_col]

    # Recency weighting — slower decay over 2 years
    max_date = pd.to_datetime(train['game_date']).max()
    days_ago = (max_date - pd.to_datetime(train['game_date'])).dt.days
    sample_weights = np.exp(-days_ago / 550)

    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.04,
        max_depth=3,
        min_samples_leaf=15,
        subsample=0.7,
        max_features=0.8,
        random_state=42
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def evaluate_model(model_home, model_away, test):
    """Evaluate both models on the test set."""
    X_test = test[FEATURE_COLS]

    # Predictions
    home_preds = model_home.predict(X_test)
    away_preds = model_away.predict(X_test)

    # Actual scores
    home_actual = test[TARGET_HOME].values
    away_actual = test[TARGET_AWAY].values

    # MAE
    home_mae = mean_absolute_error(home_actual, home_preds)
    away_mae = mean_absolute_error(away_actual, away_preds)
    total_mae = (home_mae + away_mae) / 2

    # Winner accuracy
    pred_winner_home   = home_preds > away_preds
    actual_winner_home = home_actual > away_actual
    winner_accuracy    = (pred_winner_home == actual_winner_home).mean()

    # Within N points accuracy
    total_error = np.abs(home_preds - home_actual) + \
                  np.abs(away_preds - away_actual)
    within_10 = (total_error <= 10).mean()
    within_15 = (total_error <= 15).mean()
    within_20 = (total_error <= 20).mean()

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Home Score MAE:     {home_mae:.2f} pts")
    print(f"Away Score MAE:     {away_mae:.2f} pts")
    print(f"Average MAE:        {total_mae:.2f} pts")
    print(f"Winner Accuracy:    {winner_accuracy*100:.1f}%")
    print(f"Within 10 pts:      {within_10*100:.1f}%")
    print(f"Within 15 pts:      {within_15*100:.1f}%")
    print(f"Within 20 pts:      {within_20*100:.1f}%")
    print("="*50)

    # Show mix — 5 recent regular season + 5 playoff
    regular = test[test['season_type'] == 'Regular Season'].tail(5)
    playoffs = test[test['season_type'] == 'Playoffs'].tail(5)

    reg_preds_home  = model_home.predict(regular[FEATURE_COLS])
    reg_preds_away  = model_away.predict(regular[FEATURE_COLS])
    play_preds_home = model_home.predict(playoffs[FEATURE_COLS])
    play_preds_away = model_away.predict(playoffs[FEATURE_COLS])

    print("\nSample predictions vs actuals:")
    print("--- Regular Season ---")
    for i, (_, row) in enumerate(regular.iterrows()):
        pred_h = int(np.round(reg_preds_home[i]))
        pred_a = int(np.round(reg_preds_away[i]))
        correct = "✅" if (pred_h > pred_a) == \
                         (row['home_score'] > row['away_score']) else "❌"
        print(f"  {correct} {row['away_team']} @ {row['home_team']} | "
              f"Pred: {pred_h}-{pred_a} | "
              f"Actual: {int(row['home_score'])}-{int(row['away_score'])}")

    print("--- Playoffs ---")
    for i, (_, row) in enumerate(playoffs.iterrows()):
        pred_h = int(np.round(play_preds_home[i]))
        pred_a = int(np.round(play_preds_away[i]))
        correct = "✅" if (pred_h > pred_a) == \
                         (row['home_score'] > row['away_score']) else "❌"
        print(f"  {correct} {row['away_team']} @ {row['home_team']} | "
              f"Pred: {pred_h}-{pred_a} | "
              f"Actual: {int(row['home_score'])}-{int(row['away_score'])}")

    return {
        'home_mae': home_mae,
        'away_mae': away_mae,
        'total_mae': total_mae,
        'winner_accuracy': winner_accuracy,
        'within_10': within_10,
        'within_15': within_15,
        'within_20': within_20,
        'home_preds': home_preds,
        'away_preds': away_preds,
    }


def plot_feature_importance(model_home, model_away):
    """Plot which features matter most."""
    importance_home = pd.Series(
        model_home.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    importance_away = pd.Series(
        model_away.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    importance_home.head(15).plot(
        kind='barh', ax=axes[0], color='steelblue'
    )
    axes[0].set_title('Home Score — Top 15 Features')
    axes[0].invert_yaxis()

    importance_away.head(15).plot(
        kind='barh', ax=axes[1], color='coral'
    )
    axes[1].set_title('Away Score — Top 15 Features')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(
        'model/feature_importance.png', dpi=150, bbox_inches='tight'
    )
    print("\n✅ Feature importance saved to model/feature_importance.png")


def plot_predictions_vs_actual(results, test):
    """Scatter plot of predicted vs actual scores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(
        test[TARGET_HOME], results['home_preds'],
        alpha=0.3, color='steelblue', s=10
    )
    axes[0].plot([80, 160], [80, 160], 'r--', linewidth=1)
    axes[0].set_xlabel('Actual Home Score')
    axes[0].set_ylabel('Predicted Home Score')
    axes[0].set_title(f"Home Score (MAE: {results['home_mae']:.1f})")

    axes[1].scatter(
        test[TARGET_AWAY], results['away_preds'],
        alpha=0.3, color='coral', s=10
    )
    axes[1].plot([80, 160], [80, 160], 'r--', linewidth=1)
    axes[1].set_xlabel('Actual Away Score')
    axes[1].set_ylabel('Predicted Away Score')
    axes[1].set_title(f"Away Score (MAE: {results['away_mae']:.1f})")

    plt.tight_layout()
    plt.savefig(
        'model/predictions_vs_actual.png', dpi=150, bbox_inches='tight'
    )
    print("✅ Predictions vs actual saved to model/predictions_vs_actual.png")


def save_models(model_home, model_away):
    """Save trained models to disk."""
    os.makedirs('model', exist_ok=True)

    with open('model/model_home.pkl', 'wb') as f:
        pickle.dump(model_home, f)

    with open('model/model_away.pkl', 'wb') as f:
        pickle.dump(model_away, f)

    with open('model/feature_cols.pkl', 'wb') as f:
        pickle.dump(FEATURE_COLS, f)

    print("✅ Models saved to model/")

def check_overfitting(model_home, model_away, train, test):
    """Compare train vs test performance to detect overfitting."""
    
    for split_name, split_df in [('TRAIN', train), ('TEST', test)]:
        X = split_df[FEATURE_COLS]
        home_preds = model_home.predict(X)
        away_preds = model_away.predict(X)
        
        home_mae = mean_absolute_error(
            split_df[TARGET_HOME], home_preds
        )
        away_mae = mean_absolute_error(
            split_df[TARGET_AWAY], away_preds
        )
        winner_acc = (
            (home_preds > away_preds) == 
            (split_df[TARGET_HOME].values > 
             split_df[TARGET_AWAY].values)
        ).mean()
        
        print(f"{split_name}: MAE={((home_mae+away_mae)/2):.2f} | "
              f"Winner={winner_acc*100:.1f}%")
        

if __name__ == "__main__":
    # Load data
    df = load_and_prepare_data(window=10)

    # Split chronologically
    print("\nSplitting data...")
    train, test = chronological_split(df, test_ratio=0.15)

    # Train
    print("\nTraining home score model...")
    model_home = train_model(train, TARGET_HOME)
    print("Training away score model...")
    model_away = train_model(train, TARGET_AWAY)

    # Evaluate
    print("\nEvaluating...")
    results = evaluate_model(model_home, model_away, test)

    # Plots
    plot_feature_importance(model_home, model_away)
    plot_predictions_vs_actual(results, test)

    # Save
    save_models(model_home, model_away)
    check_overfitting(model_home, model_away, train, test)
    print("\n✅ Training complete")