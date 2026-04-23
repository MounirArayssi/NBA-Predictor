import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from data.ingestion.build_features import build_feature_dataset
from model.train import FEATURE_COLS, TARGET_HOME, TARGET_AWAY

# Style
plt.style.use('dark_background')
COLORS = {
    'home':     '#4FC3F7',
    'away':     '#FF8A65',
    'correct':  '#81C784',
    'wrong':    '#E57373',
    'neutral':  '#B0BEC5',
    'accent':   '#FFD54F',
}


def load_models_and_data():
    """Load saved models and rebuild test dataset."""
    print("Loading models and data...")

    with open('model/model_home.pkl', 'rb') as f:
        model_home = pickle.load(f)
    with open('model/model_away.pkl', 'rb') as f:
        model_away = pickle.load(f)

    df = build_feature_dataset(window=10)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_HOME, TARGET_AWAY])
    df = df.sort_values('game_date').reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    test = df.iloc[split_idx:].copy()

    X_test = test[FEATURE_COLS]
    test['pred_home'] = model_home.predict(X_test)
    test['pred_away'] = model_away.predict(X_test)
    test['pred_home_rounded'] = np.round(test['pred_home']).astype(int)
    test['pred_away_rounded'] = np.round(test['pred_away']).astype(int)
    test['home_error'] = abs(test['pred_home'] - test[TARGET_HOME])
    test['away_error'] = abs(test['pred_away'] - test[TARGET_AWAY])
    test['total_error'] = test['home_error'] + test['away_error']
    test['winner_correct'] = (
        (test['pred_home'] > test['pred_away']) ==
        (test[TARGET_HOME] > test[TARGET_AWAY])
    )
    test['pred_margin'] = test['pred_home'] - test['pred_away']
    test['actual_margin'] = test[TARGET_HOME] - test[TARGET_AWAY]

    print(f"  Test set: {len(test)} games")
    return model_home, model_away, test, df


def plot_feature_importance(model_home, model_away):
    """Enhanced feature importance with color coding by category."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    categories = {
        'Home Form':       [f for f in FEATURE_COLS if f.startswith('home_') and
                           any(x in f for x in ['avg', 'off', 'def', 'pace',
                                                 'fg', 'win', 'pts'])],
        'Away Form':       [f for f in FEATURE_COLS if f.startswith('away_') and
                           any(x in f for x in ['avg', 'off', 'def', 'pace',
                                                 'fg', 'win', 'pts'])],
        'Matchup':         ['pace_differential', 'combined_pace',
                            'home_off_vs_away_def', 'away_off_vs_home_def',
                            'net_rating_diff', 'implied_total',
                            'home_3pt_matchup', 'away_3pt_matchup'],
        'H2H':             ['h2h_home_avg_score', 'h2h_away_avg_score',
                            'h2h_home_win_pct', 'h2h_games_count'],
        'Situational':     ['home_rest_days', 'away_rest_days',
                            'rest_advantage', 'home_back_to_back',
                            'away_back_to_back', 'home_court_strength'],
        'Vegas':           ['vegas_spread', 'vegas_total',
                            'vegas_home_implied', 'vegas_away_implied',
                            'vegas_vs_home_avg', 'vegas_vs_away_avg'],
        'Playoff':         ['is_playoff', 'series_game_num',
                            'home_series_wins', 'away_series_wins',
                            'is_elimination', 'series_momentum',
                            'series_pressure', 'home_playoff_elevation',
                            'away_playoff_elevation',
                            'playoff_elevation_diff'],
        'Similarity':      ['home_proxy_off_rating', 'home_proxy_avg_pts',
                            'away_proxy_off_rating', 'away_proxy_avg_pts',
                            'home_sim_off_rating', 'away_sim_off_rating'],
        'Variance':        ['home_scoring_std', 'away_scoring_std',
                            'home_consistency', 'away_consistency',
                            'variance_differential'],
    }

    cat_colors = {
        'Home Form':   '#4FC3F7',
        'Away Form':   '#FF8A65',
        'Matchup':     '#FFD54F',
        'H2H':         '#81C784',
        'Situational': '#CE93D8',
        'Vegas':       '#F48FB1',
        'Playoff':     '#80CBC4',
        'Similarity':  '#FFCC02',
        'Variance':    '#B0BEC5',
    }

    for ax, (model, title) in zip(
        axes,
        [(model_home, 'Home Score Model'),
         (model_away, 'Away Score Model')]
    ):
        importance = pd.Series(
            model.feature_importances_,
            index=FEATURE_COLS
        ).sort_values(ascending=True).tail(20)

        colors = []
        for feat in importance.index:
            color = '#B0BEC5'
            for cat, feats in categories.items():
                if feat in feats:
                    color = cat_colors.get(cat, '#B0BEC5')
                    break
            colors.append(color)

        bars = ax.barh(importance.index, importance.values, color=colors)
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xlabel('Feature Importance', fontsize=11)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=v, label=k)
            for k, v in cat_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='lower right',
                 fontsize=8, framealpha=0.3)

    plt.suptitle('Feature Importance by Category', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('model/feature_importance_enhanced.png',
                dpi=150, bbox_inches='tight')
    print("  Saved: feature_importance_enhanced.png")
    plt.close()


def plot_error_distribution(test):
    """Show where errors come from — regular season vs playoffs."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    reg  = test[test['season_type'] == 'Regular Season']
    play = test[test['season_type'] == 'Playoffs']

    # 1. Error distribution overall
    ax = axes[0, 0]
    ax.hist(test['home_error'], bins=30, alpha=0.7,
            color=COLORS['home'], label='Home Error', density=True)
    ax.hist(test['away_error'], bins=30, alpha=0.7,
            color=COLORS['away'], label='Away Error', density=True)
    ax.axvline(test['home_error'].mean(), color=COLORS['home'],
               linestyle='--', linewidth=2)
    ax.axvline(test['away_error'].mean(), color=COLORS['away'],
               linestyle='--', linewidth=2)
    ax.set_title('Error Distribution — All Games')
    ax.set_xlabel('Absolute Error (pts)')
    ax.legend()

    # 2. Regular season vs playoff error
    ax = axes[0, 1]
    ax.hist(reg['total_error'], bins=30, alpha=0.7,
            color=COLORS['correct'], label=f'Regular Season (n={len(reg)})',
            density=True)
    if len(play) > 0:
        ax.hist(play['total_error'], bins=20, alpha=0.7,
                color=COLORS['accent'],
                label=f'Playoffs (n={len(play)})', density=True)
    ax.axvline(reg['total_error'].mean(), color=COLORS['correct'],
               linestyle='--')
    if len(play) > 0:
        ax.axvline(play['total_error'].mean(), color=COLORS['accent'],
                   linestyle='--')
    ax.set_title('Total Error: Regular Season vs Playoffs')
    ax.set_xlabel('Total Error (pts)')
    ax.legend()

    # 3. Winner accuracy by margin
    ax = axes[1, 0]
    test['margin_bucket'] = pd.cut(
        abs(test['pred_margin']),
        bins=[0, 3, 6, 10, 15, 50],
        labels=['0-3', '3-6', '6-10', '10-15', '15+']
    )
    margin_acc = test.groupby('margin_bucket', observed=True)[
        'winner_correct'
    ].agg(['mean', 'count']).reset_index()

    bars = ax.bar(margin_acc['margin_bucket'].astype(str),
                  margin_acc['mean'] * 100,
                  color=COLORS['home'], alpha=0.8)
    ax.axhline(50, color='white', linestyle='--', alpha=0.3,
               label='Random (50%)')
    ax.axhline(test['winner_correct'].mean() * 100,
               color=COLORS['accent'], linestyle='--',
               label=f'Overall ({test["winner_correct"].mean()*100:.1f}%)')

    for bar, (_, row) in zip(bars, margin_acc.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'n={int(row["count"])}',
                ha='center', va='bottom', fontsize=8)

    ax.set_title('Winner Accuracy by Predicted Margin')
    ax.set_xlabel('Predicted Margin (pts)')
    ax.set_ylabel('Winner Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.legend()

    # 4. Residual plot — does model systematically over/under predict?
    ax = axes[1, 1]
    residuals_home = test['pred_home'] - test[TARGET_HOME]
    residuals_away = test['pred_away'] - test[TARGET_AWAY]

    ax.scatter(test['pred_home'], residuals_home, alpha=0.2,
               color=COLORS['home'], s=8, label='Home')
    ax.scatter(test['pred_away'], residuals_away, alpha=0.2,
               color=COLORS['away'], s=8, label='Away')
    ax.axhline(0, color='white', linewidth=1)
    ax.axhline(residuals_home.mean(), color=COLORS['home'],
               linestyle='--', linewidth=1.5,
               label=f'Home bias: {residuals_home.mean():.1f}')
    ax.axhline(residuals_away.mean(), color=COLORS['away'],
               linestyle='--', linewidth=1.5,
               label=f'Away bias: {residuals_away.mean():.1f}')
    ax.set_title('Residuals — Systematic Bias Check')
    ax.set_xlabel('Predicted Score')
    ax.set_ylabel('Predicted - Actual (pts)')
    ax.legend(fontsize=8)

    plt.suptitle('Model Error Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('model/error_analysis.png',
                dpi=150, bbox_inches='tight')
    print("  Saved: error_analysis.png")
    plt.close()


def plot_team_errors(test):
    """Which teams does the model consistently get wrong?"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Home team errors
    home_errors = test.groupby('home_team').agg(
        avg_error=('home_error', 'mean'),
        games=('home_error', 'count'),
        win_acc=('winner_correct', 'mean')
    ).sort_values('avg_error', ascending=True)

    # Away team errors
    away_errors = test.groupby('away_team').agg(
        avg_error=('away_error', 'mean'),
        games=('away_error', 'count'),
        win_acc=('winner_correct', 'mean')
    ).sort_values('avg_error', ascending=True)

    for ax, (errors, title) in zip(
        axes,
        [(home_errors, 'Home Team Score Error'),
         (away_errors, 'Away Team Score Error')]
    ):
        colors = [
            COLORS['correct'] if e < errors['avg_error'].median()
            else COLORS['wrong']
            for e in errors['avg_error']
        ]
        bars = ax.barh(errors.index, errors['avg_error'],
                       color=colors, alpha=0.8)
        ax.axvline(errors['avg_error'].mean(), color=COLORS['accent'],
                   linestyle='--', linewidth=2,
                   label=f'Avg: {errors["avg_error"].mean():.1f}')
        ax.set_title(title)
        ax.set_xlabel('Mean Absolute Error (pts)')
        ax.legend()

    plt.suptitle('Error by Team — Which Teams Are Hard to Predict?',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('model/team_errors.png',
                dpi=150, bbox_inches='tight')
    print("  Saved: team_errors.png")
    plt.close()


def plot_score_ranges(test):
    """Does the model handle high and low scoring games differently?"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (score_col, pred_col, title, color) in zip(axes, [
        (TARGET_HOME, 'pred_home', 'Home Score', COLORS['home']),
        (TARGET_AWAY, 'pred_away', 'Away Score', COLORS['away']),
    ]):
        ax.scatter(test[score_col], test[pred_col],
                   alpha=0.3, color=color, s=10)
        ax.plot([80, 160], [80, 160], 'w--',
                linewidth=1, label='Perfect prediction')

        # Add trend line
        z = np.polyfit(test[score_col], test[pred_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(80, 160, 100)
        ax.plot(x_line, p(x_line), color=COLORS['accent'],
                linewidth=2, label='Actual trend')

        mae = abs(test[score_col] - test[pred_col]).mean()
        ax.set_title(f'{title} — MAE: {mae:.1f} pts')
        ax.set_xlabel('Actual Score')
        ax.set_ylabel('Predicted Score')
        ax.legend()
        ax.set_xlim(75, 165)
        ax.set_ylim(75, 165)

    plt.suptitle('Predicted vs Actual Scores', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('model/score_ranges.png',
                dpi=150, bbox_inches='tight')
    print("  Saved: score_ranges.png")
    plt.close()


def print_summary(test):
    """Print a clean summary of model performance."""
    reg  = test[test['season_type'] == 'Regular Season']
    play = test[test['season_type'] == 'Playoffs']

    print("\n" + "="*55)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*55)
    print(f"{'Metric':<30} {'All':>8} {'RegSeason':>10} {'Playoffs':>10}")
    print("-"*55)

    metrics = [
        ('Games', len(test), len(reg), len(play)),
        ('Home MAE',
         test['home_error'].mean(),
         reg['home_error'].mean(),
         play['home_error'].mean() if len(play) else 0),
        ('Away MAE',
         test['away_error'].mean(),
         reg['away_error'].mean(),
         play['away_error'].mean() if len(play) else 0),
        ('Winner Accuracy %',
         test['winner_correct'].mean()*100,
         reg['winner_correct'].mean()*100,
         play['winner_correct'].mean()*100 if len(play) else 0),
        ('Within 10 pts %',
         (test['total_error'] <= 10).mean()*100,
         (reg['total_error'] <= 10).mean()*100,
         (play['total_error'] <= 10).mean()*100 if len(play) else 0),
        ('Within 15 pts %',
         (test['total_error'] <= 15).mean()*100,
         (reg['total_error'] <= 15).mean()*100,
         (play['total_error'] <= 15).mean()*100 if len(play) else 0),
        ('Avg Total Error',
         test['total_error'].mean(),
         reg['total_error'].mean(),
         play['total_error'].mean() if len(play) else 0),
    ]

    for name, all_val, reg_val, play_val in metrics:
        if isinstance(all_val, float):
            print(f"  {name:<28} {all_val:>8.1f} {reg_val:>10.1f} "
                  f"{play_val:>10.1f}")
        else:
            print(f"  {name:<28} {all_val:>8} {reg_val:>10} "
                  f"{play_val:>10}")

    print("="*55)

    # Biggest misses
    print("\nTop 10 Biggest Misses:")
    worst = test.nlargest(10, 'total_error')[[
        'game_date', 'home_team', 'away_team',
        'pred_home_rounded', 'pred_away_rounded',
        TARGET_HOME, TARGET_AWAY, 'total_error',
        'season_type'
    ]]
    for _, row in worst.iterrows():
        print(f"  {row['away_team']} @ {row['home_team']} "
              f"({row['game_date'].strftime('%Y-%m-%d')}) "
              f"[{row['season_type'][:3]}] | "
              f"Pred: {row['home_team']} {row['pred_home_rounded']} — "
              f"{row['away_team']} {row['pred_away_rounded']} | "
              f"Actual: {row['home_team']} {int(row[TARGET_HOME])} — "
              f"{row['away_team']} {int(row[TARGET_AWAY])} | "
              f"Error: {row['total_error']:.0f} pts")


if __name__ == "__main__":
    model_home, model_away, test, full_df = load_models_and_data()

    print("\nGenerating visualizations...")
    plot_feature_importance(model_home, model_away)
    plot_error_distribution(test)
    plot_team_errors(test)
    plot_score_ranges(test)
    print_summary(test)

    print("\n✅ All visualizations saved to model/")