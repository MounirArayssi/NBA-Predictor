"""
Generate all visualization charts for the NBA Predictor.
Saves everything to visuals/ folder.

Sources (in order of preference):
  1. Evaluated official predictions from the DB  → real-world accuracy
  2. Model test set from build_feature_dataset()  → larger sample, always available

Run from project root:
    python visuals/generate_charts.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.orm import Session

from data.storage.db import engine
from data.storage.models import Prediction, Game, Team
from data.ingestion.build_features import build_feature_dataset
from model.train import FEATURE_COLS, TARGET_HOME, TARGET_AWAY

# ── Style ────────────────────────────────────────────────────────────────────
plt.style.use('dark_background')
C = {
    'home':    '#4FC3F7',
    'away':    '#FF8A65',
    'good':    '#81C784',
    'bad':     '#E57373',
    'neutral': '#B0BEC5',
    'accent':  '#FFD54F',
    'purple':  '#CE93D8',
    'teal':    '#80CBC4',
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_models():
    root = os.path.dirname(OUT_DIR)
    with open(os.path.join(root, 'model', 'model_home.pkl'), 'rb') as f:
        m_home = pickle.load(f)
    with open(os.path.join(root, 'model', 'model_away.pkl'), 'rb') as f:
        m_away = pickle.load(f)
    return m_home, m_away


def load_db_predictions():
    """
    Return a DataFrame of evaluated official predictions joined with team names.
    Columns: game_date, home_team, away_team, season_type,
             pred_home, pred_away, actual_home, actual_away,
             home_error, away_error, total_error, winner_correct,
             margin_error
    """
    query = text("""
        SELECT
            g.game_date,
            ht.abbreviation  AS home_team,
            at.abbreviation  AS away_team,
            g.season_type,
            p.home_score_predicted  AS pred_home,
            p.away_score_predicted  AS pred_away,
            g.home_score            AS actual_home,
            g.away_score            AS actual_away,
            p.home_score_error,
            p.away_score_error,
            p.total_score_error,
            p.winner_correct,
            p.margin_error
        FROM predictions p
        JOIN games  g  ON p.game_id      = g.game_id
        JOIN teams  ht ON g.home_team_id = ht.team_id
        JOIN teams  at ON g.away_team_id = at.team_id
        WHERE p.is_official   = TRUE
          AND p.evaluated_at IS NOT NULL
          AND g.home_score   IS NOT NULL
        ORDER BY g.game_date
    """)
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        'game_date', 'home_team', 'away_team', 'season_type',
        'pred_home', 'pred_away', 'actual_home', 'actual_away',
        'home_error', 'away_error', 'total_error', 'winner_correct',
        'margin_error',
    ])
    for col in ['pred_home', 'pred_away', 'actual_home', 'actual_away',
                'home_error', 'away_error', 'total_error', 'margin_error']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_test_set():
    """Build feature dataset and return the 15% test split (matching training)."""
    print("  Building feature dataset from DB (this takes ~30-60s)...")
    df = build_feature_dataset(window=10)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_HOME, TARGET_AWAY])
    df = df.sort_values('game_date').reset_index(drop=True)

    split_idx = int(len(df) * 0.85)
    test = df.iloc[split_idx:].copy()

    m_home, m_away = load_models()
    X = test[FEATURE_COLS]
    test['pred_home']  = m_home.predict(X)
    test['pred_away']  = m_away.predict(X)
    test['home_error'] = (test['pred_home'] - test[TARGET_HOME]).abs()
    test['away_error'] = (test['pred_away'] - test[TARGET_AWAY]).abs()
    test['total_error'] = test['home_error'] + test['away_error']
    test['winner_correct'] = (
        (test['pred_home'] > test['pred_away']) ==
        (test[TARGET_HOME] > test[TARGET_AWAY])
    )
    test['pred_margin']   = test['pred_home'] - test['pred_away']
    test['actual_margin'] = test[TARGET_HOME]  - test[TARGET_AWAY]
    test['margin_error']  = (test['pred_margin'] - test['actual_margin']).abs()
    test = test.rename(columns={TARGET_HOME: 'actual_home', TARGET_AWAY: 'actual_away'})
    return test, m_home, m_away


# ── Chart 1 — Feature Importance by Category ─────────────────────────────────

CATEGORIES = {
    'Home Form':   [f for f in FEATURE_COLS if f.startswith('home_') and
                    any(x in f for x in ['avg', 'off', 'def', 'pace',
                                          'fg', 'win', 'pts', 'rating',
                                          'ft_rate', 'oreb', 'tov', 'efg'])],
    'Away Form':   [f for f in FEATURE_COLS if f.startswith('away_') and
                    any(x in f for x in ['avg', 'off', 'def', 'pace',
                                          'fg', 'win', 'pts', 'rating',
                                          'ft_rate', 'oreb', 'tov', 'efg'])],
    'Matchup':     ['combined_pace', 'home_off_vs_away_def', 'away_off_vs_home_def',
                    'net_rating_diff', 'pace_differential'],
    'Situational': ['home_rest_days', 'away_rest_days', 'rest_advantage',
                    'home_back_to_back', 'away_back_to_back'],
    'Playoff':     ['home_series_wins', 'is_elimination', 'series_pressure',
                    'series_wins_diff', 'in_series_home_avg_pts',
                    'in_series_away_avg_pts', 'in_series_total_avg', 'in_series_margin'],
    'Variance':    ['home_scoring_std', 'away_scoring_std', 'home_consistency',
                    'away_consistency', 'variance_differential',
                    'home_bad_night_pct', 'away_bad_night_pct'],
    'Clutch/Q4':   ['home_q4_avg', 'away_q4_avg', 'home_q4_diff', 'away_q4_diff',
                    'home_clutch', 'away_clutch', 'clutch_diff', 'q4_diff_spread'],
    'Location':    ['home_home_avg_pts', 'away_away_avg_pts'],
}

CAT_COLORS = {
    'Home Form':   '#4FC3F7',
    'Away Form':   '#FF8A65',
    'Matchup':     '#FFD54F',
    'Situational': '#CE93D8',
    'Playoff':     '#80CBC4',
    'Variance':    '#B0BEC5',
    'Clutch/Q4':   '#F48FB1',
    'Location':    '#81C784',
}


def plot_feature_importance(m_home, m_away):
    fig, axes = plt.subplots(1, 2, figsize=(20, 11))

    for ax, (model, title) in zip(axes,
            [(m_home, 'Home Score Model'), (m_away, 'Away Score Model')]):
        imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
        imp = imp.sort_values(ascending=True).tail(20)

        colors = []
        for feat in imp.index:
            c = '#B0BEC5'
            for cat, feats in CATEGORIES.items():
                if feat in feats:
                    c = CAT_COLORS[cat]
                    break
            colors.append(c)

        ax.barh(imp.index, imp.values, color=colors, alpha=0.85, edgecolor='none')
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xlabel('Feature Importance (Gini)', fontsize=10)
        ax.tick_params(axis='y', labelsize=8)

        legend_els = [mpatches.Patch(facecolor=v, label=k)
                      for k, v in CAT_COLORS.items()]
        ax.legend(handles=legend_els, loc='lower right', fontsize=7.5,
                  framealpha=0.25)

    plt.suptitle('Feature Importance by Category\n(Top 20 per model)',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'feature_importance_by_category.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")
    return out


# ── Chart 2 — Error by Team ───────────────────────────────────────────────────

def plot_error_by_team(df, source_label):
    home_err = (df.groupby('home_team')['home_error']
                  .agg(['mean', 'count'])
                  .rename(columns={'mean': 'avg_error', 'count': 'games'}))
    away_err = (df.groupby('away_team')['away_error']
                  .agg(['mean', 'count'])
                  .rename(columns={'mean': 'avg_error', 'count': 'games'}))

    # Combined: average of home and away errors per team
    all_teams = sorted(set(home_err.index) | set(away_err.index))
    rows = []
    for t in all_teams:
        h = home_err.loc[t, 'avg_error'] if t in home_err.index else np.nan
        a = away_err.loc[t, 'avg_error'] if t in away_err.index else np.nan
        hg = home_err.loc[t, 'games']    if t in home_err.index else 0
        ag = away_err.loc[t, 'games']    if t in away_err.index else 0
        vals = [v for v in [h, a] if not np.isnan(v)]
        rows.append({'team': t, 'avg_error': np.mean(vals),
                     'home_error': h, 'away_error': a, 'games': hg + ag})
    combined = pd.DataFrame(rows).set_index('team').sort_values('avg_error')

    fig, axes = plt.subplots(1, 2, figsize=(20, max(8, len(combined) * 0.28 + 2)))

    # Left: home errors
    he = home_err.sort_values('avg_error')
    threshold = he['avg_error'].median()
    clrs = [C['good'] if v <= threshold else C['bad'] for v in he['avg_error']]
    axes[0].barh(he.index, he['avg_error'], color=clrs, alpha=0.85, edgecolor='none')
    axes[0].axvline(he['avg_error'].mean(), color=C['accent'], linestyle='--',
                    linewidth=1.8, label=f"Avg {he['avg_error'].mean():.1f} pts")
    axes[0].set_title('Home Score MAE by Team', fontsize=12)
    axes[0].set_xlabel('Mean Absolute Error (pts)')
    axes[0].tick_params(axis='y', labelsize=7.5)
    axes[0].legend(fontsize=9)

    # Right: away errors
    ae = away_err.sort_values('avg_error')
    threshold_a = ae['avg_error'].median()
    clrs_a = [C['good'] if v <= threshold_a else C['bad'] for v in ae['avg_error']]
    axes[1].barh(ae.index, ae['avg_error'], color=clrs_a, alpha=0.85, edgecolor='none')
    axes[1].axvline(ae['avg_error'].mean(), color=C['accent'], linestyle='--',
                    linewidth=1.8, label=f"Avg {ae['avg_error'].mean():.1f} pts")
    axes[1].set_title('Away Score MAE by Team', fontsize=12)
    axes[1].set_xlabel('Mean Absolute Error (pts)')
    axes[1].tick_params(axis='y', labelsize=7.5)
    axes[1].legend(fontsize=9)

    plt.suptitle('Error by Team — Which Teams Are Hardest to Predict?',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'error_by_team.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")
    return out


# ── Chart 3 — Predictions vs Actual Scatter ───────────────────────────────────

def plot_predictions_vs_actual(df, source_label):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    score_range = [70, 165]
    trend_x = np.linspace(score_range[0], score_range[1], 200)

    for ax, (actual_col, pred_col, title, color) in zip(axes, [
        ('actual_home', 'pred_home', 'Home Score', C['home']),
        ('actual_away', 'pred_away', 'Away Score', C['away']),
    ]):
        ax.scatter(df[actual_col], df[pred_col], alpha=0.25, color=color,
                   s=12, edgecolors='none', label='Individual game')
        ax.plot(score_range, score_range, 'w--', linewidth=1.2,
                alpha=0.6, label='Perfect prediction')

        z = np.polyfit(df[actual_col].dropna(), df[pred_col].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(trend_x, p(trend_x), color=C['accent'], linewidth=2,
                label=f'Trend (slope={z[0]:.2f})')

        mae = (df[actual_col] - df[pred_col]).abs().mean()
        r   = df[[actual_col, pred_col]].dropna().corr().iloc[0, 1]
        ax.set_title(f'{title}  |  MAE {mae:.1f} pts  |  r = {r:.3f}',
                     fontsize=11)
        ax.set_xlabel('Actual Score')
        ax.set_ylabel('Predicted Score')
        ax.set_xlim(*score_range)
        ax.set_ylim(*score_range)
        ax.legend(fontsize=8)

    plt.suptitle('Predicted vs Actual Scores',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'predictions_vs_actual.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")
    return out


# ── Chart 4 — Error Analysis / Distribution ───────────────────────────────────

def plot_error_analysis(df, source_label):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── 4a: Error distribution (home vs away) ─────────────────────────────
    ax = axes[0, 0]
    ax.hist(df['home_error'].dropna(), bins=35, alpha=0.7, color=C['home'],
            label=f"Home  (mean={df['home_error'].mean():.1f})", density=True)
    ax.hist(df['away_error'].dropna(), bins=35, alpha=0.7, color=C['away'],
            label=f"Away  (mean={df['away_error'].mean():.1f})", density=True)
    ax.axvline(df['home_error'].mean(), color=C['home'], linestyle='--', linewidth=1.8)
    ax.axvline(df['away_error'].mean(), color=C['away'], linestyle='--', linewidth=1.8)
    ax.set_title('Score Error Distribution')
    ax.set_xlabel('Absolute Error (pts)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)

    # ── 4b: Regular season vs playoffs ────────────────────────────────────
    ax = axes[0, 1]
    reg  = df[df['season_type'] == 'Regular Season']
    play = df[df['season_type'] == 'Playoffs']
    ax.hist(reg['total_error'].dropna(),  bins=35, alpha=0.7, color=C['good'],
            label=f'Reg Season (n={len(reg)}, mean={reg["total_error"].mean():.1f})',
            density=True)
    if len(play) >= 5:
        ax.hist(play['total_error'].dropna(), bins=20, alpha=0.7, color=C['accent'],
                label=f'Playoffs (n={len(play)}, mean={play["total_error"].mean():.1f})',
                density=True)
        ax.axvline(play['total_error'].mean(), color=C['accent'], linestyle='--', linewidth=1.8)
    ax.axvline(reg['total_error'].mean(), color=C['good'], linestyle='--', linewidth=1.8)
    ax.set_title('Total Error: Regular Season vs Playoffs')
    ax.set_xlabel('Total Error (pts)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)

    # ── 4c: Winner accuracy by predicted margin bucket ────────────────────
    ax = axes[1, 0]
    if 'pred_margin' not in df.columns:
        df['pred_margin'] = df['pred_home'] - df['pred_away']
    df['margin_bucket'] = pd.cut(
        df['pred_margin'].abs(),
        bins=[0, 3, 6, 10, 15, 100],
        labels=['0–3', '3–6', '6–10', '10–15', '15+']
    )
    bucket_stats = (df.groupby('margin_bucket', observed=True)['winner_correct']
                      .agg(['mean', 'count']).reset_index())
    bars = ax.bar(bucket_stats['margin_bucket'].astype(str),
                  bucket_stats['mean'] * 100,
                  color=C['home'], alpha=0.85, edgecolor='none')
    ax.axhline(50, color='white', linestyle='--', alpha=0.3, label='Chance (50%)')
    overall_acc = df['winner_correct'].mean() * 100
    ax.axhline(overall_acc, color=C['accent'], linestyle='--', linewidth=1.8,
               label=f'Overall ({overall_acc:.1f}%)')
    for bar, (_, row) in zip(bars, bucket_stats.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'n={int(row["count"])}', ha='center', va='bottom', fontsize=8)
    ax.set_title('Winner Accuracy by Predicted Margin')
    ax.set_xlabel('Predicted Margin (pts)')
    ax.set_ylabel('Winner Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)

    # ── 4d: Residual bias plot ─────────────────────────────────────────────
    ax = axes[1, 1]
    res_home = df['pred_home'] - df['actual_home']
    res_away = df['pred_away'] - df['actual_away']
    ax.scatter(df['pred_home'], res_home, alpha=0.18, color=C['home'], s=8,
               label='Home', edgecolors='none')
    ax.scatter(df['pred_away'], res_away, alpha=0.18, color=C['away'], s=8,
               label='Away', edgecolors='none')
    ax.axhline(0,              color='white',   linewidth=1.0, alpha=0.5)
    ax.axhline(res_home.mean(), color=C['home'], linestyle='--', linewidth=1.8,
               label=f'Home bias {res_home.mean():+.1f} pts')
    ax.axhline(res_away.mean(), color=C['away'], linestyle='--', linewidth=1.8,
               label=f'Away bias {res_away.mean():+.1f} pts')
    ax.set_title('Residuals — Systematic Bias Check')
    ax.set_xlabel('Predicted Score')
    ax.set_ylabel('Predicted − Actual (pts)')
    ax.legend(fontsize=8)

    plt.suptitle('Model Error Analysis',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'error_analysis.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    generated = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    print(f"\n{'='*60}")
    print(f"NBA Predictor — Chart Generator")
    print(f"Run at: {timestamp}")
    print(f"{'='*60}\n")

    # ── Load models (needed for feature importance) ────────────────────────
    print("[1/4] Feature Importance by Category")
    m_home, m_away = load_models()
    generated.append(plot_feature_importance(m_home, m_away))

    # ── Decide data source for the other 3 charts ──────────────────────────
    print("\n[Checking DB for evaluated predictions...]")
    db_df = load_db_predictions()

    if len(db_df) >= 30:
        df       = db_df
        source   = f"{len(db_df)} evaluated DB predictions as of {timestamp}"
        use_db   = True
        print(f"  Using DB predictions ({len(db_df)} games).")
    else:
        print(f"  DB has only {len(db_df)} evaluated predictions — "
              f"falling back to model test set.")
        df, m_home, m_away = load_test_set()
        source   = f"model test set (15% chronological holdout, n={len(df)})"
        use_db   = False

    print(f"\n[2/4] Error by Team")
    generated.append(plot_error_by_team(df, source))

    print(f"\n[3/4] Predictions vs Actual Scatter")
    generated.append(plot_predictions_vs_actual(df, source))

    print(f"\n[4/4] Error Analysis / Distribution")
    # Need season_type — fall back to 'Regular Season' if missing
    if 'season_type' not in df.columns:
        df['season_type'] = 'Regular Season'
    generated.append(plot_error_analysis(df, source))

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DONE — Charts generated:")
    for path in generated:
        print(f"  {os.path.basename(path)}")
    print(f"\nData source for error/prediction charts:")
    print(f"  {'DB predictions' if use_db else 'Model test set'}")
    print(f"  {source}")
    print(f"\nAll saved to: {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
