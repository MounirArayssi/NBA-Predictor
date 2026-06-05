"""
Feature Analysis for NBA Predictor Models

This script performs comprehensive feature importance analysis for both team and player models.
It generates visualizations, correlation matrices, and detailed reports to help understand
what features are driving predictions.

Usage:
    python feature_analysis.py --output analysis_report.txt
    python feature_analysis.py --model team --plots
    python feature_analysis.py --model player --stat points
"""

import sys
import os
import pickle
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from train import FEATURE_COLS as TEAM_FEATURE_COLS

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid tkinter threading issues
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr, spearmanr

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent  # Go up from model/ to NBA-Predictor/
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATABASE_URL
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Create database session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def get_session():
    """Create a new database session."""
    return SessionLocal()

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

TEAM_FEATURE_COLS = [
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


PLAYER_FEATURE_COLS = [
    # Player recent form (10 game)
    'player_avg_points_l10',
    'player_avg_reb_l10',
    'player_avg_ast_l10',
    'player_avg_min_l10',
    'player_avg_usage_l10',
    'player_avg_ts_l10',
    'player_avg_fg_l10',
    'player_avg_fg3_l10',
    'player_avg_pm_l10',
    'player_avg_tov_l10',

    # Recent form (5 game)
    'player_avg_points_l5',
    'player_avg_reb_l5',
    'player_avg_ast_l5',
    'player_avg_usage_l5',
    'player_avg_ts_l5',

    # Momentum
    'points_momentum',
    'usage_momentum',
    'ts_momentum',

    # Game context
    'is_home',
    'is_playoff',
    'pace_factor',
    'off_environment',
    'usage_rate_clean',

    # Opponent context
    'opp_def_rating',
    'opp_pace',
    'team_off_rating',

    'player_avg_efg_l10',
    'player_avg_oreb_pct_l10',
    'player_avg_dreb_pct_l10',
    'player_avg_ast_pct_l10',
    'player_avg_net_rating_l10',
    'player_avg_off_rating_l10',
]

FEATURE_GROUPS = {
    'Home Team Form': [f for f in TEAM_FEATURE_COLS if f.startswith('home_') and any(x in f for x in ['avg', 'rating', 'pace', 'pct', 'rate'])],
    'Away Team Form': [f for f in TEAM_FEATURE_COLS if f.startswith('away_') and any(x in f for x in ['avg', 'rating', 'pace', 'pct', 'rate'])],
    'Rest & Schedule': ['home_rest_days', 'away_rest_days', 'rest_advantage', 'home_back_to_back', 'away_back_to_back'],
    'Head to Head': [f for f in TEAM_FEATURE_COLS if 'h2h_' in f],
    'Playoff Context': ['is_playoff', 'series_game_num', 'home_series_wins', 'away_series_wins', 'is_elimination', 'series_momentum', 'series_pressure', 'series_wins_diff'],
    'Team Similarity': [f for f in TEAM_FEATURE_COLS if 'proxy' in f or 'sim_' in f],
    'Playoff Elevation': [f for f in TEAM_FEATURE_COLS if 'elevation' in f],
    'Scoring Variance': [f for f in TEAM_FEATURE_COLS if any(x in f for x in ['std', 'consistency', 'variance'])],
    'Matchup Features': [f for f in TEAM_FEATURE_COLS if any(x in f for x in ['matchup', 'differential', 'combined', 'vs'])],
    'Momentum': [f for f in TEAM_FEATURE_COLS if 'momentum' in f],
    'Clutch Performance': [f for f in TEAM_FEATURE_COLS if any(x in f for x in ['q4', 'clutch'])],
    'Defensive Matchups': [f for f in TEAM_FEATURE_COLS if 'def_vs' in f or 'def_style' in f],
    'Shooting Consistency': [f for f in TEAM_FEATURE_COLS if 'bad_night' in f],
    'Vegas Context': ['has_vegas_odds'],
}

PLAYER_FEATURE_GROUPS = {
    'Recent Scoring (L10)': ['player_avg_points_l10', 'player_avg_fg_l10', 'player_avg_fg3_l10'],
    'Recent Form (L5)': ['player_avg_points_l5', 'player_avg_reb_l5', 'player_avg_ast_l5', 'player_avg_usage_l5', 'player_avg_ts_l5'],
    'Playmaking': ['player_avg_ast_l10', 'player_avg_ast_pct_l10'],
    'Rebounding': ['player_avg_reb_l10', 'player_avg_oreb_pct_l10', 'player_avg_dreb_pct_l10'],
    'Efficiency': ['player_avg_ts_l10', 'player_avg_efg_l10', 'player_avg_off_rating_l10'],
    'Usage & Minutes': ['player_avg_min_l10', 'player_avg_usage_l10', 'usage_rate_clean'],
    'Impact': ['player_avg_pm_l10', 'player_avg_net_rating_l10'],
    'Momentum': ['points_momentum', 'usage_momentum', 'ts_momentum'],
    'Context': ['is_home', 'is_playoff', 'pace_factor', 'off_environment'],
    'Opponent': ['opp_def_rating', 'opp_pace'],
    'Team Environment': ['team_off_rating'],
    'Ball Security': ['player_avg_tov_l10'],
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_team_model_data():
    """Load team model and training data."""
    print("Loading team model data...")
    
    model_home_path = PROJECT_ROOT / 'model' / 'model_home.pkl'
    model_away_path = PROJECT_ROOT / 'model' / 'model_away.pkl'
    
    if not model_home_path.exists() or not model_away_path.exists():
        raise FileNotFoundError("Team models not found. Run model/train.py first.")
    
    with open(model_home_path, 'rb') as f:
        model_home = pickle.load(f)
    
    with open(model_away_path, 'rb') as f:
        model_away = pickle.load(f)
    
    # Load data the same way train.py does - by calling build_feature_dataset
    print("Building feature dataset (this may take a minute)...")
    from data.ingestion.build_features import build_feature_dataset
    
    df = build_feature_dataset(window=10)
    
    # Drop rows with missing features
    before = len(df)
    df = df.dropna(subset=TEAM_FEATURE_COLS + ['home_score', 'away_score'])
    after = len(df)
    
    if before > after:
        print(f"Dropped {before - after} rows with missing values ({after} remaining)")
    
    print(f"Loaded {len(df)} games with all features")
    
    return model_home, model_away, df


def load_player_model_data(stat='points'):
    """Load player model and training data."""
    print(f"Loading player model data for {stat}...")
    
    # Player models are saved in a single file
    model_path = PROJECT_ROOT / 'model' / 'player_models.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Player models not found at {model_path}. Run model/train_players.py first.")
    
    with open(model_path, 'rb') as f:
        models_dict = pickle.load(f)
    
    if stat not in models_dict:
        raise ValueError(f"Stat '{stat}' not found in player models. Available: {list(models_dict.keys())}")
    
    model = models_dict[stat]
    
    # Load data the same way train_players.py does
    print("Loading player training data from database...")
    
    # Import the functions from train_players.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_players", 
        PROJECT_ROOT / 'model' / 'train_players.py'
    )
    train_players_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_players_module)
    
    # Use the same data loading functions
    df = train_players_module.load_player_training_data()
    df = train_players_module.add_player_features(df)
    
    print(f"Loaded {len(df)} player-game records")
    
    return model, df


# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def calculate_feature_importance(model, X, y, feature_names, n_repeats=10):
    """Calculate multiple types of feature importance."""
    print("Calculating feature importance...")
    
    # 1. Built-in feature importance (Gini/impurity-based)
    gini_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 2. Permutation importance (more reliable for correlated features)
    print("  Computing permutation importance (this may take a minute)...")
    perm_importance = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )
    
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    return gini_importance, perm_df


def plot_feature_importance(importance_df, title, output_path, top_n=30):
    """Plot feature importance."""
    plt.figure(figsize=(12, 10))
    
    top_features = importance_df.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def analyze_feature_groups(importance_df, feature_groups):
    """Analyze importance by feature groups."""
    group_importance = {}
    
    for group_name, features in feature_groups.items():
        group_features = importance_df[importance_df['feature'].isin(features)]
        if len(group_features) > 0:
            group_importance[group_name] = {
                'total_importance': group_features['importance'].sum(),
                'mean_importance': group_features['importance'].mean(),
                'num_features': len(group_features),
                'top_feature': group_features.iloc[0]['feature'] if len(group_features) > 0 else None,
                'top_importance': group_features.iloc[0]['importance'] if len(group_features) > 0 else 0
            }
    
    return pd.DataFrame(group_importance).T.sort_values('total_importance', ascending=False)


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def analyze_correlations(df, features, target, output_dir):
    """Analyze feature correlations with target and each other."""
    print("Analyzing correlations...")
    
    # Ensure target is numeric
    if df[target].dtype == 'object':
        df[target] = pd.to_numeric(df[target], errors='coerce')
    
    # Correlation with target
    correlations = []
    for feature in features:
        if feature in df.columns:
            # Ensure feature is numeric
            if df[feature].dtype == 'object':
                try:
                    df[feature] = pd.to_numeric(df[feature], errors='coerce')
                except:
                    print(f"  Skipping non-numeric feature: {feature}")
                    continue
            
            # Skip if all NaN after conversion
            if df[feature].isna().all():
                continue
                
            try:
                corr, pval = pearsonr(df[feature].fillna(0), df[target].fillna(0))
                correlations.append({
                    'feature': feature,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'p_value': pval
                })
            except Exception as e:
                print(f"  Skipping {feature} due to error: {e}")
                continue
    
    corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
    
    # Feature correlation matrix (top correlated features only)
    top_features = corr_df.head(20)['feature'].tolist()
    
    # Ensure all top features are numeric
    for feat in top_features:
        if df[feat].dtype == 'object':
            df[feat] = pd.to_numeric(df[feat], errors='coerce')
    
    corr_matrix = df[top_features].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix (Top 20 Features)')
    plt.tight_layout()
    
    output_path = output_dir / 'correlation_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    return corr_df, corr_matrix


# ============================================================================
# PREDICTION INFLUENCE ANALYSIS
# ============================================================================

def analyze_prediction_influence(model, X, feature_names, sample_size=1000):
    """Analyze how features influence predictions across different ranges."""
    print("Analyzing prediction influence...")
    
    # Sample data for efficiency
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # Get predictions
    predictions = model.predict(X_sample)
    
    # Analyze feature value ranges and their impact
    influence_data = []
    
    for i, feature in enumerate(feature_names):
        feature_values = X_sample[:, i]
        
        # Split into quartiles
        q1, q2, q3 = np.percentile(feature_values, [25, 50, 75])
        
        # Predictions by quartile
        low = predictions[feature_values <= q1]
        mid_low = predictions[(feature_values > q1) & (feature_values <= q2)]
        mid_high = predictions[(feature_values > q2) & (feature_values <= q3)]
        high = predictions[feature_values > q3]
        
        influence_data.append({
            'feature': feature,
            'mean_value': np.mean(feature_values),
            'std_value': np.std(feature_values),
            'pred_q1': np.mean(low) if len(low) > 0 else np.nan,
            'pred_q2': np.mean(mid_low) if len(mid_low) > 0 else np.nan,
            'pred_q3': np.mean(mid_high) if len(mid_high) > 0 else np.nan,
            'pred_q4': np.mean(high) if len(high) > 0 else np.nan,
            'range_impact': (np.mean(high) if len(high) > 0 else np.nan) - 
                          (np.mean(low) if len(low) > 0 else np.nan)
        })
    
    influence_df = pd.DataFrame(influence_data)
    influence_df['abs_range_impact'] = influence_df['range_impact'].abs()
    influence_df = influence_df.sort_values('abs_range_impact', ascending=False)
    
    return influence_df


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_team_report(output_dir):
    """Generate comprehensive team model analysis report."""
    print("\n" + "="*80)
    print("TEAM MODEL FEATURE ANALYSIS")
    print("="*80 + "\n")
    
    # Load models and data
    model_home, model_away, df = load_team_model_data()
    
    # Check which features are available in the dataframe
    available_features = [f for f in TEAM_FEATURE_COLS if f in df.columns]
    missing_features = [f for f in TEAM_FEATURE_COLS if f not in df.columns]
    
    if missing_features:
        print(f"\nWARNING: {len(missing_features)} features not found in data:")
        print(f"Available: {len(available_features)}/{len(TEAM_FEATURE_COLS)} features")
        if len(available_features) < 10:
            print("\nERROR: Too few features available for meaningful analysis.")
            print("Please run: python data/ingestion/build_features.py")
            print("This will create data/features.csv with all required features.")
            return
    
    # Use only available features
    feature_cols = available_features
    print(f"\nAnalyzing {len(feature_cols)} features...")
    
    # Prepare features and targets
    X = df[feature_cols].fillna(0).values
    y_home = df['home_score'].values
    y_away = df['away_score'].values
    
    # ========================================================================
    # HOME MODEL ANALYSIS
    # ========================================================================
    print("\n--- HOME MODEL ANALYSIS ---\n")
    
    gini_home, perm_home = calculate_feature_importance(
        model_home, X, y_home, feature_cols
    )
    
    # Plot importance
    plot_feature_importance(
        gini_home, 
        'Home Model - Gini Feature Importance (Top 30)',
        output_dir / 'team_home_gini_importance.png'
    )
    
    plot_feature_importance(
        perm_home,
        'Home Model - Permutation Feature Importance (Top 30)',
        output_dir / 'team_home_perm_importance.png'
    )
    
    # Group analysis
    group_importance_home = analyze_feature_groups(gini_home, FEATURE_GROUPS)
    
    # Correlation analysis
    corr_home, corr_matrix_home = analyze_correlations(
        df, feature_cols, 'home_score', output_dir
    )
    
    # Prediction influence
    influence_home = analyze_prediction_influence(model_home, X, feature_cols)
    
    # ========================================================================
    # AWAY MODEL ANALYSIS
    # ========================================================================
    print("\n--- AWAY MODEL ANALYSIS ---\n")
    
    gini_away, perm_away = calculate_feature_importance(
        model_away, X, y_away, feature_cols
    )
    
    plot_feature_importance(
        gini_away,
        'Away Model - Gini Feature Importance (Top 30)',
        output_dir / 'team_away_gini_importance.png'
    )
    
    plot_feature_importance(
        perm_away,
        'Away Model - Permutation Feature Importance (Top 30)',
        output_dir / 'team_away_perm_importance.png'
    )
    
    group_importance_away = analyze_feature_groups(gini_away, FEATURE_GROUPS)
    influence_away = analyze_prediction_influence(model_away, X, feature_cols)
    
    # ========================================================================
    # GENERATE TEXT REPORT
    # ========================================================================
    report_path = output_dir / 'team_model_analysis.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("NBA PREDICTOR - TEAM MODEL FEATURE ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Data: {len(df)} games\n")
        f.write(f"Features Analyzed: {len(feature_cols)}/{len(TEAM_FEATURE_COLS)}\n")
        if missing_features:
            f.write(f"Missing Features: {len(missing_features)}\n")
        f.write("\n")
        
        # ====================================================================
        # HOME MODEL REPORT
        # ====================================================================
        f.write("\n" + "="*80 + "\n")
        f.write("HOME MODEL ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("TOP 20 FEATURES (Gini Importance):\n")
        f.write("-"*80 + "\n")
        for idx, row in gini_home.head(20).iterrows():
            f.write(f"{row['feature']:40s} {row['importance']:.6f}\n")
        
        f.write("\n\nTOP 20 FEATURES (Permutation Importance):\n")
        f.write("-"*80 + "\n")
        for idx, row in perm_home.head(20).iterrows():
            f.write(f"{row['feature']:40s} {row['importance']:.6f} (±{row['std']:.6f})\n")
        
        f.write("\n\nFEATURE GROUP IMPORTANCE:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Group':30s} {'Total':>12s} {'Mean':>12s} {'Count':>8s} {'Top Feature':>25s}\n")
        f.write("-"*80 + "\n")
        for group, data in group_importance_home.iterrows():
            f.write(f"{group:30s} {data['total_importance']:>12.6f} {data['mean_importance']:>12.6f} "
                   f"{int(data['num_features']):>8d} {str(data['top_feature'])[:25]:>25s}\n")
        
        f.write("\n\nTOP 15 CORRELATIONS WITH HOME SCORE:\n")
        f.write("-"*80 + "\n")
        for idx, row in corr_home.head(15).iterrows():
            f.write(f"{row['feature']:40s} {row['correlation']:>8.4f} (p={row['p_value']:.4e})\n")
        
        f.write("\n\nPREDICTION INFLUENCE (Top 15 by Range Impact):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Feature':35s} {'Q1->Q4':>12s} {'Impact':>12s}\n")
        f.write("-"*80 + "\n")
        for idx, row in influence_home.head(15).iterrows():
            if not np.isnan(row['range_impact']):
                f.write(f"{row['feature']:35s} {row['pred_q1']:>5.1f}->{row['pred_q4']:>5.1f} "
                       f"{row['range_impact']:>12.2f}\n")
        
        # ====================================================================
        # AWAY MODEL REPORT
        # ====================================================================
        f.write("\n\n" + "="*80 + "\n")
        f.write("AWAY MODEL ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("TOP 20 FEATURES (Gini Importance):\n")
        f.write("-"*80 + "\n")
        for idx, row in gini_away.head(20).iterrows():
            f.write(f"{row['feature']:40s} {row['importance']:.6f}\n")
        
        f.write("\n\nTOP 20 FEATURES (Permutation Importance):\n")
        f.write("-"*80 + "\n")
        for idx, row in perm_away.head(20).iterrows():
            f.write(f"{row['feature']:40s} {row['importance']:.6f} (±{row['std']:.6f})\n")
        
        f.write("\n\nFEATURE GROUP IMPORTANCE:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Group':30s} {'Total':>12s} {'Mean':>12s} {'Count':>8s} {'Top Feature':>25s}\n")
        f.write("-"*80 + "\n")
        for group, data in group_importance_away.iterrows():
            f.write(f"{group:30s} {data['total_importance']:>12.6f} {data['mean_importance']:>12.6f} "
                   f"{int(data['num_features']):>8d} {str(data['top_feature'])[:25]:>25s}\n")
        
        f.write("\n\nPREDICTION INFLUENCE (Top 15 by Range Impact):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Feature':35s} {'Q1->Q4':>12s} {'Impact':>12s}\n")
        f.write("-"*80 + "\n")
        for idx, row in influence_away.head(15).iterrows():
            if not np.isnan(row['range_impact']):
                f.write(f"{row['feature']:35s} {row['pred_q1']:>5.1f}->{row['pred_q4']:>5.1f} "
                       f"{row['range_impact']:>12.2f}\n")
        
        # ====================================================================
        # KEY INSIGHTS
        # ====================================================================
        f.write("\n\n" + "="*80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("="*80 + "\n\n")
        
        # Most important features overall
        top_home = set(gini_home.head(10)['feature'])
        top_away = set(gini_away.head(10)['feature'])
        common_top = top_home & top_away
        
        f.write(f"Features in Top 10 for BOTH models ({len(common_top)}):\n")
        for feat in common_top:
            home_rank = gini_home[gini_home['feature'] == feat].index[0] + 1
            away_rank = gini_away[gini_away['feature'] == feat].index[0] + 1
            f.write(f"  {feat:40s} (Home: #{home_rank}, Away: #{away_rank})\n")
        
        # Most different features
        f.write("\n\nFeatures with Biggest Importance Difference:\n")
        importance_diff = gini_home.merge(gini_away, on='feature', suffixes=('_home', '_away'))
        importance_diff['diff'] = abs(importance_diff['importance_home'] - importance_diff['importance_away'])
        importance_diff = importance_diff.sort_values('diff', ascending=False)
        
        for idx, row in importance_diff.head(10).iterrows():
            f.write(f"  {row['feature']:40s} Home: {row['importance_home']:.6f}, Away: {row['importance_away']:.6f}\n")
        
        # Group dominance
        f.write("\n\nMost Important Feature Groups (Home):\n")
        for group, data in group_importance_home.head(5).iterrows():
            f.write(f"  {group:30s} {data['total_importance']:.4f}\n")
        
        f.write("\n\nMost Important Feature Groups (Away):\n")
        for group, data in group_importance_away.head(5).iterrows():
            f.write(f"  {group:30s} {data['total_importance']:.4f}\n")
    
    print(f"\nTeam model report saved to: {report_path}")
    
    # Save CSVs for further analysis
    gini_home.to_csv(output_dir / 'team_home_gini_importance.csv', index=False)
    perm_home.to_csv(output_dir / 'team_home_perm_importance.csv', index=False)
    gini_away.to_csv(output_dir / 'team_away_gini_importance.csv', index=False)
    perm_away.to_csv(output_dir / 'team_away_perm_importance.csv', index=False)
    corr_home.to_csv(output_dir / 'team_correlations.csv', index=False)
    influence_home.to_csv(output_dir / 'team_home_influence.csv', index=False)
    influence_away.to_csv(output_dir / 'team_away_influence.csv', index=False)
    
    print("\nCSV files saved:")
    print(f"  - team_home_gini_importance.csv")
    print(f"  - team_home_perm_importance.csv")
    print(f"  - team_away_gini_importance.csv")
    print(f"  - team_away_perm_importance.csv")
    print(f"  - team_correlations.csv")
    print(f"  - team_home_influence.csv")
    print(f"  - team_away_influence.csv")


def generate_player_report(output_dir, stat='points'):
    """Generate comprehensive player model analysis report."""
    print("\n" + "="*80)
    print(f"PLAYER MODEL FEATURE ANALYSIS - {stat.upper()}")
    print("="*80 + "\n")
    
    # Load model and data
    model, df = load_player_model_data(stat)
    
    # Prepare features
    X = df[PLAYER_FEATURE_COLS].fillna(0).values
    y = df[stat].values
    
    # Feature importance
    gini_importance, perm_importance = calculate_feature_importance(
        model, X, y, PLAYER_FEATURE_COLS
    )
    
    # Plot importance
    plot_feature_importance(
        gini_importance,
        f'Player {stat.title()} Model - Gini Feature Importance',
        output_dir / f'player_{stat}_gini_importance.png'
    )
    
    plot_feature_importance(
        perm_importance,
        f'Player {stat.title()} Model - Permutation Feature Importance',
        output_dir / f'player_{stat}_perm_importance.png'
    )
    
    # Group analysis
    group_importance = analyze_feature_groups(gini_importance, PLAYER_FEATURE_GROUPS)
    
    # Correlation analysis
    corr_df, corr_matrix = analyze_correlations(
        df, PLAYER_FEATURE_COLS, stat, output_dir
    )
    
    # Prediction influence
    influence_df = analyze_prediction_influence(model, X, PLAYER_FEATURE_COLS)
    
    # ========================================================================
    # GENERATE TEXT REPORT
    # ========================================================================
    report_path = output_dir / f'player_{stat}_analysis.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"NBA PREDICTOR - PLAYER {stat.upper()} MODEL FEATURE ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Data: {len(df)} player-game records\n")
        f.write(f"Features: {len(PLAYER_FEATURE_COLS)}\n\n")
        
        f.write("TOP 25 FEATURES (Gini Importance):\n")
        f.write("-"*80 + "\n")
        for idx, row in gini_importance.head(25).iterrows():
            f.write(f"{row['feature']:40s} {row['importance']:.6f}\n")
        
        f.write("\n\nTOP 25 FEATURES (Permutation Importance):\n")
        f.write("-"*80 + "\n")
        for idx, row in perm_importance.head(25).iterrows():
            f.write(f"{row['feature']:40s} {row['importance']:.6f} (±{row['std']:.6f})\n")
        
        f.write("\n\nFEATURE GROUP IMPORTANCE:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Group':30s} {'Total':>12s} {'Mean':>12s} {'Count':>8s} {'Top Feature':>25s}\n")
        f.write("-"*80 + "\n")
        for group, data in group_importance.iterrows():
            f.write(f"{group:30s} {data['total_importance']:>12.6f} {data['mean_importance']:>12.6f} "
                   f"{int(data['num_features']):>8d} {str(data['top_feature'])[:25]:>25s}\n")
        
        f.write("\n\nTOP 20 CORRELATIONS WITH ACTUAL " + stat.upper() + ":\n")
        f.write("-"*80 + "\n")
        for idx, row in corr_df.head(20).iterrows():
            f.write(f"{row['feature']:40s} {row['correlation']:>8.4f} (p={row['p_value']:.4e})\n")
        
        f.write("\n\nPREDICTION INFLUENCE (Top 20 by Range Impact):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Feature':35s} {'Q1->Q4':>12s} {'Impact':>12s}\n")
        f.write("-"*80 + "\n")
        for idx, row in influence_df.head(20).iterrows():
            if not np.isnan(row['range_impact']):
                f.write(f"{row['feature']:35s} {row['pred_q1']:>5.1f}->{row['pred_q4']:>5.1f} "
                       f"{row['range_impact']:>12.2f}\n")
        
        # ====================================================================
        # KEY INSIGHTS
        # ====================================================================
        f.write("\n\n" + "="*80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("="*80 + "\n\n")
        
        # Top predictors
        f.write(f"Most Predictive Features for {stat.title()}:\n")
        top_5 = gini_importance.head(5)
        for idx, row in top_5.iterrows():
            f.write(f"  #{idx+1}: {row['feature']:35s} ({row['importance']:.4f})\n")
        
        # Recent vs historical
        recent_feats = [f for f in PLAYER_FEATURE_COLS if '_l5' in f]
        historical_feats = [f for f in PLAYER_FEATURE_COLS if '_l10' in f and '_l5' not in f]
        
        recent_importance = gini_importance[gini_importance['feature'].isin(recent_feats)]['importance'].sum()
        historical_importance = gini_importance[gini_importance['feature'].isin(historical_feats)]['importance'].sum()
        
        f.write(f"\nRecent Form (L5) vs Historical (L10):\n")
        f.write(f"  L5 Total Importance:  {recent_importance:.4f}\n")
        f.write(f"  L10 Total Importance: {historical_importance:.4f}\n")
        f.write(f"  Ratio (L5/L10):       {recent_importance/historical_importance if historical_importance > 0 else 0:.2f}\n")
        
        # Context importance
        context_feats = ['is_home', 'is_playoff', 'opp_def_rating', 'opp_pace']
        context_importance = gini_importance[gini_importance['feature'].isin(context_feats)]
        
        f.write(f"\nContext Feature Importance:\n")
        for idx, row in context_importance.iterrows():
            f.write(f"  {row['feature']:20s} {row['importance']:.6f}\n")
        
        # Momentum features
        momentum_feats = [f for f in PLAYER_FEATURE_COLS if 'momentum' in f]
        momentum_importance = gini_importance[gini_importance['feature'].isin(momentum_feats)]
        
        f.write(f"\nMomentum Feature Importance:\n")
        for idx, row in momentum_importance.iterrows():
            f.write(f"  {row['feature']:20s} {row['importance']:.6f}\n")
    
    print(f"\nPlayer {stat} model report saved to: {report_path}")
    
    # Save CSVs
    gini_importance.to_csv(output_dir / f'player_{stat}_gini_importance.csv', index=False)
    perm_importance.to_csv(output_dir / f'player_{stat}_perm_importance.csv', index=False)
    corr_df.to_csv(output_dir / f'player_{stat}_correlations.csv', index=False)
    influence_df.to_csv(output_dir / f'player_{stat}_influence.csv', index=False)
    
    print(f"\nCSV files saved:")
    print(f"  - player_{stat}_gini_importance.csv")
    print(f"  - player_{stat}_perm_importance.csv")
    print(f"  - player_{stat}_correlations.csv")
    print(f"  - player_{stat}_influence.csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive feature analysis for NBA Predictor models'
    )
    parser.add_argument(
        '--model',
        choices=['team', 'player', 'all'],
        default='all',
        help='Which model to analyze (default: all)'
    )
    parser.add_argument(
        '--stat',
        choices=['points', 'rebounds', 'assists'],
        default='points',
        help='Player stat to analyze (default: points)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis',
        help='Output directory for reports and visualizations (default: analysis/)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Run analysis
    if args.model in ['team', 'all']:
        generate_team_report(output_dir)
    
    if args.model in ['player', 'all']:
        generate_player_report(output_dir, args.stat)
    
    print("\n" + "="*80)
    print("FEATURE ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll reports and visualizations saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - Text reports (.txt)")
    print("  - Feature importance plots (.png)")
    print("  - Correlation matrices (.png)")
    print("  - CSV exports for further analysis (.csv)")


if __name__ == '__main__':
    main()