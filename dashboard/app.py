import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import streamlit as st
import pandas as pd
from sqlalchemy import text
from datetime import datetime, timedelta
from data.storage.db import engine

# Try to import plotly, but make it optional
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="NBA Predictor Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# Streamlit Native Styling (CSS only for colors/backgrounds)
# -----------------------------
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }
        
        .stMetric {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid #334155;
        }
        
        div[data-testid="stMetricValue"] {
            color: #3b82f6;
            font-size: 2rem;
        }
        
        div[data-testid="stMetricLabel"] {
            color: #94a3b8;
        }
        
        .stAlert {
            background-color: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
        }
        
        h1, h2, h3 {
            color: #f8fafc !important;
        }
        
        p {
            color: #cbd5e1 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Data Loading Functions
# -----------------------------
@st.cache_data(ttl=60)
def load_query(query: str) -> pd.DataFrame:
    """Load data from database with caching"""
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


@st.cache_data(ttl=60)
def load_recent_performance() -> pd.DataFrame:
    """Load recent performance trends"""
    query = """
    SELECT
        DATE_TRUNC('day', g.game_date) as date,
        COUNT(*) as games,
        AVG(CASE WHEN p.winner_correct THEN 1 ELSE 0 END) as win_pct,
        AVG(p.margin_error) as avg_margin_err,
        AVG(p.total_score_error) as avg_total_err
    FROM predictions p
    JOIN games g ON p.game_id = g.game_id
    WHERE p.is_official = true
      AND p.evaluated_at IS NOT NULL
      AND g.game_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY DATE_TRUNC('day', g.game_date)
    ORDER BY date DESC;
    """
    return load_query(query)


# -----------------------------
# Helper Functions
# -----------------------------
def safe_num(value, default=0):
    """Safely convert to number"""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def format_score(value):
    """Format score for display"""
    try:
        return f"{float(value):.0f}"
    except Exception:
        return "-"


def format_percentage(value):
    """Format percentage for display"""
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "-"


def get_confidence_emoji(score):
    """Get confidence emoji"""
    score = safe_num(score, 0)
    if score >= 0.75:
        return "🟢"
    if score >= 0.50:
        return "🟡"
    return "🔴"


def get_confidence_text(score):
    """Get confidence text"""
    score = safe_num(score, 0)
    if score >= 0.75:
        return "High Confidence"
    if score >= 0.50:
        return "Medium Confidence"
    return "Low Confidence"


def get_status_emoji(is_final, evaluated_at):
    """Get game status emoji"""
    if evaluated_at is not None and not pd.isna(evaluated_at):
        return "✅"
    if bool(is_final):
        return "⏳"
    return "🕐"


def get_status_text(is_final, evaluated_at):
    """Get game status text"""
    if evaluated_at is not None and not pd.isna(evaluated_at):
        return "Evaluated"
    if bool(is_final):
        return "Ready to Evaluate"
    return "Pending Result"


def winner_from_row(row):
    """Determine winner from prediction"""
    home = row["home_team"]
    away = row["away_team"]
    home_score = safe_num(row["home_score_predicted"])
    away_score = safe_num(row["away_score_predicted"])
    return home if home_score > away_score else away


def readable_lock_reason(reason):
    """Convert lock reason to readable text"""
    mapping = {
        "first_game_day_prediction": "🎯 First official prediction of the day",
        "pre_tipoff_lock_window_override": "🔒 Updated inside pre-tip lock window",
        "non_official_rerun": "🔄 Extra rerun, not official",
        "after_tipoff_not_official": "⏰ Generated after tipoff",
        "snapshot_only": "📸 Snapshot only",
    }
    return mapping.get(str(reason), str(reason))


def create_performance_chart(df):
    """Create performance trend chart"""
    if df.empty or not PLOTLY_AVAILABLE:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['win_pct'] * 100,
        mode='lines+markers',
        name='Win %',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='<b>%{x|%b %d}</b><br>Win Rate: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Win Rate Trend (Last 30 Days)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1'),
        hovermode='x unified',
        xaxis=dict(
            showgrid=True,
            gridcolor='#334155',
            title='Date'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#334155',
            title='Win Rate (%)',
            range=[0, 100]
        ),
        height=350
    )
    
    return fig


# -----------------------------
# Header
# -----------------------------
st.title("🏀 NBA Predictor Dashboard")
st.caption("Track official predictions, compare results, and monitor model performance in real-time")
st.divider()


# -----------------------------
# Sidebar Filters
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("📅 Date Range")
    date_filter = st.radio(
        "Show predictions from:",
        ["Today", "Last 7 Days", "Last 30 Days", "All Time"],
        index=1
    )
    
    st.subheader("🎯 Status Filter")
    status_filter = st.multiselect(
        "Filter by status:",
        ["Pending Result", "Ready to Evaluate", "Evaluated"],
        default=["Pending Result", "Ready to Evaluate", "Evaluated"]
    )
    
    st.subheader("📊 Display Options")
    show_charts = st.checkbox("Show performance charts", value=True)
    show_raw_data = st.checkbox("Show raw data tables", value=False)
    
    st.divider()
    st.subheader("🔄 Refresh Data")
    if st.button("Refresh Dashboard", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# -----------------------------
# Load Data
# -----------------------------
official_df = load_query("""
    SELECT
        p.prediction_id,
        p.game_id,
        g.game_date,
        p.predicted_at,
        ht.abbreviation AS home_team,
        at.abbreviation AS away_team,
        p.home_score_predicted,
        p.away_score_predicted,
        p.model_margin,
        p.model_total,
        p.confidence_score,
        p.is_official,
        p.prediction_type,
        p.lock_reason,
        g.status,
        g.is_final,
        g.home_score,
        g.away_score,
        p.evaluated_at,
        p.winner_correct,
        p.home_score_error,
        p.away_score_error,
        p.margin_error,
        p.total_score_error
    FROM predictions p
    JOIN games g ON p.game_id = g.game_id
    JOIN teams ht ON g.home_team_id = ht.team_id
    JOIN teams at ON g.away_team_id = at.team_id
    WHERE p.is_official = true
    ORDER BY g.game_date DESC, p.predicted_at DESC;
""")

performance_df = load_query("""
    SELECT
        COUNT(*) AS games_evaluated,
        AVG(CASE WHEN winner_correct THEN 1 ELSE 0 END) AS winner_accuracy,
        AVG(home_score_error) AS avg_home_error,
        AVG(away_score_error) AS avg_away_error,
        AVG(margin_error) AS avg_margin_error,
        AVG(total_score_error) AS avg_total_error
    FROM predictions
    WHERE is_official = true
      AND evaluated_at IS NOT NULL;
""")


# -----------------------------
# Apply Filters
# -----------------------------
if not official_df.empty:
    # Date filter
    if date_filter == "Today":
        official_df = official_df[official_df["game_date"] == pd.Timestamp.now().date()]
    elif date_filter == "Last 7 Days":
        cutoff = pd.Timestamp.now().date() - timedelta(days=7)
        official_df = official_df[official_df["game_date"] >= cutoff]
    elif date_filter == "Last 30 Days":
        cutoff = pd.Timestamp.now().date() - timedelta(days=30)
        official_df = official_df[official_df["game_date"] >= cutoff]
    
    # Status filter
    official_df["dashboard_status"] = official_df.apply(
        lambda r: (
            "Evaluated"
            if pd.notna(r["evaluated_at"])
            else "Ready to Evaluate"
            if bool(r["is_final"])
            else "Pending Result"
        ),
        axis=1
    )
    
    official_df = official_df[official_df["dashboard_status"].isin(status_filter)]


# -----------------------------
# Summary Metrics
# -----------------------------
total_official = len(official_df)
pending_count = 0
ready_count = 0
evaluated_count = 0

if not official_df.empty:
    pending_count = len(
        official_df[
            (official_df["is_final"] == False) &
            (official_df["evaluated_at"].isna())
        ]
    )
    ready_count = len(
        official_df[
            (official_df["is_final"] == True) &
            (official_df["evaluated_at"].isna())
        ]
    )
    evaluated_count = len(
        official_df[
            official_df["evaluated_at"].notna()
        ]
    )

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📊 Official Predictions",
        f"{total_official}",
        help="Total number of official predictions made"
    )

with col2:
    st.metric(
        "🕐 Pending Results",
        f"{pending_count}",
        help="Games that haven't finished yet"
    )

with col3:
    st.metric(
        "⏳ Ready to Evaluate",
        f"{ready_count}",
        help="Finished games awaiting evaluation"
    )

with col4:
    st.metric(
        "✅ Evaluated",
        f"{evaluated_count}",
        help="Games that have been evaluated"
    )

st.divider()


# -----------------------------
# Performance Summary
# -----------------------------
if not performance_df.empty and performance_df.iloc[0]["games_evaluated"] > 0:
    st.header("📈 Model Performance Overview")
    
    perf_row = performance_df.iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Win Accuracy",
            format_percentage(perf_row["winner_accuracy"]),
            help="Percentage of games where winner was correctly predicted"
        )
    
    with col2:
        st.metric(
            "📏 Avg Margin Error",
            f"{safe_num(perf_row['avg_margin_error']):.2f} pts",
            help="Average error in predicted margin"
        )
    
    with col3:
        st.metric(
            "🎲 Avg Total Error",
            f"{safe_num(perf_row['avg_total_error']):.2f} pts",
            help="Average error in predicted total score"
        )
    
    with col4:
        games_eval = int(perf_row["games_evaluated"])
        st.metric(
            "📊 Sample Size",
            f"{games_eval} games",
            help="Number of evaluated predictions"
        )
    
    st.divider()


# -----------------------------
# Performance Charts
# -----------------------------
if show_charts and evaluated_count > 0 and PLOTLY_AVAILABLE:
    st.header("📊 Performance Trends")
    
    recent_perf = load_recent_performance()
    
    if not recent_perf.empty:
        perf_chart = create_performance_chart(recent_perf)
        if perf_chart:
            st.plotly_chart(perf_chart, use_container_width=True)
    
    st.divider()


# -----------------------------
# Main Content Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "🏀 Predictions",
    "📋 Full Tracker",
    "🔍 Game Details"
])


# -----------------------------
# Tab 1: Predictions (Using Native Streamlit Components)
# -----------------------------
with tab1:
    st.header("🏀 Official Predictions")
    
    if official_df.empty:
        st.info("📭 No predictions found matching your current filters. Try adjusting the date range or status filters in the sidebar.")
    else:
        # Group by date
        dates = sorted(
            official_df["game_date"].dropna().astype(str).unique(),
            reverse=True
        )
        
        for date in dates:
            day_df = official_df[official_df["game_date"].astype(str) == date]
            
            st.subheader(f"📅 {date}")
            st.caption(f"{len(day_df)} prediction(s)")
            
            for idx, row in day_df.iterrows():
                # Use container and columns for layout
                with st.container():
                    # Header row with badges
                    badge_col1, badge_col2 = st.columns([3, 1])
                    
                    with badge_col1:
                        conf_emoji = get_confidence_emoji(row["confidence_score"])
                        conf_text = get_confidence_text(row["confidence_score"])
                        status_emoji = get_status_emoji(row["is_final"], row["evaluated_at"])
                        status_text = get_status_text(row["is_final"], row["evaluated_at"])
                        
                        st.markdown(f"**{conf_emoji} {conf_text}** | **{status_emoji} {status_text}**")
                    
                    # Game matchup
                    home = row["home_team"]
                    away = row["away_team"]
                    st.subheader(f"{away} @ {home}")
                    
                    # Score display
                    score_col1, score_col2, score_col3 = st.columns([2, 1, 2])
                    
                    with score_col1:
                        st.metric(
                            label=f"🏠 {home}",
                            value=format_score(row["home_score_predicted"])
                        )
                    
                    with score_col2:
                        st.markdown("<div style='text-align: center; padding-top: 20px;'>", unsafe_allow_html=True)
                        st.markdown("**vs**")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with score_col3:
                        st.metric(
                            label=f"✈️ {away}",
                            value=format_score(row["away_score_predicted"])
                        )
                    
                    # Game details
                    detail_col1, detail_col2, detail_col3 = st.columns(3)
                    
                    with detail_col1:
                        winner = winner_from_row(row)
                        st.markdown(f"**🏆 Winner:** {winner}")
                    
                    with detail_col2:
                        margin = safe_num(row["model_margin"])
                        st.markdown(f"**📊 Margin:** {margin:+.1f}")
                    
                    with detail_col3:
                        total = safe_num(row["model_total"])
                        st.markdown(f"**🎯 Total:** {total:.1f}")
                    
                    # Add actual results if available
                    if row["home_score"] is not None and not pd.isna(row["home_score"]):
                        st.markdown("---")
                        st.markdown("**Actual Final Score**")
                        
                        actual_col1, actual_col2, actual_col3 = st.columns([2, 1, 2])
                        
                        with actual_col1:
                            st.metric(
                                label=f"{home}",
                                value=format_score(row["home_score"]),
                                delta=None
                            )
                        
                        with actual_col2:
                            st.markdown("<div style='text-align: center; padding-top: 20px;'>", unsafe_allow_html=True)
                            st.markdown("**-**")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with actual_col3:
                            st.metric(
                                label=f"{away}",
                                value=format_score(row["away_score"]),
                                delta=None
                            )
                        
                        if pd.notna(row["evaluated_at"]):
                            correct = bool(row["winner_correct"])
                            result_emoji = "✅" if correct else "❌"
                            result_text = "Correct" if correct else "Incorrect"
                            margin_err = safe_num(row["margin_error"])
                            
                            result_col1, result_col2 = st.columns(2)
                            with result_col1:
                                st.markdown(f"**{result_emoji} Winner Prediction:** {result_text}")
                            with result_col2:
                                st.markdown(f"**Error:** {margin_err:.1f} pts")
                    
                    # Footer info
                    st.caption(readable_lock_reason(row["lock_reason"]))
                    st.caption(f"Locked: {row['predicted_at']}")
                    
                    st.markdown("---")
            
            st.markdown("")


# -----------------------------
# Tab 2: Full Tracker
# -----------------------------
with tab2:
    st.header("📋 Official Prediction Tracker")
    st.caption("Complete overview of all official predictions and their status")
    
    if official_df.empty:
        st.info("No predictions to track with current filters.")
    else:
        tracker = official_df.copy()
        tracker["matchup"] = tracker["away_team"] + " @ " + tracker["home_team"]
        
        # Summary cards
        st.subheader("Status Breakdown")
        
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        
        with sum_col1:
            st.metric("🕐 Pending Result", pending_count)
        
        with sum_col2:
            st.metric("⏳ Ready to Evaluate", ready_count)
        
        with sum_col3:
            st.metric("✅ Evaluated", evaluated_count)
        
        st.divider()
        
        # Full table
        st.subheader("All Predictions")
        
        display_cols = [
            "game_date",
            "matchup",
            "dashboard_status",
            "home_score_predicted",
            "away_score_predicted",
            "home_score",
            "away_score",
            "winner_correct",
            "margin_error",
            "confidence_score",
        ]
        
        display_df = tracker[display_cols].copy()
        display_df.columns = [
            "Date",
            "Matchup",
            "Status",
            "Pred Home",
            "Pred Away",
            "Actual Home",
            "Actual Away",
            "Winner ✓",
            "Margin Error",
            "Confidence"
        ]
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        if show_raw_data:
            with st.expander("📄 Show complete raw data"):
                st.dataframe(tracker, use_container_width=True)


# -----------------------------
# Tab 3: Game Details
# -----------------------------
with tab3:
    st.header("🔍 Game Detail Inspector")
    st.caption("Deep dive into individual predictions")
    
    if official_df.empty:
        st.info("No games available with current filters.")
    else:
        detail_df = official_df.copy()
        detail_df["label"] = (
            detail_df["game_date"].astype(str)
            + " — "
            + detail_df["away_team"]
            + " @ "
            + detail_df["home_team"]
            + " (ID: "
            + detail_df["prediction_id"].astype(str)
            + ")"
        )
        
        selected_label = st.selectbox(
            "🎯 Select a game to inspect:",
            options=detail_df["label"].tolist(),
            key="game_detail_selector"
        )
        
        row = detail_df[detail_df["label"] == selected_label].iloc[0]
        
        home = row["home_team"]
        away = row["away_team"]
        
        # Game Header
        st.subheader(f"{away} @ {home}")
        st.caption(f"📅 Game Date: {row['game_date']} | 🔒 Locked: {row['predicted_at']}")
        
        st.divider()
        
        # Prediction Details
        st.subheader("🎯 Prediction Details")
        
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        with pred_col1:
            conf_emoji = get_confidence_emoji(row["confidence_score"])
            conf_text = get_confidence_text(row["confidence_score"])
            st.metric(
                "Confidence",
                f"{conf_emoji} {conf_text}",
                delta=f"{safe_num(row['confidence_score']):.1%}"
            )
        
        with pred_col2:
            st.metric("Predicted Winner", winner_from_row(row))
        
        with pred_col3:
            status_emoji = get_status_emoji(row["is_final"], row["evaluated_at"])
            status_text = get_status_text(row["is_final"], row["evaluated_at"])
            st.metric("Status", f"{status_emoji} {status_text}")
        
        st.divider()
        
        # Score Comparison
        st.subheader("📊 Score Comparison")
        
        score_col1, score_col2 = st.columns(2)
        
        with score_col1:
            st.markdown("**Predicted Score**")
            pred_score_col1, pred_score_col2 = st.columns(2)
            with pred_score_col1:
                st.metric(home, format_score(row['home_score_predicted']))
            with pred_score_col2:
                st.metric(away, format_score(row['away_score_predicted']))
        
        with score_col2:
            if row["home_score"] is not None and not pd.isna(row["home_score"]):
                st.markdown("**Actual Score**")
                actual_score_col1, actual_score_col2 = st.columns(2)
                with actual_score_col1:
                    st.metric(home, format_score(row['home_score']))
                with actual_score_col2:
                    st.metric(away, format_score(row['away_score']))
            else:
                st.info("Game not final yet")
        
        st.divider()
        
        # Model Insights
        st.subheader("🧠 Model Insights")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            margin = safe_num(row["model_margin"])
            favored = home if margin > 0 else away
            st.metric(
                "Predicted Margin",
                f"{margin:+.1f}",
                delta=f"{favored} favored"
            )
        
        with insight_col2:
            st.metric("Predicted Total", f"{safe_num(row['model_total']):.1f}")
        
        with insight_col3:
            st.metric("Prediction Type", row['prediction_type'])
        
        st.divider()
        
        # Evaluation Results
        if pd.notna(row["evaluated_at"]):
            st.subheader("✅ Evaluation Results")
            
            eval_col1, eval_col2, eval_col3, eval_col4 = st.columns(4)
            
            with eval_col1:
                correct = bool(row["winner_correct"])
                st.metric(
                    "Winner Prediction",
                    "✅ Correct" if correct else "❌ Incorrect"
                )
            
            with eval_col2:
                st.metric("Margin Error", f"{safe_num(row['margin_error']):.2f} pts")
            
            with eval_col3:
                st.metric("Total Error", f"{safe_num(row['total_score_error']):.2f} pts")
            
            with eval_col4:
                home_err = safe_num(row['home_score_error'])
                away_err = safe_num(row['away_score_error'])
                st.metric("Score Errors", f"H: {home_err:.1f} | A: {away_err:.1f}")
        
        st.divider()
        
        # Additional Info
        st.subheader("ℹ️ Additional Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("**Lock Reason:**")
            st.caption(readable_lock_reason(row['lock_reason']))
        
        with info_col2:
            st.markdown("**Game Status:**")
            final_text = "(Final)" if bool(row['is_final']) else "(In Progress)"
            st.caption(f"{row['status']} {final_text}")
        
        # Raw data expander
        if show_raw_data:
            with st.expander("📄 View complete raw data"):
                st.json(row.to_dict())


# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("🏀 NBA Predictor Dashboard | Built with Streamlit | Data updates every 60 seconds")