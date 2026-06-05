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

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
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
# Data Loading
# -----------------------------
@st.cache_data(ttl=60)
def load_query(query: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


@st.cache_data(ttl=60)
def load_recent_performance() -> pd.DataFrame:
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
# Helpers
# -----------------------------
def safe_num(value, default=0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def format_score(value):
    try:
        return f"{float(value):.0f}"
    except Exception:
        return "-"


def format_percentage(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "-"


def get_confidence_label(score):
    score = safe_num(score, 0)
    if score >= 0.75:
        return "High"
    if score >= 0.50:
        return "Medium"
    return "Low"


def get_status_emoji(is_final, evaluated_at):
    if evaluated_at is not None and not pd.isna(evaluated_at):
        return "✅"
    if bool(is_final):
        return "⏳"
    return "🕐"


def get_status_text(is_final, evaluated_at):
    if evaluated_at is not None and not pd.isna(evaluated_at):
        return "Evaluated"
    if bool(is_final):
        return "Ready to Evaluate"
    return "Pending Result"


def winner_from_row(row):
    home = row["home_team"]
    away = row["away_team"]
    home_score = safe_num(row["home_score_predicted"])
    away_score = safe_num(row["away_score_predicted"])
    return home if home_score > away_score else away


def readable_lock_reason(reason):
    mapping = {
        "first_game_day_prediction": "🎯 First official prediction of the day",
        "pre_tipoff_lock_window_override": "🔒 Updated inside pre-tip lock window",
        "non_official_rerun": "🔄 Extra rerun, not official",
        "after_tipoff_not_official": "⏰ Generated after tipoff",
        "snapshot_only": "📸 Snapshot only",
        "backfill": "📦 Backfill prediction",
        "backfill_duplicate": "📦 Backfill (duplicate skipped)",
    }
    return mapping.get(str(reason), str(reason))


def series_label(row):
    """Return a human-readable playoff series context string."""
    season_type = row.get("season_type", "")
    if season_type != "Playoffs":
        return None
    h_wins = int(safe_num(row.get("home_series_wins"), 0))
    a_wins = int(safe_num(row.get("away_series_wins"), 0))
    game_num = int(safe_num(row.get("series_game_num"), 1))
    home = row["home_team"]
    away = row["away_team"]
    if h_wins == 0 and a_wins == 0:
        return f"Game {game_num} — series begins"
    if h_wins == a_wins:
        return f"Series tied {h_wins}–{a_wins} | Game {game_num}"
    leader = home if h_wins > a_wins else away
    l_wins = max(h_wins, a_wins)
    t_wins = min(h_wins, a_wins)
    return f"{leader} leads {l_wins}–{t_wins} | Game {game_num}"


def create_performance_chart(df):
    if df.empty or not PLOTLY_AVAILABLE:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['win_pct'] * 100,
        mode='lines+markers',
        name='Win %',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='<b>%{x|%b %d}</b><br>Win Rate: %{y:.1f}%<extra></extra>'
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['avg_margin_err'],
        mode='lines+markers',
        name='Avg Margin Error',
        line=dict(color='#f59e0b', width=2, dash='dot'),
        marker=dict(size=6, symbol='diamond'),
        hovertemplate='<b>%{x|%b %d}</b><br>Margin Error: %{y:.1f} pts<extra></extra>'
    ), secondary_y=True)

    fig.update_layout(
        title='Win Rate & Margin Error Trend (Last 30 Days)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1'),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(showgrid=True, gridcolor='#334155'),
        height=350,
    )
    fig.update_yaxes(
        title_text="Win Rate (%)", range=[0, 100],
        showgrid=True, gridcolor='#334155', secondary_y=False
    )
    fig.update_yaxes(
        title_text="Margin Error (pts)",
        showgrid=False, secondary_y=True
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
        p.home_win_probability,
        p.key_factors,
        p.vegas_spread,
        p.vegas_total,
        p.spread_edge,
        p.total_edge,
        p.is_official,
        p.prediction_type,
        p.lock_reason,
        g.status,
        g.is_final,
        g.home_score,
        g.away_score,
        g.season_type,
        g.series_game_num,
        g.home_series_wins,
        g.away_series_wins,
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
    if date_filter == "Today":
        official_df = official_df[official_df["game_date"] == pd.Timestamp.now().date()]
    elif date_filter == "Last 7 Days":
        cutoff = pd.Timestamp.now().date() - timedelta(days=7)
        official_df = official_df[official_df["game_date"] >= cutoff]
    elif date_filter == "Last 30 Days":
        cutoff = pd.Timestamp.now().date() - timedelta(days=30)
        official_df = official_df[official_df["game_date"] >= cutoff]

    official_df["dashboard_status"] = official_df.apply(
        lambda r: (
            "Evaluated" if pd.notna(r["evaluated_at"])
            else "Ready to Evaluate" if bool(r["is_final"])
            else "Pending Result"
        ),
        axis=1
    )
    official_df = official_df[official_df["dashboard_status"].isin(status_filter)]


# -----------------------------
# Summary Metrics
# -----------------------------
total_official = len(official_df)
pending_count = ready_count = evaluated_count = 0

if not official_df.empty:
    pending_count   = len(official_df[(official_df["is_final"] == False) & (official_df["evaluated_at"].isna())])
    ready_count     = len(official_df[(official_df["is_final"] == True) & (official_df["evaluated_at"].isna())])
    evaluated_count = len(official_df[official_df["evaluated_at"].notna()])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Official Predictions", total_official, help="Total number of official predictions made")
with col2:
    st.metric("🕐 Pending Results", pending_count, help="Games that haven't finished yet")
with col3:
    st.metric("⏳ Ready to Evaluate", ready_count, help="Finished games awaiting evaluation")
with col4:
    st.metric("✅ Evaluated", evaluated_count, help="Games that have been evaluated")

st.divider()


# -----------------------------
# Performance Summary
# -----------------------------
if not performance_df.empty and performance_df.iloc[0]["games_evaluated"] > 0:
    st.header("📈 Model Performance Overview")

    perf_row = performance_df.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Win Accuracy", format_percentage(perf_row["winner_accuracy"]),
                  help="Percentage of games where winner was correctly predicted")
    with col2:
        st.metric("📏 Avg Margin Error", f"{safe_num(perf_row['avg_margin_error']):.2f} pts",
                  help="Average error in predicted margin")
    with col3:
        st.metric("🎲 Avg Total Error", f"{safe_num(perf_row['avg_total_error']):.2f} pts",
                  help="Average error in predicted total score")
    with col4:
        st.metric("📊 Sample Size", f"{int(perf_row['games_evaluated'])} games",
                  help="Number of evaluated predictions")

    st.divider()


# -----------------------------
# Performance Charts
# -----------------------------
if show_charts and evaluated_count > 0 and PLOTLY_AVAILABLE:
    st.header("📊 Performance Trends")

    recent_perf = load_recent_performance()
    if not recent_perf.empty:
        chart = create_performance_chart(recent_perf)
        if chart:
            st.plotly_chart(chart, use_container_width=True)

    st.divider()


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🏀 Predictions", "📋 Full Tracker", "🔍 Game Details"])


# -----------------------------
# Tab 1: Predictions
# -----------------------------
with tab1:
    st.header("🏀 Official Predictions")

    if official_df.empty:
        st.info("📭 No predictions found matching your current filters. Try adjusting the date range or status filters in the sidebar.")
    else:
        dates = sorted(official_df["game_date"].dropna().astype(str).unique(), reverse=True)

        for date in dates:
            day_df = official_df[official_df["game_date"].astype(str) == date]
            st.subheader(f"📅 {date}")
            st.caption(f"{len(day_df)} prediction(s)")

            for idx, row in day_df.iterrows():
                home = row["home_team"]
                away = row["away_team"]
                home_pred = int(safe_num(row["home_score_predicted"]))
                away_pred = int(safe_num(row["away_score_predicted"]))
                winner = home if home_pred > away_pred else away

                conf = safe_num(row["confidence_score"], 0)
                conf_label = get_confidence_label(conf)
                conf_dot = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[conf_label]

                home_win_prob = safe_num(row.get("home_win_probability"), 0.5) * 100
                away_win_prob = 100 - home_win_prob

                status_emoji = get_status_emoji(row["is_final"], row["evaluated_at"])
                status_text = get_status_text(row["is_final"], row["evaluated_at"])

                with st.container(border=True):
                    # Matchup header + status
                    h_col, s_col = st.columns([3, 1])
                    with h_col:
                        st.markdown(f"### {away} @ {home}")
                        s_lbl = series_label(row)
                        if s_lbl:
                            st.caption(f"🏆 Playoffs — {s_lbl}")
                    with s_col:
                        st.markdown(f"**{status_emoji} {status_text}**")

                    # Score line — single clean row
                    home_disp = f"**{home_pred}**" if home_pred > away_pred else str(home_pred)
                    away_disp = f"**{away_pred}**" if away_pred > home_pred else str(away_pred)
                    st.markdown(
                        f"<div style='font-size:1.4rem; font-weight:600; text-align:center; padding:8px 0;'>"
                        f"🏠 {home} {home_disp} &nbsp;—&nbsp; {away} {away_disp} ✈️"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    # Confidence + win probability
                    conf_col, prob_col = st.columns(2)
                    with conf_col:
                        st.markdown(f"**{conf_dot} Confidence: {conf_label}** ({conf:.0%})")
                        st.progress(conf)
                    with prob_col:
                        st.markdown(f"**Win Probability:** {home} {home_win_prob:.0f}% / {away} {away_win_prob:.0f}%")
                        st.progress(home_win_prob / 100)

                    # Model + Vegas metrics
                    m1, m2, m3, m4 = st.columns(4)
                    margin = safe_num(row["model_margin"])
                    vegas_spread = row.get("vegas_spread")
                    vegas_total  = row.get("vegas_total")
                    spread_edge  = row.get("spread_edge")
                    total_edge   = row.get("total_edge")

                    with m1:
                        st.metric("🏆 Winner", winner)
                    with m2:
                        st.metric("📊 Model Margin", f"{margin:+.1f}")
                    with m3:
                        if vegas_spread is not None and not pd.isna(vegas_spread):
                            edge_val = safe_num(spread_edge, 0)
                            edge_str = f"{edge_val:+.1f} edge" if edge_val != 0 else "—"
                            st.metric("Vegas Spread", f"{safe_num(vegas_spread):+.1f}", delta=edge_str)
                        else:
                            st.metric("Vegas Spread", "N/A")
                    with m4:
                        if vegas_total is not None and not pd.isna(vegas_total):
                            t_edge = safe_num(total_edge, 0)
                            t_str  = f"{t_edge:+.1f}" if t_edge != 0 else "—"
                            st.metric("Vegas Total", f"{safe_num(vegas_total):.1f}", delta=t_str)
                        else:
                            st.metric("Vegas Total", "—")

                    # Actual results
                    if row["home_score"] is not None and not pd.isna(row["home_score"]):
                        actual_home = int(safe_num(row["home_score"]))
                        actual_away = int(safe_num(row["away_score"]))
                        actual_winner = home if actual_home > actual_away else away
                        correct = bool(row["winner_correct"]) if pd.notna(row.get("winner_correct")) else None
                        result_emoji = "✅" if correct else ("❌" if correct is not None else "⏳")

                        st.markdown("---")
                        r1, r2, r3 = st.columns(3)
                        with r1:
                            st.markdown(f"**Final:** 🏠 {home} **{actual_home}** — {away} **{actual_away}** ✈️")
                        with r2:
                            st.markdown(f"**{result_emoji} Winner:** {actual_winner}")
                        with r3:
                            if pd.notna(row.get("margin_error")):
                                st.markdown(f"**Margin error:** {safe_num(row['margin_error']):.1f} pts")

                    # Key factors
                    key_factors = row.get("key_factors")
                    if key_factors and not (isinstance(key_factors, float) and pd.isna(key_factors)):
                        factors = key_factors if isinstance(key_factors, list) else []
                        if factors:
                            with st.expander("🔍 Key Factors"):
                                for f in factors:
                                    st.markdown(f"• {f}")

                    st.caption(f"{readable_lock_reason(row['lock_reason'])} | Locked: {row['predicted_at']}")

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

        sum_col1, sum_col2, sum_col3 = st.columns(3)
        with sum_col1:
            st.metric("🕐 Pending Result", pending_count)
        with sum_col2:
            st.metric("⏳ Ready to Evaluate", ready_count)
        with sum_col3:
            st.metric("✅ Evaluated", evaluated_count)

        st.divider()
        st.subheader("All Predictions")

        display_df = tracker[[
            "game_date", "matchup", "dashboard_status",
            "home_score_predicted", "away_score_predicted",
            "home_score", "away_score",
            "winner_correct", "margin_error", "confidence_score",
        ]].copy()

        display_df["winner_correct"] = display_df["winner_correct"].apply(
            lambda x: "✅" if x is True else ("❌" if x is False else "—")
        )
        display_df["confidence_score"] = display_df["confidence_score"].apply(
            lambda x: f"{float(x) * 100:.0f}%" if pd.notna(x) else "—"
        )
        display_df["margin_error"] = display_df["margin_error"].apply(
            lambda x: f"{float(x):.1f}" if pd.notna(x) else "—"
        )

        display_df.columns = [
            "Date", "Matchup", "Status",
            "Pred Home", "Pred Away",
            "Actual Home", "Actual Away",
            "Winner ✓", "Margin Err", "Confidence",
        ]

        st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)

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

        st.subheader(f"{away} @ {home}")
        s_lbl = series_label(row)
        if s_lbl:
            st.caption(f"🏆 Playoffs — {s_lbl}")
        st.caption(f"📅 Game Date: {row['game_date']} | 🔒 Locked: {row['predicted_at']}")
        st.divider()

        # Prediction details
        st.subheader("🎯 Prediction Details")
        conf = safe_num(row["confidence_score"], 0)
        conf_label = get_confidence_label(conf)
        conf_dot = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[conf_label]
        home_win_prob = safe_num(row.get("home_win_probability"), 0.5) * 100
        away_win_prob = 100 - home_win_prob

        pd1, pd2, pd3 = st.columns(3)
        with pd1:
            st.markdown(f"**{conf_dot} Confidence: {conf_label}**")
            st.progress(conf)
            st.caption(f"{conf:.0%}")
        with pd2:
            st.metric("Predicted Winner", winner_from_row(row))
        with pd3:
            status_emoji = get_status_emoji(row["is_final"], row["evaluated_at"])
            status_text  = get_status_text(row["is_final"], row["evaluated_at"])
            st.metric("Status", f"{status_emoji} {status_text}")

        st.markdown(f"**Win Probability:** {home} {home_win_prob:.0f}% / {away} {away_win_prob:.0f}%")
        st.progress(home_win_prob / 100)
        st.divider()

        # Score comparison
        st.subheader("📊 Score Comparison")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**Predicted Score**")
            c1, c2 = st.columns(2)
            with c1:
                st.metric(home, format_score(row['home_score_predicted']))
            with c2:
                st.metric(away, format_score(row['away_score_predicted']))
        with sc2:
            if row["home_score"] is not None and not pd.isna(row["home_score"]):
                st.markdown("**Actual Score**")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric(home, format_score(row['home_score']))
                with c2:
                    st.metric(away, format_score(row['away_score']))
            else:
                st.info("Game not final yet")
        st.divider()

        # Vegas context
        st.subheader("💰 Vegas Context")
        v1, v2, v3, v4 = st.columns(4)
        vs = row.get("vegas_spread")
        vt = row.get("vegas_total")
        se = row.get("spread_edge")
        te = row.get("total_edge")
        with v1:
            st.metric("Vegas Spread", f"{safe_num(vs):+.1f}" if pd.notna(vs) else "N/A")
        with v2:
            st.metric("Vegas Total", f"{safe_num(vt):.1f}" if pd.notna(vt) else "N/A")
        with v3:
            st.metric("Spread Edge", f"{safe_num(se):+.1f}" if pd.notna(se) else "N/A")
        with v4:
            st.metric("Total Edge", f"{safe_num(te):+.1f}" if pd.notna(te) else "N/A")
        st.divider()

        # Model insights
        st.subheader("🧠 Model Insights")
        i1, i2, i3 = st.columns(3)
        with i1:
            margin = safe_num(row["model_margin"])
            favored = home if margin > 0 else away
            st.metric("Predicted Margin", f"{margin:+.1f}", delta=f"{favored} favored")
        with i2:
            st.metric("Predicted Total", f"{safe_num(row['model_total']):.1f}")
        with i3:
            st.metric("Prediction Type", row['prediction_type'])

        # Key factors
        key_factors = row.get("key_factors")
        if key_factors and not (isinstance(key_factors, float) and pd.isna(key_factors)):
            factors = key_factors if isinstance(key_factors, list) else []
            if factors:
                st.subheader("🔍 Key Factors")
                for f in factors:
                    st.markdown(f"• {f}")
        st.divider()

        # Evaluation results
        if pd.notna(row["evaluated_at"]):
            st.subheader("✅ Evaluation Results")
            e1, e2, e3, e4 = st.columns(4)
            correct = bool(row["winner_correct"])
            with e1:
                st.metric("Winner Prediction", "✅ Correct" if correct else "❌ Incorrect")
            with e2:
                st.metric("Margin Error", f"{safe_num(row['margin_error']):.2f} pts")
            with e3:
                st.metric("Total Error", f"{safe_num(row['total_score_error']):.2f} pts")
            with e4:
                h_err = safe_num(row['home_score_error'])
                a_err = safe_num(row['away_score_error'])
                st.metric("Score Errors", f"H: {h_err:.1f} | A: {a_err:.1f}")
            st.divider()

        # Additional info
        st.subheader("ℹ️ Additional Information")
        ai1, ai2 = st.columns(2)
        with ai1:
            st.markdown("**Lock Reason:**")
            st.caption(readable_lock_reason(row['lock_reason']))
        with ai2:
            st.markdown("**Game Status:**")
            final_text = "(Final)" if bool(row['is_final']) else "(In Progress)"
            st.caption(f"{row['status']} {final_text}")

        if show_raw_data:
            with st.expander("📄 View complete raw data"):
                st.json(row.to_dict())


# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("🏀 NBA Predictor Dashboard | Built with Streamlit | Data updates every 60 seconds")
