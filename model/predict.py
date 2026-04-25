import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
from datetime import date
from sqlalchemy.orm import Session
from sqlalchemy import text
from data.storage.db import engine
from data.storage.models import Game, Team, GameOdds
from model.train import FEATURE_COLS
from data.ingestion.fetch_injuries import fetch_injury_report
from data.ingestion.fetch_odds import fetch_todays_player_props


injury_report = fetch_injury_report()
all_player_props = fetch_todays_player_props()

# Ensemble weights
# Dynamic ensemble weights based on star power
def get_ensemble_weights(
    home_player_preds,
    away_player_preds,
    home_team_pts,
    away_team_pts,
    home_player_pts,
    away_player_pts,
    is_playoff=False
):
    # Base weights
    team_w = 0.72
    player_w = 0.28

    # --- 1. Star-driven adjustment ---
    stars = sum(
        1 for p in home_player_preds + away_player_preds
        if p['usage_rate'] > 0.25
    )

    player_w += min(stars * 0.03, 0.09)

    # --- 2. Playoff boost ---
    if is_playoff:
        player_w += 0.05

    # --- 3. Disagreement penalty (MOST IMPORTANT) ---
    team_home_winner = home_team_pts > away_team_pts
    player_home_winner = home_player_pts > away_player_pts

    if team_home_winner != player_home_winner:
        player_w *= 0.680  
    
    team_total = home_team_pts + away_team_pts
    player_total = home_player_pts + away_player_pts
    total_gap = player_total - team_total

    if total_gap > 20:
        player_w *= 0.78
    elif total_gap > 12:
        player_w *= 0.08

        
    # --- 4. Clamp weights ---
    player_w = max(0.15, min(player_w, 0.35))
    team_w = 1 - player_w

    return round(team_w, 2), round(player_w, 2)


def load_models():
    with open('model/model_home.pkl', 'rb') as f:
        model_home = pickle.load(f)
    with open('model/model_away.pkl', 'rb') as f:
        model_away = pickle.load(f)
    return model_home, model_away

def save_predictions(predictions):
    """Save predictions to database and CSV."""
    if not predictions:
        return

    from sqlalchemy.orm import Session
    from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date
    from data.storage.db import engine
    from data.storage.models import Base
    from datetime import datetime

    # Save to CSV log
    import csv
    import os

    os.makedirs('logs', exist_ok=True)
    csv_path = 'logs/predictions_log.csv'
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'date', 'home_team', 'away_team',
            'home_pred', 'away_pred',
            'predicted_winner', 'margin', 'confidence',
            'home_pred_team', 'away_pred_team',
            'home_pred_player', 'away_pred_player',
            'is_playoff', 'season_type'
        ])
        if not file_exists:
            writer.writeheader()

        for pred in predictions:
            writer.writerow({
                'date':              date.today().isoformat(),
                'home_team':         pred['home_team'],
                'away_team':         pred['away_team'],
                'home_pred':         pred['home_pred'],
                'away_pred':         pred['away_pred'],
                'predicted_winner':  pred['predicted_winner'],
                'margin':            pred['margin'],
                'confidence':        pred['confidence'],
                'home_pred_team':    pred.get('home_pred_team', ''),
                'away_pred_team':    pred.get('away_pred_team', ''),
                'home_pred_player':  pred.get('home_pred_player', ''),
                'away_pred_player':  pred.get('away_pred_player', ''),
                'is_playoff':        pred['is_playoff'],
                'season_type':       pred['season_type'],
            })

    print(f"✅ Predictions saved to {csv_path}")

def get_todays_games():
    """Get all scheduled games for today."""
    today = date.today()
    with Session(engine) as session:
        games = session.query(Game).filter(
            Game.game_date == today,
            Game.status == 'scheduled'
        ).all()

        result = []
        for g in games:
            home = session.query(Team).filter_by(
                team_id=g.home_team_id
            ).first()
            away = session.query(Team).filter_by(
                team_id=g.away_team_id
            ).first()
            odds = session.query(GameOdds).filter_by(
                game_id=g.game_id
            ).first()

            result.append({
                'game_id':           g.game_id,
                'nba_game_id':       g.nba_game_id,
                'home_team':         home.abbreviation if home else '???',
                'away_team':         away.abbreviation if away else '???',
                'home_team_id':      g.home_team_id,
                'away_team_id':      g.away_team_id,
                'season_type':       g.season_type,
                'game_date':         g.game_date,
                'series_game_num':   g.series_game_num or 1,
                'home_series_wins':  g.home_series_wins or 0,
                'away_series_wins':  g.away_series_wins or 0,
                'vegas_total':       odds.total_line if odds else None,
                'vegas_spread':      odds.home_spread if odds else None,
                'vegas_home':        odds.vegas_home_implied if odds else None,
                'vegas_away':        odds.vegas_away_implied if odds else None,
            })

    return result

def get_bad_night_pct(team_id, n_games=50):
    """Compute probability of a bad shooting night for a team."""
    from scipy import stats as scipy_stats

    q = text("""
        SELECT tbs.fg_pct
        FROM team_box_scores tbs
        JOIN games g ON tbs.game_id = g.game_id
        WHERE tbs.team_id = :team_id
        AND g.is_final = TRUE
        AND tbs.fg_pct IS NOT NULL
        ORDER BY g.game_date DESC
        LIMIT :n
    """)

    with engine.connect() as conn:
        result = conn.execute(q, {
            'team_id': int(team_id),
            'n': n_games
        }).fetchall()

    if not result or len(result) < 10:
        return 0.25

    fg_vals = [float(r[0]) for r in result]
    avg_fg  = np.mean(fg_vals)
    std_fg  = np.std(fg_vals)

    if std_fg == 0:
        return 0.25

    # Bad night = FG% more than 1 std below average
    threshold    = avg_fg - std_fg
    bad_night_pct = float(scipy_stats.norm.cdf(threshold, avg_fg, std_fg))
    return bad_night_pct


def build_features_for_upcoming_game(home_team_id, away_team_id,
                                      game_date, season_type,
                                      series_game_num=0,
                                      home_series_wins=0,
                                      away_series_wins=0):
    """
    Build feature vector for a game that hasn't been played yet.
    Pulls the most recent rolling stats for each team.
    """
    query = text("""
        SELECT
            ht.abbreviation AS home_team,
            at.abbreviation AS away_team,
            :home_team_id   AS home_team_id,
            :away_team_id   AS away_team_id,
            :game_date      AS game_date,
            :season_type    AS season_type,

            -- Home rolling stats (most recent)
            h.avg_points              AS home_avg_points,
            h.avg_offensive_rating    AS home_off_rating,
            h.avg_defensive_rating    AS home_def_rating,
            h.avg_pace                AS home_pace,
            h.avg_fg_pct              AS home_fg_pct,
            h.avg_fg3_pct             AS home_fg3_pct,
            h.three_point_rate        AS home_3pt_rate,
            h.win_pct                 AS home_win_pct,
            h.home_avg_points         AS home_home_avg_pts,
            h.home_avg_points_allowed AS home_home_avg_pts_allowed,

            -- Away rolling stats (most recent)
            a.avg_points              AS away_avg_points,
            a.avg_offensive_rating    AS away_off_rating,
            a.avg_defensive_rating    AS away_def_rating,
            a.avg_pace                AS away_pace,
            a.avg_fg_pct              AS away_fg_pct,
            a.avg_fg3_pct             AS away_fg3_pct,
            a.three_point_rate        AS away_3pt_rate,
            a.win_pct                 AS away_win_pct,
            a.away_avg_points         AS away_away_avg_pts,
            a.away_avg_points_allowed AS away_away_avg_pts_allowed,

            ABS(h.avg_pace - a.avg_pace) AS pace_differential,

            -- 7-game recent form
            h7.avg_points           AS home_avg_points_l7,
            h7.avg_offensive_rating AS home_off_rating_l7,
            h7.avg_defensive_rating AS home_def_rating_l7,
            h7.win_pct              AS home_win_pct_l7,
            a7.avg_points           AS away_avg_points_l7,
            a7.avg_offensive_rating AS away_off_rating_l7,
            a7.avg_defensive_rating AS away_def_rating_l7,
            a7.win_pct              AS away_win_pct_l7,

            -- Vegas
            go.home_spread        AS vegas_spread,
            go.total_line         AS vegas_total,
            go.vegas_home_implied AS vegas_home_implied,
            go.vegas_away_implied AS vegas_away_implied

        FROM teams ht
        JOIN teams at ON at.team_id = :away_team_id

        JOIN team_rolling_stats h ON h.team_id = :home_team_id
            AND h."window" = 10
            AND h.as_of_date = (
                SELECT MAX(as_of_date) FROM team_rolling_stats
                WHERE team_id = :home_team_id AND "window" = 10
            )

        JOIN team_rolling_stats a ON a.team_id = :away_team_id
            AND a."window" = 10
            AND a.as_of_date = (
                SELECT MAX(as_of_date) FROM team_rolling_stats
                WHERE team_id = :away_team_id AND "window" = 10
            )

        JOIN team_rolling_stats h7 ON h7.team_id = :home_team_id
            AND h7."window" = 7
            AND h7.as_of_date = (
                SELECT MAX(as_of_date) FROM team_rolling_stats
                WHERE team_id = :home_team_id AND "window" = 7
            )

        JOIN team_rolling_stats a7 ON a7.team_id = :away_team_id
            AND a7."window" = 7
            AND a7.as_of_date = (
                SELECT MAX(as_of_date) FROM team_rolling_stats
                WHERE team_id = :away_team_id AND "window" = 7
            )

        LEFT JOIN games g2
            ON g2.home_team_id = :home_team_id
            AND g2.away_team_id = :away_team_id
            AND g2.game_date = :game_date
        LEFT JOIN game_odds go ON go.game_id = g2.game_id

        WHERE ht.team_id = :home_team_id
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {
            'home_team_id': int(home_team_id),
            'away_team_id': int(away_team_id),
            'game_date':    str(game_date),
            'season_type':  season_type,
        })
        row = result.fetchone()

    if not row:
        return None

    row_dict = dict(row._mapping)
    row_dict['game_date'] = pd.Timestamp(game_date)

    # Rest days
    row_dict['home_rest_days']    = 2
    row_dict['away_rest_days']    = 2
    row_dict['rest_advantage']    = 0
    row_dict['home_back_to_back'] = 0
    row_dict['away_back_to_back'] = 0

    # H2H defaults
    row_dict['h2h_home_avg_score'] = row_dict.get('home_avg_points', 112)
    row_dict['h2h_away_avg_score'] = row_dict.get('away_avg_points', 110)
    row_dict['h2h_home_win_pct']   = 0.5
    row_dict['h2h_games_count']    = 0

    # Playoff context
    row_dict['is_playoff']       = 1 if season_type == 'Playoffs' else 0
    row_dict['series_game_num']  = series_game_num
    row_dict['home_series_wins'] = home_series_wins
    row_dict['away_series_wins'] = away_series_wins
    row_dict['is_elimination']   = 1 if (
        home_series_wins == 3 or away_series_wins == 3
    ) else 0
    row_dict['series_momentum']  = (
        1 if home_series_wins > away_series_wins
        else -1 if away_series_wins > home_series_wins
        else 0
    )
    row_dict['series_pressure']  = home_series_wins + away_series_wins

    # Vegas features
    has_odds = row_dict.get('vegas_total') is not None
    row_dict['has_vegas_odds']     = 1 if has_odds else 0
    row_dict['vegas_spread']       = float(row_dict['vegas_spread']) \
                                     if has_odds else 0
    row_dict['vegas_total']        = float(row_dict['vegas_total']) \
                                     if has_odds else 0
    row_dict['vegas_home_implied'] = float(row_dict['vegas_home_implied']) \
                                     if has_odds else 0
    row_dict['vegas_away_implied'] = float(row_dict['vegas_away_implied']) \
                                     if has_odds else 0
    row_dict['vegas_vs_home_avg']  = (
        float(row_dict['vegas_home_implied']) -
        float(row_dict['home_avg_points'])
    ) if has_odds else 0
    row_dict['vegas_vs_away_avg']  = (
        float(row_dict['vegas_away_implied']) -
        float(row_dict['away_avg_points'])
    ) if has_odds else 0

    # Matchup interactions
    home_pace = float(row_dict.get('home_pace', 98))
    away_pace = float(row_dict.get('away_pace', 98))
    home_pts  = float(row_dict.get('home_avg_points', 112))
    away_pts  = float(row_dict.get('away_avg_points', 110))

    row_dict['combined_pace']        = (home_pace + away_pace) / 2
    row_dict['home_3pt_matchup']     = (
        float(row_dict.get('home_3pt_rate', 0.35)) -
        float(row_dict.get('away_3pt_rate', 0.35))
    )
    row_dict['away_3pt_matchup']     = -row_dict['home_3pt_matchup']
    row_dict['home_off_vs_away_def'] = (
        float(row_dict.get('home_off_rating', 115)) -
        float(row_dict.get('away_def_rating', 112))
    )
    row_dict['away_off_vs_home_def'] = (
        float(row_dict.get('away_off_rating', 113)) -
        float(row_dict.get('home_def_rating', 112))
    )
    row_dict['net_rating_diff']      = (
        (float(row_dict.get('home_off_rating', 115)) -
         float(row_dict.get('home_def_rating', 112))) -
        (float(row_dict.get('away_off_rating', 113)) -
         float(row_dict.get('away_def_rating', 112)))
    )
    row_dict['implied_total']        = home_pts + away_pts

    # Momentum
    row_dict['home_momentum'] = (
        float(row_dict.get('home_avg_points_l7', home_pts)) - home_pts
    )
    row_dict['away_momentum'] = (
        float(row_dict.get('away_avg_points_l7', away_pts)) - away_pts
    )
    row_dict['home_def_momentum'] = (
        float(row_dict.get('home_def_rating', 112)) -
        float(row_dict.get('home_def_rating_l7', 112))
    )
    row_dict['away_def_momentum'] = (
        float(row_dict.get('away_def_rating', 112)) -
        float(row_dict.get('away_def_rating_l7', 112))
    )

    # Similarity defaults
    row_dict['home_proxy_off_rating'] = row_dict.get('home_off_rating', 115)
    row_dict['home_proxy_avg_pts']    = home_pts
    row_dict['away_proxy_off_rating'] = row_dict.get('away_off_rating', 113)
    row_dict['away_proxy_avg_pts']    = away_pts
    row_dict['home_sim_off_rating']   = row_dict.get('home_off_rating', 115)
    row_dict['away_sim_off_rating']   = row_dict.get('away_off_rating', 113)

    # Playoff elevation defaults
    row_dict['home_playoff_elevation'] = 0.0
    row_dict['away_playoff_elevation'] = 0.0
    row_dict['playoff_elevation_diff'] = 0.0

    # Home court strength default
    row_dict['home_court_strength']   = 0.0

    # Scoring variance defaults
    row_dict['home_scoring_std']      = 12.0
    row_dict['away_scoring_std']      = 12.0
    row_dict['home_consistency']      = 1 / (1 + 12.0)
    row_dict['away_consistency']      = 1 / (1 + 12.0)
    row_dict['variance_differential'] = 0.0

    # Style defensive matchup defaults
    row_dict['home_def_vs_away_style'] = row_dict.get('home_def_rating', 112)
    row_dict['away_def_vs_home_style'] = row_dict.get('away_def_rating', 112)
    row_dict['home_def_style_edge']    = 0.0
    row_dict['away_def_style_edge']    = 0.0

    # Compute real bad night probability
    home_bad = get_bad_night_pct(home_team_id)
    away_bad = get_bad_night_pct(away_team_id)
    row_dict['home_bad_night_pct'] = home_bad
    row_dict['away_bad_night_pct'] = away_bad

    # Playoff pace penalty
    if season_type == 'Playoffs':
        row_dict['home_avg_points'] = float(
            row_dict['home_avg_points']
        ) * 0.90
        row_dict['away_avg_points'] = float(
            row_dict['away_avg_points']
        ) * 0.90
        row_dict['implied_total'] = (
            row_dict['home_avg_points'] +
            row_dict['away_avg_points']
        )

    # Apply downward adjustment for volatile teams
    if home_bad > 0.35:
        row_dict['home_avg_points'] = float(
            row_dict['home_avg_points']
        ) * (1 - (home_bad - 0.35) * 0.3)

    if away_bad > 0.35:
        row_dict['away_avg_points'] = float(
            row_dict['away_avg_points']
        ) * (1 - (away_bad - 0.35) * 0.3)

    return row_dict


def get_key_factors(row, home_pred, away_pred,
                    home_player_preds=None,
                    away_player_preds=None,
                    injury_report=None,
                    confidence=None,
                    max_factors=4):
    scored = []

    home = row["home_team"]
    away = row["away_team"]

    def val(key, default=0):
        try:
            return float(row.get(key, default) or default)
        except Exception:
            return default

    def add(score, text):
        if text not in [x[1] for x in scored]:
            scored.append((score, text))

    # -------------------------
    # Core ratings
    # -------------------------
    home_off = val("home_off_rating")
    away_def = val("away_def_rating")
    away_off = val("away_off_rating")
    home_def = val("home_def_rating")

    home_edge = home_off - away_def
    away_edge = away_off - home_def

    if home_edge >= 8:
        add(
            abs(home_edge),
            f"{home} offense has a strong efficiency edge vs {away}'s defense"
        )

    if away_edge >= 8:
        add(
            abs(away_edge),
            f"{away} offense has a strong efficiency edge vs {home}'s defense"
        )

    # -------------------------
    # Pace
    # -------------------------
    pace = val("combined_pace", 98)

    if pace >= 101:
        add(
            6,
            "up-tempo game environment expected — pace should boost scoring chances"
        )
    elif pace <= 95:
        add(
            6,
            "slower pace projected — fewer possessions raises upset/under risk"
        )

    # -------------------------
    # Vegas disagreement
    # -------------------------
    vegas_total = val("vegas_total")
    if vegas_total > 0:
        our_total = home_pred + away_pred
        diff = our_total - vegas_total

        if abs(diff) >= 10:
            direction = "higher" if diff > 0 else "lower"
            add(
                abs(diff),
                f"model total is {abs(diff):.0f} points {direction} than Vegas"
            )

    # -------------------------
    # Spread/value context
    # -------------------------
    vegas_spread = row.get("vegas_spread", None)

    if vegas_spread is not None:
        try:
            vegas_spread = float(vegas_spread)
            model_margin = home_pred - away_pred

            # Adjust this if your spread convention is different.
            spread_gap = model_margin + vegas_spread

            if abs(spread_gap) >= 5:
                team = home if spread_gap > 0 else away
                add(
                    abs(spread_gap),
                    f"{team} shows spread value — model margin differs from market by {abs(spread_gap):.1f}"
                )
        except Exception:
            pass

    # -------------------------
    # Momentum
    # -------------------------
    home_mom = val("home_momentum")
    away_mom = val("away_momentum")

    if home_mom >= 5:
        add(
            abs(home_mom),
            f"{home} offense is trending up recently — scoring above baseline"
        )
    elif home_mom <= -5:
        add(
            abs(home_mom),
            f"{home} offense is trending down recently — scoring below baseline"
        )

    if away_mom >= 5:
        add(
            abs(away_mom),
            f"{away} offense is trending up recently — scoring above baseline"
        )
    elif away_mom <= -5:
        add(
            abs(away_mom),
            f"{away} offense is trending down recently — scoring below baseline"
        )

    # -------------------------
    # Style matchup: 3PT profile
    # -------------------------
    home_3pa = val("home_3pa_rate")
    away_opp_3pa = val("away_opp_3pa_rate")
    away_3pa = val("away_3pa_rate")
    home_opp_3pa = val("home_opp_3pa_rate")

    if home_3pa >= 0.38 and away_opp_3pa >= 0.37:
        add(
            9,
            f"{home} 3PT volume matches up well vs a {away} defense that allows perimeter looks"
        )

    if away_3pa >= 0.38 and home_opp_3pa >= 0.37:
        add(
            9,
            f"{away} 3PT volume matches up well vs a {home} defense that allows perimeter looks"
        )

    # -------------------------
    # Style matchup: paint scoring
    # -------------------------
    home_paint = val("home_paint_pts")
    away_paint_allowed = val("away_paint_pts_allowed")
    away_paint = val("away_paint_pts")
    home_paint_allowed = val("home_paint_pts_allowed")

    if home_paint >= 50 and away_paint_allowed >= 50:
        add(
            8,
            f"{home} has a paint-pressure edge vs {away}'s interior defense"
        )

    if away_paint >= 50 and home_paint_allowed >= 50:
        add(
            8,
            f"{away} has a paint-pressure edge vs {home}'s interior defense"
        )

    # -------------------------
    # Style matchup: transition
    # -------------------------
    home_trans = val("home_transition_freq")
    away_trans_def = val("away_transition_def_rating")
    away_trans = val("away_transition_freq")
    home_trans_def = val("home_transition_def_rating")

    if home_trans >= 0.17 and away_trans_def >= 120:
        add(
            8,
            f"{home} can create transition pressure vs a vulnerable {away} transition defense"
        )

    if away_trans >= 0.17 and home_trans_def >= 120:
        add(
            8,
            f"{away} can create transition pressure vs a vulnerable {home} transition defense"
        )

    # -------------------------
    # Style matchup: isolation
    # -------------------------
    home_iso = val("home_iso_freq")
    away_iso_def = val("away_iso_def_rating")
    away_iso = val("away_iso_freq")
    home_iso_def = val("home_iso_def_rating")

    if home_iso >= 0.10 and away_iso_def >= 1.05:
        add(
            7,
            f"{home} isolation scoring profile could punish {away}'s one-on-one defense"
        )

    if away_iso >= 0.10 and home_iso_def >= 1.05:
        add(
            7,
            f"{away} isolation scoring profile could punish {home}'s one-on-one defense"
        )

    # -------------------------
    # Style matchup: pick-and-roll
    # -------------------------
    home_pnr = val("home_pnr_ball_handler_ppp")
    away_pnr_def = val("away_pnr_def_ppp")
    away_pnr = val("away_pnr_ball_handler_ppp")
    home_pnr_def = val("home_pnr_def_ppp")

    if home_pnr >= 0.95 and away_pnr_def >= 0.95:
        add(
            8,
            f"{home} pick-and-roll attack has a favorable matchup vs {away}'s coverage"
        )

    if away_pnr >= 0.95 and home_pnr_def >= 0.95:
        add(
            8,
            f"{away} pick-and-roll attack has a favorable matchup vs {home}'s coverage"
        )

    # -------------------------
    # Extra possessions: rebounding
    # -------------------------
    home_oreb = val("home_oreb_pct")
    away_dreb = val("away_dreb_pct")
    away_oreb = val("away_oreb_pct")
    home_dreb = val("home_dreb_pct")

    if home_oreb >= 0.30 and away_dreb <= 0.70:
        add(
            7,
            f"{home} could generate extra possessions through offensive rebounding"
        )

    if away_oreb >= 0.30 and home_dreb <= 0.70:
        add(
            7,
            f"{away} could generate extra possessions through offensive rebounding"
        )

    # -------------------------
    # Turnover pressure
    # -------------------------
    home_tov = val("home_tov_pct")
    away_force_tov = val("away_opp_tov_pct")
    away_tov = val("away_tov_pct")
    home_force_tov = val("home_opp_tov_pct")

    if home_tov >= 0.145 and away_force_tov >= 0.145:
        add(
            7,
            f"{home} turnover risk is elevated vs {away}'s defensive pressure"
        )

    if away_tov >= 0.145 and home_force_tov >= 0.145:
        add(
            7,
            f"{away} turnover risk is elevated vs {home}'s defensive pressure"
        )

    # -------------------------
    # Volatility / bad-night risk
    # -------------------------
    home_std = val("home_scoring_std")
    away_std = val("away_scoring_std")

    if home_std >= 9:
        add(
            home_std,
            f"{home} has high scoring volatility — wider boom/bust range"
        )

    if away_std >= 9:
        add(
            away_std,
            f"{away} has high scoring volatility — wider boom/bust range"
        )

    home_bad_night = val("home_bad_night_pct", 0.15)
    away_bad_night = val("away_bad_night_pct", 0.15)

    if home_bad_night > 0.35:
        add(
            home_bad_night * 20,
            f"{home} has elevated bad-night risk — projection is more fragile"
        )

    if away_bad_night > 0.35:
        add(
            away_bad_night * 20,
            f"{away} has elevated bad-night risk — projection is more fragile"
        )

    # -------------------------
    # Star dependency
    # -------------------------
    home_usage = val("home_top_player_usage")
    away_usage = val("away_top_player_usage")

    if home_usage >= 0.32:
        add(
            7,
            f"{home} is heavily dependent on its lead creator — star performance could swing the game"
        )

    if away_usage >= 0.32:
        add(
            7,
            f"{away} is heavily dependent on its lead creator — star performance could swing the game"
        )

    # -------------------------
    # Confidence explanation
    # -------------------------
    if confidence == "LOW":
        add(
            9,
            "low-confidence projection — model sees a narrow margin or high volatility"
        )
    elif confidence == "HIGH":
        add(
            5,
            "high-confidence projection — model sees clear separation between teams"
        )

    # -------------------------
    # Team model vs player model disagreement
    # -------------------------
    home_team_pred = val("home_team_pred", None)
    away_team_pred = val("away_team_pred", None)
    home_player_pred = val("home_player_pred", None)
    away_player_pred = val("away_player_pred", None)

    if home_team_pred and home_player_pred:
        gap = home_player_pred - home_team_pred
        if abs(gap) >= 7:
            direction = "more upside" if gap > 0 else "more downside"
            add(
                abs(gap),
                f"{home} player model shows {direction} than team trends suggest"
            )

    if away_team_pred and away_player_pred:
        gap = away_player_pred - away_team_pred
        if abs(gap) >= 7:
            direction = "more upside" if gap > 0 else "more downside"
            add(
                abs(gap),
                f"{away} player model shows {direction} than team trends suggest"
            )

    # -------------------------
    # Playoff context
    # -------------------------
    if row.get("is_playoff") and row.get("series_game_num", 0):
        game_num = int(row.get("series_game_num", 0))
        h_wins = int(row.get("home_series_wins", 0))
        a_wins = int(row.get("away_series_wins", 0))

        if h_wins == 3 or a_wins == 3:
            trailer = away if h_wins == 3 else home
            add(
                10,
                f"elimination pressure — {trailer} must win to keep the series alive"
            )
        elif h_wins == 2 and a_wins == 0:
            add(
                8,
                f"{home} leads 2-0 — {away} is in desperation mode"
            )
        elif a_wins == 2 and h_wins == 0:
            add(
                8,
                f"{away} leads 2-0 — {home} is in desperation mode"
            )
        elif h_wins == 1 and a_wins == 1:
            add(
                6,
                "series tied 1-1 — Game 3 creates a major leverage spot"
            )
        elif game_num == 1:
            add(
                5,
                "Game 1 tone-setter — early series adjustments matter"
            )

    # -------------------------
    # Player hot/cold form
    # -------------------------
    def player_flags(team, preds):
        if not preds:
            return

        hot = [p for p in preds if p.get("hot_flag")]
        cold = [p for p in preds if p.get("slump_flag")]

        if hot:
            names = ", ".join(
                p.get("full_name", "").split()[-1]
                for p in hot[:2]
                if p.get("full_name")
            )
            if names:
                add(
                    6,
                    f"{team} has recent player-form upside from {names}"
                )

        if cold:
            names = ", ".join(
                p.get("full_name", "").split()[-1]
                for p in cold[:2]
                if p.get("full_name")
            )
            if names:
                add(
                    6,
                    f"{team} has player-form risk from {names}"
                )

    player_flags(home, home_player_preds)
    player_flags(away, away_player_preds)

    # -------------------------
    # Injuries
    # -------------------------
    if injury_report:
        for inj in injury_report:
            try:
                team = inj.get("team")
                player = inj.get("player") or inj.get("full_name")
                status = str(inj.get("status", "")).lower()
                impact = float(inj.get("impact_score", 0) or 0)

                if status in ["out", "doubtful"] and impact >= 7:
                    add(
                        impact + 3,
                        f"{team} missing high-impact player {player} — rotation/usage shift matters"
                    )
            except Exception:
                continue

    # -------------------------
    # Fallback if nothing fires
    # -------------------------
    if not scored:
        winner = home if home_pred > away_pred else away
        add(
            1,
            f"{winner} projects slightly better overall, but no major matchup edge stands out"
        )

    scored = sorted(scored, key=lambda x: x[0], reverse=True)

    return [text for score, text in scored[:max_factors]]


def predict_todays_games():
    print(f"\n{'='*55}")
    print(f"NBA PREDICTIONS — {date.today().strftime('%B %d, %Y')}")
    print(f"{'='*55}\n")

    model_home, model_away = load_models()
    games = get_todays_games()

    if not games:
        print("No games scheduled today.")
        return []

    print(f"Found {len(games)} games today\n")

    # Fetch injury report and player props once for all games
    from data.ingestion.fetch_injuries import fetch_injury_report
    from data.ingestion.fetch_odds import fetch_todays_player_props
    from model.predict_players import predict_team_player_stats

    print("Fetching injury report...")
    injury_report = fetch_injury_report()

    print("Fetching player props...")
    all_player_props = fetch_todays_player_props()

    predictions = []

    for game in games:
        print(f"\nBuilding features for "
              f"{game['away_team']} @ {game['home_team']}...")

        row_dict = build_features_for_upcoming_game(
            home_team_id     = game['home_team_id'],
            away_team_id     = game['away_team_id'],
            game_date        = game['game_date'],
            season_type      = game['season_type'],
            series_game_num  = game['series_game_num'],
            home_series_wins = game['home_series_wins'],
            away_series_wins = game['away_series_wins'],
        )

        if not row_dict:
            print(f"  ⚠️  Could not build features")
            continue

        row_dict['home_team'] = game['home_team']
        row_dict['away_team'] = game['away_team']

        # Build feature vector
        features = np.array([
            float(row_dict.get(f, 0) or 0)
            for f in FEATURE_COLS
        ]).reshape(1, -1)

        # Team model predictions
        home_pred_team = float(model_home.predict(features)[0])
        away_pred_team = float(model_away.predict(features)[0])

        # Match props to this game
        game_props_key = next(
            (k for k in all_player_props
             if game['away_team'] in k or game['home_team'] in k),
            None
        )
        game_props = all_player_props.get(game_props_key, {})

        # Player model predictions
        try:
            home_player_preds, home_player_pts, _, _ = \
                predict_team_player_stats(
                    team_id          = game['home_team_id'],
                    opponent_team_id = game['away_team_id'],
                    is_home          = True,
                    is_playoff       = game['season_type'] == 'Playoffs',
                    team_off_rating  = float(
                        row_dict.get('home_off_rating', 115)
                    ),
                    opp_def_rating   = float(
                        row_dict.get('away_def_rating', 112)
                    ),
                    opp_pace         = float(
                        row_dict.get('away_pace', 98)
                    ),
                    team_pace        = float(
                        row_dict.get('home_pace', 98)
                    ),
                    injury_report    = injury_report,
                    team_abbr        = game['home_team'],
                    player_props     = game_props
                )

            away_player_preds, away_player_pts, _, _ = \
                predict_team_player_stats(
                    team_id          = game['away_team_id'],
                    opponent_team_id = game['home_team_id'],
                    is_home          = False,
                    is_playoff       = game['season_type'] == 'Playoffs',
                    team_off_rating  = float(
                        row_dict.get('away_off_rating', 113)
                    ),
                    opp_def_rating   = float(
                        row_dict.get('home_def_rating', 114)
                    ),
                    opp_pace         = float(
                        row_dict.get('home_pace', 98)
                    ),
                    team_pace        = float(
                        row_dict.get('away_pace', 98)
                    ),
                    injury_report    = injury_report,
                    team_abbr        = game['away_team'],
                    player_props     = game_props
                )

            player_model_available = (
                home_player_pts > 0 and away_player_pts > 0
            )

        except Exception as e:
            print(f"  ⚠️  Player model failed: {e}")
            home_player_preds      = []
            away_player_preds      = []
            home_player_pts        = 0
            away_player_pts        = 0
            player_model_available = False

        if player_model_available:
            home_disagreement = abs(home_pred_team - home_player_pts)
            away_disagreement = abs(away_pred_team - away_player_pts)

            if home_disagreement > 15 or away_disagreement > 15:
                home_tw, home_pw = 0.90, 0.10
                away_tw, away_pw = 0.90, 0.10
                print(f"  ⚠️  High model disagreement — "
                      f"trusting team model "
                      f"(home diff: {home_disagreement:.0f}, "
                      f"away diff: {away_disagreement:.0f})")
            else:
                home_tw, home_pw = get_ensemble_weights(
    home_player_preds,
    away_player_preds,
    home_pred_team,
    away_pred_team,
    home_player_pts,
    away_player_pts,
    is_playoff=game['season_type'] == 'Playoffs'
)
                away_tw, away_pw = get_ensemble_weights(
    home_player_preds,
    away_player_preds,
    home_pred_team,
    away_pred_team,
    home_player_pts,
    away_player_pts,
    is_playoff=game['season_type'] == 'Playoffs'
)

            home_pred_final = round(
                home_pred_team * home_tw +
                home_player_pts * home_pw
            )
            away_pred_final = round(
                away_pred_team * away_tw +
                away_player_pts * away_pw
            )
        else:
            home_pred_final = round(home_pred_team)
            away_pred_final = round(away_pred_team)

        # Tiebreaker
        if home_pred_final == away_pred_final:
            home_pred_final += 1

        predicted_winner = (
            game['home_team'] if home_pred_final > away_pred_final
            else game['away_team']
        )
        margin = abs(home_pred_final - away_pred_final)

        home_win_prob = 1 / (1 + np.exp(-margin / 8))
        if home_pred_final < away_pred_final:
            home_win_prob = 1 - home_win_prob

        if margin >= 12:
            confidence = "HIGH"
            conf_emoji = "🔒"
        elif margin >= 6:
            confidence = "MEDIUM"
            conf_emoji = "📊"
        else:
            confidence = "LOW"
            conf_emoji = "🎲"

        # Downgrade confidence for high-variance teams
        home_volatile = float(
            row_dict.get('home_bad_night_pct', 0.15)
        ) > 0.35
        away_volatile = float(
            row_dict.get('away_bad_night_pct', 0.15)
        ) > 0.35
        if home_volatile or away_volatile:
            if confidence == "HIGH":
                confidence = "MEDIUM"
                conf_emoji = "📊"
            elif confidence == "MEDIUM":
                confidence = "LOW"
                conf_emoji = "🎲"

        factors = get_key_factors(
            row_dict, home_pred_final, away_pred_final,
            home_player_preds = home_player_preds,
            away_player_preds = away_player_preds,
            injury_report     = injury_report,
            confidence= confidence
        )

        vegas_line = ""
        if game.get('vegas_total'):
            veg_total  = float(game['vegas_total'])
            veg_spread = float(game['vegas_spread'] or 0)
            vegas_line = (
                f"Vegas: O/U {veg_total} | "
                f"Spread: {veg_spread:+.1f} | "
                f"Implied: {game['home_team']} "
                f"{float(game['vegas_home']):.0f} — "
                f"{game['away_team']} "
                f"{float(game['vegas_away']):.0f}"
            )

        pred = {
            'game_id':           game['game_id'],
            'home_team':         game['home_team'],
            'away_team':         game['away_team'],
            'home_pred':         home_pred_final,
            'away_pred':         away_pred_final,
            'home_pred_team':    round(home_pred_team),
            'away_pred_team':    round(away_pred_team),
            'home_pred_player':  home_player_pts,
            'away_pred_player':  away_player_pts,
            'predicted_winner':  predicted_winner,
            'margin':            round(margin, 1),
            'home_win_prob':     round(home_win_prob * 100, 1),
            'away_win_prob':     round((1 - home_win_prob) * 100, 1),
            'confidence':        confidence,
            'conf_emoji':        conf_emoji,
            'factors':           factors,
            'vegas_line':        vegas_line,
            'is_playoff':        game['season_type'] == 'Playoffs',
            'season_type':       game['season_type'],
            'home_player_preds': home_player_preds,
            'away_player_preds': away_player_preds,
        }
        predictions.append(pred)

        # Print output
        playoff_tag = "🏆 PLAYOFFS" if pred['is_playoff'] else "🏀 NBA"
        print(f"\n{playoff_tag} | {conf_emoji} {confidence} CONFIDENCE")
        print(f"{game['away_team']} @ {game['home_team']}")
        print(f"Prediction: {game['home_team']} {home_pred_final} "
              f"— {game['away_team']} {away_pred_final}")

        if player_model_available:
            print(f"  Team model:   "
                  f"{game['home_team']} {round(home_pred_team)} — "
                  f"{game['away_team']} {round(away_pred_team)}")
            print(f"  Player model: "
                  f"{game['home_team']} {home_player_pts} — "
                  f"{game['away_team']} {away_player_pts}")

        win_pct = (
            pred['home_win_prob']
            if predicted_winner == game['home_team']
            else pred['away_win_prob']
        )
        print(f"Winner: {predicted_winner} ({win_pct}%)")

        if vegas_line:
            print(f"{vegas_line}")

        if factors:
            print("Key factors:")
            for f in factors:
                print(f"  → {f}")

        if home_player_preds:
            print(f"\n  {game['home_team']} key players:")
            for p in home_player_preds[:8]:
                rti   = " ⚠️RTI" if p.get('returning_from_injury') else ""
                slump = " 📉"     if p.get('slump_flag')            else ""
                hot   = " 🔥"     if p.get('hot_flag')              else ""
                print(f"    {p['full_name']:<22} "
                      f"{p['pred_points']:.0f}pts "
                      f"{p['pred_rebounds']:.0f}reb "
                      f"{p['pred_assists']:.0f}ast"
                      f"{rti}{slump}{hot}")

        if away_player_preds:
            print(f"\n  {game['away_team']} key players:")
            for p in away_player_preds[:8]:
                rti   = " ⚠️RTI" if p.get('returning_from_injury') else ""
                slump = " 📉"     if p.get('slump_flag')            else ""
                hot   = " 🔥"     if p.get('hot_flag')              else ""
                print(f"    {p['full_name']:<22} "
                      f"{p['pred_points']:.0f}pts "
                      f"{p['pred_rebounds']:.0f}reb "
                      f"{p['pred_assists']:.0f}ast"
                      f"{rti}{slump}{hot}")

        # Injuries
        for team_abbr in [game['home_team'], game['away_team']]:
            team_injuries = injury_report.get(team_abbr, [])
            key = [
                p for p in team_injuries
                if p['status'] in ['Out', 'Doubtful']
            ]
            if key:
                print(f"\n  ⚠️  {team_abbr} injuries:")
                for p in key:
                    detail = f" ({p['type']})" if p['type'] else ""
                    print(f"    ❌ {p['name']}{detail}")

        print()

    print(f"\n{'='*55}")
    print(f"Generated {len(predictions)} predictions")
    print(f"{'='*55}\n")
    save_predictions(predictions)
    return predictions


if __name__ == "__main__":
    predictions = predict_todays_games()