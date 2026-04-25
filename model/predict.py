import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date
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


# Ensemble weights
# Dynamic ensemble weights based on star power
def clamp(value, low, high):
    """Small helper so calibration code stays readable."""
    return max(low, min(high, value))


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


# Ensemble weights
# Dynamic ensemble weights based on star power, but without crushing disagreement.

def get_ensemble_weights(
    home_player_preds,
    away_player_preds,
    home_team_pts,
    away_team_pts,
    home_player_pts,
    away_player_pts,
    is_playoff=False
):
    """
    Dynamic team/player weights.

    Big philosophy change:
    - Disagreement should mainly lower confidence, not erase the margin.
    - The player model matters more in playoffs and star-heavy games.
    - Totals can be tempered if the player model is way off, but no signal gets nuked.
    """
    player_w = 0.31

    all_players = (home_player_preds or []) + (away_player_preds or [])
    stars = sum(1 for p in all_players if safe_float(p.get('usage_rate'), 0) > 0.25)
    alpha_stars = sum(1 for p in all_players if safe_float(p.get('usage_rate'), 0) > 0.31)

    player_w += min(stars * 0.025, 0.08)
    player_w += min(alpha_stars * 0.015, 0.04)

    if is_playoff:
        player_w += 0.04

    team_total = home_team_pts + away_team_pts
    player_total = home_player_pts + away_player_pts
    total_gap = abs(player_total - team_total)

    # Temper total outliers, but keep the player model alive.
    if total_gap >= 28:
        player_w *= 0.70
    elif total_gap >= 20:
        player_w *= 0.80
    elif total_gap >= 12:
        player_w *= 0.90

    player_w = clamp(player_w, 0.22, 0.46)
    team_w = 1.0 - player_w
    return team_w, player_w

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
            Game.is_final == False  # Changed from status == 'scheduled'
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
                'home_series_wins':  g.series_home_wins or 0,
                'away_series_wins':  g.series_away_wins or 0,
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
        if isinstance(injury_report, dict):
            injury_items = []
            for team_abbr, players in injury_report.items():
                for p in players or []:
                    item = dict(p)
                    item.setdefault("team", team_abbr)
                    injury_items.append(item)
        else:
            injury_items = injury_report

        for inj in injury_items:
            try:
                team = inj.get("team")
                player = inj.get("player") or inj.get("full_name") or inj.get("name")
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
    """Main prediction workflow"""
    print_header()
    
    model_home, model_away = load_models()
    games = get_todays_games()
    
    if not games:
        print("No games scheduled today.")
        return []
    
    print(f"Found {len(games)} games today\n")
    
    # Fetch shared data once
    injury_report, all_player_props = fetch_shared_game_data()
    
    predictions = []
    for game in games:
        prediction = predict_single_game(
            game, model_home, model_away, 
            injury_report, all_player_props
        )
        if prediction:
            predictions.append(prediction)
            print_prediction(prediction, game, injury_report)
    
    print_footer(len(predictions))
    save_predictions(predictions)
    return predictions


def print_header():
    """Print predictions header"""
    print(f"\n{'='*55}")
    print(f"NBA PREDICTIONS — {date.today().strftime('%B %d, %Y')}")
    print(f"{'='*55}\n")


def fetch_shared_game_data():
    """Fetch injury report and player props once for all games"""
    from data.ingestion.fetch_injuries import fetch_injury_report
    from data.ingestion.fetch_odds import fetch_todays_player_props
    
    print("Fetching injury report...")
    injury_report = fetch_injury_report()
    
    print("Fetching player props...")
    all_player_props = fetch_todays_player_props()
    
    return injury_report, all_player_props


def predict_single_game(game, model_home, model_away, injury_report, all_player_props):
    """Generate prediction for a single game."""
    print(f"\nBuilding features for {game['away_team']} @ {game['home_team']}...")

    row_dict = build_game_features(game)
    if not row_dict:
        print("  ⚠️  Could not build features")
        return None

    row_dict['home_team'] = game['home_team']
    row_dict['away_team'] = game['away_team']

    home_pred_team, away_pred_team = get_team_predictions(model_home, model_away, row_dict)

    game_props = match_player_props(all_player_props, game)
    player_results = get_player_predictions(game, row_dict, injury_report, game_props)

    home_pred_final, away_pred_final, ensemble_debug = calculate_ensemble_prediction(
        game,
        row_dict,
        home_pred_team,
        away_pred_team,
        player_results
    )

    home_pred_final, away_pred_final, calibration_debug = calibrate_final_score(
        game,
        row_dict,
        home_pred_final,
        away_pred_final,
        home_pred_team,
        away_pred_team,
        player_results
    )

    if home_pred_final == away_pred_final:
        edge = get_directional_edge(game, row_dict, home_pred_team, away_pred_team, player_results)
        if edge < 0:
            away_pred_final += 1
        else:
            home_pred_final += 1

    confidence, conf_emoji = calculate_confidence(
        home_pred_final,
        away_pred_final,
        row_dict,
        home_pred_team=home_pred_team,
        away_pred_team=away_pred_team,
        player_results=player_results,
        debug={**ensemble_debug, **calibration_debug}
    )

    row_dict['home_team_pred'] = home_pred_team
    row_dict['away_team_pred'] = away_pred_team
    row_dict['home_player_pred'] = player_results.get('home_pts', 0)
    row_dict['away_player_pred'] = player_results.get('away_pts', 0)

    factors = get_key_factors(
        row_dict,
        home_pred_final,
        away_pred_final,
        home_player_preds=player_results['home_preds'],
        away_player_preds=player_results['away_preds'],
        injury_report=injury_report,
        confidence=confidence
    )

    return build_prediction_dict(
        game,
        home_pred_final,
        away_pred_final,
        home_pred_team,
        away_pred_team,
        player_results,
        confidence,
        conf_emoji,
        factors,
        debug={**ensemble_debug, **calibration_debug}
    )


def build_game_features(game):
    """Build feature vector for a game"""
    return build_features_for_upcoming_game(
        home_team_id=game['home_team_id'],
        away_team_id=game['away_team_id'],
        game_date=game['game_date'],
        season_type=game['season_type'],
        series_game_num=game['series_game_num'],
        home_series_wins=game['home_series_wins'],
        away_series_wins=game['away_series_wins'],
    )


def get_team_predictions(model_home, model_away, row_dict):
    """Get predictions from team models"""
    features = np.array([
        float(row_dict.get(f, 0) or 0)
        for f in FEATURE_COLS
    ]).reshape(1, -1)
    
    home_pred = float(model_home.predict(features)[0])
    away_pred = float(model_away.predict(features)[0])
    
    return home_pred, away_pred


def match_player_props(all_player_props, game):
    """Match player props to current game"""
    game_props_key = next(
        (k for k in all_player_props
         if game['away_team'] in k or game['home_team'] in k),
        None
    )
    return all_player_props.get(game_props_key, {})


def get_player_predictions(game, row_dict, injury_report, game_props):
    """Get predictions from player models for both teams"""
    from model.predict_players import predict_team_player_stats
    
    try:
        home_preds, home_pts, _, _ = predict_team_player_stats(
            team_id=game['home_team_id'],
            opponent_team_id=game['away_team_id'],
            is_home=True,
            is_playoff=game['season_type'] == 'Playoffs',
            team_off_rating=float(row_dict.get('home_off_rating', 115)),
            opp_def_rating=float(row_dict.get('away_def_rating', 112)),
            opp_pace=float(row_dict.get('away_pace', 98)),
            team_pace=float(row_dict.get('home_pace', 98)),
            injury_report=injury_report,
            team_abbr=game['home_team'],
            player_props=game_props
        )
        
        away_preds, away_pts, _, _ = predict_team_player_stats(
            team_id=game['away_team_id'],
            opponent_team_id=game['home_team_id'],
            is_home=False,
            is_playoff=game['season_type'] == 'Playoffs',
            team_off_rating=float(row_dict.get('away_off_rating', 113)),
            opp_def_rating=float(row_dict.get('home_def_rating', 114)),
            opp_pace=float(row_dict.get('home_pace', 98)),
            team_pace=float(row_dict.get('away_pace', 98)),
            injury_report=injury_report,
            team_abbr=game['away_team'],
            player_props=game_props
        )
        
        return {
            'home_preds': home_preds,
            'home_pts': home_pts,
            'away_preds': away_preds,
            'away_pts': away_pts,
            'available': home_pts > 0 and away_pts > 0
        }
        
    except Exception as e:
        print(f"  ⚠️  Player model failed: {e}")
        return {
            'home_preds': [],
            'home_pts': 0,
            'away_preds': [],
            'away_pts': 0,
            'available': False
        }



def same_sign(a, b):
    """True when both margins point to the same winner."""
    return (a > 0 and b > 0) or (a < 0 and b < 0)


def sign_or_zero(value):
    """Return -1, 0, or 1 without numpy edge-case surprises."""
    value = safe_float(value, 0.0)
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def get_market_home_margin(game):
    """
    Convert stored home_spread to expected home margin.

    In your printed output, a positive home spread means the home team is the
    underdog. Example: PHX +9.5 means market expects PHX by -9.5, so the
    expected home margin is -spread.
    """
    if game.get('vegas_spread') is None:
        return None
    return -safe_float(game.get('vegas_spread'), 0.0)


def estimate_player_projection_risk(player_results, team_total=None):
    """
    Lightweight risk estimate for the player model.

    This does not require retraining or changes inside predict_team_player_stats.
    It looks for common reasons player aggregation gets too aggressive:
    - a very different total than the team model
    - too much scoring coming from lower-usage / likely bench players
    - many hot/slump flags, which usually means more volatility
    """
    if not player_results or not player_results.get('available'):
        return {
            'risk': 1.0,
            'bench_scoring_share': 0.0,
            'flagged_players': 0,
            'player_total_gap': 0.0,
        }

    all_players = (player_results.get('home_preds') or []) + (player_results.get('away_preds') or [])
    player_total = safe_float(player_results.get('home_pts'), 0) + safe_float(player_results.get('away_pts'), 0)

    low_usage_pts = 0.0
    flagged_players = 0
    for p in all_players:
        pts = safe_float(p.get('pred_points'), 0)
        usage = safe_float(p.get('usage_rate'), 0)
        if usage and usage < 0.18:
            low_usage_pts += pts
        if p.get('hot_flag') or p.get('slump_flag') or p.get('returning_from_injury'):
            flagged_players += 1

    bench_share = (low_usage_pts / player_total) if player_total > 0 else 0.0
    total_gap = abs(player_total - team_total) if team_total is not None else 0.0

    risk = 0.0
    if total_gap >= 24:
        risk += 2.0
    elif total_gap >= 16:
        risk += 1.25
    elif total_gap >= 10:
        risk += 0.5

    if bench_share >= 0.34:
        risk += 1.0
    elif bench_share >= 0.28:
        risk += 0.5

    if flagged_players >= 6:
        risk += 0.75
    elif flagged_players >= 3:
        risk += 0.35

    return {
        'risk': round(risk, 3),
        'bench_scoring_share': round(bench_share, 3),
        'flagged_players': flagged_players,
        'player_total_gap': round(total_gap, 2),
    }


def score_team_model_reliability(row_dict, game, team_margin, player_margin, market_margin):
    """
    Reliability score for using the team model as the side/margin anchor.
    This is deterministic arbitration, not noise or an artificial margin floor.
    """
    score = 2.0  # Team model is the baseline side/margin anchor.

    abs_team = abs(team_margin)
    if abs_team >= 8:
        score += 2.0
    elif abs_team >= 5:
        score += 1.25
    elif abs_team <= 2:
        score -= 0.75

    net_rating_diff = clamp(safe_float(row_dict.get('net_rating_diff'), 0.0), -12, 12)
    if abs(net_rating_diff) >= 7 and same_sign(team_margin, net_rating_diff):
        score += 1.25
    elif abs(net_rating_diff) >= 4 and same_sign(team_margin, net_rating_diff):
        score += 0.75

    home_edge = safe_float(row_dict.get('home_off_vs_away_def'), 0.0)
    away_edge = safe_float(row_dict.get('away_off_vs_home_def'), 0.0)
    matchup_edge = home_edge - away_edge
    if abs(matchup_edge) >= 5 and same_sign(team_margin, matchup_edge):
        score += 0.75

    if market_margin is not None:
        if same_sign(team_margin, market_margin):
            score += 1.75
            if abs(market_margin) >= 5:
                score += 0.75
        elif abs(market_margin) >= 4:
            score -= 0.75

    if not same_sign(team_margin, player_margin) and abs(player_margin) >= 10:
        score -= 0.35

    return round(score, 3)


def score_player_model_reliability(row_dict, game, player_results, team_margin, player_margin, market_margin):
    """
    Reliability score for letting the player model override team/market reads.
    It can win, but it should need evidence when it points opposite the team model.
    """
    score = 1.0
    all_players = (player_results.get('home_preds') or []) + (player_results.get('away_preds') or [])

    stars = sum(1 for p in all_players if safe_float(p.get('usage_rate'), 0) >= 0.27)
    alpha_stars = sum(1 for p in all_players if safe_float(p.get('usage_rate'), 0) >= 0.32)
    score += min(stars * 0.30, 1.20)
    score += min(alpha_stars * 0.35, 0.90)

    if game.get('season_type') == 'Playoffs':
        score += 0.65

    abs_player = abs(player_margin)
    if abs_player >= 10:
        score += 1.25
    elif abs_player >= 6:
        score += 0.75
    elif abs_player <= 2:
        score -= 0.75

    if market_margin is not None:
        if same_sign(player_margin, market_margin):
            score += 1.25
            if abs(market_margin) >= 5:
                score += 0.50
        elif abs(market_margin) >= 4:
            score -= 0.90

    team_total = safe_float(row_dict.get('_team_total_for_reliability'), 0.0)
    risk_info = estimate_player_projection_risk(player_results, team_total=team_total if team_total else None)
    score -= safe_float(risk_info.get('risk'), 0.0) * 0.65

    if market_margin is not None:
        if not same_sign(player_margin, team_margin) and not same_sign(player_margin, market_margin):
            score -= 0.75

    return round(score, 3), risk_info


def choose_margin_from_arbitration(row_dict, game, team_margin, player_margin, player_results):
    """
    Pick a final margin using arbitration.

    Big change from the old version:
    - Consensus margins can be blended and preserved.
    - Disagreement margins are NOT directly averaged because that collapses
      +6 and -15 into fake coin flips.
    - Disagreement chooses the most reliable signal, then keeps a tempered
      portion of that signal's edge.
    """
    market_margin = get_market_home_margin(game)
    winner_disagreement = not same_sign(team_margin, player_margin)
    model_disagreement = abs(team_margin - player_margin)

    team_score = score_team_model_reliability(
        row_dict, game, team_margin, player_margin, market_margin
    )
    player_score, player_risk = score_player_model_reliability(
        row_dict, game, player_results, team_margin, player_margin, market_margin
    )

    candidates = [
        {'source': 'team', 'margin': team_margin, 'score': team_score},
        {'source': 'player', 'margin': player_margin, 'score': player_score},
    ]
    if market_margin is not None:
        market_score = 1.35
        if same_sign(market_margin, team_margin):
            market_score += 0.75
        if same_sign(market_margin, player_margin):
            market_score += 0.50
        if abs(market_margin) >= 6:
            market_score += 0.40
        candidates.append({'source': 'market', 'margin': market_margin, 'score': round(market_score, 3)})

    if not winner_disagreement:
        team_w, player_w = get_ensemble_weights(
            player_results.get('home_preds') or [],
            player_results.get('away_preds') or [],
            safe_float(row_dict.get('_home_pred_team'), 0.0),
            safe_float(row_dict.get('_away_pred_team'), 0.0),
            safe_float(player_results.get('home_pts'), 0.0),
            safe_float(player_results.get('away_pts'), 0.0),
            is_playoff=game.get('season_type') == 'Playoffs'
        )
        raw_margin = team_margin * team_w + player_margin * player_w
        min_abs = min(abs(team_margin), abs(player_margin))
        max_abs = max(abs(team_margin), abs(player_margin))

        if min_abs >= 3.5:
            preserved_abs = max(abs(raw_margin), 0.72 * min_abs + 0.18 * max_abs)
            margin = sign_or_zero(raw_margin) * preserved_abs
            method = 'consensus_preserved'
        else:
            margin = raw_margin
            method = 'consensus_blended'

        if game.get('season_type') == 'Playoffs' and abs(margin) >= 3:
            margin *= 1.04

        debug = {
            'team_reliability': team_score,
            'player_reliability': player_score,
            'player_projection_risk': player_risk.get('risk'),
            'player_bench_scoring_share': player_risk.get('bench_scoring_share'),
            'flagged_players': player_risk.get('flagged_players'),
            'chosen_margin_source': 'consensus',
            'margin_method': method,
            'model_disagreement': round(model_disagreement, 2),
            'winner_disagreement': False,
        }
        return clamp(margin, -24, 24), debug

    sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    best = sorted_candidates[0]
    second = sorted_candidates[1]
    reliability_gap = best['score'] - second['score']

    if reliability_gap < 0.65:
        directional = get_directional_edge(game, row_dict,
                                          safe_float(row_dict.get('_home_pred_team'), 0.0),
                                          safe_float(row_dict.get('_away_pred_team'), 0.0),
                                          player_results)
        if abs(directional) < 0.75:
            directional = team_margin if abs(team_margin) >= abs(player_margin) else player_margin

        margin = sign_or_zero(directional) * 2.6
        method = 'disagreement_true_tossup'
        chosen_source = 'mixed'
    else:
        chosen_source = best['source']
        chosen_margin = best['margin']

        if chosen_source == 'team':
            target_abs = 0.78 * abs(chosen_margin)
        elif chosen_source == 'player':
            target_abs = 0.66 * abs(chosen_margin)
        else:
            target_abs = 0.62 * abs(chosen_margin)

        if market_margin is not None and same_sign(chosen_margin, market_margin):
            target_abs = max(
                target_abs,
                0.75 * abs(chosen_margin),
                0.70 * abs(market_margin)
            )

        losing_margin = second['margin']
        if not same_sign(chosen_margin, losing_margin) and abs(losing_margin) >= 12:
            target_abs *= 0.90

        target_abs = clamp(target_abs, min(2.0, abs(chosen_margin)), 16.0)
        margin = sign_or_zero(chosen_margin) * target_abs
        method = f'disagreement_arbitrated_to_{chosen_source}'

    if game.get('season_type') == 'Playoffs' and abs(margin) >= 3:
        margin *= 1.04

    debug = {
        'team_reliability': team_score,
        'player_reliability': player_score,
        'player_projection_risk': player_risk.get('risk'),
        'player_bench_scoring_share': player_risk.get('bench_scoring_share'),
        'flagged_players': player_risk.get('flagged_players'),
        'chosen_margin_source': chosen_source,
        'reliability_gap': round(reliability_gap, 3),
        'margin_method': method,
        'model_disagreement': round(model_disagreement, 2),
        'winner_disagreement': True,
    }
    return clamp(margin, -24, 24), debug


def blend_game_total(game, team_total, player_total, player_results):
    """
    Blend scoring total separately from margin.
    Totals are allowed to use Vegas more directly because total calibration
    does not decide the winner.
    """
    team_total = safe_float(team_total, 0.0)
    player_total = safe_float(player_total, 0.0)

    if not player_results.get('available') or player_total <= 0:
        blended_total = team_total
        team_w = 1.0
        player_w = 0.0
    else:
        total_gap = abs(player_total - team_total)
        player_w = 0.30
        if game.get('season_type') == 'Playoffs':
            player_w += 0.04
        if total_gap >= 28:
            player_w *= 0.65
        elif total_gap >= 20:
            player_w *= 0.75
        elif total_gap >= 12:
            player_w *= 0.88
        player_w = clamp(player_w, 0.18, 0.40)
        team_w = 1.0 - player_w
        blended_total = team_total * team_w + player_total * player_w

    vegas_total = game.get('vegas_total')
    has_vegas_total = vegas_total is not None and safe_float(vegas_total, 0) > 0
    market_total_weight = 0.0

    if has_vegas_total:
        vegas_total = safe_float(vegas_total)
        total_gap_vs_market = abs(blended_total - vegas_total)
        market_total_weight = 0.18 if total_gap_vs_market <= 10 else 0.28
        blended_total = blended_total * (1 - market_total_weight) + vegas_total * market_total_weight

    total_drag = 0.0
    if game.get('season_type') == 'Playoffs':
        game_num = int(game.get('series_game_num', 1) or 1)
        if game_num >= 3:
            total_drag += 0.8
        if game_num >= 5:
            total_drag += 0.6
        if int(game.get('home_series_wins', 0) or 0) == 3 or int(game.get('away_series_wins', 0) or 0) == 3:
            total_drag += 0.6
        blended_total -= total_drag

    return clamp(blended_total, 185, 255), {
        'team_weight': round(team_w, 3),
        'player_weight': round(player_w, 3),
        'market_total_weight': round(market_total_weight, 3),
        'total_drag': round(total_drag, 2),
    }


def calculate_ensemble_prediction(game, row_dict, home_pred_team, away_pred_team, player_results):
    """
    Final ensemble v2.

    Core design:
    - Blend/calibrate total separately.
    - Arbitrate margin instead of averaging opposing margin signals.
    - Disagreement lowers confidence through debug flags; it does not
      automatically shrink every game to 1 point.
    """
    team_total = float(home_pred_team + away_pred_team)
    team_margin = float(home_pred_team - away_pred_team)

    row_dict['_home_pred_team'] = float(home_pred_team)
    row_dict['_away_pred_team'] = float(away_pred_team)
    row_dict['_team_total_for_reliability'] = team_total

    if not player_results.get('available'):
        market_margin = get_market_home_margin(game)
        final_total, total_debug = blend_game_total(
            game, team_total, 0.0, {'available': False}
        )
        final_margin = team_margin
        home_pred = (final_total + final_margin) / 2
        away_pred = (final_total - final_margin) / 2
        debug = {
            **total_debug,
            'team_total': round(team_total, 2),
            'player_total': 0.0,
            'team_margin': round(team_margin, 2),
            'player_margin': 0.0,
            'ensemble_margin': round(final_margin, 2),
            'model_disagreement': 0.0,
            'winner_disagreement': False,
            'margin_method': 'team_only',
            'chosen_margin_source': 'team',
            'team_reliability': None,
            'player_reliability': None,
            'market_margin': None if market_margin is None else round(market_margin, 2),
        }
        return home_pred, away_pred, debug

    player_total = float(player_results.get('home_pts', 0) + player_results.get('away_pts', 0))
    player_margin = float(player_results.get('home_pts', 0) - player_results.get('away_pts', 0))

    final_total, total_debug = blend_game_total(game, team_total, player_total, player_results)
    final_margin, margin_debug = choose_margin_from_arbitration(
        row_dict, game, team_margin, player_margin, player_results
    )

    market_margin = get_market_home_margin(game)

    home_pred = (final_total + final_margin) / 2
    away_pred = (final_total - final_margin) / 2

    debug = {
        **total_debug,
        **margin_debug,
        'team_total': round(team_total, 2),
        'player_total': round(player_total, 2),
        'team_margin': round(team_margin, 2),
        'player_margin': round(player_margin, 2),
        'ensemble_margin': round(final_margin, 2),
        'market_margin': None if market_margin is None else round(market_margin, 2),
    }

    return home_pred, away_pred, debug


def get_directional_edge(game, row_dict, home_pred_team, away_pred_team, player_results):
    """
    Positive = home edge, negative = away edge.
    Used only for rounding tiebreaks and true toss-up direction.
    This intentionally favors stable signals when models disagree.
    """
    team_margin = float(home_pred_team - away_pred_team)
    player_margin = 0.0
    if player_results.get('available'):
        player_margin = float(player_results.get('home_pts', 0) - player_results.get('away_pts', 0))

    market_margin = get_market_home_margin(game)
    net_rating_diff = clamp(safe_float(row_dict.get('net_rating_diff'), 0.0), -10, 10)

    if market_margin is not None:
        edge = 0.47 * team_margin + 0.24 * player_margin + 0.24 * market_margin + 0.05 * net_rating_diff
    else:
        edge = 0.58 * team_margin + 0.32 * player_margin + 0.10 * net_rating_diff

    return edge


def calibrate_final_score(game, row_dict, home_pred, away_pred, home_pred_team, away_pred_team, player_results):
    """
    Final calibration after ensemble v2.

    This is intentionally light because calculate_ensemble_prediction now already:
    - calibrates total toward Vegas
    - applies playoff total drag
    - arbitrates margin

    This function mainly rounds scores and avoids accidental ties.
    """
    model_total = float(home_pred + away_pred)
    model_margin = float(home_pred - away_pred)
    directional_edge = get_directional_edge(game, row_dict, home_pred_team, away_pred_team, player_results)
    market_margin = get_market_home_margin(game)

    calibrated_total = clamp(model_total, 185, 255)
    calibrated_margin = clamp(model_margin, -24, 24)

    if abs(calibrated_margin) < 1.25 and abs(directional_edge) >= 2.25:
        calibrated_margin = sign_or_zero(directional_edge) * min(3.2, max(2.2, abs(directional_edge) * 0.55))

    if market_margin is not None and same_sign(calibrated_margin, market_margin):
        if abs(market_margin) >= 7 and abs(calibrated_margin) < 4:
            calibrated_margin = sign_or_zero(calibrated_margin) * min(6.0, max(4.0, abs(market_margin) * 0.55))

    home_final = round((calibrated_total + calibrated_margin) / 2)
    away_final = round((calibrated_total - calibrated_margin) / 2)

    if home_final == away_final:
        if directional_edge < 0:
            away_final += 1
        else:
            home_final += 1

    debug = {
        'model_total_before_calibration': round(model_total, 2),
        'model_margin_before_calibration': round(model_margin, 2),
        'calibrated_total': round(calibrated_total, 2),
        'calibrated_margin': round(calibrated_margin, 2),
        'directional_edge': round(directional_edge, 2),
        'market_margin': None if market_margin is None else round(market_margin, 2),
    }
    return home_final, away_final, debug


def apply_playoff_compression(home_pred, away_pred, game):
    """
    Deprecated compatibility function.

    Kept so imports/tests do not break, but no longer used in predict_single_game.
    The old version multiplied both teams by the same factor, which crushed margins.
    """
    total = float(home_pred + away_pred)
    margin = float(home_pred - away_pred)

    drag = 0.0
    game_num = int(game.get('series_game_num', 1) or 1)
    if game_num >= 3:
        drag += 1.0
    if game_num >= 5:
        drag += 0.8

    total -= drag
    return round((total + margin) / 2), round((total - margin) / 2)



def calculate_confidence(home_pred, away_pred, row_dict,
                         home_pred_team=None, away_pred_team=None,
                         player_results=None, debug=None):
    """
    Confidence is not the same thing as margin.

    A 7-point projected win with team/player disagreement should not be HIGH confidence.
    A 2-point projected win can be correctly low confidence without forcing every score to be close.
    """
    margin = abs(home_pred - away_pred)
    debug = debug or {}

    model_disagreement = safe_float(debug.get('model_disagreement'), 0.0)
    winner_disagreement = bool(debug.get('winner_disagreement', False))
    bad_night_max = max(
        safe_float(row_dict.get('home_bad_night_pct'), 0.15),
        safe_float(row_dict.get('away_bad_night_pct'), 0.15)
    )

    if margin >= 10:
        confidence = "HIGH"
        conf_emoji = "🔒"
    elif margin >= 5:
        confidence = "MEDIUM"
        conf_emoji = "📊"
    else:
        confidence = "LOW"
        conf_emoji = "🎲"

    # Disagreement lowers confidence but does not alter the score.
    if winner_disagreement or model_disagreement >= 12 or bad_night_max > 0.35:
        if confidence == "HIGH":
            confidence = "MEDIUM"
            conf_emoji = "📊"
        elif confidence == "MEDIUM" and (winner_disagreement or model_disagreement >= 14):
            confidence = "LOW"
            conf_emoji = "🎲"

    return confidence, conf_emoji

def build_prediction_dict(game, home_pred, away_pred, home_pred_team, away_pred_team,
                          player_results, confidence, conf_emoji, factors, debug=None):
    """Build prediction dictionary with all metadata."""
    debug = debug or {}
    margin = abs(home_pred - away_pred)
    predicted_winner = game['home_team'] if home_pred > away_pred else game['away_team']

    win_prob_favorite = 1 / (1 + np.exp(-margin / 9.5))
    home_win_prob = win_prob_favorite if home_pred > away_pred else 1 - win_prob_favorite

    vegas_line = build_vegas_line(game)

    return {
        'game_id': game['game_id'],
        'home_team': game['home_team'],
        'away_team': game['away_team'],
        'home_pred': home_pred,
        'away_pred': away_pred,
        'home_pred_team': round(home_pred_team),
        'away_pred_team': round(away_pred_team),
        'home_pred_player': player_results['home_pts'],
        'away_pred_player': player_results['away_pts'],
        'predicted_winner': predicted_winner,
        'margin': round(margin, 1),
        'home_win_prob': round(home_win_prob * 100, 1),
        'away_win_prob': round((1 - home_win_prob) * 100, 1),
        'confidence': confidence,
        'conf_emoji': conf_emoji,
        'factors': factors,
        'vegas_line': vegas_line,
        'is_playoff': game['season_type'] == 'Playoffs',
        'season_type': game['season_type'],
        'home_player_preds': player_results['home_preds'],
        'away_player_preds': player_results['away_preds'],
        'series_home_wins': game.get('home_series_wins'),
        'series_away_wins': game.get('away_series_wins'),
        'debug': debug,
    }


def build_vegas_line(game):
    """Format Vegas line string"""
    if not game.get('vegas_total'):
        return ""
    
    return (
        f"Vegas: O/U {float(game['vegas_total'])} | "
        f"Spread: {float(game['vegas_spread'] or 0):+.1f} | "
        f"Implied: {game['home_team']} {float(game['vegas_home']):.0f} — "
        f"{game['away_team']} {float(game['vegas_away']):.0f}"
    )



def print_prediction(pred, game, injury_report):
    """Print formatted prediction output."""
    playoff_tag = "🏆 PLAYOFFS" if pred['is_playoff'] else "🏀 NBA"
    print(f"\n{playoff_tag} | {pred['conf_emoji']} {pred['confidence']} CONFIDENCE")

    if pred['is_playoff'] and pred['series_home_wins'] is not None:
        home_wins = int(pred['series_home_wins'])
        away_wins = int(pred['series_away_wins'])

        if home_wins == away_wins:
            series_text = f"📊 Series tied {home_wins}-{away_wins}"
        elif home_wins > away_wins:
            series_text = f"📊 Series: {game['home_team']} leads {home_wins}-{away_wins}"
        else:
            series_text = f"📊 Series: {game['away_team']} leads {away_wins}-{home_wins}"

        print(series_text)

    print(f"{game['away_team']} @ {game['home_team']}")
    print(f"Prediction: {game['home_team']} {pred['home_pred']} — "
          f"{game['away_team']} {pred['away_pred']}")

    if pred['home_pred_player'] > 0:
        print(f"  Team model:   {game['home_team']} {pred['home_pred_team']} — "
              f"{game['away_team']} {pred['away_pred_team']}")
        print(f"  Player model: {game['home_team']} {pred['home_pred_player']} — "
              f"{game['away_team']} {pred['away_pred_player']}")

    if os.environ.get("NBA_PREDICT_DEBUG", "0") == "1":
        dbg = pred.get('debug', {}) or {}
        print("  Debug:")
        print(f"    weights: team={dbg.get('team_weight')} player={dbg.get('player_weight')} | method={dbg.get('margin_method')}")
        print(f"    margins: team={dbg.get('team_margin')} player={dbg.get('player_margin')} ensemble={dbg.get('ensemble_margin')} final={dbg.get('calibrated_margin')}")
        print(f"    totals: team={dbg.get('team_total')} player={dbg.get('player_total')} final={dbg.get('calibrated_total')}")
        print(f"    edge={dbg.get('directional_edge')} market_margin={dbg.get('market_margin')} disagreement={dbg.get('model_disagreement')}")

    win_pct = pred['home_win_prob'] if pred['predicted_winner'] == game['home_team'] else pred['away_win_prob']
    print(f"Winner: {pred['predicted_winner']} ({win_pct}%)")

    if pred['vegas_line']:
        print(f"{pred['vegas_line']}")

    if pred['factors']:
        print("Key factors:")
        for f in pred['factors']:
            print(f"  → {f}")

    print_player_predictions(game['home_team'], pred['home_player_preds'])
    print_player_predictions(game['away_team'], pred['away_player_preds'])

    print_injuries(game['home_team'], injury_report)
    print_injuries(game['away_team'], injury_report)

    print()

def print_player_predictions(team_abbr, player_preds):
    """Print player predictions for a team"""
    if not player_preds:
        return
    
    print(f"\n  {team_abbr} key players:")
    for p in player_preds[:8]:
        rti = " ⚠️RTI" if p.get('returning_from_injury') else ""
        slump = " 📉" if p.get('slump_flag') else ""
        hot = " 🔥" if p.get('hot_flag') else ""
        print(f"    {p['full_name']:<22} "
              f"{p['pred_points']:.0f}pts "
              f"{p['pred_rebounds']:.0f}reb "
              f"{p['pred_assists']:.0f}ast"
              f"{rti}{slump}{hot}")


def print_injuries(team_abbr, injury_report):
    """Print injury report for a team"""
    team_injuries = injury_report.get(team_abbr, [])
    key_injuries = [p for p in team_injuries if p['status'] in ['Out', 'Doubtful']]
    
    if key_injuries:
        print(f"\n  ⚠️  {team_abbr} injuries:")
        for p in key_injuries:
            detail = f" ({p['type']})" if p['type'] else ""
            print(f"    ❌ {p['name']}{detail}")


def print_footer(count):
    """Print predictions footer"""
    print(f"\n{'='*55}")
    print(f"Generated {count} predictions")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    predictions = predict_todays_games()