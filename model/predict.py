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

# Ensemble weights
TEAM_WEIGHT   = 0.83
PLAYER_WEIGHT = 0.17


def load_models():
    with open('model/model_home.pkl', 'rb') as f:
        model_home = pickle.load(f)
    with open('model/model_away.pkl', 'rb') as f:
        model_away = pickle.load(f)
    return model_home, model_away


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


def get_key_factors(row, home_pred, away_pred):
    """Generate human-readable key factors driving the prediction."""
    factors = []

    if row.get('home_avg_points', 0) > row.get('away_def_rating', 0):
        factors.append(
            f"{row['home_team']} averaging "
            f"{float(row['home_avg_points']):.0f} pts "
            f"vs {row['away_team']} allowing "
            f"{float(row['away_def_rating']):.0f} per 100"
        )

    rest_adv = row.get('rest_advantage', 0)
    if rest_adv >= 2:
        factors.append(
            f"{row['home_team']} has {int(rest_adv)} more rest days"
        )
    elif rest_adv <= -2:
        factors.append(
            f"{row['away_team']} has "
            f"{int(abs(rest_adv))} more rest days"
        )

    home_mom = row.get('home_momentum', 0)
    away_mom = row.get('away_momentum', 0)
    if home_mom > 3:
        factors.append(
            f"{row['home_team']} trending up "
            f"(+{home_mom:.1f} pts vs 10-game avg)"
        )
    elif home_mom < -3:
        factors.append(
            f"{row['home_team']} trending down "
            f"({home_mom:.1f} pts vs 10-game avg)"
        )
    if away_mom > 3:
        factors.append(
            f"{row['away_team']} trending up "
            f"(+{away_mom:.1f} pts vs 10-game avg)"
        )

    combined = row.get('combined_pace', 98)
    if combined > 101:
        factors.append(
            f"High pace game expected "
            f"({combined:.0f} possessions)"
        )
    elif combined < 95:
        factors.append(
            f"Slow, defensive game expected "
            f"({combined:.0f} possessions)"
        )

    if row.get('is_playoff', 0) and row.get('series_game_num', 0) > 1:
        game_num = int(row['series_game_num'])
        h_wins   = int(row['home_series_wins'])
        a_wins   = int(row['away_series_wins'])
        factors.append(
            f"Game {game_num} — Series: "
            f"{row['home_team']} {h_wins} — "
            f"{row['away_team']} {a_wins}"
        )

    vegas_total = row.get('vegas_total', 0)
    if vegas_total and float(vegas_total) > 0:
        our_total = home_pred + away_pred
        diff = our_total - float(vegas_total)
        if abs(diff) > 8:
            direction = "higher" if diff > 0 else "lower"
            factors.append(
                f"Model predicts {abs(diff):.0f} pts "
                f"{direction} than Vegas total "
                f"({float(vegas_total):.0f})"
            )

    return factors[:3]


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
    predictions = []

    for game in games:
        print(f"Building features for "
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

        # Player model predictions
        try:
            from model.predict_players import predict_team_player_stats
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
                    )
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
                    )
                )
            player_model_available = (
                home_player_pts > 0 and away_player_pts > 0
            )
        except Exception as e:
            print(f"  ⚠️  Player model failed: {e}")
            home_player_preds = []
            away_player_preds = []
            home_player_pts   = 0
            away_player_pts   = 0
            player_model_available = False

        # Ensemble blend
        if player_model_available:
            home_pred_final = round(
                home_pred_team * TEAM_WEIGHT +
                home_player_pts * PLAYER_WEIGHT
            )
            away_pred_final = round(
                away_pred_team * TEAM_WEIGHT +
                away_player_pts * PLAYER_WEIGHT
            )
        else:
            home_pred_final = round(home_pred_team)
            away_pred_final = round(away_pred_team)

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
            row_dict, home_pred_final, away_pred_final
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

        # Show model breakdown
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

        # Player projections
        if home_player_preds:
            print(f"\n  {game['home_team']} key players:")
            for p in home_player_preds[:8]:
                rti = " ⚠️RTI" if p.get('returning_from_injury') else ""
                print(f"    {p['full_name']:<22} "
                      f"{p['pred_points']:.0f}pts "
                      f"{p['pred_rebounds']:.0f}reb "
                      f"{p['pred_assists']:.0f}ast{rti}")

        if away_player_preds:
            print(f"\n  {game['away_team']} key players:")
            for p in away_player_preds[:8]:
                rti = " ⚠️RTI" if p.get('returning_from_injury') else ""
                print(f"    {p['full_name']:<22} "
                      f"{p['pred_points']:.0f}pts "
                      f"{p['pred_rebounds']:.0f}reb "
                      f"{p['pred_assists']:.0f}ast{rti}")

        print()

    print(f"\n{'='*55}")
    print(f"Generated {len(predictions)} predictions")
    print(f"{'='*55}\n")

    return predictions


if __name__ == "__main__":
    predictions = predict_todays_games()