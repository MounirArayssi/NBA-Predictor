import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
from datetime import date
from sqlalchemy import text
from sqlalchemy.orm import Session
from data.storage.db import engine
from data.storage.models import Team
from model.train_players import PLAYER_FEATURE_COLS


def load_player_models():
    with open('model/player_models.pkl', 'rb') as f:
        return pickle.load(f)

def get_playoff_series_minutes(team_id, season='2025-26'):
    """
    Get each player's actual minutes from games already
    played in this playoff series.
    Returns dict: {player_id: avg_playoff_minutes}
    """
    query = text("""
        SELECT 
            pbs.player_id,
            AVG(pbs.minutes_played) as playoff_avg_minutes,
            COUNT(*) as games_played
        FROM player_box_scores pbs
        JOIN games g ON pbs.game_id = g.game_id
        WHERE pbs.team_id   = :team_id
        AND g.season_type   = 'Playoffs'
        AND g.season        = :season
        AND g.is_final      = TRUE
        AND pbs.minutes_played >= 1
        GROUP BY pbs.player_id
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {
            'team_id': int(team_id),
            'season':  season
        })
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    if df.empty:
        return {}

    return dict(zip(
        df['player_id'].astype(int),
        df['playoff_avg_minutes'].astype(float)
    ))

def get_team_roster(team_id, game_date=None, is_playoff=False):
    """
    Get current rotation players for a team.
    Filters to current season only.
    """
    limit = 9 if is_playoff else 12
    min_minutes = 16 if is_playoff else 15

    query = text("""
        SELECT
            p.player_id,
            p.full_name,
            p.position,
            prs.avg_points          AS player_avg_points_l10,
            prs.avg_rebounds        AS player_avg_reb_l10,
            prs.avg_assists         AS player_avg_ast_l10,
            prs.avg_minutes         AS player_avg_min_l10,
            prs.avg_usage_rate      AS player_avg_usage_l10,
            prs.avg_true_shooting   AS player_avg_ts_l10,
            prs.avg_fg_pct          AS player_avg_fg_l10,
            prs.avg_fg3_pct         AS player_avg_fg3_l10,
            prs.avg_plus_minus      AS player_avg_pm_l10,
            prs.avg_turnovers       AS player_avg_tov_l10,
            prs5.avg_points         AS player_avg_points_l5,
            prs5.avg_rebounds       AS player_avg_reb_l5,
            prs5.avg_assists        AS player_avg_ast_l5,
            prs5.avg_usage_rate     AS player_avg_usage_l5,
            prs5.avg_true_shooting  AS player_avg_ts_l5,
            recent.recent_games
        FROM players p

        JOIN (
            SELECT
                pbs.player_id,
                COUNT(*) AS recent_games
            FROM player_box_scores pbs
            JOIN games g ON pbs.game_id = g.game_id
            WHERE pbs.team_id = :team_id
            AND g.season = '2025-26'
            AND pbs.minutes_played >= 10
            AND g.game_date >= (
                SELECT MAX(g2.game_date) - INTERVAL '65 days'
                FROM games g2
                WHERE g2.season = '2025-26'
                AND g2.is_final = TRUE
            )    
                AND (
        :is_playoff = false
        OR EXISTS (
            SELECT 1
            FROM player_box_scores pbs3
            JOIN games g3 ON pbs3.game_id = g3.game_id
            WHERE pbs3.player_id = pbs.player_id
            AND pbs3.team_id = :team_id
            AND g3.season_type = 'Playoffs'
            AND g3.season = '2025-26'
            AND pbs3.minutes_played >= 3
        )
    )
            GROUP BY pbs.player_id
            HAVING COUNT(*) >= 5
        ) recent ON recent.player_id = p.player_id

JOIN player_rolling_stats prs
    ON prs.player_id   = p.player_id
    AND prs.team_id    = :team_id
    AND prs."window"   = 10
    AND prs.as_of_date = (
        SELECT MAX(as_of_date)
        FROM player_rolling_stats
        WHERE player_id = p.player_id
        AND team_id     = :team_id
        AND "window"    = 10
    )

JOIN player_rolling_stats prs5
    ON prs5.player_id   = p.player_id
    AND prs5.team_id    = :team_id
    AND prs5."window"   = 5
    AND prs5.as_of_date = (
        SELECT MAX(as_of_date)
        FROM player_rolling_stats
        WHERE player_id = p.player_id
        AND team_id     = :team_id
        AND "window"    = 5
    )

        WHERE prs.avg_minutes >= :min_minutes
        AND prs.avg_points    >= 4
        ORDER BY prs.avg_minutes DESC
        LIMIT :limit
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {
            'team_id': int(team_id),
            'limit':   limit,
            'min_minutes': min_minutes,
            'is_playoff': is_playoff
        })
        roster = pd.DataFrame(
            result.fetchall(),
            columns=result.keys()
        )
    return roster

def predict_team_player_stats(team_id, opponent_team_id,
                               is_home, is_playoff,
                               team_off_rating, opp_def_rating,
                               opp_pace, team_pace,
                               game_date=None):
    """
    Predict stat lines for all rotation players on a team.
    Returns player predictions and team totals.
    """
    if game_date is None:
        game_date = date.today()

    models  = load_player_models()
    roster  = get_team_roster(team_id, game_date)
    playoff_minutes_map = {}
    if is_playoff:
        playoff_minutes_map = get_playoff_series_minutes(team_id)


    if roster.empty:
        return [], 0, 0, 0

    # Add game context features
    roster['is_home']          = int(is_home)
    roster['is_playoff']       = int(is_playoff)
    roster['opp_def_rating']   = float(opp_def_rating)
    roster['opp_pace']         = float(opp_pace)
    roster['team_off_rating']  = float(team_off_rating)
    roster['pace_factor']      = float(opp_pace) / 98.0
    roster['off_environment']  = (
        float(team_off_rating) - float(opp_def_rating)
    )
    roster['usage_rate_clean'] = roster[
        'player_avg_usage_l10'
    ].fillna(0.20)
    roster['points_momentum']  = (
        roster['player_avg_points_l5'] -
        roster['player_avg_points_l10']
    ).fillna(0)
    roster['usage_momentum']   = (
        roster['player_avg_usage_l5'] -
        roster['player_avg_usage_l10']
    ).fillna(0)
    roster['ts_momentum']      = (
        roster['player_avg_ts_l5'] -
        roster['player_avg_ts_l10']
    ).fillna(0)
    roster['game_date']        = pd.Timestamp(game_date)
    roster                     = roster.fillna(0)

    X = roster[PLAYER_FEATURE_COLS].values

    all_players = []
    for i, (_, player) in enumerate(roster.iterrows()):
        usage   = float(player['player_avg_usage_l10'])
        minutes = float(player['player_avg_min_l10'])

    # Override with actual playoff minutes if available
        player_id_int = int(player['player_id'])
        if player_id_int in playoff_minutes_map:
            playoff_min = playoff_minutes_map[player_id_int]
            if playoff_min >= 1:
                minutes = float(playoff_min)

        # Skip very low usage or low minute players
        if minutes < 15:
            print(f"  FILTERED OUT: {player['full_name']} - {minutes:.1f} min")
            continue

        features  = X[i].reshape(1, -1)
        pred_pts  = max(0, float(
            models['points'].predict(features)[0]
        ))
        pred_reb  = max(0, float(
            models['rebounds'].predict(features)[0]
        ))
        pred_ast  = max(0, float(
            models['assists'].predict(features)[0]
        ))

        # Injury return flag
        recent_games = int(player.get('recent_games', 20))
        returning_from_injury = recent_games < 7

        all_players.append({
            'player_id':              int(player['player_id']),
            'full_name':              player['full_name'],
            'position':               player.get('position', ''),
            'pred_points':            round(pred_pts, 1),
            'pred_rebounds':          round(pred_reb, 1),
            'pred_assists':           round(pred_ast, 1),
            'avg_minutes':            round(minutes, 1),
            'usage_rate':             round(usage, 3),
            'returning_from_injury':  returning_from_injury,
            'recent_games':           recent_games,
        })

    # Sort by usage rate — best proxy for rotation order
    all_players.sort(key=lambda x: x['avg_minutes'], reverse=True)

    if is_playoff:
        all_players = redistribute_playoff_minutes(all_players)

    if not all_players:
        return [], 0, 0, 0
    
    starters = all_players[:5]
    bench    = all_players[5:]

    team_points = round(
        sum(p['pred_points'] for p in starters) +
        sum(p['pred_points'] * (p['avg_minutes'] / 35.0) for p in bench)
    )
    team_rebounds = round(
        sum(p['pred_rebounds'] for p in starters) +
        sum(p['pred_rebounds'] * (p['avg_minutes'] / 35.0) for p in bench)
    )
    team_assists = round(
        sum(p['pred_assists'] for p in starters) +
        sum(p['pred_assists'] * (p['avg_minutes'] / 35.0) for p in bench)
    )

    all_players.sort(key=lambda x: x['avg_minutes'], reverse=True)
    # Return top 8 for display
    predictions = all_players[:8]

    return predictions, team_points, team_rebounds, team_assists

def redistribute_playoff_minutes(all_players):
    if not all_players or len(all_players) < 2:
        return all_players

    MAX_MINUTES = 40
    sorted_players = sorted(
        all_players, key=lambda x: x['avg_minutes'], reverse=True
    )

    starters = sorted_players[:5]
    bench    = sorted_players[5:]

    # Boost starters by 15% regardless of injury status
    # RTI players who are playing full minutes shouldn't be penalized
    for p in starters:
        original = p['avg_minutes']
        boosted  = min(original * 1.15, MAX_MINUTES)
        ratio    = boosted / original if original > 0 else 1.0
        p['pred_points']   = round(p['pred_points']   * ratio, 1)
        p['pred_rebounds'] = round(p['pred_rebounds'] * ratio, 1)
        p['pred_assists']  = round(p['pred_assists']  * ratio, 1)
        p['avg_minutes']   = round(boosted, 1)

    for p in bench:
        original = p['avg_minutes']
        reduced  = original * 0.90
        ratio    = reduced / original if original > 0 else 1.0
        p['pred_points']   = round(p['pred_points']   * ratio, 1)
        p['pred_rebounds'] = round(p['pred_rebounds'] * ratio, 1)
        p['pred_assists']  = round(p['pred_assists']  * ratio, 1)
        p['avg_minutes']   = round(reduced, 1)

    return sorted_players

if __name__ == "__main__":
    from model.predict import get_todays_games

    games = get_todays_games()

    if not games:
        print("No games today")
    else:
        for game in games:
            print(f"\n{game['away_team']} @ {game['home_team']}")
            print("=" * 50)

            home_preds, home_pts, home_reb, home_ast = \
                predict_team_player_stats(
                    team_id          = game['home_team_id'],
                    opponent_team_id = game['away_team_id'],
                    is_home          = True,
                    is_playoff       = game['season_type'] == 'Playoffs',
                    team_off_rating  = 115.0,
                    opp_def_rating   = 112.0,
                    opp_pace         = 98.0,
                    team_pace        = 98.0
                )

            away_preds, away_pts, away_reb, away_ast = \
                predict_team_player_stats(
                    team_id          = game['away_team_id'],
                    opponent_team_id = game['home_team_id'],
                    is_home          = False,
                    is_playoff       = game['season_type'] == 'Playoffs',
                    team_off_rating  = 113.0,
                    opp_def_rating   = 114.0,
                    opp_pace         = 98.0,
                    team_pace        = 98.0
                )

            print(f"\n{game['home_team']} Rotation:")
            print(f"  {'Player':<24} {'Pts':>5} "
                  f"{'Reb':>5} {'Ast':>5} {'Min':>5}")
            print("  " + "-" * 46)
            for p in home_preds[:8]:
                rti = " ⚠️RTI" if p['returning_from_injury'] else ""
                print(f"  {p['full_name']:<24} "
                      f"{p['pred_points']:>5.1f} "
                      f"{p['pred_rebounds']:>5.1f} "
                      f"{p['pred_assists']:>5.1f} "
                      f"{p['avg_minutes']:>5.1f}"
                      f"{rti}")
            print(f"  {'Team Total':<24} "
                  f"{home_pts:>5} "
                  f"{home_reb:>5} "
                  f"{home_ast:>5}")

            print(f"\n{game['away_team']} Rotation:")
            print(f"  {'Player':<24} {'Pts':>5} "
                  f"{'Reb':>5} {'Ast':>5} {'Min':>5}")
            print("  " + "-" * 46)
            for p in away_preds[:8]:
                rti = " ⚠️RTI" if p['returning_from_injury'] else ""
                print(f"  {p['full_name']:<24} "
                      f"{p['pred_points']:>5.1f} "
                      f"{p['pred_rebounds']:>5.1f} "
                      f"{p['pred_assists']:>5.1f} "
                      f"{p['avg_minutes']:>5.1f}"
                      f"{rti}")
            print(f"  {'Team Total':<24} "
                  f"{away_pts:>5} "
                  f"{away_reb:>5} "
                  f"{away_ast:>5}")