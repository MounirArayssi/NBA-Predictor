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

def get_player_weights(usage_rate):
    """Stars get more weight in team total, role players less."""
    player_w = 0.15 + (
        min(max(usage_rate - 0.10, 0), 0.20) / 0.20
    ) * 0.20
    return round(1 - player_w, 2), round(player_w, 2)

def get_playoff_performance_factor(player_id, team_id,
                                    reg_usage, reg_points,
                                    season='2025-26'):
    """
    Compare playoff efficiency to regular season efficiency.
    Uses points-per-usage as the metric so usage changes
    don't falsely trigger slump detection.
    """
    query = text("""
        SELECT
            pbs.points,
            pbs.usage_rate,
            pbs.true_shooting,
            pbs.plus_minus,
            pbs.minutes_played
        FROM player_box_scores pbs
        JOIN games g ON pbs.game_id = g.game_id
        WHERE pbs.player_id = :player_id
        AND pbs.team_id     = :team_id
        AND g.season_type   = 'Playoffs'
        AND g.season        = :season
        AND g.is_final      = TRUE
        AND pbs.minutes_played >= 10
        ORDER BY g.game_date DESC
        LIMIT 4
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {
            'player_id': int(player_id),
            'team_id':   int(team_id),
            'season':    season
        }).fetchall()

    if not result:
        return 1.0, False, False

    playoff_df = pd.DataFrame(result, columns=[
        'points', 'usage_rate', 'true_shooting',
        'plus_minus', 'minutes_played'
    ])

    playoff_pts   = float(playoff_df['points'].mean())
    playoff_usage = float(playoff_df['usage_rate'].mean()) \
                    if playoff_df['usage_rate'].notna().any() else reg_usage
    playoff_ts    = float(playoff_df['true_shooting'].mean()) \
                    if playoff_df['true_shooting'].notna().any() else 0.55

    # Points per usage — efficiency metric
    reg_ppu    = reg_points / max(reg_usage, 0.05)
    playoff_ppu = playoff_pts / max(playoff_usage, 0.05)

    # Efficiency drop — normalized by regular season baseline
    efficiency_diff = (playoff_ppu - reg_ppu) / max(reg_ppu, 1.0)

    # Minimum absolute threshold — ignore tiny changes
    # A player going 13pts/15% → 8pts/9% usage
    # reg_ppu = 13/0.15 = 86.7
    # playoff_ppu = 8/0.09 = 88.9 → barely changed, ignore
    # vs Ingram: 21pts/28% → 12pts/28% 
    # reg_ppu = 75, playoff_ppu = 42.8 → -43% efficiency drop

    slump_flag = False
    hot_flag   = False

    if efficiency_diff < -0.20:   # 20%+ efficiency drop
        penalty    = max(0.82, 1.0 + efficiency_diff * 0.4)
        slump_flag = True
        return penalty, slump_flag, hot_flag

    elif efficiency_diff > 0.20:  # 20%+ efficiency boost
        boost    = min(1.15, 1.0 + efficiency_diff * 0.3)
        hot_flag = True
        return boost, slump_flag, hot_flag

    return 1.0, False, False


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
            HAVING COUNT(*) >= 7
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
        AND prs.avg_points    >= 6
        AND prs.avg_usage_rate >= 0.08
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
                               game_date=None,
                               injury_report=None,
                               team_abbr=None,
                               player_props=None):
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

    # Remove confirmed Out/Doubtful players from roster
    if injury_report and team_abbr:
        out_players = {
            p['name'].lower()
            for p in injury_report.get(team_abbr, [])
            if p['status'] in ['Out', 'Doubtful']
        }
        if out_players:
            before = len(roster)
            roster = roster[
                ~roster['full_name'].str.lower().isin(out_players)
            ].reset_index(drop=True)
            removed = before - len(roster)
            if removed > 0:
                print(f"  🚫 Removed {removed} injured player(s) "
                      f"from {team_abbr} rotation")
    # After injury removal
    if is_playoff and len(roster) < 7:
        # Boost remaining players minutes by injury absence factor
        absent_factor = 1 + (0.15 * (7 - len(roster)))
        roster['player_avg_min_l10'] = roster[
            'player_avg_min_l10'
        ] * absent_factor
    # rest of the function stays exactly the same...
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

        player_id_int = int(player['player_id'])
        if player_id_int in playoff_minutes_map:
            playoff_min = playoff_minutes_map[player_id_int]
            if playoff_min >= 1:
                minutes = float(playoff_min)

        is_rotation = minutes >= 15

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

        slump_flag = False
        hot_flag   = False

        if is_playoff:
            factor, slump_flag, hot_flag = get_playoff_performance_factor(
                int(player['player_id']),
                team_id,
                reg_usage  = float(player.get('player_avg_usage_l10', 0.15)),
                reg_points = float(player.get('player_avg_points_l10', pred_pts))
            )
            
            if factor != 1.0:
                pred_pts     = pred_pts * factor
                pred_reb     = pred_reb * (1 + (factor - 1) * 0.4)
                pred_ast     = pred_ast * (1 + (factor - 1) * 0.4)
                playoff_flag = True
        else:
            # Regular season — use L5 vs L10 momentum
            l5_pts  = float(player.get('player_avg_points_l5', pred_pts))
            l10_pts = float(player.get('player_avg_points_l10', pred_pts))
            momentum = l5_pts - l10_pts

            if momentum < -4:
                pred_pts  *= 0.90
                pred_reb  *= 0.95
                pred_ast  *= 0.95
                slump_flag = True
            elif momentum > 4:
                pred_pts  *= 1.08
                pred_reb  *= 1.03
                pred_ast  *= 1.03
                hot_flag   = True


        # Only anchor top 4 players by usage
        top_4_names = {
        p['full_name'] 
        for p in sorted(all_players, 
                    key=lambda x: x['usage_rate'], 
                    reverse=True)[:4]
        }

        # Then in the loop:
        player_props_filtered = player_props if player['full_name'] in top_4_names else {}
        pred_pts, pred_reb, pred_ast = apply_vegas_props_anchor(
        pred_pts, pred_reb, pred_ast,   
        player['full_name'], player_props_filtered
        )
        recent_games          = int(player.get('recent_games', 20))
        returning_from_injury = recent_games < 7

        all_players.append({
            'player_id':             int(player['player_id']),
            'full_name':             player['full_name'],
            'position':              player.get('position', ''),
            'pred_points':           round(pred_pts, 1),
            'pred_rebounds':         round(pred_reb, 1),
            'pred_assists':          round(pred_ast, 1),
            'avg_minutes':           round(minutes, 1),
            'usage_rate':            round(usage, 3),
            'returning_from_injury': returning_from_injury,
            'recent_games':          recent_games,
            'is_rotation':           is_rotation,
            'slump_flag':            slump_flag,
            'hot_flag':              hot_flag,
            'playoff_flag': False,
        })

    all_players.sort(key=lambda x: x['avg_minutes'], reverse=True)

    if is_playoff:
        all_players = redistribute_playoff_minutes(all_players)

    if not all_players:
        return [], 0, 0, 0

    rotation = [p for p in all_players if p['is_rotation']]
    fringe   = [p for p in all_players if not p['is_rotation']]

    rotation.sort(key=lambda x: x['avg_minutes'], reverse=True)
    starters = sorted(rotation,
                  key=lambda x: x['avg_minutes'],
                  reverse=True)[:5]
    bench    = rotation[5:]

    team_points = round(
    sum(p['pred_points'] for p in starters) +
    sum(p['pred_points'] * 0.5 for p in bench) +
    sum(p['pred_points'] * 0.2 for p in fringe)
)
    team_rebounds = round(
    sum(p['pred_rebounds'] for p in starters) +
    sum(p['pred_rebounds'] * 0.5 for p in bench) +
    sum(p['pred_rebounds'] * 0.2 for p in fringe)
)
    team_assists = round(
    sum(p['pred_assists'] for p in starters) +
    sum(p['pred_assists'] * 0.5 for p in bench) +
    sum(p['pred_assists'] * 0.2 for p in fringe)
)

    all_players.sort(key=lambda x: x['avg_minutes'], reverse=True)
    predictions = all_players[:8]

    return predictions, team_points, team_rebounds, team_assists


def apply_vegas_props_anchor(pred_pts, pred_reb, pred_ast,
                              player_name, player_props,
                              anchor_weight=0.3):
    if not player_props:
        return pred_pts, pred_reb, pred_ast

    # Normalize name for matching
    def normalize(name):
        return name.lower().replace('.', '').replace('-', ' ').strip()

    player_norm = normalize(player_name)

    props = None
    for name, lines in player_props.items():
        if normalize(name) == player_norm:
            props = lines
            break

    # Fallback — last name + first initial match
    if not props:
        last  = player_name.split()[-1].lower()
        first = player_name[0].lower()
        for name, lines in player_props.items():
            parts = name.split()
            if (len(parts) >= 2 and
                    parts[-1].lower() == last and
                    parts[0][0].lower() == first):
                props = lines
                break

    if not props:
        return pred_pts, pred_reb, pred_ast

    def nudge(pred, line, weight, cap=1.5):
        if not line:
            return pred
        adj = max(-cap, min(cap, (line - pred) * weight))
        return round(pred + adj, 1)

    new_pts = nudge(pred_pts, props.get('points'), anchor_weight)
    new_reb = nudge(pred_reb, props.get('rebounds'), anchor_weight)
    new_ast = nudge(pred_ast, props.get('assists'), anchor_weight)

    if abs(new_pts - pred_pts) > 0.1:
        print(f"    📊 [{player_name}] "
              f"pts {pred_pts:.1f}→{new_pts:.1f} "
              f"(Vegas: {props.get('points','—')}) | "
              f"reb {pred_reb:.1f}→{new_reb:.1f} | "
              f"ast {pred_ast:.1f}→{new_ast:.1f}")

    return new_pts, new_reb, new_ast



def redistribute_playoff_minutes(all_players):
    if not all_players or len(all_players) < 2:
        return all_players

    MAX_MINUTES = 39
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