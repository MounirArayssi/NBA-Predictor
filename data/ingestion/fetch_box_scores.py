import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import pandas as pd
from nba_api.stats.endpoints import (
    boxscoretraditionalv3,
    boxscoreadvancedv3,
    boxscorefourfactorsv3,
    boxscoresummaryv2,
    boxscoresummaryv3
)
from sqlalchemy.orm import Session
from data.storage.db import engine
from data.storage.models import Game, Team, Player, PlayerBoxScore, TeamBoxScore
from config.settings import NBA_API_DELAY
from data.ingestion.update_series import update_series_scores

def safe_int(val):
    try:
        return int(val) if pd.notna(val) else None
    except:
        return None


def safe_float(val):
    try:
        return float(val) if pd.notna(val) else None
    except:
        return None


def parse_minutes(min_str):
    try:
        if pd.isna(min_str):
            return None
        parts = str(min_str).split(':')
        return float(parts[0]) + float(parts[1]) / 60 \
               if len(parts) == 2 else float(parts[0])
    except:
        return None


def sync_game_scores(session, game):
    home_bs = session.query(TeamBoxScore).filter_by(
        game_id=game.game_id, is_home=True
    ).first()
    away_bs = session.query(TeamBoxScore).filter_by(
        game_id=game.game_id, is_home=False
    ).first()
    if home_bs and away_bs:
        game.home_score = home_bs.points
        game.away_score = away_bs.points
        session.commit()


def get_quarter_points(nba_game_id, team_id):
    """Get per-quarter points using BoxScoreSummaryV3."""
    try:
        summary = boxscoresummaryv3.BoxScoreSummaryV3(
            game_id=nba_game_id
        )
        dfs = summary.get_data_frames()

        # Print available dataframes to find line score
        line_score = None
        for i, df in enumerate(dfs):
            if 'PTS_QTR1' in df.columns or 'q1Points' in df.columns:
                line_score = df
                break

        if line_score is None or line_score.empty:
            return None, None, None, None

        # Try both column naming conventions
        team_col = 'TEAM_ID' if 'TEAM_ID' in line_score.columns \
                   else 'teamId'
        team_row = line_score[line_score[team_col] == team_id]

        if team_row.empty:
            return None, None, None, None

        row = team_row.iloc[0]
        q1 = safe_int(row.get('PTS_QTR1') or row.get('q1Points'))
        q2 = safe_int(row.get('PTS_QTR2') or row.get('q2Points'))
        q3 = safe_int(row.get('PTS_QTR3') or row.get('q3Points'))
        q4 = safe_int(row.get('PTS_QTR4') or row.get('q4Points'))
        return q1, q2, q3, q4

    except Exception as e:
        return None, None, None, None
    


def fetch_box_scores_for_game(session, game):
    try:
        # Traditional V3
        trad = boxscoretraditionalv3.BoxScoreTraditionalV3(
            game_id=game.nba_game_id
        )
        player_df = trad.get_data_frames()[0]
        team_df   = trad.get_data_frames()[2]
        time.sleep(NBA_API_DELAY)

        # Advanced V3
        adv = boxscoreadvancedv3.BoxScoreAdvancedV3(
            game_id=game.nba_game_id
        )
        player_adv_df = adv.get_data_frames()[0]
        team_adv_df   = adv.get_data_frames()[1]
        time.sleep(NBA_API_DELAY)

    except Exception as e:
        print(f"    ❌ API error for game {game.nba_game_id}: {e}")
        return False

    if player_df.empty or team_df.empty:
        print(f"    ⚠️  No data for game {game.nba_game_id}")
        return False

    # --- Player box scores ---
    for _, row in player_df.iterrows():
        if pd.isna(row.get('minutes')) or row.get('minutes') == '0:00':
            continue

        player = session.query(Player).filter_by(
            nba_player_id=int(row['personId'])
        ).first()
        team = session.query(Team).filter_by(
            nba_team_id=int(row['teamId'])
        ).first()

        if not player or not team:
            continue

        existing = session.query(PlayerBoxScore).filter_by(
            game_id=game.game_id,
            player_id=player.player_id
        ).first()

        if existing:
            continue

        adv_row = player_adv_df[
            player_adv_df['personId'] == row['personId']
        ]

        def padv(col):
            try:
                v = adv_row[col].iloc[0]
                return safe_float(v) if not adv_row.empty \
                       and pd.notna(v) else None
            except:
                return None

        oreb = safe_int(row.get('reboundsOffensive'))
        dreb = safe_int(row.get('reboundsDefensive'))
        fga  = safe_int(row.get('fieldGoalsAttempted'))
        fgm  = safe_int(row.get('fieldGoalsMade'))
        fg3a = safe_int(row.get('threePointersAttempted'))
        fta  = safe_int(row.get('freeThrowsAttempted'))
        pts  = safe_int(row.get('points'))

        # Effective FG% = (FGM + 0.5 * FG3M) / FGA
        fg3m   = safe_int(row.get('threePointersMade'))
        efg    = None
        if fga and fga > 0 and fgm is not None and fg3m is not None:
            efg = (fgm + 0.5 * fg3m) / fga

        pbs = PlayerBoxScore(
            game_id        = game.game_id,
            player_id      = player.player_id,
            team_id        = team.team_id,
            minutes_played = parse_minutes(row.get('minutes')),
            points         = pts,
            rebounds       = safe_int(row.get('reboundsTotal')),
            assists        = safe_int(row.get('assists')),
            steals         = safe_int(row.get('steals')),
            blocks         = safe_int(row.get('blocks')),
            turnovers      = safe_int(row.get('turnovers')),
            fouls          = safe_int(row.get('foulsPersonal')),
            fgm            = fgm,
            fga            = fga,
            fg_pct         = safe_float(row.get('fieldGoalsPercentage')),
            fg3m           = fg3m,
            fg3a           = fg3a,
            fg3_pct        = safe_float(row.get('threePointersPercentage')),
            ftm            = safe_int(row.get('freeThrowsMade')),
            fta            = fta,
            ft_pct         = safe_float(row.get('freeThrowsPercentage')),
            plus_minus     = safe_float(row.get('plusMinusPoints')),

            # New fields
            oreb           = oreb,
            dreb           = dreb,
            efg_pct        = efg,
            usage_rate     = padv('usagePercentage'),
            true_shooting  = padv('trueShootingPercentage'),
            oreb_pct       = padv('offensiveReboundPercentage'),
            dreb_pct       = padv('defensiveReboundPercentage'),
            ast_pct        = padv('assistPercentage'),
            tov_pct        = padv('turnoverPercentage'),
            blk_pct        = padv('blockPercentage'),
            stl_pct        = padv('stealPercentage'),
            off_rating     = padv('offensiveRating'),
            def_rating     = padv('defensiveRating'),
            net_rating     = padv('netRating'),
            pace           = padv('pace'),
        )
        session.add(pbs)

    # --- Team box scores ---
    for _, row in team_df.iterrows():
        team = session.query(Team).filter_by(
            nba_team_id=int(row['teamId'])
        ).first()

        if not team:
            continue

        existing = session.query(TeamBoxScore).filter_by(
            game_id=game.game_id,
            team_id=team.team_id
        ).first()

        if existing:
            continue

        adv_row = team_adv_df[
            team_adv_df['teamId'] == row['teamId']
        ]

        def adv_float(col):
            try:
                return float(adv_row[col].iloc[0]) \
                       if not adv_row.empty \
                       and pd.notna(adv_row[col].iloc[0]) \
                       else None
            except:
                return None

        is_home = (team.team_id == game.home_team_id)

        # Quarter points
        q1, q2, q3, q4 = get_quarter_points(game.nba_game_id, int(row['teamId']))

        tbs = TeamBoxScore(
            game_id          = game.game_id,
            team_id          = team.team_id,
            is_home          = is_home,
            points           = safe_int(row.get('points')),
            rebounds         = safe_int(row.get('reboundsTotal')),
            assists          = safe_int(row.get('assists')),
            steals           = safe_int(row.get('steals')),
            blocks           = safe_int(row.get('blocks')),
            turnovers        = safe_int(row.get('turnovers')),
            fgm              = safe_int(row.get('fieldGoalsMade')),
            fga              = safe_int(row.get('fieldGoalsAttempted')),
            fg_pct           = safe_float(row.get('fieldGoalsPercentage')),
            fg3m             = safe_int(row.get('threePointersMade')),
            fg3a             = safe_int(row.get('threePointersAttempted')),
            fg3_pct          = safe_float(
                row.get('threePointersPercentage')
            ),
            ftm              = safe_int(row.get('freeThrowsMade')),
            fta              = safe_int(row.get('freeThrowsAttempted')),
            ft_pct           = safe_float(row.get('freeThrowsPercentage')),
            offensive_rating = adv_float('offensiveRating'),
            defensive_rating = adv_float('defensiveRating'),
            net_rating       = adv_float('netRating'),
            pace             = adv_float('pace'),
            true_shooting    = adv_float('trueShootingPercentage'),
            efg_pct          = adv_float('effectiveFieldGoalPercentage'),
            tov_pct          = adv_float('turnoverRatio'),
            oreb_pct         = adv_float('offensiveReboundPercentage'),
            ft_rate          = adv_float('assistRatio'),
            q1_points        = q1,
            q2_points        = q2,
            q3_points        = q3,
            q4_points        = q4,
        )
        session.add(tbs)

    sync_game_scores(session, game)
    session.commit()
    return True


def fetch_all_box_scores():
    with Session(engine) as session:
        games_with_scores = session.query(
            PlayerBoxScore.game_id
        ).distinct()

        games = session.query(Game).filter(
            Game.is_final == True,
            Game.game_id.notin_(games_with_scores)
        ).order_by(Game.game_date).all()

        total = len(games)
        print(f"Found {total} games needing box scores")

        for i, game in enumerate(games):
            home = session.query(Team).filter_by(
                team_id=game.home_team_id
            ).first()
            away = session.query(Team).filter_by(
                team_id=game.away_team_id
            ).first()

            print(f"  [{i+1}/{total}] {away.abbreviation} @ "
                  f"{home.abbreviation} on {game.game_date}...")

            success = fetch_box_scores_for_game(session, game)
            print(f"    {'✅ Done' if success else '⚠️  Skipped'}")

    print("✅ All box scores fetched")

def backfill_advanced_stats():
    """
    Refetch box scores for games that are missing 
    the new advanced stat columns.
    """
    with Session(engine) as session:
        # Find games where oreb is NULL (new columns not yet populated)
        games_missing_advanced = session.query(
            PlayerBoxScore.game_id
        ).filter(
            PlayerBoxScore.oreb == None
        ).distinct().all()

        game_ids = [g.game_id for g in games_missing_advanced]

        if not game_ids:
            print("No games missing advanced stats")
            return

        games = session.query(Game).filter(
            Game.game_id.in_(game_ids),
            Game.is_final == True
        ).order_by(Game.game_date).all()

        print(f"Backfilling advanced stats for {len(games)} games...")

        for i, game in enumerate(games):
            home = session.query(Team).filter_by(
                team_id=game.home_team_id
            ).first()
            away = session.query(Team).filter_by(
                team_id=game.away_team_id
            ).first()

            print(f"  [{i+1}/{len(games)}] "
                  f"{away.abbreviation} @ {home.abbreviation} "
                  f"{game.game_date}...")

            try:
                # Delete existing box scores and refetch
                session.query(PlayerBoxScore).filter_by(
                    game_id=game.game_id
                ).delete()
                session.query(TeamBoxScore).filter_by(
                    game_id=game.game_id
                ).delete()
                session.commit()

                success = fetch_box_scores_for_game(session, game)
                print(f"    {'✅' if success else '⚠️'}")

            except Exception as e:
                print(f"    ❌ {e}")
                session.rollback()

    print("✅ Backfill complete")

def backfill_recent_advanced_stats(days=80):
    """Backfill only recent games — faster than full backfill."""
    from datetime import date, timedelta
    cutoff = date.today() - timedelta(days=days)

    with Session(engine) as session:
        games = session.query(Game).filter(
            Game.is_final == True,
            Game.game_date >= cutoff
        ).order_by(Game.game_date).all()

        print(f"Backfilling {len(games)} games from last {days} days...")

        for i, game in enumerate(games):
            home = session.query(Team).filter_by(
                team_id=game.home_team_id
            ).first()
            away = session.query(Team).filter_by(
                team_id=game.away_team_id
            ).first()

            print(f"  [{i+1}/{len(games)}] "
                  f"{away.abbreviation} @ {home.abbreviation} "
                  f"{game.game_date}...")

            try:
                session.query(PlayerBoxScore).filter_by(
                    game_id=game.game_id
                ).delete()
                session.query(TeamBoxScore).filter_by(
                    game_id=game.game_id
                ).delete()
                session.commit()

                success = fetch_box_scores_for_game(session, game)
                print(f"    {'✅' if success else '⚠️'}")

            except Exception as e:
                print(f"    ❌ {e}")
                session.rollback()

    print("✅ Recent backfill complete")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backfill',        action='store_true')
    parser.add_argument('--backfill-recent', action='store_true')
    parser.add_argument('--days', type=int,  default=60)
    args = parser.parse_args()

    if args.backfill:
        backfill_advanced_stats()
    elif args.backfill_recent:
        backfill_recent_advanced_stats(days=args.days)
    else:
        fetch_all_box_scores()

    print("\n" + "="*60)
    print("Updating playoff series scores...")
    print("="*60)
    
    try:
        update_series_scores()
    except Exception as e:
        print(f"Warning: Series update failed but box scores were saved: {e}")