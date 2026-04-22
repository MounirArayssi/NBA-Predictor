import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv3, boxscoreadvancedv3
from sqlalchemy.orm import Session
from data.storage.db import engine
from data.storage.models import Game, Team, Player, PlayerBoxScore, TeamBoxScore
from config.settings import NBA_API_DELAY


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
    """Update game scores from box scores to ensure accuracy."""
    home_bs = session.query(TeamBoxScore).filter_by(
        game_id=game.game_id,
        is_home=True
    ).first()

    away_bs = session.query(TeamBoxScore).filter_by(
        game_id=game.game_id,
        is_home=False
    ).first()

    if home_bs and away_bs:
        game.home_score = home_bs.points
        game.away_score = away_bs.points
        session.commit()


def fetch_box_scores_for_game(session, game):
    try:
        # Traditional V3
        trad = boxscoretraditionalv3.BoxScoreTraditionalV3(
            game_id=game.nba_game_id
        )
        player_df = trad.get_data_frames()[0]  # player stats
        team_df   = trad.get_data_frames()[2]  # full team totals
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

    # Skip if no data returned
    if player_df.empty or team_df.empty:
        print(f"    ⚠️  No data for game {game.nba_game_id}")
        return False

    # --- Player box scores ---
    for _, row in player_df.iterrows():
        # Skip DNP players
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

        # Match advanced row
        adv_row = player_adv_df[
            player_adv_df['personId'] == row['personId']
        ]

        pbs = PlayerBoxScore(
            game_id        = game.game_id,
            player_id      = player.player_id,
            team_id        = team.team_id,
            minutes_played = parse_minutes(row.get('minutes')),
            points         = safe_int(row.get('points')),
            rebounds       = safe_int(row.get('reboundsTotal')),
            assists        = safe_int(row.get('assists')),
            steals         = safe_int(row.get('steals')),
            blocks         = safe_int(row.get('blocks')),
            turnovers      = safe_int(row.get('turnovers')),
            fouls          = safe_int(row.get('foulsPersonal')),
            fgm            = safe_int(row.get('fieldGoalsMade')),
            fga            = safe_int(row.get('fieldGoalsAttempted')),
            fg_pct         = safe_float(row.get('fieldGoalsPercentage')),
            fg3m           = safe_int(row.get('threePointersMade')),
            fg3a           = safe_int(row.get('threePointersAttempted')),
            fg3_pct        = safe_float(row.get('threePointersPercentage')),
            ftm            = safe_int(row.get('freeThrowsMade')),
            fta            = safe_int(row.get('freeThrowsAttempted')),
            ft_pct         = safe_float(row.get('freeThrowsPercentage')),
            plus_minus     = safe_float(row.get('plusMinusPoints')),
            usage_rate     = safe_float(
                adv_row['usagePercentage'].iloc[0]
            ) if not adv_row.empty else None,
            true_shooting  = safe_float(
                adv_row['trueShootingPercentage'].iloc[0]
            ) if not adv_row.empty else None,
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

        # Match advanced row
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
            fg3_pct          = safe_float(row.get('threePointersPercentage')),
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
        )
        session.add(tbs)

    # Sync game scores from box scores to ensure accuracy
    sync_game_scores(session, game)
    session.commit()
    return True


def fetch_all_box_scores():
    """Fetch box scores for all final games missing them."""
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


if __name__ == "__main__":
    fetch_all_box_scores()