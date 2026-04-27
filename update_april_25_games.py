import sys
import os
import time
from datetime import date, timedelta

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

import pandas as pd
from sqlalchemy.orm import Session

from nba_api.stats.endpoints import scoreboardv2

from data.storage.db import engine
from data.storage.models import Game, Team


FINAL_GAME_STATUS_IDS = {3}
FINAL_STATUS_TEXT_KEYWORDS = ("final", "final/ot", "final/2ot", "final/3ot")


def normalize_game_date(value):
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def fetch_nba_scoreboard(game_date):
    """
    Fetch NBA scoreboard for a specific date using nba_api.

    Returns a pandas DataFrame from the GameHeader result set.
    """
    game_date_str = normalize_game_date(game_date)

    # NBA stats endpoints can occasionally be sensitive to rapid requests.
    # Keep timeout generous and retry once.
    last_error = None

    for attempt in range(2):
        try:
            board = scoreboardv2.ScoreboardV2(
                game_date=game_date_str,
                league_id="00",
                day_offset=0,
                timeout=30,
            )

            frames = board.get_data_frames()

            if not frames:
                return pd.DataFrame()

            # ScoreboardV2 normally returns GameHeader as the first frame.
            game_header = frames[0]
            return game_header

        except Exception as e:
            last_error = e
            time.sleep(2)

    print(f"⚠️ Failed to fetch scoreboard for {game_date_str}: {last_error}")
    return pd.DataFrame()


def is_final_game(row):
    """
    NBA scoreboard rows usually include GAME_STATUS_ID:
    1 = scheduled, 2 = live/in progress, 3 = final.
    """
    status_id = row.get("GAME_STATUS_ID")

    try:
        if int(status_id) in FINAL_GAME_STATUS_IDS:
            return True
    except Exception:
        pass

    status_text = str(row.get("GAME_STATUS_TEXT", "")).lower().strip()
    return any(keyword in status_text for keyword in FINAL_STATUS_TEXT_KEYWORDS)


def get_score(row, col):
    value = row.get(col)
    if pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        return None


def update_final_scores_for_date(game_date=None):
    """
    Update local games table with final scores from NBA scoreboard.

    Matches in this order:
    1. nba_game_id from your local Game table to NBA GAME_ID
    2. fallback by local date + home/away team_id
    """

    if game_date is None:
        game_date = date.today() - timedelta(days=1)

    game_date_str = normalize_game_date(game_date)

    print(f"Fetching NBA scoreboard for {game_date_str}...")

    scoreboard_df = fetch_nba_scoreboard(game_date_str)

    if scoreboard_df.empty:
        print(f"No scoreboard rows returned for {game_date_str}.")
        return {
            "date": game_date_str,
            "updated": 0,
            "skipped": 0,
            "not_found": 0,
        }

    updated_count = 0
    skipped_count = 0
    not_found_count = 0

    with Session(engine) as session:
        for _, row in scoreboard_df.iterrows():
            nba_game_id = str(row.get("GAME_ID", "")).strip()

            home_team_id = row.get("HOME_TEAM_ID")
            away_team_id = row.get("VISITOR_TEAM_ID")

            home_score = get_score(row, "PTS_HOME")
            away_score = get_score(row, "PTS_AWAY")

            final = is_final_game(row)

            if not final:
                skipped_count += 1
                print(f"⏳ Skipping non-final game NBA_GAME_ID={nba_game_id}")
                continue

            if home_score is None or away_score is None:
                skipped_count += 1
                print(f"⚠️ Skipping final game with missing score NBA_GAME_ID={nba_game_id}")
                continue

            game = None

            # Best match: NBA official game id.
            if nba_game_id:
                game = session.query(Game).filter(
                    Game.nba_game_id == nba_game_id
                ).first()

                # Sometimes local DB stores nba_game_id as int-like or without same type.
                if not game:
                    game = session.query(Game).filter(
                        Game.nba_game_id == int(nba_game_id)
                    ).first() if nba_game_id.isdigit() else None

            # Fallback match: game date + NBA team IDs.
            if not game and home_team_id and away_team_id:
                game = session.query(Game).filter(
                    Game.game_date == game_date_str,
                    Game.home_team_id == int(home_team_id),
                    Game.away_team_id == int(away_team_id),
                ).first()

            if not game:
                not_found_count += 1
                print(
                    f"❌ Local game not found for NBA_GAME_ID={nba_game_id} "
                    f"home_team_id={home_team_id} away_team_id={away_team_id}"
                )
                continue

            game.home_score = home_score
            game.away_score = away_score
            game.is_final = True

            updated_count += 1

            home = session.query(Team).filter_by(team_id=game.home_team_id).first()
            away = session.query(Team).filter_by(team_id=game.away_team_id).first()

            print(
                f"✓ Updated game_id={game.game_id}: "
                f"{home.abbreviation if home else game.home_team_id} {home_score} - "
                f"{away.abbreviation if away else game.away_team_id} {away_score}"
            )

        session.commit()

    print(f"\n✅ Updated {updated_count} final games")
    print(f"⏭️ Skipped {skipped_count} non-final/missing-score games")
    print(f"❌ Not found locally: {not_found_count}")

    return {
        "date": game_date_str,
        "updated": updated_count,
        "skipped": skipped_count,
        "not_found": not_found_count,
    }


if __name__ == "__main__":
    update_final_scores_for_date("2026-04-24")