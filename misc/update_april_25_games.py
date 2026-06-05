import sys
import os
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

from data.storage.db import engine, DATABASE_URL
from data.storage.models import Game, Team

try:
    from nba_api.stats.endpoints import scoreboardv3
    SCOREBOARD_VERSION = "v3"
except ImportError:
    from nba_api.stats.endpoints import scoreboardv2
    SCOREBOARD_VERSION = "v2"


def safe_score(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(float(value))
    except Exception:
        return None


def fetch_scoreboard(game_date):
    """
    Returns normalized rows:
    gameId, gameStatus, gameStatusText, homeTeamId, awayTeamId,
    homeTeamTricode, awayTeamTricode, homeScore, awayScore
    """

    game_date_str = str(game_date)
    print(f"Using Scoreboard{SCOREBOARD_VERSION.upper()}")

    if SCOREBOARD_VERSION == "v3":
        board = scoreboardv3.ScoreboardV3(
            game_date=game_date_str,
            league_id="00",
            timeout=30,
        )
    else:
        board = scoreboardv2.ScoreboardV2(
            game_date=game_date_str,
            league_id="00",
            day_offset=0,
            timeout=30,
        )

    frames = board.get_data_frames()

    if not frames:
        return pd.DataFrame()

    print("Scoreboard frames:")
    for i, frame in enumerate(frames):
        print(f"  frame {i}: rows={len(frame)}, cols={list(frame.columns)[:10]}")

    if SCOREBOARD_VERSION == "v2":
        game_frame = None
        for frame in frames:
            cols = set(frame.columns)
            if {"GAME_ID", "GAME_STATUS_ID", "PTS_HOME", "PTS_AWAY"}.issubset(cols):
                game_frame = frame
                break

        if game_frame is None:
            return pd.DataFrame()

        rows = []
        for _, r in game_frame.iterrows():
            rows.append({
                "gameId": str(r.get("GAME_ID")).strip(),
                "gameStatus": r.get("GAME_STATUS_ID"),
                "gameStatusText": r.get("GAME_STATUS_TEXT"),
                "homeTeamId": r.get("HOME_TEAM_ID"),
                "awayTeamId": r.get("VISITOR_TEAM_ID"),
                "homeTeamTricode": r.get("HOME_TEAM_ABBREVIATION"),
                "awayTeamTricode": r.get("VISITOR_TEAM_ABBREVIATION"),
                "homeScore": safe_score(r.get("PTS_HOME")),
                "awayScore": safe_score(r.get("PTS_AWAY")),
            })

        return pd.DataFrame(rows)

    # ScoreboardV3:
    # frame 1 = game status/info
    # frame 2 = team rows with score, two rows per game
    game_info = None
    team_scores = None

    for frame in frames:
        cols = set(frame.columns)

        if {"gameId", "gameStatus", "gameStatusText"}.issubset(cols):
            game_info = frame.copy()

        if {"gameId", "teamId", "teamTricode", "score"}.issubset(cols):
            team_scores = frame.copy()

    if game_info is None or team_scores is None:
        print("⚠️ Could not find required ScoreboardV3 frames.")
        return pd.DataFrame()

    rows = []

    for _, game_row in game_info.iterrows():
        game_id = str(game_row.get("gameId")).strip()
        teams = team_scores[team_scores["gameId"].astype(str) == game_id].copy()

        if teams.empty or len(teams) < 2:
            print(f"⚠️ Missing team score rows for gameId={game_id}")
            continue

        # Determine home/away from gameCode.
        # Example often looks like: 20260427/MINden or similar.
        # But if we cannot parse it confidently, we will use local DB match later.
        game_code = str(game_row.get("gameCode", ""))
        home_row = None
        away_row = None

        # Use team order from V3 team score frame if no better signal:
        # Usually the two team rows are away then home in scoreboard output.
        # We still verify against local DB in match/update step.
        if len(teams) >= 2:
            away_row = teams.iloc[0]
            home_row = teams.iloc[1]

        rows.append({
            "gameId": game_id,
            "gameStatus": game_row.get("gameStatus"),
            "gameStatusText": game_row.get("gameStatusText"),
            "gameCode": game_code,

            "homeTeamId": int(home_row.get("teamId")) if home_row is not None else None,
            "awayTeamId": int(away_row.get("teamId")) if away_row is not None else None,
            "homeTeamTricode": home_row.get("teamTricode") if home_row is not None else None,
            "awayTeamTricode": away_row.get("teamTricode") if away_row is not None else None,
            "homeScore": safe_score(home_row.get("score")) if home_row is not None else None,
            "awayScore": safe_score(away_row.get("score")) if away_row is not None else None,
        })

    return pd.DataFrame(rows)


def is_final(row):
    try:
        status = row.get("gameStatus")
        if status is not None and int(status) == 3:
            return True
    except Exception:
        pass

    text = str(row.get("gameStatusText", "")).lower()
    return "final" in text


def match_local_game(session, row, game_date):
    nba_game_id = str(row.get("gameId") or "").strip()

    game = None

    # Match by NBA game id first.
    if nba_game_id:
        game = (
            session.query(Game)
            .filter(Game.nba_game_id == nba_game_id)
            .first()
        )

        if not game and nba_game_id.isdigit():
            try:
                game = (
                    session.query(Game)
                    .filter(Game.nba_game_id == int(nba_game_id))
                    .first()
                )
            except Exception:
                pass

    # Fallback by date + team ids.
    if not game:
        home_team_id = row.get("homeTeamId")
        away_team_id = row.get("awayTeamId")

        if home_team_id and away_team_id:
            game = (
                session.query(Game)
                .filter(
                    Game.game_date == game_date,
                    Game.home_team_id == int(home_team_id),
                    Game.away_team_id == int(away_team_id),
                )
                .first()
            )

    # Fallback by date + team abbreviations.
    if not game:
        home_abbr = row.get("homeTeamTricode")
        away_abbr = row.get("awayTeamTricode")

        if home_abbr and away_abbr:
            home = session.query(Team).filter_by(abbreviation=home_abbr).first()
            away = session.query(Team).filter_by(abbreviation=away_abbr).first()

            if home and away:
                game = (
                    session.query(Game)
                    .filter(
                        Game.game_date == game_date,
                        Game.home_team_id == home.team_id,
                        Game.away_team_id == away.team_id,
                    )
                    .first()
                )

    return game


def update_scores_for_date(game_date):
    print(f"\nFetching scores for {game_date}...")

    df = fetch_scoreboard(game_date)

    if df.empty:
        print(f"No NBA scoreboard data found for {game_date}")
        return 0

    updated = 0
    skipped = 0
    missing = 0

    with Session(engine) as session:
        for _, row in df.iterrows():
            game_id = row.get("gameId")
            home_score = safe_score(row.get("homeScore"))
            away_score = safe_score(row.get("awayScore"))

            if not is_final(row):
                skipped += 1
                print(f"⏳ Skipping non-final game {game_id}")
                continue

            if home_score is None or away_score is None:
                skipped += 1
                print(f"⚠️ Missing score for final game {game_id}")
                print(row.to_dict())
                continue

            game = match_local_game(session, row, game_date)

            if not game:
                missing += 1
                print(
                    f"❌ Could not match local game for gameId={game_id} | "
                    f"{row.get('awayTeamTricode')} @ {row.get('homeTeamTricode')} "
                    f"{away_score}-{home_score}"
                )
                continue

            # If the API team order was wrong, use local game team abbreviations
            # to verify and swap if needed.
            home = session.query(Team).filter_by(team_id=game.home_team_id).first()
            away = session.query(Team).filter_by(team_id=game.away_team_id).first()

            local_home_abbr = home.abbreviation if home else None
            local_away_abbr = away.abbreviation if away else None

            api_home_abbr = row.get("homeTeamTricode")
            api_away_abbr = row.get("awayTeamTricode")

            if (
                local_home_abbr
                and local_away_abbr
                and api_home_abbr
                and api_away_abbr
                and local_home_abbr == api_away_abbr
                and local_away_abbr == api_home_abbr
            ):
                home_score, away_score = away_score, home_score

            game.home_score = home_score
            game.away_score = away_score
            game.is_final = True

            print(
                f"✅ Updated game_id={game.game_id}: "
                f"{local_away_abbr} @ {local_home_abbr} | "
                f"{local_home_abbr} {home_score} — {local_away_abbr} {away_score}"
            )

            updated += 1

        session.commit()

    print(f"\nDate: {game_date}")
    print(f"Updated: {updated}")
    print(f"Skipped non-final/missing: {skipped}")
    print(f"Missing local matches: {missing}")

    return updated


def update_recent_scores(days_back=5):
    print("Database target:")
    print(f"  Uses Neon: {'neon.tech' in str(DATABASE_URL)}")
    print(f"  URL prefix: {str(DATABASE_URL)[:35]}...")

    total = 0
    today = date.today()

    for i in range(days_back + 1):
        game_date = today - timedelta(days=i)
        total += update_scores_for_date(game_date)

    print(f"\n✅ Total games updated: {total}")
    return total


def update_specific_date(game_date_str):
    target = date.fromisoformat(game_date_str)
    return update_scores_for_date(target)


if __name__ == "__main__":
    #update_recent_scores(days_back=5)

    # For one date only, comment the line above and uncomment:
    update_specific_date("2026-04-30")