"""
Refetch player box scores for games where the players' points don't add up to
the team score. Those gaps come from fetch_box_scores skipping players who
weren't in the roster-based players table (now fixed). Only missing rows are
inserted, so it is safe to stop and rerun.

    python misc/backfill_missing_players.py [--limit N]
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.orm import Session
from data.storage.db import engine
from data.storage.models import Game
from data.ingestion.fetch_box_scores import fetch_box_scores_for_game

GAP_QUERY = text("""
    WITH team_pts AS (
        SELECT b.game_id, b.team_id, SUM(b.points) AS pts
        FROM player_box_scores b
        GROUP BY 1, 2
    )
    SELECT DISTINCT g.game_id, g.game_date
    FROM games g
    JOIN team_pts h ON h.game_id = g.game_id AND h.team_id = g.home_team_id
    JOIN team_pts a ON a.game_id = g.game_id AND a.team_id = g.away_team_id
    WHERE g.is_final = TRUE
    AND (h.pts <> g.home_score OR a.pts <> g.away_score)
    ORDER BY g.game_date
""")


def backfill(limit=None):
    with engine.connect() as conn:
        game_ids = [r[0] for r in conn.execute(GAP_QUERY)]
    if limit:
        game_ids = game_ids[:limit]
    print(f"{len(game_ids)} games with missing player rows", flush=True)

    with Session(engine) as session:
        for i, game_id in enumerate(game_ids):
            game = session.get(Game, game_id)
            try:
                ok = fetch_box_scores_for_game(session, game)
            except Exception as e:
                session.rollback()
                ok = False
                print(f"    error: {e}", flush=True)
            print(f"[{i+1}/{len(game_ids)}] {game.game_date} "
                  f"game {game_id} {'ok' if ok else 'FAILED'}", flush=True)


if __name__ == "__main__":
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
    backfill(limit)
