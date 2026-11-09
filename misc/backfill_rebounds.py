"""
Fill player_box_scores.oreb / dreb where they are NULL (1,387 games in 2024-25 and
2025-26 were stored without them).

Only UPDATEs those two columns on existing rows; nothing is deleted or re-inserted.
A row is updated only when the API's offensive + defensive rebounds equal the
stored total rebounds. Resumable: each run picks up games that still have NULLs.

    python misc/backfill_rebounds.py            # all remaining games
    python misc/backfill_rebounds.py --limit 20 # a few, to test
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import pandas as pd
from sqlalchemy import text
from nba_api.stats.endpoints import boxscoretraditionalv3

from data.storage.db import engine
from config.settings import NBA_API_DELAY


def pending_games(limit=None):
    q = """
        SELECT DISTINCT g.game_id, g.nba_game_id, g.game_date
        FROM player_box_scores b JOIN games g USING (game_id)
        WHERE b.oreb IS NULL AND g.is_final = TRUE
        ORDER BY g.game_date, g.game_id
    """
    with engine.connect() as c:
        rows = c.execute(text(q)).fetchall()
    return rows[:limit] if limit else rows


def fetch(nba_game_id, tries=3):
    for i in range(tries):
        try:
            df = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=nba_game_id, timeout=60).get_data_frames()[0]
            return df
        except Exception as e:  # network / rate limit: back off and retry
            err = e
            time.sleep(5 * (i + 1))
    raise err


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    games = pending_games(limit)
    print(f"{len(games)} games with missing rebounds", flush=True)
    done = rows_ok = rows_bad = failed = 0
    t0 = time.time()
    for k, (gid, nba_id, gdate) in enumerate(games, 1):
        try:
            df = fetch(nba_id)
        except Exception as e:
            failed += 1
            print(f"  [{k}/{len(games)}] {gdate} {nba_id}: API error {e}", flush=True)
            continue
        df = df[["personId", "reboundsOffensive", "reboundsDefensive", "reboundsTotal"]].dropna()
        with engine.begin() as c:
            stored = pd.read_sql(text("""
                SELECT b.id, p.nba_player_id, b.rebounds FROM player_box_scores b
                JOIN players p USING (player_id)
                WHERE b.game_id = :g AND b.oreb IS NULL
            """), c, params={"g": gid})
            m = stored.merge(df, left_on="nba_player_id", right_on="personId", how="left")
            for r in m.itertuples(index=False):
                if pd.isna(r.reboundsOffensive):
                    continue
                o, d = int(r.reboundsOffensive), int(r.reboundsDefensive)
                if r.rebounds is not None and o + d != int(r.rebounds):
                    rows_bad += 1
                    continue
                c.execute(text("UPDATE player_box_scores SET oreb = :o, dreb = :d WHERE id = :i AND oreb IS NULL"),
                          {"o": o, "d": d, "i": int(r.id)})
                rows_ok += 1
        done += 1
        if k % 25 == 0 or k == len(games):
            el = time.time() - t0
            print(f"  [{k}/{len(games)}] {gdate}  rows filled {rows_ok}, mismatched {rows_bad}, "
                  f"failed games {failed}, {el / k:.1f}s/game, ~{(len(games) - k) * el / k / 60:.0f} min left", flush=True)
        time.sleep(NBA_API_DELAY)
    print(f"Done: {done} games, {rows_ok} rows filled, {rows_bad} skipped (totals didn't match), "
          f"{failed} games failed (re-run to retry)", flush=True)


if __name__ == "__main__":
    main()
