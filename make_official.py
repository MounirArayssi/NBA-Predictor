"""
Mark latest predictions as official.

Usage:
    python make_official.py
    python make_official.py --date 2026-04-28
    python make_official.py --all-recent
"""

import sys
from pathlib import Path
from datetime import date, timedelta

from dotenv import load_dotenv
from sqlalchemy import text

# Add project root to path
project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "twitter"
    else Path(__file__).parent
)
sys.path.insert(0, str(project_root))

load_dotenv()


def get_db_engine():
    """
    Use the shared project database engine.

    This is important because data.storage.db already knows how to use:
    - Streamlit secrets
    - DATABASE_URL environment variable
    - local config fallback
    """
    from data.storage.db import engine, DATABASE_URL

    print("Database target:")
    print(f"  Uses Neon: {'neon.tech' in str(DATABASE_URL)}")
    print(f"  URL prefix: {str(DATABASE_URL)[:35]}...")

    return engine


def mark_latest_as_official(target_date: str = None):
    """
    Mark the latest prediction for each game on a given game_date as official.

    Keeps all older predictions in the database as snapshots.
    Does NOT delete rows.
    Does NOT clear evaluation data.
    """

    if target_date is None:
        target_date = date.today().isoformat()

    engine = get_db_engine()

    try:
        with engine.begin() as conn:
            # Step 1: Mark all predictions for this game date as non-official snapshots.
            clear_query = text("""
                UPDATE predictions p
                SET
                    is_official = false,
                    prediction_type = 'snapshot',
                    lock_reason = 'older_snapshot'
                FROM games g
                WHERE p.game_id = g.game_id
                  AND DATE(g.game_date) = :target_date
            """)

            result = conn.execute(clear_query, {"target_date": target_date})
            cleared_count = result.rowcount

            print(
                f"✓ Marked {cleared_count} prediction(s) as snapshots "
                f"for {target_date}"
            )

            # Step 2: Mark the latest prediction per game as official.
            #
            # DISTINCT ON keeps one row per game.
            # ORDER BY predicted_at DESC, prediction_id DESC chooses the latest row.
            mark_query = text("""
                UPDATE predictions p
                SET
                    is_official = true,
                    prediction_type = 'official',
                    lock_reason = 'latest_selected'
                WHERE prediction_id IN (
                    SELECT DISTINCT ON (p2.game_id)
                        p2.prediction_id
                    FROM predictions p2
                    JOIN games g
                        ON p2.game_id = g.game_id
                    WHERE DATE(g.game_date) = :target_date
                    ORDER BY
                        p2.game_id,
                        p2.predicted_at DESC,
                        p2.prediction_id DESC
                )
            """)

            result = conn.execute(mark_query, {"target_date": target_date})
            marked_count = result.rowcount

            print(f"✓ Marked {marked_count} latest prediction(s) as official")

            # Step 3: Verify official rows.
            verify_query = text("""
                SELECT
                    g.game_date,
                    ht.abbreviation AS home,
                    at.abbreviation AS away,
                    p.prediction_id,
                    p.home_score_predicted,
                    p.away_score_predicted,
                    wt.abbreviation AS winner,
                    p.predicted_at,
                    p.model_version,
                    p.is_official
                FROM predictions p
                JOIN games g
                    ON p.game_id = g.game_id
                JOIN teams ht
                    ON g.home_team_id = ht.team_id
                JOIN teams at
                    ON g.away_team_id = at.team_id
                LEFT JOIN teams wt
                    ON p.predicted_winner_id = wt.team_id
                WHERE DATE(g.game_date) = :target_date
                  AND p.is_official = true
                ORDER BY
                    g.game_date,
                    ht.abbreviation,
                    at.abbreviation
            """)

            rows = conn.execute(
                verify_query,
                {"target_date": target_date}
            ).fetchall()

            if rows:
                print(f"\n{'=' * 72}")
                print(f"Official predictions for {target_date}:")
                print(f"{'=' * 72}")

                for row in rows:
                    game_date = row[0]
                    home = row[1]
                    away = row[2]
                    prediction_id = row[3]
                    home_score = row[4]
                    away_score = row[5]
                    winner = row[6]
                    predicted_at = row[7]
                    model_version = row[8]

                    print(
                        f"  prediction_id={prediction_id} | "
                        f"{away} @ {home} | "
                        f"{home} {home_score:.0f} - {away} {away_score:.0f} | "
                        f"winner={winner} | model={model_version}"
                    )
                    print(f"    Game date: {game_date} | Predicted at: {predicted_at}")

                print(f"{'=' * 72}")
            else:
                print(f"\n⚠️  No predictions found for {target_date}")
                print("   Run 'python model/predict.py' first")

        return marked_count

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


def verify_one_official_per_game(start_date: str = None, end_date: str = None):
    """
    Show whether every game has exactly one official prediction.
    """

    engine = get_db_engine()

    if start_date is None:
        start_date = (date.today() - timedelta(days=7)).isoformat()

    if end_date is None:
        end_date = date.today().isoformat()

    query = text("""
        SELECT
            p.game_id,
            g.game_date,
            ht.abbreviation AS home_team,
            at.abbreviation AS away_team,
            COUNT(*) AS total_predictions,
            COUNT(*) FILTER (WHERE p.is_official = true) AS official_predictions,
            STRING_AGG(
                CASE WHEN p.is_official THEN p.prediction_id::text END,
                ', '
            ) AS official_prediction_id,
            MAX(
                CASE WHEN p.is_official THEN p.predicted_at END
            ) AS official_predicted_at
        FROM predictions p
        JOIN games g
            ON g.game_id = p.game_id
        JOIN teams ht
            ON ht.team_id = g.home_team_id
        JOIN teams at
            ON at.team_id = g.away_team_id
        WHERE DATE(g.game_date) BETWEEN :start_date AND :end_date
        GROUP BY
            p.game_id,
            g.game_date,
            ht.abbreviation,
            at.abbreviation
        ORDER BY
            g.game_date,
            ht.abbreviation,
            at.abbreviation
    """)

    with engine.connect() as conn:
        rows = conn.execute(query, {
            "start_date": start_date,
            "end_date": end_date,
        }).fetchall()

    print(f"\nOfficial prediction check: {start_date} to {end_date}")
    print("=" * 72)

    if not rows:
        print("No prediction rows found.")
        return

    for row in rows:
        game_id = row[0]
        game_date = row[1]
        home = row[2]
        away = row[3]
        total = row[4]
        official_count = row[5]
        official_id = row[6]
        official_at = row[7]

        status = "✅" if official_count == 1 else "⚠️"

        print(
            f"{status} game_id={game_id} | {game_date} | {away} @ {home} | "
            f"total={total} | official={official_count} | "
            f"official_id={official_id} | official_at={official_at}"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Mark latest predictions as official"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Game date in YYYY-MM-DD format. Default=today.",
    )
    parser.add_argument(
        "--all-recent",
        action="store_true",
        help="Mark official predictions for all games in last 7 days.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify one official prediction per game.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for --verify. Default=7 days ago.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for --verify. Default=today.",
    )

    args = parser.parse_args()

    if args.verify:
        verify_one_official_per_game(
            start_date=args.start_date,
            end_date=args.end_date,
        )
        return

    if args.all_recent:
        print("Marking official predictions for last 7 days...")
        total = 0

        for i in range(7):
            check_date = (date.today() - timedelta(days=i)).isoformat()
            print(f"\nMarking latest predictions as official for {check_date}...")
            count = mark_latest_as_official(check_date)
            total += count

        print(f"\n✅ Total: {total} prediction(s) marked as official")
        return

    target_date = args.date or date.today().isoformat()

    print(f"Marking latest predictions as official for {target_date}...")
    count = mark_latest_as_official(target_date)

    if count > 0:
        print(f"\n✅ Success! {count} prediction(s) are now official")
        print("\nVerify with:")
        print(f"  python make_official.py --verify --start-date {target_date} --end-date {target_date}")
    else:
        print("\n⚠️  No predictions were marked")


if __name__ == "__main__":
    main()