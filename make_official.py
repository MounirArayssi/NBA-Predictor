"""
Mark latest predictions as official.
Run this after generating predictions with predict.py

Usage:
    python make_official.py
    python make_official.py --date 2026-04-27
"""

import os
import sys
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Add project root to path
project_root = Path(__file__).parent.parent if Path(__file__).parent.name == 'twitter' else Path(__file__).parent
sys.path.insert(0, str(project_root))

load_dotenv()


def get_db_engine():
    """Create database engine from environment variables."""
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'nba_predictor')
    
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(connection_string)


def mark_latest_as_official(target_date: str = None):
    """
    Mark the latest predictions for a given date as official.
    
    Steps:
    1. Clear is_official flag for all predictions on this date
    2. Set is_official = true for the most recent prediction per game
    """
    if target_date is None:
        target_date = date.today().isoformat()
    
    engine = get_db_engine()
    
    try:
        with engine.begin() as conn:
            # Step 1: Clear all is_official flags for this date
            clear_query = text("""
                UPDATE predictions p
                SET is_official = false
                FROM games g
                WHERE p.game_id = g.game_id
                  AND DATE(g.game_date) = :target_date
            """)
            
            result = conn.execute(clear_query, {"target_date": target_date})
            cleared_count = result.rowcount
            print(f"✓ Cleared is_official flag for {cleared_count} predictions on {target_date}")
            
            # Step 2: Mark the latest prediction per game as official
            mark_query = text("""
                UPDATE predictions p
                SET is_official = true
                WHERE prediction_id IN (
                    SELECT DISTINCT ON (p2.game_id) p2.prediction_id
                    FROM predictions p2
                    JOIN games g ON p2.game_id = g.game_id
                    WHERE DATE(g.game_date) = :target_date
                    ORDER BY p2.game_id, p2.predicted_at DESC
                )
            """)
            
            result = conn.execute(mark_query, {"target_date": target_date})
            marked_count = result.rowcount
            print(f"✓ Marked {marked_count} latest predictions as official")
            
            # Step 3: Show which games were marked
            verify_query = text("""
                SELECT 
                    g.game_date,
                    ht.abbreviation as home,
                    at.abbreviation as away,
                    p.home_score_predicted,
                    p.away_score_predicted,
                    wt.abbreviation as winner,
                    p.predicted_at
                FROM predictions p
                JOIN games g ON p.game_id = g.game_id
                JOIN teams ht ON g.home_team_id = ht.team_id
                JOIN teams at ON g.away_team_id = at.team_id
                LEFT JOIN teams wt ON p.predicted_winner_id = wt.team_id
                WHERE DATE(g.game_date) = :target_date
                  AND p.is_official = true
                ORDER BY g.game_date
            """)
            
            result = conn.execute(verify_query, {"target_date": target_date})
            rows = result.fetchall()
            
            if rows:
                print(f"\n{'='*60}")
                print(f"Official predictions for {target_date}:")
                print(f"{'='*60}")
                for row in rows:
                    print(f"  {row[1]} @ {row[2]}: {row[5]} wins ({row[2]} {row[3]:.0f} - {row[1]} {row[4]:.0f})")
                    print(f"    Predicted at: {row[6]}")
                print(f"{'='*60}")
            else:
                print(f"\n⚠️  No predictions found for {target_date}")
                print("   Run 'python model/predict.py' first")
        
        return marked_count
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Mark latest predictions as official')
    parser.add_argument('--date', type=str, help='Date (YYYY-MM-DD), default=today')
    parser.add_argument('--all-recent', action='store_true', help='Mark official predictions for all games in last 7 days')
    
    args = parser.parse_args()
    
    if args.all_recent:
        from datetime import timedelta
        print("Marking official predictions for last 7 days...")
        total = 0
        for i in range(7):
            check_date = (date.today() - timedelta(days=i)).isoformat()
            count = mark_latest_as_official(check_date)
            total += count
        print(f"\n✓ Total: {total} predictions marked as official")
    else:
        target_date = args.date or date.today().isoformat()
        print(f"Marking latest predictions as official for {target_date}...")
        count = mark_latest_as_official(target_date)
        
        if count > 0:
            print(f"\n✅ Success! {count} predictions are now official")
            print(f"\nNow you can post to Twitter:")
            print(f"  python twitter/post_from_db.py --dry-run")
        else:
            print(f"\n⚠️  No predictions were marked")


if __name__ == "__main__":
    main()