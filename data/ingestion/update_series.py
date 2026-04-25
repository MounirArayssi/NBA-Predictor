"""
Automatically update series wins after games are completed.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from sqlalchemy.orm import Session
from data.storage.db import engine


def update_series_scores():
    """
    Update series_home_wins and series_away_wins for playoff games.
    Shows the CURRENT series score entering each game.
    """
    session = Session(engine)
    try:
        query = text("""
            WITH series_games AS (
                SELECT 
                    game_id,
                    season,
                    home_team_id,
                    away_team_id,
                    series_id,
                    game_date,
                    home_score,
                    away_score,
                    CASE 
                        WHEN home_score > away_score THEN home_team_id
                        WHEN away_score > home_score THEN away_team_id
                        ELSE NULL
                    END as winner_team_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY series_id 
                        ORDER BY game_date, game_id
                    ) as game_number
                FROM games
                WHERE season_type = 'Playoffs'
                  AND series_id IS NOT NULL
            ),
            series_wins_before AS (
                SELECT 
                    curr.game_id,
                    curr.series_id,
                    curr.home_team_id,
                    curr.away_team_id,
                    curr.game_number,
                    COALESCE(SUM(CASE 
                        WHEN prev.winner_team_id = curr.home_team_id 
                        THEN 1 
                        ELSE 0 
                    END), 0) as home_wins_before,
                    COALESCE(SUM(CASE 
                        WHEN prev.winner_team_id = curr.away_team_id 
                        THEN 1 
                        ELSE 0 
                    END), 0) as away_wins_before
                FROM series_games curr
                LEFT JOIN series_games prev 
                    ON curr.series_id = prev.series_id 
                    AND prev.game_number < curr.game_number
                    AND prev.winner_team_id IS NOT NULL
                GROUP BY 
                    curr.game_id,
                    curr.series_id,
                    curr.home_team_id,
                    curr.away_team_id,
                    curr.game_number
            )
            UPDATE games g
            SET 
                series_home_wins = sw.home_wins_before,
                series_away_wins = sw.away_wins_before
            FROM series_wins_before sw
            WHERE g.game_id = sw.game_id;
        """)
        
        result = session.execute(query)
        session.commit()
        rows_updated = result.rowcount
        
        if rows_updated > 0:
            print(f"✓ Updated series scores for {rows_updated} playoff games")
        else:
            print("✓ Series scores already up to date")
        
        return rows_updated
        
    except Exception as e:
        session.rollback()
        print(f"✗ Error updating series scores: {e}")
        raise
    finally:
        session.close()


if __name__ == '__main__':
    update_series_scores()