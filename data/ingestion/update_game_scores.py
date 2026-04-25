import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
from datetime import date, timedelta
from nba_api.stats.endpoints import scoreboardv2
from sqlalchemy.orm import Session
from data.storage.db import engine
from data.storage.models import Game
from config.settings import NBA_API_DELAY


def update_recent_game_scores(days_back=10):
    """
    Update scores for games in the past N days.
    Checks ScoreboardV2 for final scores and updates the database.
    """
    print(f"\nUpdating game scores for past {days_back} days...")
    
    updated_count = 0
    
    for i in range(days_back):
        target_date = date.today() - timedelta(days=i)
        date_str = target_date.strftime("%m/%d/%Y")
        print(f"\n  Checking {date_str}...")
        
        try:
            board = scoreboardv2.ScoreboardV2(
                game_date=date_str,
                league_id="00"
            )
            games_df = board.get_data_frames()[0]
            line_score = board.get_data_frames()[1]  # LineScore has final scores
            
            if games_df.empty:
                print(f"    No games found")
                continue
            
            print(f"    Found {len(games_df)} games")
            
            with Session(engine) as session:
                for _, row in games_df.iterrows():
                    game_id = str(row['GAME_ID'])
                    game_status = row.get('GAME_STATUS_TEXT', '')
                    
                    # Find game in database
                    db_game = session.query(Game).filter_by(
                        nba_game_id=game_id
                    ).first()
                    
                    if not db_game:
                        print(f"    ⚠️  Game {game_id} not in database")
                        continue
                    
                    # Check if game is final
                    is_final = 'Final' in game_status
                    
                    if is_final:
                        # Get scores from LineScore dataframe
                        home_line = line_score[
                            (line_score['GAME_ID'] == game_id) & 
                            (line_score['TEAM_ID'] == row['HOME_TEAM_ID'])
                        ]
                        away_line = line_score[
                            (line_score['GAME_ID'] == game_id) & 
                            (line_score['TEAM_ID'] == row['VISITOR_TEAM_ID'])
                        ]
                        
                        if not home_line.empty and not away_line.empty:
                            home_pts = int(home_line.iloc[0]['PTS'])
                            away_pts = int(away_line.iloc[0]['PTS'])
                            
                            # Update if different from stored values
                            if (db_game.home_score != home_pts or 
                                db_game.away_score != away_pts or 
                                not db_game.is_final):
                                
                                db_game.home_score = home_pts
                                db_game.away_score = away_pts
                                db_game.is_final = True
                                db_game.status = 'final'
                                
                                print(f"    ✅ Updated {game_id}: "
                                      f"{away_pts} - {home_pts} (Final)")
                                updated_count += 1
                            else:
                                print(f"    ✓ {game_id} already up to date")
                        else:
                            print(f"    ⚠️  Could not find scores for {game_id}")
                    else:
                        print(f"    ⏳ {game_id} status: {game_status}")
                
                session.commit()
            
            time.sleep(NBA_API_DELAY)
            
        except Exception as e:
            print(f"    ❌ Error for {date_str}: {e}")
            continue
    
    print(f"\n✅ Updated {updated_count} games")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=10,
                        help='Number of days back to check')
    args = parser.parse_args()
    
    update_recent_game_scores(days_back=args.days)