

import random
import os
import sys
import time
from typing import Any, Dict, List
from pathlib import Path
from datetime import datetime, date
from dotenv import load_dotenv
import tweepy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# NBA Team Hashtags
TEAM_HASHTAGS = {
    # Eastern Conference
    "Celtics": "#DifferentHere",
    "Nets": "#NetsWorld",
    "Knicks": "#NewYorkForever",
    "76ers": "#HereTheyCome",
    "Raptors": "#WeTheNorth",
    "Bulls": "#SeeRed",
    "Cavaliers": "#LetEmKnow",
    "Pistons": "#DetroitBasketball",
    "Pacers": "#BoomBaby",
    "Bucks": "#FearTheDeer",
    "Hawks": "#TrueToAtlanta",
    "Hornets": "#LetsFly",
    "Heat": "#HEATCulture",
    "Magic": "#MagicTogether",
    "Wizards": "#DCAboveAll",
    # Western Conference
    "Nuggets": "#MileHighPlayoffs",
    "Timberwolves": "#RaisedByWolves",
    "Thunder": "#ThunderUp",
    "Trail Blazers": "#RipCity",
    "Jazz": "#TakeNote",
    "Warriors": "#DubNation",
    "Clippers": "#ClipperNation",
    "Lakers": "#LakeShow",
    "Suns": "#SunsUp",
    "Kings": "#BeamTeam",
    "Mavericks": "#MFFL",
    "Rockets": "#RunAsOne",
    "Grizzlies": "#GrindCity",
    "Pelicans": "#Pelicans",
    "Spurs": "#GoSpursGo",
}


def get_team_hashtag(team_name: str) -> str:
    """Get the team hashtag for a given team name."""
    if team_name in TEAM_HASHTAGS:
        return TEAM_HASHTAGS[team_name]
    
    for key, hashtag in TEAM_HASHTAGS.items():
        if key in team_name or team_name in key:
            return hashtag
    
    return ""


def get_db_engine():
    """Create database engine from environment variables."""
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'nba_predictor')
    
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(connection_string)


def get_twitter_client():
    """Create and return authenticated Twitter API client."""
    try:
        client = tweepy.Client(
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        )
        return client
    except Exception as e:
        print(f"ERROR: Error creating Twitter client: {e}")
        return None


def post_tweet(text: str) -> bool:
    """Post a tweet to Twitter."""
    try:
        client = get_twitter_client()
        if not client:
            return False
            
        response = client.create_tweet(text=text)
        print(f"SUCCESS: Tweet posted! ID: {response.data['id']}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def generate_tweet(prediction: Dict[str, Any]) -> str:
    """Generate a creative tweet from a game prediction dictionary."""
    home = prediction["home_team"]
    away = prediction["away_team"]
    h_pred = round(float(prediction["home_pred"]))
    a_pred = round(float(prediction["away_pred"]))
    winner = prediction["predicted_winner"]
    loser = away if winner == home else home
    margin = abs(float(prediction["margin"]))
    conf = str(prediction.get("confidence", "MEDIUM")).upper()
    is_playoff = bool(prediction.get("is_playoff", False))

    # Get team hashtags
    home_tag = get_team_hashtag(home)
    away_tag = get_team_hashtag(away)
    winner_tag = get_team_hashtag(winner)

    # Top scorer info
    home_star_name = prediction.get("home_top_scorer")
    home_star_pts = prediction.get("home_top_scorer_pts")
    away_star_name = prediction.get("away_top_scorer")
    away_star_pts = prediction.get("away_top_scorer_pts")
    
    winner_star_name = home_star_name if winner == home else away_star_name
    winner_star_pts = home_star_pts if winner == home else away_star_pts

    game_tag = "🏆" if is_playoff else "🏀"
    conf_line = {
        "HIGH": "i'm calling it early —",
        "MEDIUM": "the numbers say —",
        "LOW": "this one's a coin flip but —",
    }.get(conf, "model says —")

    # Player highlight
    star_line = ""
    if winner_star_name and winner_star_pts:
        star_line = f"\n{winner_star_name} goes {round(float(winner_star_pts))}pts"

    # Base hashtag
    base_hashtag = "#NBAPlayoffs" if is_playoff else "#NBA"
    
    # Collect all unique team hashtags
    hashtags = [base_hashtag]
    for tag in [home_tag, away_tag]:
        if tag and tag not in hashtags:
            hashtags.append(tag)
    hashtag_line = " ".join(hashtags)

    templates = [
        (
            f"{game_tag} {away} @ {home}\n"
            f"{conf_line} {winner} wins\n"
            f"predicted: {home} {h_pred} — {away} {a_pred}"
            f"{star_line}\n{hashtag_line}"
        ),
        (
            f"ran the model. {loser} fans look away 👀\n"
            f"{game_tag} {winner} {max(h_pred, a_pred)} — "
            f"{loser} {min(h_pred, a_pred)}"
            f"{star_line}\n{hashtag_line}"
        ),
        (
            f"{game_tag} {away} @ {home} prediction\n"
            f"{conf_line} {winner} by {margin:.0f}\n"
            f"final: {home} {h_pred} — {away} {a_pred}"
            f"{star_line}\n{hashtag_line}"
        ),
        (
            f"good morning to everyone except {loser} fans 🙂\n"
            f"{winner} wins tonight — {home} {h_pred} {away} {a_pred}"
            f"{star_line}\n{hashtag_line}"
        ),
        (
            f"{game_tag} {away} @ {home}\n"
            f"{winner} takes it — {winner_tag}\n"
            f"{home} {h_pred} — {away} {a_pred}"
            f"{star_line}\n{base_hashtag}"
        ),
    ]

    tweet = random.choice(templates)

    # Keep under 280 chars
    if len(tweet) > 275:
        if len(hashtag_line) > 20:
            tweet = tweet.replace(hashtag_line, base_hashtag)
        
        if len(tweet) > 275:
            tweet = tweet[:272] + "..."

    return tweet


def load_predictions_from_db(target_date: str = None) -> List[Dict[str, Any]]:
    """
    Load predictions from PostgreSQL database.
    
    Args:
        target_date: Date string (YYYY-MM-DD), defaults to today
        
    Returns:
        List of prediction dictionaries
    """
    if target_date is None:
        target_date = date.today().isoformat()
    
    try:
        engine = get_db_engine()
        
        # First, check teams table structure to find name column
        with engine.connect() as conn:
            check_query = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'teams'
                ORDER BY ordinal_position
            """)
            columns_result = conn.execute(check_query)
            team_columns = [row[0] for row in columns_result.fetchall()]
            print(f"DEBUG: Teams table columns: {team_columns}")
            
            # Determine which column has team names
            # Check for 'name', 'full_name', 'abbreviation', etc.
            team_name_col = None
            if 'name' in team_columns:
                team_name_col = 'name'
            elif 'full_name' in team_columns:
                team_name_col = 'full_name'
            elif 'abbreviation' in team_columns:
                team_name_col = 'abbreviation'
            else:
                # Use first text-like column
                team_name_col = team_columns[1] if len(team_columns) > 1 else 'team_id'
            
            print(f"DEBUG: Using '{team_name_col}' for team names")
        
        # Build query with correct column name
        query = text(f"""
            SELECT 
                g.game_date,
                ht.{team_name_col} as home_team,
                at.{team_name_col} as away_team,
                p.home_score_predicted as home_pred,
                p.away_score_predicted as away_pred,
                wt.{team_name_col} as predicted_winner,
                p.model_margin as margin,
                CASE 
                    WHEN p.confidence_score >= 0.70 THEN 'HIGH'
                    WHEN p.confidence_score >= 0.55 THEN 'MEDIUM'
                    ELSE 'LOW'
                END as confidence,
                g.season_type = 'Playoffs' as is_playoff,
                p.predicted_at,
                p.key_factors
            FROM predictions p
            JOIN games g ON p.game_id = g.game_id
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            LEFT JOIN teams wt ON p.predicted_winner_id = wt.team_id
            WHERE DATE(g.game_date) = :target_date
              AND p.is_official = true
            ORDER BY g.game_date, p.predicted_at DESC
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {"target_date": target_date})
            rows = result.fetchall()
        
        if not rows:
            print(f"INFO: No predictions found for {target_date}")
            
            # Try to get the most recent official predictions
            latest_query = text(f"""
                SELECT 
                    g.game_date,
                    ht.{team_name_col} as home_team,
                    at.{team_name_col} as away_team,
                    p.home_score_predicted as home_pred,
                    p.away_score_predicted as away_pred,
                    wt.{team_name_col} as predicted_winner,
                    p.model_margin as margin,
                    CASE 
                        WHEN p.confidence_score >= 0.70 THEN 'HIGH'
                        WHEN p.confidence_score >= 0.55 THEN 'MEDIUM'
                        ELSE 'LOW'
                    END as confidence,
                    g.season_type = 'Playoffs' as is_playoff,
                    p.predicted_at,
                    p.key_factors
                FROM predictions p
                JOIN games g ON p.game_id = g.game_id
                JOIN teams ht ON g.home_team_id = ht.team_id
                JOIN teams at ON g.away_team_id = at.team_id
                LEFT JOIN teams wt ON p.predicted_winner_id = wt.team_id
                WHERE p.is_official = true
                ORDER BY p.predicted_at DESC
                LIMIT 10
            """)
            
            with engine.connect() as conn:
                result = conn.execute(latest_query)
                rows = result.fetchall()
            
            if rows:
                latest_date = rows[0][0].date().isoformat() if rows[0][0] else None
                print(f"      Using latest predictions from {latest_date}")
        
        predictions = []
        for row in rows:
            # Extract top scorers from key_factors JSON if available
            home_star_name = None
            home_star_pts = None
            away_star_name = None
            away_star_pts = None
            
            if row[10]:  # key_factors column - already deserialized by SQLAlchemy
                key_factors = row[10]  # Already a dict or list, no need to parse
                
                # key_factors might be a dict or a list - handle both
                if isinstance(key_factors, dict):
                    home_star_name = key_factors.get('home_top_scorer')
                    home_star_pts = key_factors.get('home_top_scorer_pts')
                    away_star_name = key_factors.get('away_top_scorer')
                    away_star_pts = key_factors.get('away_top_scorer_pts')
                elif isinstance(key_factors, list):
                    # If it's a list of factor strings, try to extract player info
                    # This might need adjustment based on your actual format
                    for factor in key_factors:
                        if isinstance(factor, str):
                            # Look for player scoring predictions in factors
                            if 'home' in factor.lower() and 'pts' in factor.lower():
                                # Try to parse something like "Home: Player X 25pts"
                                pass  # Add parsing if needed
            
            predictions.append({
                'home_team': row[1],
                'away_team': row[2],
                'home_pred': float(row[3]),
                'away_pred': float(row[4]),
                'predicted_winner': row[5],
                'margin': float(row[6]) if row[6] else 0.0,
                'confidence': row[7] or 'MEDIUM',
                'is_playoff': bool(row[8]),
                'home_top_scorer': home_star_name,
                'home_top_scorer_pts': home_star_pts,
                'away_top_scorer': away_star_name,
                'away_top_scorer_pts': away_star_pts,
            })
        
        print(f"SUCCESS: Loaded {len(predictions)} predictions from database")
        return predictions
        
    except Exception as e:
        print(f"ERROR: Failed to load from database: {e}")
        import traceback
        traceback.print_exc()
        return []


def post_all_predictions(predictions: List[Dict[str, Any]], dry_run: bool = False, delay_seconds: int = 15) -> List[Dict[str, Any]]:
    """Post tweets for all predictions."""
    results = []
    total = len(predictions)
    
    print("\n" + "="*60)
    if dry_run:
        print(f"DRY RUN MODE - {total} tweet(s) to preview")
    else:
        print(f"LIVE POSTING MODE - {total} tweet(s) will be posted")
    print("="*60 + "\n")
    
    for i, prediction in enumerate(predictions, start=1):
        try:
            tweet_text = generate_tweet(prediction)
            
            print(f"\n--- Tweet {i}/{total} ---")
            print(tweet_text)
            print(f"Length: {len(tweet_text)} characters")
            
            if dry_run:
                print("[DRY RUN - NOT POSTED]")
                results.append({
                    "prediction": prediction,
                    "tweet": tweet_text,
                    "posted": False,
                    "dry_run": True,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                print(f"\nPosting to Twitter...")
                success = post_tweet(tweet_text)
                
                results.append({
                    "prediction": prediction,
                    "tweet": tweet_text,
                    "posted": success,
                    "dry_run": False,
                    "timestamp": datetime.now().isoformat()
                })
                
                if success:
                    print(f"SUCCESS: Tweet {i}/{total} posted!")
                    
                    if i < total:
                        print(f"Waiting {delay_seconds} seconds...")
                        time.sleep(delay_seconds)
                else:
                    print(f"ERROR: Failed to post tweet {i}/{total}")
                    
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "prediction": prediction,
                "tweet": None,
                "posted": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Post predictions from database to Twitter')
    parser.add_argument('--dry-run', action='store_true', help='Preview tweets')
    parser.add_argument('--post', action='store_true', help='Post tweets')
    parser.add_argument('--delay', type=int, default=15, help='Seconds between tweets')
    parser.add_argument('--date', type=str, help='Date to post (YYYY-MM-DD), default=today')
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.post:
        print("ERROR: Specify --dry-run or --post")
        sys.exit(1)
    
    # Load predictions from database
    print(f"Loading predictions from database...")
    predictions = load_predictions_from_db(args.date)
    
    if not predictions:
        print("\nERROR: No predictions found!")
        print("\nRun this first:")
        print("  python model/predict.py")
        sys.exit(1)
    
    print(f"\nFound {len(predictions)} game(s):")
    for p in predictions:
        print(f"  • {p['away_team']} @ {p['home_team']}")
    
    # Confirmation
    if args.post:
        print(f"\nWARNING: About to post {len(predictions)} tweet(s)!")
        print(f"Cost: ${len(predictions) * 0.01:.2f}")
        response = input("\nType 'yes' to confirm: ")
        
        if response.lower() != 'yes':
            print("Cancelled")
            sys.exit(0)
    
    # Post
    results = post_all_predictions(predictions, dry_run=args.dry_run, delay_seconds=args.delay)
    
    # Summary
    posted = sum(1 for r in results if r.get("posted", False))
    print(f"\n" + "="*60)
    print(f"SUMMARY: {posted}/{len(predictions)} posted successfully")
    print("="*60)


if __name__ == "__main__":
    main()