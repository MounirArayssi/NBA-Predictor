import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tweepy
from dotenv import load_dotenv
load_dotenv()

def get_twitter_client():
    client = tweepy.Client(
        bearer_token        = os.getenv('TWITTER_BEARER_TOKEN'),
        consumer_key        = os.getenv('TWITTER_API_KEY'),
        consumer_secret     = os.getenv('TWITTER_API_SECRET'),
        access_token        = os.getenv('TWITTER_ACCESS_TOKEN'),
        access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
    )
    return client

if __name__ == "__main__":
    print("Testing Twitter connection...")
    client = get_twitter_client()

    # Post a test tweet
    try:
        response = client.create_tweet(
            text="🤖 NBA Predictor bot is online. Predictions coming soon. #NBA #NBAPlayoffs"
        )
        print(f"✅ Tweet posted successfully!")
        print(f"   Tweet ID: {response.data['id']}")
        print(f"   Text: {response.data['text']}")
    except Exception as e:
        print(f"❌ Error: {e}")