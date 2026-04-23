import requests
import os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv('ODDS_API_KEY')
url = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
params = {
    'apiKey':     key,
    'regions':    'us',
    'markets':    'spreads,totals',
    'oddsFormat': 'american'
}
response = requests.get(url, params=params, timeout=10)
print(f'Remaining requests: {response.headers.get("x-requests-remaining")}')
print(f'Used requests: {response.headers.get("x-requests-used")}')