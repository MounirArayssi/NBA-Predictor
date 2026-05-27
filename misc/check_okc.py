import sys
sys.path.append('.')
from sqlalchemy import text
from data.storage.db import engine
import pandas as pd

q = text("""
SELECT 
    ht.abbreviation as home,
    at.abbreviation as away,
    g.game_date,
    go.home_spread,
    go.total_line,
    go.vegas_home_implied,
    go.vegas_away_implied
FROM game_odds go
JOIN games g ON go.game_id = g.game_id
JOIN teams ht ON g.home_team_id = ht.team_id
JOIN teams at ON g.away_team_id = at.team_id
ORDER BY g.game_date
""")

print(pd.read_sql(q, engine).to_string())