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
    tbs.is_home,
    tbs.points,
    tbs.q1_points,
    tbs.q2_points,
    tbs.q3_points,
    tbs.q4_points
FROM team_box_scores tbs
JOIN games g ON tbs.game_id = g.game_id
JOIN teams ht ON g.home_team_id = ht.team_id
JOIN teams at ON g.away_team_id = at.team_id
WHERE g.season_type = 'Playoffs'
AND g.season = '2025-26'
AND g.is_final = TRUE
ORDER BY g.game_date DESC
LIMIT 10
""")

print(pd.read_sql(q, engine).to_string())