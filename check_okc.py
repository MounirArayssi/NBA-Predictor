import sys
sys.path.append('.')
from sqlalchemy import text
from data.storage.db import engine
import pandas as pd

q = text("""
SELECT p.full_name, 
       AVG(pbs.minutes_played) as avg_minutes,
       COUNT(*) as games,
       g.season
FROM player_box_scores pbs
JOIN players p ON pbs.player_id = p.player_id
JOIN games g ON pbs.game_id = g.game_id
JOIN teams t ON pbs.team_id = t.team_id
WHERE t.abbreviation = 'DET'
AND p.full_name ILIKE '%jenkins%'
AND pbs.minutes_played >= 1
GROUP BY p.full_name, g.season
ORDER BY g.season
""")

print(pd.read_sql(q, engine).to_string())