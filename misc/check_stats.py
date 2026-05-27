import sys
sys.path.append('.')
from sqlalchemy import text
from data.storage.db import engine
import pandas as pd

q = text("""
SELECT 
    t.abbreviation,
    trs.avg_points,
    trs.avg_offensive_rating,
    trs.avg_defensive_rating,
    trs.avg_pace,
    trs.win_pct,
    trs.as_of_date,
    trs.games_counted
FROM team_rolling_stats trs
JOIN teams t ON trs.team_id = t.team_id
WHERE t.abbreviation IN ('OKC', 'PHX')
AND trs."window" = 10
AND trs.as_of_date = (
    SELECT MAX(as_of_date) 
    FROM team_rolling_stats trs2
    WHERE trs2.team_id = trs.team_id
    AND trs2."window" = 10
)
ORDER BY t.abbreviation
""")

print(pd.read_sql(q, engine).to_string())