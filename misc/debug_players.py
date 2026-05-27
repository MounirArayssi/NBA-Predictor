import sys
sys.path.append('.')
from model.predict_players import (
    get_team_roster, predict_team_player_stats,
    get_playoff_series_minutes, redistribute_playoff_minutes,
    get_player_weights
)
from sqlalchemy import text
from data.storage.db import engine
import pandas as pd

# Check DEN specifically
team_abbr = 'DEN'
q = text("""
    SELECT team_id FROM teams WHERE abbreviation = :abbr
""")
with engine.connect() as conn:
    team_id = conn.execute(q, {'abbr': team_abbr}).fetchone()[0]

print(f"\n{'='*50}")
print(f"DIAGNOSING {team_abbr} (team_id={team_id})")
print(f"{'='*50}")

# Step 1: Raw roster
roster = get_team_roster(team_id, is_playoff=True)
print(f"\n1. Raw roster ({len(roster)} players):")
print(roster[['full_name', 'player_avg_min_l10',
              'player_avg_points_l10', 'player_avg_usage_l10']].to_string())

# Step 2: Playoff minutes override
playoff_mins = get_playoff_series_minutes(team_id)
print(f"\n2. Playoff minutes override ({len(playoff_mins)} players):")
for pid, mins in playoff_mins.items():
    name = roster[roster['player_id'] == pid]['full_name'].values
    name = name[0] if len(name) > 0 else f"id={pid}"
    print(f"   {name}: {mins:.1f} min")

# Step 3: Simulate the loop
print(f"\n3. Player contributions to team total:")
print(f"{'Player':<25} {'Pred Pts':>8} {'Min':>6} {'Usage':>6} "
      f"{'Weight':>7} {'Contribution':>12}")
print("-" * 70)

total = 0
rotation_total = 0
fringe_total   = 0

for _, player in roster.iterrows():
    minutes = float(player['player_avg_min_l10'])
    pid     = int(player['player_id'])
    if pid in playoff_mins and playoff_mins[pid] >= 1:
        minutes = playoff_mins[pid]

    usage    = float(player['player_avg_usage_l10'] or 0.15)
    pred_pts = float(player['player_avg_points_l10'] or 0)

    is_rotation = minutes >= 15
    _, pw       = get_player_weights(usage)
    multiplier  = (1 + pw * 0.5) if is_rotation else 1.0
    contrib     = pred_pts * (minutes / 48.0) * multiplier

    label = "rotation" if is_rotation else "fringe"
    print(f"{player['full_name']:<25} {pred_pts:>8.1f} {minutes:>6.1f} "
          f"{usage:>6.3f} {multiplier:>7.3f} {contrib:>12.2f} ({label})")
    total += contrib
    if is_rotation:
        rotation_total += contrib
    else:
        fringe_total += contrib

print(f"\n  Rotation total: {rotation_total:.1f}")
print(f"  Fringe total:   {fringe_total:.1f}")
print(f"  Grand total:    {total:.1f}")
print(f"  Rounded:        {round(total)}")