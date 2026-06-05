import sys
sys.path.append('.')
from data.ingestion.fetch_odds import fetch_todays_player_props
from model.predict_players import apply_vegas_props_anchor

# Fetch props
print("Fetching props...")
all_props = fetch_todays_player_props()

# Show all available props
for game, players in all_props.items():
    print(f"\n{'='*50}")
    print(f"Game: {game}")
    print(f"{'='*50}")
    for player, lines in players.items():
        print(f"  {player:<25} "
              f"pts:{lines.get('points','—'):>5} | "
              f"reb:{lines.get('rebounds','—'):>5} | "
              f"ast:{lines.get('assists','—'):>5}")

# Test anchoring on specific players
print(f"\n{'='*50}")
print("ANCHORING TEST")
print(f"{'='*50}")

# Simulate our model predictions vs Vegas
test_cases = [
    # (player_name, our_pred_pts, our_pred_reb, our_pred_ast, game_key)
    ("Jalen Brunson",    22.0, 3.5, 7.0, "New York Knicks@Atlanta Hawks"),
    ("Nikola Jokic",     24.0, 12.0, 9.0, "Denver Nuggets@Minnesota Timberwolves"),
    ("Anthony Edwards",  21.0, 4.0, 4.0, "Denver Nuggets@Minnesota Timberwolves"),
    ("Donovan Mitchell", 29.0, 5.0, 6.0, "Cleveland Cavaliers@Toronto Raptors"),
    ("Jalen Johnson",    22.0, 9.0, 5.0, "New York Knicks@Atlanta Hawks"),
]

for player, pred_pts, pred_reb, pred_ast, game_key in test_cases:
    props = all_props.get(game_key, {})
    new_pts, new_reb, new_ast = apply_vegas_props_anchor(
        pred_pts, pred_reb, pred_ast, player, props
    )
    vegas = props.get(player, {})
    print(f"\n{player}")
    print(f"  Model:  {pred_pts}pts {pred_reb}reb {pred_ast}ast")
    print(f"  Vegas:  {vegas.get('points','—')}pts "
          f"{vegas.get('rebounds','—')}reb "
          f"{vegas.get('assists','—')}ast")
    print(f"  Final:  {new_pts}pts {new_reb}reb {new_ast}ast")
    diff = new_pts - pred_pts
    print(f"  Change: {diff:+.1f}pts")