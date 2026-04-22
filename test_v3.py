from nba_api.stats.endpoints import boxscoretraditionalv3

test = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id="0042500122")
dfs = test.get_data_frames()

print("Dataframe 1 (current):")
print(dfs[1][['teamTricode', 'points', 'startersBench']])

print("\nDataframe 2:")
print(dfs[2][['teamTricode', 'points']])