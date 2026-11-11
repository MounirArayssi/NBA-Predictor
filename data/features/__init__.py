"""
Feature builders, grouped by category. Each add_* function takes the game-level
DataFrame from data.ingestion.build_features and returns it with new columns.

    strength.py      A  Team strength      Elo, home court, ratings matchups, similarity, style, H2H
    recent_form.py   B  Recent performance momentum, variance, shooting consistency, quarters, four factors
    availability.py  C  Player availability rotation players listed Out/Doubtful
    rest.py          D  Rest               rest days, back-to-backs
    playoffs.py      E  Playoffs           series state, playoff elevation, in-series form

Call order matters (some builders read columns made by earlier ones); see
build_feature_dataset in data/ingestion/build_features.py.
"""
