"""
A — Team strength: how good each team is, overall and against this opponent.
Elo + season margin (data/ingestion/team_strength.py), home court, rating
matchups, similar-opponent proxies, defense vs offensive style, head-to-head.
"""
import pandas as pd
from sqlalchemy import text
from data.storage.db import engine

# Elo stays in data/ingestion/team_strength.py: model/predict.py, the web scripts
# and misc/walkforward_backfill.py (which patches ELO_START_DATE) use it there.
from data.ingestion.team_strength import add_team_strength_features  # noqa: F401


def add_home_court_strength(df):
    """
    Compute how much stronger each team's home court is
    vs league average home court advantage.
    Some arenas (OKC, BOS, DEN) have measurably stronger impact.
    """
    print("  Computing home court strength...")

    from config.settings import CURRENT_SEASON, PREVIOUS_SEASON

    query = text("""
        SELECT
            g.home_team_id AS team_id,
            AVG(tbs_home.points - tbs_away.points) AS avg_home_margin,
            COUNT(*) AS home_games
        FROM games g
        JOIN team_box_scores tbs_home
            ON tbs_home.game_id = g.game_id
            AND tbs_home.team_id = g.home_team_id
        JOIN team_box_scores tbs_away
            ON tbs_away.game_id = g.game_id
            AND tbs_away.team_id = g.away_team_id
        WHERE g.is_final = TRUE
        AND g.season IN (:current_season, :previous_season)
        GROUP BY g.home_team_id
        HAVING COUNT(*) >= 20
    """)

    home_margins = pd.read_sql(query, engine, params={
        "current_season": CURRENT_SEASON,
        "previous_season": PREVIOUS_SEASON,
    })

    # League average home margin
    league_avg_margin = float(home_margins['avg_home_margin'].mean())

    # Home court strength = how much better than league average
    home_margins['home_court_strength'] = (
        home_margins['avg_home_margin'] - league_avg_margin
    )

    strength_map = dict(zip(
        home_margins['team_id'],
        home_margins['home_court_strength']
    ))

    df['home_court_strength'] = df['home_team_id'].map(
        strength_map
    ).fillna(0.0)

    return df


def add_matchup_interactions(df):
    """
    Add interaction features between offensive and defensive styles.
    These directly affect predicted totals:
    - High 3pt rate offense vs good 3pt defense = lower scoring
    - High TOV forcing defense vs turnover prone offense = lower scoring
    - High FT rate offense vs foul prone defense = higher scoring
    (Momentum, formerly here, is recent_form.add_momentum_features.)
    """
    print("  Computing matchup interactions...")

    # 3pt matchup — home offense vs away defense
    # Positive = home team 3pt rate exceeds away team's 3pt defense
    df['home_3pt_rate'] = df['home_3pt_rate'].fillna(0.35)
    df['away_3pt_rate'] = df['away_3pt_rate'].fillna(0.35)
    df['home_3pt_matchup'] = (
        df['home_3pt_rate'] - df['away_3pt_rate']
    )
    df['away_3pt_matchup'] = -df['home_3pt_matchup']

    # Pace matchup — combined pace determines total possessions
    df['combined_pace'] = (
        df['home_pace'] + df['away_pace']
    ) / 2

    # Offensive vs defensive rating matchup
    # How much does home offense exceed away defense?
    df['home_off_vs_away_def'] = (
        df['home_off_rating'] - df['away_def_rating']
    )
    df['away_off_vs_home_def'] = (
        df['away_off_rating'] - df['home_def_rating']
    )

    # Net rating differential — overall team quality gap
    df['net_rating_diff'] = (
        (df['home_off_rating'] - df['home_def_rating']) -
        (df['away_off_rating'] - df['away_def_rating'])
    )

    # Predicted game total based on ratings
    # Rough estimate: (home_off + away_off) / 2 * pace factor
    df['implied_total'] = (
        df['home_avg_points'] + df['away_avg_points']
    )

    return df


def add_similarity_features(df):
    """
    For each game, find the 3 most similar teams to each opponent
    and compute how the home/away team performed against those proxy teams.

    Two directions:
    1. Teams similar to the OPPONENT — how did home team do vs those teams?
    2. Teams similar to the HOME TEAM — how did similar teams do vs this opponent?
    """
    print("  Computing team similarity features...")

    # Load similarity matrix
    sim_query = text("""
        SELECT team_a_id, team_b_id, similarity_score
        FROM team_similarity
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM team_similarity)
    """)
    sim_df = pd.read_sql(sim_query, engine)

    # Build similarity lookup — both directions
    # {team_id: [(similar_team_id, score), ...]}
    similarity_map = {}
    for _, row in sim_df.iterrows():
        a = int(row['team_a_id'])
        b = int(row['team_b_id'])
        score = float(row['similarity_score'])

        if a not in similarity_map:
            similarity_map[a] = []
        if b not in similarity_map:
            similarity_map[b] = []

        similarity_map[a].append((b, score))
        similarity_map[b].append((a, score))

    # Sort by similarity descending
    for team_id in similarity_map:
        similarity_map[team_id] = sorted(
            similarity_map[team_id],
            key=lambda x: x[1],
            reverse=True
        )

    # Load all team box scores for proxy lookups
    bs_query = text("""
        SELECT
            tbs.team_id,
            tbs.points,
            tbs.offensive_rating,
            tbs.defensive_rating,
            g.game_date,
            CASE
                WHEN tbs.team_id = g.home_team_id THEN g.away_team_id
                ELSE g.home_team_id
            END AS opponent_team_id,
            CASE
                WHEN tbs.team_id = g.home_team_id THEN g.away_score
                ELSE g.home_score
            END AS opponent_points
        FROM team_box_scores tbs
        JOIN games g ON tbs.game_id = g.game_id
        WHERE g.is_final = TRUE
        ORDER BY g.game_date
    """)
    bs_df = pd.read_sql(bs_query, engine)
    bs_df['game_date'] = pd.to_datetime(bs_df['game_date'])

    # Features to compute
    home_proxy_off   = []  # home team off rating vs similar opponents
    home_proxy_pts   = []  # home team avg points vs similar opponents
    away_proxy_off   = []
    away_proxy_pts   = []
    home_sim_off     = []  # similar teams' off rating vs this opponent
    away_sim_off     = []

    df = df.sort_values('game_date').copy()
    df['game_date'] = pd.to_datetime(df['game_date'])

    for _, row in df.iterrows():
        home_id   = int(row['home_team_id'])
        away_id   = int(row['away_team_id'])
        game_date = row['game_date']

        # Get top 3 similar teams to the opponent
        away_similars = [
            t for t, _ in similarity_map.get(away_id, [])[:3]
            if t != home_id
        ]
        home_similars = [
            t for t, _ in similarity_map.get(home_id, [])[:3]
            if t != away_id
        ]

        # --- Direction 1: How did HOME team perform vs teams similar to AWAY ---
        proxy_games_home = bs_df[
            (bs_df['team_id'] == home_id) &
            (bs_df['opponent_team_id'].isin(away_similars)) &
            (bs_df['game_date'] < game_date)
        ]

        if len(proxy_games_home) >= 3:
            home_proxy_off.append(
                float(proxy_games_home['offensive_rating'].mean())
            )
            home_proxy_pts.append(
                float(proxy_games_home['points'].mean())
            )
        else:
            home_proxy_off.append(None)
            home_proxy_pts.append(None)

        # --- Direction 1: How did AWAY team perform vs teams similar to HOME ---
        proxy_games_away = bs_df[
            (bs_df['team_id'] == away_id) &
            (bs_df['opponent_team_id'].isin(home_similars)) &
            (bs_df['game_date'] < game_date)
        ]

        if len(proxy_games_away) >= 3:
            away_proxy_off.append(
                float(proxy_games_away['offensive_rating'].mean())
            )
            away_proxy_pts.append(
                float(proxy_games_away['points'].mean())
            )
        else:
            away_proxy_off.append(None)
            away_proxy_pts.append(None)

        # --- Direction 2: How did teams similar to HOME perform vs AWAY ---
        sim_home_vs_away = bs_df[
            (bs_df['team_id'].isin(home_similars)) &
            (bs_df['opponent_team_id'] == away_id) &
            (bs_df['game_date'] < game_date)
        ]

        home_sim_off.append(
            float(sim_home_vs_away['offensive_rating'].mean())
            if len(sim_home_vs_away) >= 2 else None
        )

        # --- Direction 2: How did teams similar to AWAY perform vs HOME ---
        sim_away_vs_home = bs_df[
            (bs_df['team_id'].isin(away_similars)) &
            (bs_df['opponent_team_id'] == home_id) &
            (bs_df['game_date'] < game_date)
        ]

        away_sim_off.append(
            float(sim_away_vs_home['offensive_rating'].mean())
            if len(sim_away_vs_home) >= 2 else None
        )

    df['home_proxy_off_rating'] = home_proxy_off
    df['home_proxy_avg_pts']    = home_proxy_pts
    df['away_proxy_off_rating'] = away_proxy_off
    df['away_proxy_avg_pts']    = away_proxy_pts
    df['home_sim_off_rating']   = home_sim_off
    df['away_sim_off_rating']   = away_sim_off

    # Fill missing with rolling averages
    df['home_proxy_off_rating'] = df['home_proxy_off_rating'].fillna(
        df['home_off_rating']
    )
    df['home_proxy_avg_pts'] = df['home_proxy_avg_pts'].fillna(
        df['home_avg_points']
    )
    df['away_proxy_off_rating'] = df['away_proxy_off_rating'].fillna(
        df['away_off_rating']
    )
    df['away_proxy_avg_pts'] = df['away_proxy_avg_pts'].fillna(
        df['away_avg_points']
    )
    df['home_sim_off_rating'] = df['home_sim_off_rating'].fillna(
        df['home_off_rating']
    )
    df['away_sim_off_rating'] = df['away_sim_off_rating'].fillna(
        df['away_off_rating']
    )

    return df


def add_style_defensive_matchup(df):
    """
    For each game compute how each defense performs against
    teams with a similar offensive style to tonight's opponent.

    This goes beyond raw defensive rating to capture:
    - Does this defense struggle against pace?
    - Does this defense struggle against 3pt heavy teams?
    - Does this defense struggle against paint-heavy teams?
    """
    print("  Computing style-based defensive matchups...")

    # Pull all team box scores with opponent style info
    query = text("""
        SELECT
            tbs_def.team_id        AS def_team_id,
            tbs_off.team_id        AS off_team_id,
            tbs_def.defensive_rating,
            tbs_off.offensive_rating,
            tbs_off.pace,
            tbs_off.fg3_pct,
            tbs_off.fg3a,
            tbs_off.fga,
            tbs_off.oreb_pct,
            tbs_off.tov_pct,
            tbs_def.points         AS points_allowed,
            g.game_date
        FROM team_box_scores tbs_def
        JOIN games g ON tbs_def.game_id = g.game_id
        JOIN team_box_scores tbs_off
            ON tbs_off.game_id = g.game_id
            AND tbs_off.team_id != tbs_def.team_id
        WHERE g.is_final = TRUE
        AND tbs_def.defensive_rating IS NOT NULL
        AND tbs_off.pace IS NOT NULL
        ORDER BY g.game_date
    """)

    matchup_df = pd.read_sql(query, engine)
    matchup_df['game_date'] = pd.to_datetime(matchup_df['game_date'])
    matchup_df['three_point_rate'] = (
        matchup_df['fg3a'] / matchup_df['fga'].replace(0, 1)
    )
    df['game_date'] = pd.to_datetime(df['game_date'])

    home_def_vs_style = []
    away_def_vs_style = []

    for _, row in df.iterrows():
        game_date = row['game_date']
        home_id   = int(row['home_team_id'])
        away_id   = int(row['away_team_id'])

        # Home team's offensive style
        home_style_pace = row.get('home_pace', 98.0)
        home_style_3pt  = row.get('home_3pt_rate', 0.35)

        # Away team's offensive style
        away_style_pace = row.get('away_pace', 98.0)
        away_style_3pt  = row.get('away_3pt_rate', 0.35)

        # How does AWAY defense perform vs teams similar to HOME offense?
        # Find games where away team defended against teams
        # with similar pace and 3pt rate to home team
        away_def_history = matchup_df[
            (matchup_df['def_team_id'] == away_id) &
            (matchup_df['game_date'] < game_date) &
            (abs(matchup_df['pace'] - home_style_pace) < 5) &
            (abs(matchup_df['three_point_rate'] - home_style_3pt) < 0.07)
        ]

        if len(away_def_history) >= 3:
            away_def_vs_style.append(
                float(away_def_history['defensive_rating'].mean())
            )
        else:
            # Fall back to overall defensive rating
            away_def_vs_style.append(
                float(row.get('away_def_rating', 112.0))
            )

        # How does HOME defense perform vs teams similar to AWAY offense?
        home_def_history = matchup_df[
            (matchup_df['def_team_id'] == home_id) &
            (matchup_df['game_date'] < game_date) &
            (abs(matchup_df['pace'] - away_style_pace) < 5) &
            (abs(matchup_df['three_point_rate'] - away_style_3pt) < 0.07)
        ]

        if len(home_def_history) >= 3:
            home_def_vs_style.append(
                float(home_def_history['defensive_rating'].mean())
            )
        else:
            home_def_vs_style.append(
                float(row.get('home_def_rating', 112.0))
            )

    df['home_def_vs_away_style'] = home_def_vs_style
    df['away_def_vs_home_style'] = away_def_vs_style

    # How much better/worse is this defense vs this specific style
    # compared to their overall defensive rating
    df['home_def_style_edge'] = (
        df['home_def_rating'] - df['home_def_vs_away_style']
    )
    df['away_def_style_edge'] = (
        df['away_def_rating'] - df['away_def_vs_home_style']
    )

    return df


def add_h2h_features(df):
    """Add head to head historical features."""
    print("  Computing head-to-head features...")

    df = df.sort_values('game_date').copy()

    h2h_home_avg     = []
    h2h_away_avg     = []
    h2h_home_win_pct = []
    h2h_games_count  = []

    h2h_history = {}

    for _, row in df.iterrows():
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        key     = tuple(sorted([home_id, away_id]))

        if key in h2h_history and len(h2h_history[key]) > 0:
            past    = h2h_history[key]
            past_df = pd.DataFrame(past)

            home_games = past_df[past_df['home_team_id'] == home_id]
            away_games = past_df[past_df['home_team_id'] == away_id]

            all_home_scores = list(home_games['home_score']) + \
                              list(away_games['away_score'])
            all_away_scores = list(home_games['away_score']) + \
                              list(away_games['home_score'])

            h2h_home_avg.append(
                sum(all_home_scores) / len(all_home_scores)
                if all_home_scores else None
            )
            h2h_away_avg.append(
                sum(all_away_scores) / len(all_away_scores)
                if all_away_scores else None
            )

            home_wins = sum(
                1 for h, a in zip(all_home_scores, all_away_scores)
                if h > a
            )
            h2h_home_win_pct.append(
                home_wins / len(all_home_scores)
                if all_home_scores else None
            )
            h2h_games_count.append(len(all_home_scores))

        else:
            h2h_home_avg.append(None)
            h2h_away_avg.append(None)
            h2h_home_win_pct.append(None)
            h2h_games_count.append(0)

        if key not in h2h_history:
            h2h_history[key] = []
        h2h_history[key].append({
            'home_team_id': home_id,
            'away_team_id': away_id,
            'home_score':   row['home_score'],
            'away_score':   row['away_score'],
            'game_date':    row['game_date']
        })

    df['h2h_home_avg_score'] = h2h_home_avg
    df['h2h_away_avg_score'] = h2h_away_avg
    df['h2h_home_win_pct']   = h2h_home_win_pct
    df['h2h_games_count']    = h2h_games_count

    # Fill missing H2H with rolling averages
    df['h2h_home_avg_score'] = df['h2h_home_avg_score'].fillna(
        df['home_avg_points']
    )
    df['h2h_away_avg_score'] = df['h2h_away_avg_score'].fillna(
        df['away_avg_points']
    )
    df['h2h_home_win_pct'] = df['h2h_home_win_pct'].fillna(0.5)

    return df
