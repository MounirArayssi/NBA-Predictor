import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sqlalchemy import text
from data.storage.db import engine

# Rest days are capped here in training and in live prediction (model/predict.py)
REST_CAP = 5


from data.ingestion.team_strength import add_team_strength_features


def build_feature_dataset(window=10):
    print(f"Building feature dataset (window={window})...")

    query = text("""
        SELECT
            g.game_id,
            g.nba_game_id,
            g.game_date,
            g.season,
            g.season_type,
            g.home_score,
            g.away_score,
            g.home_team_id,
            g.away_team_id,
            ht.abbreviation AS home_team,
            at.abbreviation AS away_team,

            -- Home team rolling stats
            h.avg_points              AS home_avg_points,
            h.avg_offensive_rating    AS home_off_rating,
            h.avg_defensive_rating    AS home_def_rating,
            h.avg_pace                AS home_pace,
            h.avg_fg_pct              AS home_fg_pct,
            h.avg_fg3_pct             AS home_fg3_pct,
            h.three_point_rate        AS home_3pt_rate,
            h.win_pct                 AS home_win_pct,
            h.home_avg_points         AS home_home_avg_pts,
            h.home_avg_points_allowed AS home_home_avg_pts_allowed,
            h.avg_ft_rate             AS home_ft_rate,

            -- Away team rolling stats
            a.avg_points              AS away_avg_points,
            a.avg_offensive_rating    AS away_off_rating,
            a.avg_defensive_rating    AS away_def_rating,
            a.avg_pace                AS away_pace,
            a.avg_fg_pct              AS away_fg_pct,
            a.avg_fg3_pct             AS away_fg3_pct,
            a.three_point_rate        AS away_3pt_rate,
            a.win_pct                 AS away_win_pct,
            a.away_avg_points         AS away_away_avg_pts,
            a.away_avg_points_allowed AS away_away_avg_pts_allowed,
            a.avg_ft_rate             AS away_ft_rate,

            -- Matchup features
            ABS(h.avg_pace - a.avg_pace) AS pace_differential,

            h7.avg_points             AS home_avg_points_l7,
            h7.avg_offensive_rating   AS home_off_rating_l7,
            h7.avg_defensive_rating   AS home_def_rating_l7,
            h7.win_pct                AS home_win_pct_l7,
            a7.avg_points             AS away_avg_points_l7,
            a7.avg_offensive_rating   AS away_off_rating_l7,
            a7.avg_defensive_rating   AS away_def_rating_l7,
            a7.win_pct                AS away_win_pct_l7

        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id

        -- Home team rolling stats
        JOIN team_rolling_stats h ON h.team_id = g.home_team_id
            AND h."window" = :window
            AND h.as_of_date = (
                SELECT MAX(as_of_date)
                FROM team_rolling_stats
                WHERE team_id = g.home_team_id
                AND "window" = :window
                AND as_of_date <= g.game_date
            )

        -- Away team rolling stats
        JOIN team_rolling_stats a ON a.team_id = g.away_team_id
            AND a."window" = :window
            AND a.as_of_date = (
                SELECT MAX(as_of_date)
                FROM team_rolling_stats
                WHERE team_id = g.away_team_id
                AND "window" = :window
                AND as_of_date <= g.game_date
            )
                 
        JOIN team_rolling_stats h7 ON h7.team_id = g.home_team_id
    AND h7."window" = 7
    AND h7.as_of_date = (
        SELECT MAX(as_of_date)
        FROM team_rolling_stats
        WHERE team_id = g.home_team_id
        AND "window" = 7
        AND as_of_date <= g.game_date
    )

        JOIN team_rolling_stats a7 ON a7.team_id = g.away_team_id
        AND a7."window" = 7
        AND a7.as_of_date = (
        SELECT MAX(as_of_date)
        FROM team_rolling_stats
        WHERE team_id = g.away_team_id
        AND "window" = 7
        AND as_of_date <= g.game_date
    )

        WHERE g.is_final = TRUE
        AND g.home_score IS NOT NULL
        AND g.away_score IS NOT NULL
        ORDER BY g.game_date
    """)

    df = pd.read_sql(query, engine, params={"window": window})
    print(f"Loaded {len(df)} game records")

    # Add features
    df = add_rest_days(df)
    df = add_h2h_features(df)
    df = add_playoff_features(df)
    df = add_similarity_features(df)
    df = add_playoff_elevation_features(df) 
    df = add_home_court_strength(df)      # add
    df = add_scoring_variance(df)         # add
    df = add_matchup_interactions(df)
    df = add_style_defensive_matchup(df)
    df = add_shooting_consistency(df)
    df = add_quarter_features(df)
    df = add_series_form_features(df)
    df = add_advanced_rolling_features(df)
    df = add_availability_features(df)
    df = add_team_strength_features(df)
    return df


# Availability — rotation players unavailable for a game.
# Rotation = played >= ROT_MIN_GAMES of the team's last ROT_WINDOW games THIS SEASON
# at >= ROT_MIN_MINUTES avg. (Using last season's games let departed players count
# as "missing" for the first weeks of every season.)
# Unavailable = listed Out/Doubtful on the official NBA injury report — the game
# day's training-time report in training, the latest report live — so training and
# prediction use the same information.
ROT_WINDOW = 10
ROT_MIN_GAMES = 2   # 3 dropped stars hurt in week 1 (e.g. Banchero, 2 games); 2 tested slightly better in CV (2026-11-05)
ROT_MIN_MINUTES = 15


def _load_played_box_scores(before_date=None):
    query = """
        SELECT b.game_id, b.team_id, b.player_id, b.minutes_played, b.points,
               g.game_date, g.season, p.full_name
        FROM player_box_scores b
        JOIN games g USING (game_id)
        JOIN players p USING (player_id)
        WHERE g.is_final = TRUE AND b.minutes_played > 0
    """
    params = {}
    if before_date is not None:
        query += " AND g.game_date < :before_date"
        params['before_date'] = str(before_date)
    box = pd.read_sql(text(query), engine, params=params)
    box['game_date'] = pd.to_datetime(box['game_date'])
    return box


def _rotation(window):
    agg = window.groupby('player_id').agg(
        gp=('game_id', 'nunique'), mins=('minutes_played', 'mean'),
        pts=('points', 'mean'), full_name=('full_name', 'last'))
    return agg[(agg.gp >= ROT_MIN_GAMES) & (agg.mins >= ROT_MIN_MINUTES)]


def _still_on_team(players, team_id, last_team_by_date, as_of):
    """Drop players who have since appeared for another team (traded/released)."""
    if players.empty:
        return players
    prior = last_team_by_date[
        last_team_by_date.player_id.isin(players.index) &
        (last_team_by_date.game_date < as_of)]
    latest = prior.groupby('player_id')['team_id'].last()
    return players[latest.reindex(players.index).eq(team_id).values]


def _season_rotation(tb, season, before_date, team_id, last_team_by_date):
    """Rotation from the team's last ROT_WINDOW games of `season` before before_date."""
    prior = tb[(tb.season == season) & (tb.game_date < before_date)]
    if prior.empty:
        return _rotation(prior)
    recent = (prior[['game_id', 'game_date']].drop_duplicates()
              .sort_values('game_date').tail(ROT_WINDOW)['game_id'])
    rot = _rotation(prior[prior.game_id.isin(recent)])
    return _still_on_team(rot, team_id, last_team_by_date, before_date)


def compute_availability():
    """
    Historical availability: for every team-game, minutes and points per game of
    rotation players listed Out/Doubtful on that day's official injury report.
    Dates with no stored report fall back to box-score absences (counted and
    reported, since that definition differs from what's known before tip-off).
    """
    from data.ingestion.fetch_official_injuries import load_training_unavailable

    box = _load_played_box_scores()
    last_team_by_date = box.sort_values('game_date')[['player_id', 'game_date', 'team_id']]
    unavailable = load_training_unavailable()

    rows = []
    fallback = 0
    for team_id, tb in box.groupby('team_id'):
        games = (tb[['game_id', 'game_date', 'season']].drop_duplicates()
                 .sort_values('game_date'))
        played_by_game = tb.groupby('game_id')['player_id'].apply(set).to_dict()
        for gid, gdate, season in games.itertuples(index=False):
            rot = _season_rotation(tb, season, gdate, team_id, last_team_by_date)
            report = unavailable.get(gdate.date())
            if report is not None:
                absent = rot[rot.index.isin(report.get(team_id, set()))]
            else:
                fallback += 1
                absent = rot[~rot.index.isin(played_by_game.get(gid, set()))]
            rows.append((gid, team_id, float(absent['mins'].astype(float).sum()),
                         float(absent['pts'].sum())))
    print(f"    {len(rows)} team-games, {fallback} without an injury report "
          f"(box-score absences used)")
    return pd.DataFrame(rows, columns=['game_id', 'team_id', 'missing_min', 'missing_pts'])


def add_availability_features(df):
    print("  Computing availability (unavailable rotation players)...")
    avail = compute_availability()
    for side in ('home', 'away'):
        a = avail.rename(columns={'team_id': f'{side}_team_id',
                                  'missing_min': f'{side}_missing_min',
                                  'missing_pts': f'{side}_missing_pts'})
        df = df.merge(a, on=['game_id', f'{side}_team_id'], how='left')
    return df


def normalize_player_name(name):
    import unicodedata
    name = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode()
    return ' '.join(name.lower().replace('.', '').split())


def compute_live_availability(team_id, game_date, out_names):
    """
    Availability for an unplayed game: rotation players (same definition as training)
    listed Out/Doubtful on the injury report.
    Returns (missing_min, missing_pts, missing_player_names).
    """
    if not out_names:
        return 0.0, 0.0, []
    from data.ingestion.fetch_official_injuries import season_for_date

    box = _load_played_box_scores(before_date=game_date)
    tb = box[box.team_id == team_id]
    if tb.empty:
        return 0.0, 0.0, []

    last_team_by_date = box.sort_values('game_date')[['player_id', 'game_date', 'team_id']]
    gdate = pd.Timestamp(game_date)
    rot = _season_rotation(tb, season_for_date(gdate.date()), gdate, team_id,
                           last_team_by_date)

    out = {normalize_player_name(n) for n in out_names}
    absent = rot[rot['full_name'].map(normalize_player_name).isin(out)]
    return (float(absent['mins'].astype(float).sum()), float(absent['pts'].sum()),
            absent['full_name'].tolist())


def add_rest_days(df):
    """Calculate rest days for each team before each game."""
    print("  Computing rest days...")

    # Same rule as the live get_rest_days (model/predict.py): days since the
    # team's previous final game, home or away, from the full schedule (not just
    # rows in df), capped at REST_CAP; 3 when there is no previous game.
    df = df.sort_values('game_date').copy()
    df['game_date'] = pd.to_datetime(df['game_date'])

    sched = pd.read_sql(text("""
        SELECT game_date, home_team_id AS team_id FROM games WHERE is_final = TRUE
        UNION
        SELECT game_date, away_team_id AS team_id FROM games WHERE is_final = TRUE
    """), engine)
    sched['game_date'] = pd.to_datetime(sched['game_date'])
    dates_by_team = {t: np.sort(g['game_date'].unique())
                     for t, g in sched.groupby('team_id')}

    def rest_for(team_id, game_date):
        dates = dates_by_team.get(team_id)
        if dates is None:
            return 3
        i = np.searchsorted(dates, np.datetime64(game_date), side='left')
        if i == 0:
            return 3
        days = int((np.datetime64(game_date) - dates[i - 1]) / np.timedelta64(1, 'D'))
        return max(0, min(REST_CAP, days))

    df['home_rest_days'] = [rest_for(t, d) for t, d in zip(df['home_team_id'], df['game_date'])]
    df['away_rest_days'] = [rest_for(t, d) for t, d in zip(df['away_team_id'], df['game_date'])]
    df['rest_advantage']    = df['home_rest_days'] - df['away_rest_days']
    df['home_back_to_back'] = (df['home_rest_days'] == 1).astype(int)
    df['away_back_to_back'] = (df['away_rest_days'] == 1).astype(int)

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


def add_playoff_features(df):
    """Add playoff-specific context features."""
    print("  Computing playoff features...")

    df = df.sort_values('game_date').copy()

    df['is_playoff'] = (df['season_type'] == 'Playoffs').astype(int)

    series_history   = {}
    series_game_num  = []
    home_series_wins = []
    away_series_wins = []
    is_elimination   = []
    series_momentum  = []

    for _, row in df.iterrows():
        if row['season_type'] != 'Playoffs':
            series_game_num.append(0)
            home_series_wins.append(0)
            away_series_wins.append(0)
            is_elimination.append(0)
            series_momentum.append(0)
            continue

        key        = tuple(sorted([row['home_team_id'], row['away_team_id']]))
        season_key = (row['season'], key)

        # Wins are tracked per TEAM: home court alternates within a series, so
        # "wins by whoever was home" is not tonight's home team's series wins.
        if season_key not in series_history:
            series_history[season_key] = {'games': [], 'wins': {}}

        history  = series_history[season_key]
        game_num = len(history['games']) + 1
        h_wins   = history['wins'].get(row['home_team_id'], 0)
        a_wins   = history['wins'].get(row['away_team_id'], 0)

        series_game_num.append(game_num)
        home_series_wins.append(h_wins)
        away_series_wins.append(a_wins)
        is_elimination.append(int(h_wins == 3 or a_wins == 3))

        if len(history['games']) == 0:
            series_momentum.append(0)
        else:
            last_game = history['games'][-1]
            series_momentum.append(1 if last_game['home_won'] else -1)

        home_won = row['home_score'] > row['away_score']
        history['games'].append({'home_won': home_won})
        winner = row['home_team_id'] if home_won else row['away_team_id']
        history['wins'][winner] = history['wins'].get(winner, 0) + 1

    df['series_game_num']  = series_game_num
    df['home_series_wins'] = home_series_wins
    df['away_series_wins'] = away_series_wins
    df['is_elimination']   = is_elimination
    df['series_momentum']  = series_momentum
    df['series_pressure']  = df['home_series_wins'] + df['away_series_wins']
    df['series_wins_diff'] = df['home_series_wins'] - df['away_series_wins']
    return df

def add_quarter_features(df):
    """
    Add quarter-based performance features.
    Focus on 4th quarter clutch performance.
    """
    print("  Computing quarter features...")

    query = text("""
        SELECT
            tbs.team_id,
            tbs.q1_points,
            tbs.q2_points,
            tbs.q3_points,
            tbs.q4_points,
            tbs.points,
            g.game_date,
            -- Opponent Q4
            opp.q4_points AS opp_q4_points
        FROM team_box_scores tbs
        JOIN games g ON tbs.game_id = g.game_id
        JOIN team_box_scores opp
            ON opp.game_id = tbs.game_id
            AND opp.team_id != tbs.team_id
        WHERE g.is_final = TRUE
        AND tbs.q4_points IS NOT NULL
        ORDER BY tbs.team_id, g.game_date
    """)

    q_df = pd.read_sql(query, engine)
    q_df['game_date'] = pd.to_datetime(q_df['game_date'])
    df['game_date']   = pd.to_datetime(df['game_date'])

    home_q4_avg   = []
    away_q4_avg   = []
    home_q4_diff  = []
    away_q4_diff  = []
    home_q1_avg   = []
    away_q1_avg   = []
    home_clutch   = []
    away_clutch   = []

    for _, row in df.iterrows():
        game_date = row['game_date']
        home_id   = row['home_team_id']
        away_id   = row['away_team_id']

        for team_id, lists in [
            (home_id, (home_q4_avg, home_q4_diff,
                       home_q1_avg, home_clutch)),
            (away_id, (away_q4_avg, away_q4_diff,
                       away_q1_avg, away_clutch))
        ]:
            recent = q_df[
                (q_df['team_id'] == team_id) &
                (q_df['game_date'] < game_date)
            ].tail(10)

            if len(recent) < 3:
                lists[0].append(26.0)   # league avg Q4
                lists[1].append(0.0)
                lists[2].append(27.0)   # league avg Q1
                lists[3].append(0.50)
                continue

            q4_avg  = float(recent['q4_points'].mean())
            q4_diff = float(
                (recent['q4_points'] -
                 recent['opp_q4_points']).mean()
            )
            q1_avg  = float(recent['q1_points'].mean())

            # Clutch win rate — games won where Q4 margin <= 5
            close_games = recent[
                abs(recent['q4_points'] -
                    recent['opp_q4_points']) <= 5
            ]
            if len(close_games) >= 2:
                clutch_rate = float(
                    (close_games['q4_points'] >
                     close_games['opp_q4_points']).mean()
                )
            else:
                clutch_rate = 0.50

            lists[0].append(q4_avg)
            lists[1].append(q4_diff)
            lists[2].append(q1_avg)
            lists[3].append(clutch_rate)

    df['home_q4_avg']    = home_q4_avg
    df['away_q4_avg']    = away_q4_avg
    df['home_q4_diff']   = home_q4_diff
    df['away_q4_diff']   = away_q4_diff
    df['home_q1_avg']    = home_q1_avg
    df['away_q1_avg']    = away_q1_avg
    df['home_clutch']    = home_clutch
    df['away_clutch']    = away_clutch
    df['clutch_diff']    = (
        df['home_clutch'] - df['away_clutch']
    )
    df['q4_diff_spread'] = (
        df['home_q4_diff'] - df['away_q4_diff']
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

def add_playoff_elevation_features(df):
    """
    Add team playoff elevation scores as model features.
    Applies series context adjustments for playoff games.
    """
    print("  Computing playoff elevation features...")

    from data.ingestion.compute_playoff_factors import (
        compute_player_playoff_factors,
        compute_team_playoff_elevation,
        get_series_adjusted_elevations
    )

    # Compute base factors
    factor_dict, _ = compute_player_playoff_factors()
    if not factor_dict:
        df['home_playoff_elevation'] = 0.0
        df['away_playoff_elevation'] = 0.0
        df['playoff_elevation_diff'] = 0.0
        return df

    team_elevations = compute_team_playoff_elevation(factor_dict)

    home_elevations = []
    away_elevations = []

    for _, row in df.iterrows():
        if row['season_type'] != 'Playoffs':
            home_elevations.append(0.0)
            away_elevations.append(0.0)
            continue

        home_adj, away_adj = get_series_adjusted_elevations(
            team_elevations  = team_elevations,
            home_team_id     = row['home_team_id'],
            away_team_id     = row['away_team_id'],
            series_game_num  = int(row['series_game_num']) 
                               if row['series_game_num'] > 0 else 1,
            home_series_wins = int(row['home_series_wins']),
            away_series_wins = int(row['away_series_wins'])
        )

        home_elevations.append(home_adj)
        away_elevations.append(away_adj)

    df['home_playoff_elevation'] = home_elevations
    df['away_playoff_elevation'] = away_elevations
    df['playoff_elevation_diff'] = (
        df['home_playoff_elevation'] - df['away_playoff_elevation']
    )

    return df

def add_home_court_strength(df):
    """
    Compute how much stronger each team's home court is
    vs league average home court advantage.
    Some arenas (OKC, BOS, DEN) have measurably stronger impact.
    """
    print("  Computing home court strength...")

    from sqlalchemy import text
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


def add_scoring_variance(df):
    """
    Add scoring consistency features.
    High variance teams are harder to predict.
    Low variance teams are more consistent and predictable.
    """
    print("  Computing scoring variance...")

    query = text("""
        SELECT
            tbs.team_id,
            g.game_date,
            tbs.points,
            tbs.is_home
        FROM team_box_scores tbs
        JOIN games g ON tbs.game_id = g.game_id
        WHERE g.is_final = TRUE
        ORDER BY tbs.team_id, g.game_date
    """)

    scores_df = pd.read_sql(query, engine)
    scores_df['game_date'] = pd.to_datetime(scores_df['game_date'])
    df['game_date'] = pd.to_datetime(df['game_date'])

    home_var = []
    away_var = []
    home_consistency = []
    away_consistency = []

    for _, row in df.iterrows():
        game_date = row['game_date']
        home_id   = row['home_team_id']
        away_id   = row['away_team_id']

        # Last 10 games before this game
        home_recent = scores_df[
            (scores_df['team_id'] == home_id) &
            (scores_df['game_date'] < game_date)
        ].tail(10)

        away_recent = scores_df[
            (scores_df['team_id'] == away_id) &
            (scores_df['game_date'] < game_date)
        ].tail(10)

        # Scoring standard deviation
        h_std = float(home_recent['points'].std()) \
                if len(home_recent) >= 5 else 12.0
        a_std = float(away_recent['points'].std()) \
                if len(away_recent) >= 5 else 12.0

        # Consistency score — lower std = more consistent
        h_consistency = 1.0 / (1.0 + h_std)
        a_consistency = 1.0 / (1.0 + a_std)

        home_var.append(h_std)
        away_var.append(a_std)
        home_consistency.append(h_consistency)
        away_consistency.append(a_consistency)

    df['home_scoring_std']     = home_var
    df['away_scoring_std']     = away_var
    df['home_consistency']     = home_consistency
    df['away_consistency']     = away_consistency
    df['variance_differential'] = (
        df['home_scoring_std'] - df['away_scoring_std']
    )

    return df


def add_matchup_interactions(df):
    """
    Add interaction features between offensive and defensive styles.
    These directly affect predicted totals:
    - High 3pt rate offense vs good 3pt defense = lower scoring
    - High TOV forcing defense vs turnover prone offense = lower scoring
    - High FT rate offense vs foul prone defense = higher scoring
    """
    print("  Computing matchup interactions...")

    # 3pt matchup — home offense vs away defense
    # Positive = home team 3pt rate exceeds away team's 3pt defense
    # In build_features.py add_matchup_interactions function
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

    # Momentum — are they trending up or down recently?
    # Positive = scoring more in last 7 than last 10
    df['home_momentum'] = (
        df['home_avg_points_l7'] - df['home_avg_points']
    )
    df['away_momentum'] = (
        df['away_avg_points_l7'] - df['away_avg_points']
    )
    df['home_def_momentum'] = (
        df['home_def_rating'] - df['home_def_rating_l7']
    )
    df['away_def_momentum'] = (
        df['away_def_rating'] - df['away_def_rating_l7']
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

def add_shooting_consistency(df):
    """
    For each team compute:
    - FG% standard deviation (how consistent is their shooting?)
    - % of games below their average FG% by >5%
    - 3pt% variance (3pt teams are more boom/bust)
    - Free throw rate consistency
    """
    print("  Computing shooting consistency features...")

    query = text("""
        SELECT
            tbs.team_id,
            tbs.fg_pct,
            tbs.fg3_pct,
            tbs.ft_pct,
            tbs.ft_rate,
            tbs.points,
            g.game_date
        FROM team_box_scores tbs
        JOIN games g ON tbs.game_id = g.game_id
        WHERE g.is_final = TRUE
        ORDER BY tbs.team_id, g.game_date
    """)

    shoot_df = pd.read_sql(query, engine)
    shoot_df['game_date'] = pd.to_datetime(shoot_df['game_date'])
    df['game_date'] = pd.to_datetime(df['game_date'])

    home_fg_std       = []
    away_fg_std       = []
    home_3pt_std      = []
    away_3pt_std      = []
    home_bad_night_pct = []
    away_bad_night_pct = []
    home_ft_rate_std  = []
    away_ft_rate_std  = []

    for _, row in df.iterrows():
        game_date = row['game_date']
        home_id   = row['home_team_id']
        away_id   = row['away_team_id']

        for team_id, lists in [
            (home_id, (home_fg_std, home_3pt_std,
                       home_bad_night_pct, home_ft_rate_std)),
            (away_id, (away_fg_std, away_3pt_std,
                       away_bad_night_pct, away_ft_rate_std))
        ]:
            recent = shoot_df[
                (shoot_df['team_id'] == team_id) &
                (shoot_df['game_date'] < game_date)
            ].tail(50)  # last 20 games

            if len(recent) < 5:
                lists[0].append(0.05)   # default fg std
                lists[1].append(0.06)   # default 3pt std
                lists[2].append(0.25)   # default bad night %
                lists[3].append(0.02)   # default ft rate std
                continue

            fg_vals  = recent['fg_pct'].dropna()
            fg3_vals = recent['fg3_pct'].dropna()
            ft_vals  = recent['ft_rate'].dropna()
            pts_vals = recent['points'].dropna()

            fg_std  = float(fg_vals.std())  if len(fg_vals)  > 2 else 0.05
            fg3_std = float(fg3_vals.std()) if len(fg3_vals) > 2 else 0.06
            ft_std  = float(ft_vals.std())  if len(ft_vals)  > 2 else 0.02

            # Bad shooting night = FG% more than 1 std below average
            fg_mean = float(fg_vals.mean())
            bad_nights = (fg_vals < fg_mean - fg_std).sum()
            bad_night_pct = float(bad_nights / len(fg_vals))

            lists[0].append(fg_std)
            lists[1].append(fg3_std)
            lists[2].append(bad_night_pct)
            lists[3].append(ft_std)

    df['home_fg_consistency']    = home_fg_std
    df['away_fg_consistency']    = away_fg_std
    df['home_3pt_consistency']   = home_3pt_std
    df['away_3pt_consistency']   = away_3pt_std
    df['home_bad_night_pct']     = home_bad_night_pct
    df['away_bad_night_pct']     = away_bad_night_pct
    df['home_ft_rate_std']       = home_ft_rate_std
    df['away_ft_rate_std']       = away_ft_rate_std

    # Boom/bust differential
    # High = one team is much more volatile than the other
    df['shooting_volatility_diff'] = (
        df['home_fg_consistency'] - df['away_fg_consistency']
    )

    return df
def add_advanced_rolling_features(df):
    """
    Rolling averages of oreb_pct, tov_pct, efg_pct from box scores (last 10 games).
    All three columns are 100% populated in team_box_scores.
    """
    print("  Computing advanced rolling features (oreb/tov/efg)...")

    query = text("""
        SELECT tbs.team_id, g.game_date, tbs.oreb_pct, tbs.tov_pct, tbs.efg_pct
        FROM team_box_scores tbs
        JOIN games g ON tbs.game_id = g.game_id
        WHERE g.is_final = TRUE
        AND tbs.oreb_pct IS NOT NULL
        ORDER BY tbs.team_id, g.game_date
    """)

    adv_df = pd.read_sql(query, engine)
    adv_df['game_date'] = pd.to_datetime(adv_df['game_date'])
    df['game_date'] = pd.to_datetime(df['game_date'])

    home_oreb, away_oreb = [], []
    home_tov,  away_tov  = [], []
    home_efg,  away_efg  = [], []

    LEAGUE_OREB = 0.29
    LEAGUE_TOV  = 14.1
    LEAGUE_EFG  = 0.545

    for _, row in df.iterrows():
        game_date = row['game_date']
        home_id   = row['home_team_id']
        away_id   = row['away_team_id']

        for team_id, o_list, t_list, e_list in [
            (home_id, home_oreb, home_tov, home_efg),
            (away_id, away_oreb, away_tov, away_efg),
        ]:
            recent = adv_df[
                (adv_df['team_id'] == team_id) &
                (adv_df['game_date'] < game_date)
            ].tail(10)

            if len(recent) < 3:
                o_list.append(LEAGUE_OREB)
                t_list.append(LEAGUE_TOV)
                e_list.append(LEAGUE_EFG)
            else:
                o_list.append(float(recent['oreb_pct'].mean()))
                t_list.append(float(recent['tov_pct'].mean()))
                e_list.append(float(recent['efg_pct'].mean()))

    df['home_oreb_pct'] = home_oreb
    df['away_oreb_pct'] = away_oreb
    df['home_tov_pct']  = home_tov
    df['away_tov_pct']  = away_tov
    df['home_efg_pct']  = home_efg
    df['away_efg_pct']  = away_efg

    return df


# Weight (in games) the rolling average gets when blended with in-series form.
# A 1-2 game series average is mostly noise, so it's shrunk toward the rolling avg.
SERIES_SHRINK_K = 3


def shrink_series_avg(series_avg, n_games, rolling_avg):
    """Blend an n-game in-series average with the team's rolling average."""
    return (n_games * series_avg + SERIES_SHRINK_K * rolling_avg) / (n_games + SERIES_SHRINK_K)


def add_series_form_features(df):
    """
    For each playoff game, compute the avg scores from games already
    played in this series, shrunk toward the rolling avg by series length.
    Game 1 falls back to rolling avg.
    Regular-season rows get their rolling avg as a neutral fill.
    """
    print("  Computing in-series form features...")

    df = df.sort_values('game_date').copy()
    series_history = {}
    in_series_home = []
    in_series_away = []

    for _, row in df.iterrows():
        home_id = int(row['home_team_id'])
        away_id = int(row['away_team_id'])

        if row['season_type'] != 'Playoffs':
            in_series_home.append(float(row['home_avg_points']))
            in_series_away.append(float(row['away_avg_points']))
            continue

        season = row['season']
        key = (season, tuple(sorted([home_id, away_id])))
        history = series_history.get(key, [])

        if history:
            home_scores = []
            away_scores = []
            for g in history:
                if g['home_id'] == home_id:
                    home_scores.append(g['home_score'])
                    away_scores.append(g['away_score'])
                else:
                    home_scores.append(g['away_score'])
                    away_scores.append(g['home_score'])
            n = len(home_scores)
            in_series_home.append(shrink_series_avg(
                sum(home_scores) / n, n, float(row['home_avg_points'])))
            in_series_away.append(shrink_series_avg(
                sum(away_scores) / n, n, float(row['away_avg_points'])))
        else:
            in_series_home.append(float(row['home_avg_points']))
            in_series_away.append(float(row['away_avg_points']))

        if key not in series_history:
            series_history[key] = []
        series_history[key].append({
            'home_id': home_id,
            'away_id': away_id,
            'home_score': float(row['home_score']),
            'away_score': float(row['away_score']),
        })

    df['in_series_home_avg_pts'] = in_series_home
    df['in_series_away_avg_pts'] = in_series_away
    df['in_series_total_avg'] = (
        df['in_series_home_avg_pts'] + df['in_series_away_avg_pts']
    )
    df['in_series_margin'] = (
        df['in_series_home_avg_pts'] - df['in_series_away_avg_pts']
    )

    return df


if __name__ == "__main__":
    df = build_feature_dataset(window=10)

    print("\nFeature dataset sample:")
    print(df[[
        'game_date', 'home_team', 'away_team',
        'home_score', 'away_score',
        'home_avg_points', 'away_avg_points',
        'home_rest_days', 'away_rest_days',
        'is_playoff', 'series_game_num',
        'series_momentum'
    ]].tail(10))

    print(f"\nTotal features: {len(df.columns)}")
    print(f"Total games:    {len(df)}")
    print(f"\nMissing values:\n"
          f"{df.isnull().sum()[df.isnull().sum() > 0]}")

    df.to_csv('data/features.csv', index=False)
    print("\n✅ Features saved to data/features.csv")