"""
B — Recent performance: how each team has played over its last few games.
Momentum (7- vs 10-game), scoring variance, shooting consistency, quarter
splits / clutch, and rolling four-factor rates (oreb, tov, efg).
"""
import pandas as pd
from sqlalchemy import text
from data.storage.db import engine


def add_momentum_features(df):
    """
    Momentum — are they trending up or down recently?
    Positive = scoring more in last 7 than last 10 (defense: allowing less).
    """
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
