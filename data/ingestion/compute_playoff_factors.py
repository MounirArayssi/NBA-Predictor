import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from sqlalchemy import text
from data.storage.db import engine


def safe_ratio(ply_val, reg_val, pct_threshold=0.08):
    """
    Compute ratio between playoff and regular season value.
    Returns 1.0 (neutral) if difference is within threshold %.
    Prevents small fluctuations from being treated as real changes.
    """
    if pd.isna(reg_val) or reg_val == 0:
        return 1.0
    if abs(ply_val - reg_val) / abs(reg_val) < pct_threshold:
        return 1.0
    return float(ply_val / reg_val)


def compute_player_playoff_factors():
    """
    Compute playoff elevation factors aggregated across all seasons.
    One factor per player, weighted by sample size and playoff minutes.

    playoff_elevation > 1.0 = player elevates in playoffs
    playoff_elevation < 1.0 = player declines in playoffs
    """
    print("Computing player playoff elevation factors...")

    query = text("""
        SELECT
            pbs.player_id,
            MAX(p.full_name)                        AS full_name,
            g.season_type,
            AVG(pbs.points)                         AS avg_points,
            AVG(COALESCE(pbs.usage_rate, 0.2))      AS avg_usage,
            AVG(COALESCE(pbs.true_shooting, 0.55))  AS avg_ts,
            AVG(COALESCE(pbs.turnovers, 2.0))       AS avg_tov,
            AVG(COALESCE(pbs.plus_minus, 0.0))      AS avg_plus_minus,
            AVG(pbs.minutes_played)                  AS avg_minutes,
            AVG(COALESCE(pbs.assists, 0.0))         AS avg_ast,
            AVG(COALESCE(pbs.rebounds, 0.0))        AS avg_reb,
            AVG(COALESCE(pbs.steals, 0.0))          AS avg_stl,
            AVG(COALESCE(pbs.blocks, 0.0))          AS avg_blk,
            AVG(COALESCE(pbs.fg_pct, 0.45))         AS avg_fg_pct,
            COUNT(*)                                 AS games_played
        FROM player_box_scores pbs
        JOIN players p ON pbs.player_id = p.player_id
        JOIN games g ON pbs.game_id = g.game_id
        WHERE pbs.minutes_played >= 15
        AND pbs.points IS NOT NULL
        GROUP BY pbs.player_id, g.season_type
    """)

    df = pd.read_sql(query, engine)

    reg = df[df['season_type'] == 'Regular Season'].copy()
    ply = df[df['season_type'] == 'Playoffs'].copy()

    # Filter — exclude true end-of-bench garbage time players
    reg = reg[
        (reg['avg_points'] >= 7) &
        (reg['avg_minutes'] >= 9.5) &
        (reg['games_played'] >= 25)
    ]

    # Filter — minimum playoff sample
    ply = ply[ply['games_played'] >= 5]

    merged = reg.merge(ply, on='player_id', suffixes=('_reg', '_ply'))

    if merged.empty:
        print("  ⚠️  No players with sufficient data")
        return {}, pd.DataFrame()

    merged = merged.drop_duplicates(subset=['player_id'])
    print(f"  Qualified players: {len(merged)}")

    # --- Elevation metrics with percentage-based thresholds ---

    # Points — 8% threshold
    merged['points_factor'] = merged.apply(
        lambda r: safe_ratio(
            r['avg_points_ply'],
            r['avg_points_reg'],
            pct_threshold=0.08
        ), axis=1
    ).clip(0.5, 2.0)

    # True shooting — 8% threshold
    merged['ts_factor'] = merged.apply(
        lambda r: safe_ratio(
            r['avg_ts_ply'],
            r['avg_ts_reg'],
            pct_threshold=0.08
        ), axis=1
    ).clip(0.7, 1.3)

    # Usage — 8% threshold
    merged['usage_factor'] = merged.apply(
        lambda r: safe_ratio(
            r['avg_usage_ply'],
            r['avg_usage_reg'],
            pct_threshold=0.08
        ), axis=1
    ).clip(0.7, 1.5)

    # Turnovers — penalize only meaningful increases
    merged['tov_factor'] = merged.apply(
        lambda r: 1.0 if abs(
            r['avg_tov_ply'] - r['avg_tov_reg']
        ) / max(r['avg_tov_reg'], 0.5) < 0.10
        else 1.0 - (
            (r['avg_tov_ply'] - r['avg_tov_reg']) /
            max(r['avg_tov_reg'], 0.5)
        ) * 0.15,
        axis=1
    ).clip(0.85, 1.15)

    # Plus/minus — ignore < 2pt difference
    merged['pm_factor'] = merged.apply(
        lambda r: 1.0 if abs(
            r['avg_plus_minus_ply'] - r['avg_plus_minus_reg']
        ) < 2.0
        else 1.0 + (
            (r['avg_plus_minus_ply'] - r['avg_plus_minus_reg']) / 20
        ),
        axis=1
    ).clip(0.85, 1.15)

    # Assists — 15% threshold
    merged['ast_factor'] = merged.apply(
        lambda r: safe_ratio(
            r['avg_ast_ply'],
            r['avg_ast_reg'],
            pct_threshold=0.15
        ), axis=1
    ).clip(0.7, 1.5)

    # Rebounds — 10% threshold
    merged['reb_factor'] = merged.apply(
        lambda r: safe_ratio(
            r['avg_reb_ply'],
            r['avg_reb_reg'],
            pct_threshold=0.10
        ), axis=1
    ).clip(0.7, 1.5)

    # FG% — 8% threshold
    merged['fg_factor'] = merged.apply(
        lambda r: safe_ratio(
            r['avg_fg_pct_ply'],
            r['avg_fg_pct_reg'],
            pct_threshold=0.08
        ), axis=1
    ).clip(0.7, 1.3)

    # Defensive impact — 10% threshold
    merged['def_factor'] = merged.apply(
        lambda r: safe_ratio(
            r['avg_stl_ply'] + r['avg_blk_ply'],
            r['avg_stl_reg'] + r['avg_blk_reg'],
            pct_threshold=0.10
        ), axis=1
    ).clip(0.7, 1.5)

    # --- Sample size weighting ---
    # Combines games played AND playoff minutes
    games_weight = (
        np.log1p(merged['games_played_ply']) / np.log1p(50)
    ) ** 0.7

    minutes_weight = (
        merged['avg_minutes_ply'].clip(15, 40) - 15
    ) / 25

    merged['sample_weight'] = (
        games_weight * 0.6 + minutes_weight * 0.4
    ).clip(0, 1)

    # Combined elevation score
    merged['raw_elevation'] = (
        merged['points_factor']  * 0.25 +
        merged['ts_factor']      * 0.20 +
        merged['usage_factor']   * 0.10 +
        merged['tov_factor']     * 0.10 +
        merged['pm_factor']      * 0.10 +
        merged['ast_factor']     * 0.10 +
        merged['reb_factor']     * 0.05 +
        merged['fg_factor']      * 0.05 +
        merged['def_factor']     * 0.05
    )

    # Blend toward neutral based on sample size
    merged['playoff_elevation'] = (
        merged['raw_elevation'] * merged['sample_weight'] +
        1.0 * (1 - merged['sample_weight'])
    )

    print("\n  Top 15 playoff elevators:")
    top = merged.nlargest(15, 'playoff_elevation')[[
        'full_name_reg', 'avg_points_reg', 'avg_points_ply',
        'playoff_elevation', 'games_played_ply', 'sample_weight'
    ]]
    print(top.to_string(index=False))

    print("\n  Top 15 playoff decliners:")
    bottom = merged.nsmallest(15, 'playoff_elevation')[[
        'full_name_reg', 'avg_points_reg', 'avg_points_ply',
        'playoff_elevation', 'games_played_ply', 'sample_weight'
    ]]
    print(bottom.to_string(index=False))

    # Show Jokic
    jokic = merged[merged['player_id'] == 237]
    if not jokic.empty:
        print(f"\n  Jokic playoff elevation: "
              f"{jokic['playoff_elevation'].values[0]:.4f} "
              f"(sample weight: {jokic['sample_weight'].values[0]:.3f})")
        print(f"  Jokic factors: "
              f"pts={jokic['points_factor'].values[0]:.3f} "
              f"ts={jokic['ts_factor'].values[0]:.3f} "
              f"ast={jokic['ast_factor'].values[0]:.3f} "
              f"reb={jokic['reb_factor'].values[0]:.3f} "
              f"pm={jokic['pm_factor'].values[0]:.3f}")

    factor_dict = dict(zip(
        merged['player_id'],
        merged['playoff_elevation']
    ))

    return factor_dict, merged


def apply_series_context(base_elevation, series_game_num,
                         team_won_last, is_elimination,
                         team_series_wins, opp_series_wins):
    """
    Adjust team elevation based on series context.

    Game 1: 50% weight — unknown matchup, no adjustments made yet
    Game 2: 70% weight — one game of data, partial adjustments
    Game 3: 85% weight — coaching adjustments baked in
    Game 4+: 100% weight — full playoff DNA showing

    Momentum: winning last game adds slight boost
    Elimination: amplifies elevation signal
    Series deficit: team down 0-2 plays desperate, more volatile
    """
    # Game number weight
    if series_game_num == 1:
        game_weight = 0.50
    elif series_game_num == 2:
        game_weight = 0.70
    elif series_game_num == 3:
        game_weight = 0.85
    else:
        game_weight = 1.00

    # Apply game weight to elevation
    adjusted = base_elevation * game_weight

    # Momentum signal — won last game = slight confidence boost
    if team_won_last and series_game_num > 1:
        adjusted += 0.15
    elif not team_won_last and series_game_num > 1:
        adjusted -= 0.05

    # Elimination game — pressure amplifies playoff DNA
    if is_elimination:
        if base_elevation > 0:
            adjusted *= 1.2   # good playoff teams elevate more
        else:
            adjusted *= 0.8   # bad playoff teams don't collapse as much

    # Series deficit — team down 0-2 plays desperate
    # Creates more variance, slightly unpredictable
    series_deficit = opp_series_wins - team_series_wins
    if series_deficit >= 2:
        adjusted *= 0.85  # discount elevation when down badly

    return adjusted


def compute_team_playoff_elevation(factor_dict,
                                   series_game_num=1,
                                   home_won_last=False,
                                   away_won_last=False,
                                   is_elimination=False,
                                   home_series_wins=0,
                                   away_series_wins=0):
    """
    Aggregate player playoff factors to team level.
    Applies series context adjustments on top of base elevation.
    Conservative scaling (0.2) and asymmetric cap (-1.5 to +3).
    """
    print("\nComputing team playoff elevation scores...")

    query = text("""
        SELECT
            pbs.player_id,
            pbs.team_id,
            t.abbreviation,
            AVG(pbs.usage_rate)     AS avg_usage,
            AVG(pbs.points)         AS avg_points,
            AVG(pbs.minutes_played) AS avg_minutes,
            COUNT(*)                AS games
        FROM player_box_scores pbs
        JOIN teams t ON pbs.team_id = t.team_id
        JOIN games g ON pbs.game_id = g.game_id
        WHERE g.season = '2025-26'
        AND pbs.minutes_played >= 15
        AND pbs.usage_rate IS NOT NULL
        GROUP BY pbs.player_id, pbs.team_id, t.abbreviation
        HAVING COUNT(*) >= 10
        AND AVG(pbs.points) >= 6
        AND AVG(pbs.minutes_played) >= 9.5
    """)

    roster_df = pd.read_sql(query, engine)

    team_elevations = {}
    team_coverage   = {}

    for team_id in roster_df['team_id'].unique():
        team_players = roster_df[
            roster_df['team_id'] == team_id
        ].copy()

        team_players['playoff_factor'] = team_players['player_id'].map(
            factor_dict
        )

        total_usage = team_players['avg_usage'].sum()
        if total_usage == 0:
            team_elevations[int(team_id)] = 0.0
            team_coverage[int(team_id)]   = 0.0
            continue

        team_players['weight'] = (
            team_players['avg_usage'] / total_usage
        )

        has_history = team_players['playoff_factor'].notna()
        coverage = float(
            team_players.loc[has_history, 'weight'].sum()
        )
        team_coverage[int(team_id)] = coverage

        team_players['playoff_factor'] = (
            team_players['playoff_factor'].fillna(1.0)
        )

        weighted_factor = float(
            (team_players['playoff_factor'] * team_players['weight']).sum()
        )

        avg_team_pts = float(team_players['avg_points'].sum())

        # More conservative scaling — 0.2 instead of 0.4
        raw_adj = (weighted_factor - 1.0) * avg_team_pts * 0.2

        # Asymmetric cap
        base_elevation = float(np.clip(raw_adj, -1.5, 3.0))
        team_elevations[int(team_id)] = base_elevation

    # Apply series context to home and away teams
    # We store base elevations and apply context at prediction time
    # This function returns base elevations for training
    # At prediction time, apply_series_context() adjusts them

    abbrev_map = dict(zip(
        roster_df['team_id'].astype(int),
        roster_df['abbreviation']
    ))

    elevation_df = pd.DataFrame([
        {
            'team':      abbrev_map.get(t, str(t)),
            'elevation': round(v, 3),
            'coverage':  round(team_coverage.get(t, 0), 2)
        }
        for t, v in team_elevations.items()
    ]).sort_values('elevation', ascending=False)

    print("\n  Team playoff elevation (-1.5 to +3 cap) | coverage = % roster with history")
    print(elevation_df.to_string(index=False))

    return team_elevations


def get_series_adjusted_elevations(team_elevations,
                                   home_team_id, away_team_id,
                                   series_game_num,
                                   home_series_wins, away_series_wins):
    """
    Apply series context to get game-specific elevation adjustments.
    Called at prediction time for upcoming games.

    Returns adjusted elevations for home and away teams.
    """
    home_base = team_elevations.get(int(home_team_id), 0.0)
    away_base = team_elevations.get(int(away_team_id), 0.0)

    # Who won the last game?
    # If home has more series wins they likely won last
    home_won_last = home_series_wins > away_series_wins
    away_won_last = away_series_wins > home_series_wins

    # Is it an elimination game?
    is_elimination = (home_series_wins == 3 or away_series_wins == 3)

    home_adjusted = apply_series_context(
        base_elevation   = home_base,
        series_game_num  = series_game_num,
        team_won_last    = home_won_last,
        is_elimination   = is_elimination,
        team_series_wins = home_series_wins,
        opp_series_wins  = away_series_wins
    )

    away_adjusted = apply_series_context(
        base_elevation   = away_base,
        series_game_num  = series_game_num,
        team_won_last    = away_won_last,
        is_elimination   = is_elimination,
        team_series_wins = away_series_wins,
        opp_series_wins  = home_series_wins
    )

    return home_adjusted, away_adjusted


if __name__ == "__main__":
    factor_dict, player_df = compute_player_playoff_factors()

    if factor_dict:
        team_elevations = compute_team_playoff_elevation(factor_dict)

        # Example: show series-adjusted elevations for a game 3
        # BOS (home, up 2-0) vs PHI (away, down 0-2)
        print("\n  Example series context adjustments:")
        print("  BOS vs PHI — Game 3, BOS leads 2-0")

        # Find BOS and PHI team IDs
        from data.storage.models import Team
        from sqlalchemy.orm import Session
        from data.storage.db import engine as db_engine

        with Session(db_engine) as session:
            bos = session.query(Team).filter_by(abbreviation='BOS').first()
            phi = session.query(Team).filter_by(abbreviation='PHI').first()

            if bos and phi:
                home_adj, away_adj = get_series_adjusted_elevations(
                    team_elevations  = team_elevations,
                    home_team_id     = bos.team_id,
                    away_team_id     = phi.team_id,
                    series_game_num  = 3,
                    home_series_wins = 2,
                    away_series_wins = 0
                )
                print(f"  BOS elevation: {team_elevations.get(bos.team_id, 0):.3f} "
                      f"→ series adjusted: {home_adj:.3f}")
                print(f"  PHI elevation: {team_elevations.get(phi.team_id, 0):.3f} "
                      f"→ series adjusted: {away_adj:.3f}")

    print("\n✅ Playoff factors computed")