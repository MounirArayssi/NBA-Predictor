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
    This prevents small fluctuations from being treated as real changes.
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
    # 30ppg player: ignores differences < 2.4pts
    # 15ppg player: ignores differences < 1.2pts
    merged['points_factor'] = merged.apply(
        lambda r: safe_ratio(
            r['avg_points_ply'],
            r['avg_points_reg'],
            pct_threshold=0.08
        ), axis=1
    ).clip(0.5, 2.0)

    # True shooting — 5% threshold
    merged['ts_factor'] = merged.apply(
        lambda r: safe_ratio(
            r['avg_ts_ply'],
            r['avg_ts_reg'],
            pct_threshold=0.05
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

    # Turnovers — penalize only meaningful increases (10% threshold)
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

    # Plus/minus — 10% threshold
    merged['pm_factor'] = merged.apply(
        lambda r: 1.0 if abs(
            r['avg_plus_minus_ply'] - r['avg_plus_minus_reg']
        ) < 2.0  # ignore < 2pt PM difference
        else 1.0 + (
            (r['avg_plus_minus_ply'] - r['avg_plus_minus_reg']) / 20
        ),
        axis=1
    ).clip(0.85, 1.15)

    # Assists — 10% threshold
    merged['ast_factor'] = merged.apply(
        lambda r: safe_ratio(
            r['avg_ast_ply'],
            r['avg_ast_reg'],
            pct_threshold=0.10
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

    # FG% — 5% threshold
    merged['fg_factor'] = merged.apply(
        lambda r: safe_ratio(
            r['avg_fg_pct_ply'],
            r['avg_fg_pct_reg'],
            pct_threshold=0.05
        ), axis=1
    ).clip(0.7, 1.3)

    # Defensive impact (steals + blocks) — 10% threshold
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
    # Points: 25%, TS: 20%, Usage: 10%, TOV: 10%
    # PM: 10%, AST: 10%, REB: 5%, FG%: 5%, DEF: 5%
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

    # Show Jokic specifically
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


def compute_team_playoff_elevation(factor_dict):
    """
    Aggregate player playoff factors to team level.
    Weighted by usage rate so stars matter more than bench players.
    Asymmetric cap: max +3 reward, max -1.5 penalty.
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
        raw_adj = (weighted_factor - 1.0) * avg_team_pts * 0.4

        # Asymmetric cap — more room to reward elevation than penalize decline
        capped_adj = float(np.clip(raw_adj, -1.5, 3.0))
        team_elevations[int(team_id)] = capped_adj

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


if __name__ == "__main__":
    factor_dict, player_df = compute_player_playoff_factors()
    if factor_dict:
        team_elevations = compute_team_playoff_elevation(factor_dict)
    print("\n✅ Playoff factors computed")