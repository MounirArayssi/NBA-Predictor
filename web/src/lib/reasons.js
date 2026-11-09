// Short "primary read" for a forecast: the inputs that moved its projected margin
// most (exact SHAP contributions from explanations.json, summed per input), and a
// coin-flip flag for near-even games. Only forecasts that reproduce exactly have
// reasons; everything else returns [].
import ex from "../data/explanations.json";

export const COIN_FLIP = 0.55; // favorite below this = coin flip

export const REASON_GROUPS = [
  ["Elo", ["home_elo", "away_elo", "elo_diff"]],
  ["Season margin", ["home_season_margin", "away_season_margin"]],
  ["Sample size", ["home_games_played", "away_games_played"]],
  ["Offense, last 10", ["home_off_rating", "away_off_rating"]],
  ["Defense, last 10", ["home_def_rating", "away_def_rating"]],
  ["Pace", ["combined_pace"]],
  ["Injuries", ["home_missing_pts", "away_missing_pts"]],
  ["Rest", ["home_rest_days", "away_rest_days", "home_back_to_back", "away_back_to_back"]],
];

/** Top reasons for game g: [{label, pts, team}], largest first, |pts| >= min. */
export function reasons(g, n = 2, min = 0.5) {
  const e = ex.games[g.id];
  if (!e?.ok) return [];
  return REASON_GROUPS.map(([label, keys]) => {
    const v = keys.reduce((s, k) => s + (e.c[k]?.[2] ?? 0), 0);
    return { label, pts: Math.abs(v), team: v >= 0 ? g.home : g.away };
  }).filter((r) => r.pts >= min).sort((a, b) => b.pts - a.pts).slice(0, n);
}

export const isCoinFlip = (fc) => fc != null && Math.max(fc.hwp, 1 - fc.hwp) < COIN_FLIP;
