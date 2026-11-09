// Season records and win/loss streaks going into a date, computed from the final
// results in games.json. Nothing here is stored: every rebuild after new results
// are exported recomputes them, so they update as soon as a game is posted.
// Records count regular-season games only; streaks run across the whole season
// (regular season and playoffs) but never carry over from last season.
import { games } from "./data.js";

export const STREAK_MIN = 4;

// season|team -> finals sorted oldest first: { date, won, reg }
const byTeam = new Map();
for (const g of games) {
  if (g.status !== "final" || !g.score) continue;
  for (const [abbr, home] of [[g.home, true], [g.away, false]]) {
    const k = `${g.season}|${abbr}`;
    if (!byTeam.has(k)) byTeam.set(k, []);
    byTeam.get(k).push({ date: g.date, tip: g.tip || "", won: home === g.score.h > g.score.a, reg: g.stype === "Regular Season" });
  }
}
for (const list of byTeam.values()) list.sort((a, b) => (a.date + a.tip).localeCompare(b.date + b.tip));

const before = (abbr, season, date) => (byTeam.get(`${season}|${abbr}`) || []).filter((x) => x.date < date);

/** "W–L" in the regular season before `date`. */
export function record(abbr, season, date) {
  let w = 0, l = 0;
  for (const x of before(abbr, season, date)) if (x.reg) x.won ? w++ : l++;
  return `${w}–${l}`;
}

/** Current streak before `date`: { n, won } (n = 0 if no games). */
export function streak(abbr, season, date) {
  const list = before(abbr, season, date);
  if (!list.length) return { n: 0, won: false };
  const won = list.at(-1).won;
  let n = 0;
  for (let i = list.length - 1; i >= 0 && list[i].won === won; i--) n++;
  return { n, won };
}
