// Build-time data access. Everything here comes from web/scripts/export_data.py,
// which reads the database read-only. Nothing in the site computes predictions.
import site from "../data/site.json";
import teams from "../data/teams.json";
import games from "../data/games.json";
import performance from "../data/performance.json";
import teamAnalytics from "../data/team_analytics.json";

export { site, teams, games, performance, teamAnalytics };

export const TITLE = "NBA Forecasts";
export const TZ = "America/New_York";

export const teamByAbbr = Object.fromEntries(teams.map((t) => [t.abbr, t]));
export const gameById = Object.fromEntries(games.map((g) => [g.id, g]));

export const teamUrl = (abbr) => `/teams/${teamByAbbr[abbr]?.slug ?? abbr.toLowerCase()}/`;
export const gameUrl = (id) => `/games/${id}/`;

/* ---------- formatting (all display times in US Eastern, labeled ET) ---------- */
const dFmt = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", weekday: "short", month: "short", day: "numeric", year: "numeric" });
const dShort = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", month: "short", day: "numeric" });
const dLong = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", weekday: "long", month: "long", day: "numeric", year: "numeric" });
const tFmt = new Intl.DateTimeFormat("en-US", { timeZone: TZ, hour: "numeric", minute: "2-digit" });
const dtFmt = new Intl.DateTimeFormat("en-US", { timeZone: TZ, month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" });

const asDate = (ymd) => new Date(`${ymd}T12:00:00Z`);
export const fmtDate = (ymd) => dFmt.format(asDate(ymd));
export const fmtShortDate = (ymd) => dShort.format(asDate(ymd));
export const fmtLongDate = (ymd) => dLong.format(asDate(ymd));
export const fmtTime = (iso) => (iso ? `${tFmt.format(new Date(iso))} ET` : null);
export const fmtDateTime = (iso) => (iso ? `${dtFmt.format(new Date(iso))} ET` : null);
export const pct = (x, d = 0) => (x == null ? "–" : `${(100 * x).toFixed(d)}%`);
export const pct100 = (x, d = 1) => (x == null ? "–" : `${Number(x).toFixed(d)}%`);
export const signed = (x, d = 1) => (x == null ? "–" : `${x > 0 ? "+" : x < 0 ? "−" : "±"}${Math.abs(x).toFixed(d)}`);
export const fixed = (x, d = 1) => (x == null ? "–" : Number(x).toFixed(d));
export const round0 = (x) => (x == null ? "–" : String(Math.round(x)));

/** ET calendar date (YYYY-MM-DD) of an instant. */
export function etDate(d = new Date()) {
  const p = new Intl.DateTimeFormat("en-CA", { timeZone: TZ, year: "numeric", month: "2-digit", day: "2-digit" }).formatToParts(d);
  const get = (t) => p.find((x) => x.type === t).value;
  return `${get("year")}-${get("month")}-${get("day")}`;
}

/** Which team a forecast picks, and with what probability. */
export function favorite(pred, g) {
  if (!pred) return null;
  const homeFav = pred.ph > pred.pa || (pred.ph === pred.pa && pred.hwp >= 0.5);
  return { abbr: homeFav ? g.home : g.away, p: homeFav ? pred.hwp : 1 - pred.hwp, home: homeFav };
}


export const seasonGames = (season) => games.filter((g) => g.season === season);
