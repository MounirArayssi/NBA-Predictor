// Post-build checks on dist/ (run after `npm run build`):
//  1. every internal link and asset resolves to a built file
//  2. no secrets or connection strings in the output
//  3. data consistency between the exported datasets
import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join, extname } from "node:path";

const DIST = new URL("../dist/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const SRC = new URL("../src/data/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const PUB = new URL("../public/data/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
let fails = 0;
const fail = (m) => { fails++; console.log("FAIL ", m); };
const ok = (m) => console.log("PASS ", m);

const files = [];
(function walk(d) { for (const f of readdirSync(d)) { const p = join(d, f); statSync(p).isDirectory() ? walk(p) : files.push(p); } })(DIST);
const html = files.filter((f) => f.endsWith(".html"));

// 1. Links
const resolve = (href) => {
  const path = href.split("#")[0].split("?")[0];
  if (!path) return true;
  const p = join(DIST, decodeURIComponent(path));
  return existsSync(p) && (statSync(p).isFile() || existsSync(join(p, "index.html")));
};
let checked = 0; const broken = new Set();
for (const f of html) {
  const s = readFileSync(f, "utf8");
  for (const m of s.matchAll(/(?:href|src)="(\/[^"]*)"/g)) {
    if (m[1].includes("${")) continue; // client-side templates
    checked++;
    if (!resolve(m[1])) broken.add(`${m[1]}  (in ${f.slice(DIST.length)})`);
  }
}
broken.size ? [...broken].slice(0, 20).forEach((b) => fail(`broken link ${b}`)) : ok(`${checked} internal links across ${html.length} pages resolve`);

// 2. Secrets
const patterns = [/postgres(?:ql)?:\/\//i, /DATABASE_URL/, /DB_PASSWORD/, /ODDS_API_KEY/, /apiKey=/i, /BEARER_TOKEN|CONSUMER_SECRET|ACCESS_TOKEN_SECRET/i];
const text = files.filter((f) => [".html", ".js", ".json", ".css", ".xml", ".txt"].includes(extname(f)));
const hits = [];
for (const f of text) { const s = readFileSync(f, "utf8"); for (const p of patterns) if (p.test(s)) hits.push(`${f.slice(DIST.length)} matches ${p}`); }
hits.length ? hits.forEach(fail) : ok(`no secret patterns in ${text.length} output files`);

// 3. Data consistency
const games = JSON.parse(readFileSync(join(SRC, "games.json"), "utf8"));
const ta = JSON.parse(readFileSync(join(SRC, "team_analytics.json"), "utf8"));
const ids = new Set();
let bad = 0;
for (const g of games) {
  if (ids.has(g.id)) { fail(`duplicate game ${g.id}`); bad++; }
  ids.add(g.id);
  if (g.home === g.away) { fail(`game ${g.id} home == away`); bad++; }
  for (const p of [g.fc].filter(Boolean)) {
    if (!(p.hwp >= 0 && p.hwp <= 1)) { fail(`game ${g.id} probability ${p.hwp}`); bad++; }
    if (g.score && p.correct !== ((p.ph > p.pa) === (g.score.h > g.score.a))) { fail(`game ${g.id} correct flag mismatch`); bad++; }
    if (g.score && Math.abs(p.margin_err - ((g.score.h - g.score.a) - (p.ph - p.pa))) > 0.11) { fail(`game ${g.id} margin error mismatch`); bad++; }
  }
  if (g.fc?.verification === "verified" && g.tip && !(Date.parse(g.fc.at) < Date.parse(g.tip))) { fail(`game ${g.id} verified but saved after tip`); bad++; }
}
if (!bad) ok(`${games.length} game records: unique ids, orientation, probabilities, grading and timing consistent`);

// Team perspective: each archive row must appear for both teams with p and 1-p.
let persp = 0, rowsChecked = 0;
for (const src of ta.sources) {
  const arch = JSON.parse(readFileSync(join(PUB, `archive-${src.season}.json`), "utf8"));
  const byTeam = {};
  for (const [abbr, v] of Object.entries(ta.teams)) for (const r of v[src.key]?.games ?? []) byTeam[`${abbr}|${r.id}`] = r;
  for (const a of arch) {
    const h = byTeam[`${a.h}|${a.id}`], w = byTeam[`${a.a}|${a.id}`];
    rowsChecked++;
    if (!h || !w || !h.home || w.home || Math.abs(h.p - a.p) > 0.0015 || Math.abs(w.p - (1 - a.p)) > 0.0015 ||
        h.ts !== a.hs || w.ts !== a.as_ || h.won === w.won) { persp++; if (persp < 5) fail(`team perspective mismatch ${src.season} ${a.id}`); }
  }
}
if (!persp) ok(`${rowsChecked} graded games: home/away team rows use p and 1 − p with matching scores`);

// Expected-wins arithmetic
let ew = 0;
for (const [abbr, v] of Object.entries(ta.teams)) for (const [k, s] of Object.entries(v)) {
  const sum = s.games.reduce((t, r) => t + r.p, 0), wins = s.games.filter((r) => r.won).length;
  if (s.metrics.n && (Math.abs(sum - s.metrics.exp_w) > 0.2 || wins !== s.metrics.act_w)) { ew++; if (ew < 5) fail(`expected wins ${abbr} ${k}`); }
}
if (!ew) ok("expected and actual wins match the per-game rows for every team and sample");


// 4. Predictability and surprises: recompute independently from the archive rows.
const pred = JSON.parse(readFileSync(join(SRC, "predictability.json"), "utf8"));
const sur = JSON.parse(readFileSync(join(SRC, "surprises.json"), "utf8"));
const site = JSON.parse(readFileSync(join(SRC, "site.json"), "utf8"));
const archive = site.seasons.flatMap((s) => JSON.parse(readFileSync(join(PUB, `archive-${s}.json`), "utf8")));
let pb = 0;
const ids2 = new Set();
for (const r of archive) {
  if (ids2.has(r.id)) { fail(`archive duplicate ${r.id}`); pb++; } ids2.add(r.id);
  if (r.hs == null || r.as_ == null) { fail(`archive non-final ${r.id}`); pb++; }
  const hw = r.hs > r.as_, pw = hw ? r.p : 1 - r.p;
  if (r.w !== (hw ? r.h : r.a) || Math.abs(r.pw - pw) > 0.0015) { pb++; if (pb < 5) fail(`winner/probability ${r.id}`); }
  if (Math.abs(r.su - -Math.log(Math.max(1e-4, pw))) > 0.002) { pb++; if (pb < 5) fail(`surprise ${r.id}`); }
  if (Math.abs(r.se - Math.abs((r.hs - r.as_) - (r.ph - r.pa))) > 0.051) { pb++; if (pb < 5) fail(`scoreline ${r.id}`); }
}
for (const sc of pred.scopes) {
  const rows = archive.filter((r) => (sc.season === "all" || r.s === sc.season) && (sc.kind === "all" || (sc.kind === "po") === !!r.po));
  if (rows.length !== sc.n_games) { fail(`scope ${sc.key} games ${rows.length} vs ${sc.n_games}`); pb++; }
  for (const [abbr, m] of Object.entries(sc.teams)) {
    const g = rows.filter((r) => r.h === abbr || r.a === abbr).map((r) => {
      const home = r.h === abbr, ts = home ? r.hs : r.as_, os = home ? r.as_ : r.hs, tp = home ? r.ph : r.pa, op = home ? r.pa : r.ph;
      return { me: (ts - os) - (tp - op), p: home ? r.p : 1 - r.p, won: ts > os };
    });
    if (g.length !== m.n) { pb++; fail(`scope ${sc.key} ${abbr} n`); continue; }
    if (!g.length) continue;
    const mae = g.reduce((s, x) => s + Math.abs(x.me), 0) / g.length;
    const exp = g.reduce((s, x) => s + x.p, 0), act = g.filter((x) => x.won).length;
    const brier = g.reduce((s, x) => s + (x.p - (x.won ? 1 : 0)) ** 2, 0) / g.length;
    if (Math.abs(mae - m.mae) > 0.011 || Math.abs(exp - m.exp_w) > 0.011 || act !== m.act_w || Math.abs(brier - m.brier) > 0.0002) { pb++; if (pb < 8) fail(`scope ${sc.key} ${abbr} metrics`); }
  }
}
const wk = sur.week?.outcome;
if (wk && archive.some((r) => r.d >= sur.week.start && r.d <= sur.week.end && r.su > wk.su + 1e-9)) { fail("week surprise is not the maximum"); pb++; }
if (!pb) ok(`${archive.length} archive rows and ${pred.scopes.length} predictability scopes match independent recomputation`);

// 5. Explanations: verified ones reconstruct; unverified have no attribution.
const ex = JSON.parse(readFileSync(join(SRC, "explanations.json"), "utf8"));
const gmap = Object.fromEntries(games.map((g) => [g.id, g]));
let xb = 0, nx = 0;
for (const [id, e] of Object.entries(ex.games)) {
  if (!e.ok) { if (e.c) { fail(`unverified explanation has contributions ${id}`); xb++; } continue; }
  nx++;
  const total = e.base + Object.values(e.c).reduce((s, v) => s + v[2], 0);
  const fc = gmap[id]?.fc;
  if (Math.abs(total - e.pm) > 0.02 || !fc || Math.abs(e.pm - (fc.ph - fc.pa)) > 1.0001) { xb++; if (xb < 5) fail(`explanation ${id}`); }
}
if (!xb) ok(`${nx} verified explanations reconstruct their projected margins`);


console.log(fails ? `${fails} problem(s)` : "All build checks passed");
process.exit(fails ? 1 : 0);
