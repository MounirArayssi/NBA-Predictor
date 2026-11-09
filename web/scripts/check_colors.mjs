// Contrast checks for the dark palette and team UI colors.
//   node scripts/check_colors.mjs
import { TEAM_COLORS, teamColor } from "../src/lib/teamColors.js";
const lum = (hex) => {
  const v = hex.replace("#", "").match(/../g).map((h) => parseInt(h, 16) / 255).map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
};
const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((m, n) => n - m); return (x + 0.05) / (y + 0.05); };
const PAPER = "#0d1216", SURFACE = "#151c22", BOARD = "#0d1216", LIGHT = "#f3f4f1", WHITE = "#ffffff";
let fail = 0;
const req = (name, fg, bg, min) => { const r = ratio(fg, bg); const ok = r >= min; if (!ok) fail++; console.log(`${ok ? "ok  " : "FAIL"} ${name.padEnd(28)} ${r.toFixed(2)} (min ${min})`); };
for (const k of Object.keys(TEAM_COLORS)) {
  const c = teamColor(k);
  req(`${k} ui on instrument`, c.ui, BOARD, 3);
  req(`${k} light on page`, c.light, LIGHT, 3);
  req(`${k} text on primary`, c.fg, c.p, 4.5);
}
const light = { ink: "#1b1f22", "ink-2": "#434b51", muted: "#5d666d", accent: "#a94c1d", model: "#2b6b61" };
for (const [k, v] of Object.entries(light)) { req(`${k} on light page`, v, LIGHT, 4.5); req(`${k} on white`, v, WHITE, 4.5); }
const text = { ink: "#e9edf0", "ink-2": "#b4bec6", muted: "#8b969f", accent: "#f08c4f", model: "#5ec4b0", led: "#f5b94c" }; // instrument panels
for (const [k, v] of Object.entries(text)) { req(`${k} on paper`, v, PAPER, 4.5); req(`${k} on surface`, v, SURFACE, 4.5); req(`${k} on board`, v, BOARD, 4.5); }
process.exit(fail ? 1 : 0);
