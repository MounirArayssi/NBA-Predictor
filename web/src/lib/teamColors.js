// Official team colors (primary, secondary) from each team's published palette.
// `ui` is the one of those official colors that stays visible on the dark page
// ground (checked by scripts/check_colors.mjs at >= 3:1 for graphics).
// Logos are not used anywhere; color and abbreviation identify a team.
export const TEAM_COLORS = {
  ATL: { p: "#E03A3E", s: "#C1D32F", ui: "#E03A3E" },
  BOS: { p: "#007A33", s: "#BA9653", ui: "#BA9653" },
  BKN: { p: "#000000", s: "#FFFFFF", ui: "#FFFFFF" },
  CHA: { p: "#1D1160", s: "#00788C", ui: "#00788C" },
  CHI: { p: "#CE1141", s: "#000000", ui: "#CE1141" },
  CLE: { p: "#860038", s: "#FDBB30", ui: "#FDBB30" },
  DAL: { p: "#00538C", s: "#B8C4CA", ui: "#B8C4CA" },
  DEN: { p: "#0E2240", s: "#FEC524", ui: "#FEC524" },
  DET: { p: "#C8102E", s: "#1D42BA", ui: "#C8102E" },
  GSW: { p: "#1D428A", s: "#FFC72C", ui: "#FFC72C" },
  HOU: { p: "#CE1141", s: "#C4CED4", ui: "#CE1141" },
  IND: { p: "#002D62", s: "#FDBB30", ui: "#FDBB30" },
  LAC: { p: "#C8102E", s: "#1D428A", ui: "#C8102E" },
  LAL: { p: "#552583", s: "#FDB927", ui: "#FDB927" },
  MEM: { p: "#5D76A9", s: "#12173F", ui: "#5D76A9" },
  MIA: { p: "#98002E", s: "#F9A01B", ui: "#F9A01B" },
  MIL: { p: "#00471B", s: "#EEE1C6", ui: "#EEE1C6" },
  MIN: { p: "#0C2340", s: "#78BE20", ui: "#78BE20" },
  NOP: { p: "#0C2340", s: "#85714D", ui: "#85714D" },
  NYK: { p: "#006BB6", s: "#F58426", ui: "#F58426" },
  OKC: { p: "#007AC1", s: "#EF3B24", ui: "#007AC1" },
  ORL: { p: "#0077C0", s: "#C4CED4", ui: "#0077C0" },
  PHI: { p: "#006BB6", s: "#ED174C", ui: "#ED174C" },
  PHX: { p: "#1D1160", s: "#E56020", ui: "#E56020" },
  POR: { p: "#E03A3E", s: "#000000", ui: "#E03A3E" },
  SAC: { p: "#5A2D81", s: "#63727A", ui: "#63727A" },
  SAS: { p: "#C4CED4", s: "#000000", ui: "#C4CED4" },
  TOR: { p: "#CE1141", s: "#000000", ui: "#CE1141" },
  UTA: { p: "#002B5C", s: "#F9A01B", ui: "#F9A01B" },
  WAS: { p: "#002B5C", s: "#E31837", ui: "#E31837" },
};

const FALLBACK = { p: "#5a646c", s: "#8a949c", ui: "#8a949c" };

const lum = (hex) => {
  const v = hex.replace("#", "").match(/../g).map((h) => parseInt(h, 16) / 255).map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
};
export const contrast = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((m, n) => n - m); return (x + 0.05) / (y + 0.05); };
const LIGHT_PAPER = "#f3f4f1";

/** Full color record: p, s (official), ui (on dark), light (on the light page:
 *  primary unless it is too pale, then secondary), fg (text on a primary fill). */
export const teamColor = (abbr) => {
  const c = TEAM_COLORS[abbr] ?? FALLBACK;
  const light = contrast(c.p, LIGHT_PAPER) >= 3 ? c.p : c.s;
  const fg = contrast("#ffffff", c.p) >= contrast("#000000", c.p) ? "#ffffff" : "#000000";
  return { ...c, light, fg };
};
/** Inline CSS custom properties for a team: --tp, --ts (official), --tu (on dark),
 *  --tl (on light), --tf (text on primary). */
export const teamVars = (abbr) => {
  const c = teamColor(abbr);
  return `--tp:${c.p};--ts:${c.s};--tu:${c.ui};--tl:${c.light};--tf:${c.fg}`;
};
