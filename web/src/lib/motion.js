// Site-wide motion, loaded once from Base.astro.
// - Adds `.in` to revealable elements the first time they scroll into view; CSS in
//   global.css does the drawing (chart sweep, bar fill, court lines, ladders).
// - Counts numbers up in key figures and scoreboards, keeping their exact final text.
// Without JS nothing is hidden (all initial states are scoped to html.js), and with
// prefers-reduced-motion everything is shown in its final state immediately.

const SEL = ".chart, .pbar, .figures, .board, [data-reveal], [data-roll]";
const COUNT_SEL = ".figure .v, .board .digits, [data-count]";
const still = matchMedia("(prefers-reduced-motion: reduce)").matches;

// Animate every number inside an element's text nodes from 0 to its value,
// preserving decimals, thousands separators and surrounding text.
const NUM = /\d[\d,]*(?:\.\d+)?/g;
function countUp(el, dur = 1100) {
  if (el.dataset.counted) return;
  el.dataset.counted = "1";
  if (/\d{4}-\d{2}/.test(el.textContent)) return; // season labels stay as written
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) if (/\d/.test(walker.currentNode.nodeValue)) nodes.push(walker.currentNode);
  if (!nodes.length) return;
  const plans = nodes.map((n) => {
    const src = n.nodeValue;
    const parts = [];
    let last = 0;
    src.replace(NUM, (m, i) => {
      parts.push(src.slice(last, i));
      const dec = (m.split(".")[1] || "").length;
      parts.push({ v: parseFloat(m.replace(/,/g, "")), dec, comma: m.includes(",") });
      last = i + m.length;
      return m;
    });
    parts.push(src.slice(last));
    return { n, src, parts };
  });
  // Year-like values (2022-23) and tiny counts read better left alone.
  const animatable = plans.some((p) => p.parts.some((x) => typeof x === "object" && x.v >= 2 && !(x.v >= 1990 && x.v <= 2100 && !x.dec && !x.comma)));
  if (!animatable) return;
  const fmt = (x, k) => {
    if (x.v >= 1990 && x.v <= 2100 && !x.dec && !x.comma) return String(x.v);
    const v = x.v * k;
    return x.comma ? v.toLocaleString("en-US", { minimumFractionDigits: x.dec, maximumFractionDigits: x.dec }) : v.toFixed(x.dec);
  };
  const t0 = performance.now();
  const step = (t) => {
    const p = Math.min(1, (t - t0) / dur);
    const k = 1 - Math.pow(1 - p, 4);
    for (const pl of plans) pl.n.nodeValue = p < 1 ? pl.parts.map((x) => (typeof x === "string" ? x : fmt(x, k))).join("") : pl.src;
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// Odometer: each digit of a final score rolls up from 0 in its own column. The
// real number stays in the page for screen readers; the columns are decorative.
function roll(el) {
  if (el.dataset.rolled) return;
  el.dataset.rolled = "1";
  const text = el.textContent.trim();
  if (!/^\d{1,3}$/.test(text)) return;
  const sr = document.createElement("span");
  sr.className = "sr-only"; sr.textContent = text;
  const vis = document.createElement("span");
  vis.className = "roll"; vis.setAttribute("aria-hidden", "true");
  const cols = [...text].map((d, i) => {
    const c = document.createElement("span");
    c.className = "rc"; c.style.setProperty("--rd", `${0.1 + i * 0.12}s`);
    for (let k = 0; k <= 9; k++) { const s = document.createElement("span"); s.textContent = k; c.appendChild(s); }
    vis.appendChild(c);
    return [c, +d];
  });
  el.textContent = "";
  el.append(sr, vis);
  requestAnimationFrame(() => requestAnimationFrame(() => cols.forEach(([c, d]) => { c.style.transform = `translateY(-${d}em)`; })));
}

function reveal(el) {
  el.classList.add("in");
  if (still) return;
  if (el.matches(COUNT_SEL)) countUp(el);
  el.querySelectorAll(COUNT_SEL).forEach((c) => countUp(c));
  if (el.matches("[data-roll]")) roll(el);
}

// Tip-off countdowns: "Tips in 2h 14m", then "In progress" until the final is
// posted (the page only knows finals from its last rebuild).
function tick() {
  const now = Date.now();
  document.querySelectorAll(".cd[data-tip]").forEach((el) => {
    const t = Date.parse(el.dataset.tip);
    if (Number.isNaN(t)) return;
    const m = Math.round((t - now) / 6e4);
    let txt = "", live = false;
    if (m > 0 && m < 36 * 60) txt = m < 60 ? `Tips in ${m}m` : `Tips in ${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
    else if (m <= 0 && m > -210) { txt = "In progress"; live = true; }
    else if (m <= -210) txt = "Awaiting final";
    el.textContent = txt;
    el.classList.toggle("live", live);
  });
}
tick();
setInterval(tick, 30000);

const targets = [...document.querySelectorAll(SEL)];
if (still || !("IntersectionObserver" in window)) {
  targets.forEach((el) => el.classList.add("in"));
} else {
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { reveal(e.target); io.unobserve(e.target); }
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.12 });
  targets.forEach((el) => io.observe(el));
  // Never leave anything hidden: reveal whatever is still pending after a while
  // (print, very tall elements, background tabs).
  addEventListener("beforeprint", () => targets.forEach((el) => el.classList.add("in")));
}
