// Site-wide motion, loaded once from Base.astro.
// - Adds `.in` to revealable elements the first time they scroll into view; CSS in
//   global.css does the drawing (chart sweep, bar fill, court lines, ladders).
// - Counts numbers up in key figures and scoreboards, keeping their exact final text.
// Without JS nothing is hidden (all initial states are scoped to html.js), and with
// prefers-reduced-motion everything is shown in its final state immediately.

const SEL = ".chart, .pbar, .figures, .board, [data-reveal]";
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

function reveal(el) {
  el.classList.add("in");
  if (still) return;
  if (el.matches(COUNT_SEL)) countUp(el);
  el.querySelectorAll(COUNT_SEL).forEach((c) => countUp(c));
}

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
