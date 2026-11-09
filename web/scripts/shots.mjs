// Visual QA helper: screenshots pages at several widths with the locally installed
// Chrome and reports any element wider than the viewport (horizontal overflow).
//   node scripts/shots.mjs <outDir> [baseUrl]
import { chromium } from "playwright-core";
const out = process.argv[2] || "shots";
const base = process.argv[3] || "http://127.0.0.1:4321";
const pages = (process.env.PAGES || "/").split(",");
const widths = (process.env.WIDTHS || "1440,390").split(",").map(Number);
const browser = await chromium.launch({ executablePath: process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe" });
for (const w of widths) {
  const ctx = await browser.newContext({ viewport: { width: w, height: 900 }, deviceScaleFactor: 1, isMobile: w < 600, hasTouch: w < 600 });
  const page = await ctx.newPage();
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));
  for (const p of pages) {
    await page.goto(base + p, { waitUntil: "networkidle" });
    // Scroll through once so scroll-triggered reveals run, then let them settle.
    await page.evaluate(async () => {
      for (let y = 0; y < document.body.scrollHeight; y += 500) { scrollTo(0, y); await new Promise((r) => setTimeout(r, 60)); }
      scrollTo(0, 0);
    });
    await page.waitForTimeout(2600);
    const over = await page.evaluate(() => {
      const vw = document.documentElement.clientWidth;
      const bad = [];
      document.querySelectorAll("body *").forEach((el) => {
        const r = el.getBoundingClientRect();
        if (r.right > vw + 1 && !el.closest(".table-scroll, .chart-scroll") && r.width > 0) bad.push(`${el.tagName.toLowerCase()}.${[...el.classList].join(".")} right=${Math.round(r.right)}`);
      });
      return { vw, sw: document.documentElement.scrollWidth, bad: bad.slice(0, 8) };
    });
    const name = `${p.replace(/\W+/g, "_") || "home"}_${w}`;
    await page.screenshot({ path: `${out}/${name}.png`, fullPage: true });
    console.log(`${p} @${w}: scrollWidth ${over.sw} / ${over.vw}${over.bad.length ? " OVERFLOW " + over.bad.join(" | ") : ""}`);
  }
  if (errors.length) console.log(`console errors @${w}:`, errors.slice(0, 5));
  await ctx.close();
}
await browser.close();
