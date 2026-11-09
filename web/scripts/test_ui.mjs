// Interaction tests against the built site (run `npm run preview` first).
//   node scripts/test_ui.mjs [baseUrl]
import { chromium } from "playwright-core";
const base = process.argv[2] || "http://127.0.0.1:4321";
const browser = await chromium.launch({ executablePath: process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe" });
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
let fails = 0;
const check = (name, ok, detail = "") => { console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  (" + detail + ")" : ""}`); if (!ok) fails++; };
const visibleRows = () => page.$$eval(".grow", (r) => r.filter((x) => x.offsetParent !== null).map((x) => ({ d: x.dataset.date, t: x.dataset.teams, s: x.dataset.status })));

// Forecasts: date navigation, view, filters, URL state
await page.goto(`${base}/forecasts/?date=2026-11-06`);
let rows = await visibleRows();
check("forecasts: ?date selects that day", rows.length > 0 && rows.every((r) => r.d === "2026-11-06"), `${rows.length} rows`);
await page.click('[data-step="1"]');
rows = await visibleRows();
check("forecasts: next day", rows.every((r) => r.d === "2026-11-07") && page.url().includes("date=2026-11-07"));
await page.click('[data-view="week"]');
rows = await visibleRows();
check("forecasts: week view spans several days", new Set(rows.map((r) => r.d)).size > 1, `${new Set(rows.map((r) => r.d)).size} days`);
await page.selectOption("#f-team", "OKC");
rows = await visibleRows();
check("forecasts: team filter", rows.length > 0 && rows.every((r) => r.t.split(" ").includes("OKC")) && page.url().includes("team=OKC"));
await page.selectOption("#f-status", "final");
rows = await visibleRows();
check("forecasts: status filter", rows.every((r) => r.s === "final"));
await page.goto(`${base}/forecasts/?date=2026-12-25`);
const msg = await page.textContent("#empty");
check("forecasts: honest empty state outside data", (await page.isVisible("#empty")) && /no schedule/i.test(msg), msg.trim().slice(0, 60));
await page.goto(`${base}/forecasts/?date=2026-11-03`);
check("forecasts: no-games day message", /No NBA games are scheduled/.test(await page.textContent("#empty")));

// Archive: datasets, filters, paging, URL
await page.goto(`${base}/forecasts/archive/`);
await page.waitForSelector("#a-body tr");
await page.waitForFunction(() => /^[\d,]+ games/.test(document.querySelector("#summary").textContent.trim()));
const sum1 = await page.textContent("#summary");
check("archive: current season loads", /^[\d,]+ games/.test(sum1.trim()), sum1.slice(0, 40));
await page.selectOption("#a-team", "BOS");
const teamsOk = await page.$$eval("#a-body tr td:nth-child(2)", (c) => c.every((x) => x.textContent.includes("BOS")));
check("archive: team filter", teamsOk && page.url().includes("team=BOS"));
await page.selectOption("#a-res", "miss");
const missOk = await page.$$eval("#a-body .res", (c) => c.every((x) => x.textContent.includes("Missed")));
check("archive: result filter", missOk);
await page.click("#a-reset");
await page.selectOption("#a-season", "2022-23");
await page.waitForFunction(() => /1,3\d\d games/.test(document.querySelector("#summary").textContent));
await page.click("#next");
check("archive: paging", (await page.textContent("#page")).startsWith("Page 2"));

// Teams directory: search, conference, sort
await page.goto(`${base}/teams/`);
await page.fill("#t-search", "lakers");
check("teams: search", (await page.textContent("#t-count")).startsWith("1 of 30"));
await page.fill("#t-search", "");
await page.click('[data-conf="East"]');
check("teams: conference filter", (await page.textContent("#t-count")).startsWith("15 of 30"));
await page.click("#teams thead th:nth-child(4) button");
await page.click("#teams thead th:nth-child(4) button");
const elos = await page.$$eval("#teams tbody tr:not([hidden]) td:nth-child(4)", (c) => c.map((x) => parseFloat(x.dataset.v)).filter((v) => !isNaN(v)));
check("teams: sort by Elo descending", elos.every((v, i) => i === 0 || elos[i - 1] >= v));

// Team page: sample switcher
await page.goto(`${base}/teams/boston-celtics/`);
const opts = await page.$$eval("#src option", (o) => o.map((x) => x.value));
await page.selectOption("#src", opts[opts.length - 1]);
const shown = await page.$$eval(".panel", (p) => p.filter((x) => !x.hidden).map((x) => x.dataset.src));
check("team: sample switcher shows one panel", shown.length === 1 && shown[0] === opts[opts.length - 1] && page.url().includes("sample="));


// Surprise Games explorer
await page.goto(`${base}/surprises/`);
await page.waitForFunction(() => /games/.test(document.querySelector("#s-summary").textContent));
const pws = await page.$$eval("#s-body tr td.pw", (c) => c.map((x) => parseInt(x.textContent)));
check("surprises: loads, lowest winner chance first", pws.length === 50 && pws.every((v, i) => i === 0 || pws[i - 1] <= v), `${pws.slice(0, 3)}`);
await page.selectOption("#s-team", "BOS");
await page.waitForFunction(() => [...document.querySelectorAll("#s-body tr td:nth-child(3)")].every((c) => c.textContent.includes("BOS")));
check("surprises: team filter", page.url().includes("team=BOS"));
await page.click('[data-view="scoreline"]');
const ses = await page.$$eval("#s-body tr td:nth-child(7)", (c) => c.map((x) => +x.textContent));
check("surprises: margin-miss ranking sorted", ses.length > 0 && ses.every((v, i) => i === 0 || ses[i - 1] >= v));
await page.selectOption("#s-season", "2024-25");
await page.waitForFunction(() => [...document.querySelectorAll("#s-body tr td:nth-child(2)")].every((c) => /2024|2025/.test(c.textContent)));
check("surprises: season filter", page.url().includes("season=2024-25"));
await page.click('[data-view="recent"]');
const rec = await page.$$eval("#s-body tr td.pw", (c) => c.map((x) => parseInt(x.textContent)));
check("surprises: recent shows only favorite losses", rec.every((v) => v <= 50));

// Team Predictability
await page.goto(`${base}/teams/predictability/`);
const visibleScopes = () => page.$$eval(".scope", (s) => s.filter((x) => !x.hidden).map((x) => x.dataset.key));
check("predictability: one scope visible", (await visibleScopes()).length === 1);
await page.selectOption("#p-season", "2024-25");
await page.selectOption("#p-kind", "po");
check("predictability: season + playoffs", (await visibleScopes())[0] === "2024-25|po" && page.url().includes("games=po"));
await page.selectOption("#p-min", "20");
const vis = await page.$$eval(".scope:not([hidden]) tbody tr", (r) => r.filter((x) => !x.hidden).map((x) => +x.dataset.n));
check("predictability: minimum games filter", vis.every((n) => n >= 20), `${vis.length} teams`);
await page.selectOption("#p-season", "all"); await page.selectOption("#p-kind", "all"); await page.selectOption("#p-min", "0");
await page.click(".scope:not([hidden]) thead th:nth-child(3) button");
const maes = await page.$$eval(".scope:not([hidden]) tbody tr td:nth-child(3)", (c) => c.map((x) => +x.dataset.v));
check("predictability: sort by margin miss", maes.length === 30 && (maes.every((v, i) => !i || maes[i - 1] <= v) || maes.every((v, i) => !i || maes[i - 1] >= v)));

// Inside the Forecast: an explained game and an unexplained one
await page.goto(`${base}/games/0022600033/`);
check("game: Inside the Forecast present", await page.isVisible("#inside"));
const hasWf = await page.$("#inside .wf svg");
const note = await page.textContent("#inside .prob-note");
check("game: explanation or honest note", !!hasWf || /can't be broken down exactly/.test(note), hasWf ? "waterfall" : "note");
check("game: result vs forecast section", await page.isVisible("#result-vs-forecast"));
await page.goto(`${base}/teams/boston-celtics/`);
check("team: predictability + surprises sections", (await page.isVisible("#predictability")) && (await page.isVisible("#surprises")));
await page.goto(`${base}/`);
const wk = await page.$('a.wk');
check("home: surprise of the week links somewhere", !!wk && /\/(games|surprises)\//.test(await wk.getAttribute("href")));

// Mobile menu
await page.setViewportSize({ width: 390, height: 800 });
await page.goto(`${base}/`);
const before = await page.isVisible("#site-nav");
await page.click(".menu-btn");
check("mobile menu toggles", !before && (await page.isVisible("#site-nav")));
await page.keyboard.press("Escape");
check("mobile menu closes on Escape", !(await page.isVisible("#site-nav")));

check("no console or page errors", errors.length === 0, errors.slice(0, 3).join(" | "));
await browser.close();
console.log(fails ? `${fails} check(s) failed` : "All checks passed");
process.exit(fails ? 1 : 0);
