"""
Official NBA injury report (https://official.nba.com/nba-injury-report-2025-26-season/).

The league publishes a PDF every 15 minutes (hourly before 2025-26) and keeps the
archive online, so the same source serves live predictions and training:

- live:     latest report published before now
- training: the game day's TRAINING_REPORT_TIME report, which roughly matches when
            the daily pipeline makes its predictions (~6 PM ET)

Reports are stored in injury_report_fetches / injury_report_entries so each PDF is
downloaded once.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import io
import re
import time
import unicodedata
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from sqlalchemy import text
from sqlalchemy.orm import Session

from data.storage.db import engine
from data.storage.models import Base, InjuryReportFetch, InjuryReportEntry

ET = ZoneInfo("America/New_York")
BASE_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{t}.pdf"
# The CDN stalls (no response) for non-browser-looking User-Agents
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Report used for each game day in training/backfills (ET, 24h)
TRAINING_REPORT_TIME = "17:30"

STATUSES = {"Out", "Doubtful", "Questionable", "Probable", "Available"}
UNAVAILABLE = ("Out", "Doubtful")


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

def report_urls(report_date, hhmm):
    """
    Candidate URLs for a report time. 2025-26 onward: '..._05_30PM.pdf'
    (15-minute reports). Earlier seasons: hourly '..._05PM.pdf' (published :30).
    """
    h, m = map(int, hhmm.split(":"))
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    d = report_date.isoformat()
    urls = [BASE_URL.format(d=d, t=f"{h12:02d}_{m:02d}{ampm}")]
    if m == 30 or m == 0:
        urls.append(BASE_URL.format(d=d, t=f"{h12:02d}{ampm}"))
    return urls


# ---------------------------------------------------------------------------
# Name / team matching
# ---------------------------------------------------------------------------

def name_key(name):
    """Accent-free, lowercase, letters/digits only — the PDF drops spaces."""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", name.lower())


SUFFIX_RE = re.compile(r"(,?\s*(Jr\.?|Sr\.?|II|III|IV|V))+$")
SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}


def split_raw(raw):
    """
    'ButlerIII,Jimmy' -> ('Jimmy', 'ButlerIII', 'Butler')
    'Udeh,Jr.,Ernest' -> ('Ernest', 'Udeh,Jr.', 'Udeh')
    Returns (first, last as printed, last without suffix).
    """
    if "," not in raw:
        return "", raw, SUFFIX_RE.sub("", raw)
    last, first = raw.rsplit(",", 1)
    return first.strip(), last.strip(), SUFFIX_RE.sub("", last.strip())


def raw_name_key(raw):
    """'ButlerIII,Jimmy' -> key of 'Jimmy ButlerIII'."""
    first, last, _ = split_raw(raw)
    return name_key(first + last)


def raw_name_key_nosuffix(raw):
    first, _, last = split_raw(raw)
    return name_key(first + last)


def _strip_suffix_tokens(full_name):
    toks = [t for t in re.split(r"[\s,]+", str(full_name)) if t]
    while len(toks) > 1 and name_key(toks[-1]) in SUFFIX_TOKENS:
        toks.pop()
    return toks


def split_raw_name(raw):
    """Best-effort 'First Last' for display when the player isn't in our table."""
    first, last, _ = split_raw(raw)
    return f"{first} {last}".strip()


def season_for_date(d):
    start = d.year if d.month >= 8 else d.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


class PlayerMatcher:
    """
    Maps report names to players.player_id:
      1. exact full-name match
      2. match with Jr./III etc. removed on both sides
      3. unique last-name match among players on that team's box scores this
         season or last (handles 'Carrington,Carlton' -> Bub Carrington)
    """

    def __init__(self):
        with engine.connect() as conn:
            players = conn.execute(text("SELECT player_id, full_name FROM players")).fetchall()
            rosters = conn.execute(text("""
                SELECT DISTINCT b.team_id, g.season, b.player_id
                FROM player_box_scores b JOIN games g USING (game_id)
            """)).fetchall()
        self.name = {pid: full for pid, full in players}
        self.exact, self.nosuffix, self.last = {}, {}, {}
        for pid, full in players:
            self.exact.setdefault(name_key(full), []).append(pid)
            toks = _strip_suffix_tokens(full)
            self.nosuffix.setdefault(name_key("".join(toks)), []).append(pid)
            self.last.setdefault(name_key(toks[-1]), set()).add(pid)
        self.roster = {}
        for team_id, season, pid in rosters:
            self.roster.setdefault((team_id, season), set()).add(pid)

    def match(self, raw, team_id, game_date):
        season = season_for_date(game_date)
        prev = season_for_date(date(game_date.year - 1, game_date.month, 1))
        on_team = self.roster.get((team_id, season), set()) | self.roster.get((team_id, prev), set())

        for cands in (self.exact.get(raw_name_key(raw), []),
                      self.nosuffix.get(raw_name_key_nosuffix(raw), [])):
            if len(cands) == 1:
                return cands[0]
            if len(cands) > 1:
                hit = [c for c in cands if c in on_team]
                if len(hit) == 1:
                    return hit[0]

        _, _, last = split_raw(raw)
        hit = [c for c in self.last.get(name_key(last), set()) if c in on_team]
        return hit[0] if len(hit) == 1 else None


def _load_lookups():
    with engine.connect() as conn:
        teams = conn.execute(text("SELECT team_id, abbreviation, full_name FROM teams")).fetchall()
    team_by_key = {}
    for team_id, abbr, full in teams:
        team_by_key[name_key(full)] = (team_id, abbr)
    # Names the PDF prints differently from our table
    team_by_key.setdefault("laclippers", team_by_key.get(name_key("Los Angeles Clippers")))
    return team_by_key, PlayerMatcher()


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

COLUMNS = ["GameDate", "GameTime", "Matchup", "Team", "PlayerName", "CurrentStatus", "Reason"]


def _find_header(words):
    """
    Column x-positions from the header line, or (None, None) if the page has none.
    Newer PDFs print each header as one word ('PlayerName'); 2021-22 PDFs print
    'Player Name', so neighbouring words on the header line are merged first.
    """
    anchor = next((w for w in words if w["text"] == "Matchup"), None)
    if anchor is None:
        return None, None
    line = sorted((w for w in words if abs(w["top"] - anchor["top"]) < 2), key=lambda w: w["x0"])
    merged = []
    for w in line:
        if merged and w["x0"] - merged[-1]["x1"] < 6:
            merged[-1] = {"text": merged[-1]["text"] + w["text"],
                          "x0": merged[-1]["x0"], "x1": w["x1"]}
        else:
            merged.append({"text": w["text"], "x0": w["x0"], "x1": w["x1"]})
    header = {m["text"]: m["x0"] for m in merged if m["text"] in COLUMNS}
    if len(header) != len(COLUMNS):
        return None, None
    return header, anchor["top"]


def parse_report_pdf(pdf_bytes):
    """
    Returns a list of dicts: game_date, matchup, team (as printed), raw_name,
    status, reason. Date/matchup/team cells are only printed on the first row of
    their group, so they are carried forward. Teams that have not submitted yet
    produce no rows.
    """
    import pdfplumber

    rows = []
    cur = {"game_date": None, "matchup": None, "team": None}

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        bounds = None
        for page in pdf.pages:
            words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
            header, header_top = _find_header(words)
            if header is not None:
                xs = [header[c] for c in COLUMNS]
                # Column i spans [x_i - 3, x_{i+1} - 3)
                bounds = [x - 3 for x in xs] + [float("inf")]
            else:
                header_top = 0
            if bounds is None:
                continue

            # Group words into lines by vertical position
            lines = {}
            for w in words:
                if w["top"] <= header_top + 2:
                    continue
                key = round(w["top"] / 3)
                lines.setdefault(key, []).append(w)

            for key in sorted(lines):
                line_text = "".join(w["text"] for w in lines[key])
                # Page title ('Injury Report: 11/05/26 11:15 AM'), footer, repeated header
                if line_text.startswith("InjuryReport:") or re.fullmatch(r"Page\d+of\d+", line_text) \
                        or line_text.startswith("GameDateGameTime"):
                    continue
                cells = {c: [] for c in COLUMNS}
                for w in sorted(lines[key], key=lambda w: w["x0"]):
                    for i, c in enumerate(COLUMNS):
                        if bounds[i] <= w["x0"] < bounds[i + 1]:
                            cells[c].append(w["text"])
                            break
                cells = {c: " ".join(v) for c, v in cells.items()}

                if re.fullmatch(r"\d{2}/\d{2}/\d{4}", cells["GameDate"]):
                    cur["game_date"] = datetime.strptime(cells["GameDate"], "%m/%d/%Y").date()
                if re.fullmatch(r"[A-Z]{2,3}@[A-Z]{2,3}", cells["Matchup"]):
                    cur["matchup"] = cells["Matchup"]
                if cells["Team"]:
                    cur["team"] = cells["Team"]

                status = cells["CurrentStatus"]
                if cells["PlayerName"] and status in STATUSES:
                    rows.append({
                        "game_date": cur["game_date"],
                        "matchup": cur["matchup"],
                        "team": cur["team"],
                        "raw_name": cells["PlayerName"],
                        "status": status,
                        "reason": cells["Reason"][:200] or None,
                    })
    return rows


# ---------------------------------------------------------------------------
# Fetch + store
# ---------------------------------------------------------------------------

def ensure_tables():
    Base.metadata.create_all(engine, tables=[
        InjuryReportFetch.__table__, InjuryReportEntry.__table__])


def fetch_and_store_report(report_date, hhmm, session=None, lookups=None, verbose=True):
    """
    Download, parse and store one report. Returns the fetch_id, or None if the
    report doesn't exist. Already-fetched reports are not downloaded again.
    """
    own_session = session is None
    session = session or Session(engine)
    try:
        urls = report_urls(report_date, hhmm)
        existing = session.query(InjuryReportFetch).filter(
            InjuryReportFetch.url.in_(urls)).all()
        ok = [f for f in existing if f.http_status == 200]
        if ok:
            return ok[0].fetch_id
        if len(existing) == len(urls):
            return None  # every candidate URL already returned an error

        tried = {f.url for f in existing}
        team_by_key, matcher = lookups or _load_lookups()

        for url in urls:
            if url in tried:
                continue
            try:
                resp = requests.get(url, headers=HEADERS, timeout=(5, 20))
                code = resp.status_code
            except requests.RequestException as e:
                if verbose:
                    print(f"  ⚠️  {url}: {e}")
                return None  # network trouble: don't record, retry next time

            fetch = InjuryReportFetch(report_date=report_date, report_time=hhmm,
                                      url=url, http_status=code)
            session.add(fetch)
            session.flush()

            if code != 200:
                session.commit()
                continue

            parsed = parse_report_pdf(resp.content)
            n = 0
            unmatched_teams = set()
            seen = set()
            for r in parsed:
                team = team_by_key.get(name_key(r["team"] or ""))
                if team is None or r["game_date"] is None:
                    unmatched_teams.add(r["team"])
                    continue
                k = (team[0], r["raw_name"], r["game_date"])
                if k in seen:
                    continue
                seen.add(k)
                pid = matcher.match(r["raw_name"], team[0], r["game_date"])
                session.add(InjuryReportEntry(
                    fetch_id=fetch.fetch_id,
                    game_date=r["game_date"],
                    matchup=r["matchup"],
                    team_id=team[0],
                    player_id=pid,
                    player_name=matcher.name[pid] if pid else split_raw_name(r["raw_name"]),
                    raw_name=r["raw_name"],
                    status=r["status"],
                    reason=r["reason"],
                ))
                n += 1
            fetch.n_entries = n
            session.commit()
            if verbose:
                extra = f" (unmatched teams: {unmatched_teams})" if unmatched_teams else ""
                print(f"  📋 {report_date} {hhmm} ET: {n} entries{extra}")
            return fetch.fetch_id
        return None
    finally:
        if own_session:
            session.close()


def fetch_latest_report(now_et=None, lookback_hours=6, verbose=True):
    """Most recent report at or before now (ET). Returns fetch_id or None."""
    ensure_tables()
    now_et = now_et or datetime.now(ET)
    t = now_et.replace(minute=now_et.minute - now_et.minute % 15, second=0, microsecond=0)
    lookups = _load_lookups()
    with Session(engine) as session:
        for _ in range(lookback_hours * 4):
            fid = fetch_and_store_report(t.date(), t.strftime("%H:%M"),
                                         session=session, lookups=lookups, verbose=verbose)
            if fid:
                return fid
            t -= timedelta(minutes=15)
    return None


def fetch_training_report(game_date, session=None, lookups=None, verbose=True):
    """
    The game day's TRAINING_REPORT_TIME report, falling back to the nearest
    earlier report (up to 3 hours) if that one is missing.
    """
    base = datetime.combine(game_date, datetime.strptime(TRAINING_REPORT_TIME, "%H:%M").time())
    for step in range(0, 13):
        t = base - timedelta(minutes=15 * step)
        fid = fetch_and_store_report(t.date(), t.strftime("%H:%M"), session=session,
                                     lookups=lookups, verbose=verbose)
        if fid:
            return fid
    return None


def backfill_training_reports(start_date=None, end_date=None, delay=0.3):
    """Fetch the training report for every date with a final game."""
    ensure_tables()
    q = "SELECT DISTINCT game_date FROM games WHERE is_final = TRUE"
    params = {}
    if start_date:
        q += " AND game_date >= :s"; params["s"] = start_date
    if end_date:
        q += " AND game_date <= :e"; params["e"] = end_date
    with engine.connect() as conn:
        dates = [r[0] for r in conn.execute(text(q + " ORDER BY game_date"), params)]

    lookups = _load_lookups()
    missing = []
    with Session(engine) as session:
        for i, d in enumerate(dates):
            if not fetch_training_report(d, session=session, lookups=lookups, verbose=False):
                missing.append(d)
            if i % 25 == 0:
                print(f"  {i}/{len(dates)} dates ({d}), missing so far: {len(missing)}")
            time.sleep(delay)
    print(f"Done: {len(dates)} dates, {len(missing)} without a report")
    return missing


# ---------------------------------------------------------------------------
# Reading stored reports
# ---------------------------------------------------------------------------

def _entries_for_fetch(fetch_id, game_date):
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT t.abbreviation, e.player_name, e.status, e.reason, e.player_id, e.team_id
            FROM injury_report_entries e JOIN teams t USING (team_id)
            WHERE e.fetch_id = :f AND e.game_date = :d
        """), {"f": fetch_id, "d": game_date}).fetchall()
    return rows


def _as_report_dict(rows):
    """Same shape fetch_injuries.fetch_injury_report returns."""
    out = {}
    for abbr, name, status, reason, player_id, _ in rows:
        if status == "Available":
            continue
        out.setdefault(abbr, []).append({
            "name": name, "status": status, "detail": reason or "",
            "raw_status": status, "player_id": player_id,
        })
    return out


def get_report_for_game_date(game_date, live=False):
    """
    Injury report dict {abbr: [{name, status, detail, ...}]} for games on
    game_date. live=True uses the latest report published so far; otherwise the
    game day's training-time report. Returns None if no report is available.
    """
    ensure_tables()
    fid = fetch_latest_report() if live else fetch_training_report(game_date)
    if fid is None:
        return None
    return _as_report_dict(_entries_for_fetch(fid, game_date))


def load_training_unavailable():
    """
    For training: {(game_date, team_id): set(player_id)} of players listed
    Out/Doubtful on each game day's training report. Dates with no stored report
    are absent from the dict.
    """
    with engine.connect() as conn:
        fetches = conn.execute(text("""
            SELECT report_date, report_time, fetch_id FROM injury_report_fetches
            WHERE http_status = 200
        """)).fetchall()
        rows = conn.execute(text("""
            SELECT f.fetch_id, e.game_date, e.team_id, e.player_id, e.status
            FROM injury_report_entries e JOIN injury_report_fetches f USING (fetch_id)
            WHERE e.game_date = f.report_date
        """)).fetchall()

    # Pick, per date, the latest stored report at or before TRAINING_REPORT_TIME
    best = {}
    for d, t, fid in fetches:
        if t <= TRAINING_REPORT_TIME and (d not in best or t > best[d][0]):
            best[d] = (t, fid)
    chosen = {fid: d for d, (t, fid) in best.items()}

    out = {d: {} for d in best}
    for fid, d, team_id, player_id, status in rows:
        if chosen.get(fid) != d:
            continue
        s = out[d].setdefault(team_id, set())
        if status in UNAVAILABLE and player_id is not None:
            s.add(player_id)
    return out


if __name__ == "__main__":
    if "--backfill" in sys.argv:
        backfill_training_reports()
    else:
        print(get_report_for_game_date(date.today(), live=True))


def get_injury_report(game_date=None):
    """
    Injury report used by predictions, {abbr: [{name, status, detail}]}.

    - game_date today/future (or None): latest official report, falling back to
      ESPN if the NBA report can't be fetched.
    - game_date in the past (backfills): that day's training-time official
      report — never today's report, which would leak/mismatch.
    """
    today = datetime.now(ET).date()
    game_date = game_date or today
    if game_date < today:
        report = get_report_for_game_date(game_date, live=False)
        if report is None:
            print(f"  ⚠️  No official injury report stored for {game_date} — using none")
            return {}
        print(f"  📋 Official injury report for {game_date} "
              f"({TRAINING_REPORT_TIME} ET): {sum(len(v) for v in report.values())} entries")
        return report

    try:
        report = get_report_for_game_date(game_date, live=True)
    except Exception as e:
        print(f"  ⚠️  Official injury report failed: {e}")
        report = None
    if report is not None:
        print(f"  📋 Official NBA injury report: {sum(len(v) for v in report.values())} entries")
        return report

    print("  ⚠️  Official injury report unavailable — falling back to ESPN")
    from data.ingestion.fetch_injuries import fetch_injury_report
    return fetch_injury_report()
