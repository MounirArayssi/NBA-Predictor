"""
Plain-English "why" for a game, written by fixed rules from the raw-stat
breakdowns (analytics/breakdown.py). No text generation: every number in a
sentence is a value from the breakdown, and a sentence about an input is
written only when that input's breakdown reproduced the stored value.

Per team, in order:
  1. Results:  record and scoring margin this season (plus last season's margin
               while it still carries weight).
  2. Offense / defense: the four-factor parts. The largest part is spelled out
               ("...which alone costs about 6 points per 100 possessions"); other
               notable parts are listed; "at everything" when all four agree.
  3. Availability: rotation players listed Out/Doubtful.
  4. Rest: second night of a back-to-back.
"""
from __future__ import annotations

NOTABLE = 1.0      # rating points per 100 possessions worth mentioning
ALL_MIN = 0.3      # each factor at least this size for "at everything"
NAMES = {
    "off": {"efg": "shooting", "tov": "ball security", "orb": "offensive rebounding", "ftr": "getting to the line"},
    "def": {"efg": "shooting defense", "tov": "forcing turnovers", "orb": "defensive rebounding", "ftr": "free throws allowed"},
}


def _pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def _signed(v):
    r = round(v)
    return f"+{r}" if r > 0 else f"−{abs(r)}" if r < 0 else "±0"


def _pts(v):
    a = abs(v)
    return f"{a:.1f}" if a < 3 else f"{a:.0f}"


def phrase(kind: str, k: str, team: float, lg: float, helps: float = 0.0) -> str:
    """What the team does on factor k, vs the league, as a verb phrase."""
    if kind == "def" and k == "efg" and helps > 0:
        return f"holds opponents to {_pct(team)} eFG against a {_pct(lg)} league average"
    if kind == "off":
        return {
            "efg": f"shoots {_pct(team)} eFG against a {_pct(lg)} league average",
            "tov": f"turns it over on {_pct(team, 1)} of possessions (league {_pct(lg, 1)})",
            "orb": f"rebounds {_pct(team)} of its own misses (league {_pct(lg)})",
            "ftr": f"makes {100 * team:.0f} free throws per 100 shots (league {100 * lg:.0f})",
        }[k]
    return {
        "efg": f"lets opponents shoot {_pct(team)} eFG against a {_pct(lg)} league average",
        "tov": f"forces turnovers on {_pct(team, 1)} of opponent possessions (league {_pct(lg, 1)})",
        "orb": f"lets opponents rebound {_pct(team)} of their misses (league {_pct(lg)})",
        "ftr": f"allows {100 * team:.0f} free throws per 100 opponent shots (league {100 * lg:.0f})",
    }[k]


def side_items(kind: str, x: dict) -> list[tuple[str, float]]:
    """(factor, helps) pairs; helps > 0 means the factor helps this team.
    Defense parts are 'more points allowed', so they are flipped."""
    sgn = -1 if kind == "def" else 1
    out = []
    for k in ("efg", "tov", "orb", "ftr"):
        if k == "orb" and x.get("orb_missing"):
            continue
        out.append((k, sgn * x["parts"][k]))
    return out


def unit_sentences(kind: str, x: dict, lg: dict) -> list[str]:
    if not x or not x.get("ok") or not x.get("parts"):
        return []
    unit = "offense" if kind == "off" else "defense"
    items = side_items(kind, x)
    ranked = sorted(items, key=lambda kv: -abs(kv[1]))
    notable = [kv for kv in ranked if abs(kv[1]) >= NOTABLE]
    out = []
    if notable:
        k, h = notable[0]
        out.append(f"Their {unit} {phrase(kind, k, x['f'][k], lg['f'][k], h)}, which alone "
                   f"{'adds' if h > 0 else 'costs'} about {_pts(h)} points per 100 possessions.")
    full = len(items) == 4
    if full and all(h <= -ALL_MIN for _, h in items):
        out.append(f"Their {unit} is below average at everything: shooting, turnovers, rebounding and free throws.")
    elif full and all(h >= ALL_MIN for _, h in items):
        out.append(f"Their {unit} is above average at everything: shooting, turnovers, rebounding and free throws.")
    elif len(notable) > 1:
        bits = [f"{NAMES[kind][k]} {'+' if h > 0 else '−'}{_pts(h)}" for k, h in notable[1:3]]
        out.append(f"Smaller factors: {', '.join(bits)}.")
    if not notable and not out:
        out.append(f"Their {unit} is close to league average on all four factors.")
    return out


def results_sentence(abbr: str, el: dict, m: dict) -> str | None:
    if not m or not m.get("ok"):
        return None
    if not m["n"]:
        base = f"{abbr} hasn't played this season yet"
        return base + (f"; last season they were a {_signed(m['last'])} team." if m.get("last") is not None else ".")
    w = m.get("w_n")
    rec = f"{w}–{m['n'] - w}" if w is not None else f"{m['n']} games in"
    cur = m["cur"]
    verb = "winning" if cur > 0 else "losing"
    s = (f"{abbr} is {rec}, {verb} by {abs(cur):.0f} a game on average "
         f"(scoring {m['pf']:.0f}, allowing {m['pa']:.0f}).") if abs(cur) >= 0.5 else (
        f"{abbr} is {rec}, about even on points (scoring {m['pf']:.0f}, allowing {m['pa']:.0f}).")
    if m.get("w", 1) < 1 and m.get("last") is not None:
        s += f" Last season they were a {_signed(m['last'])} team, which still counts for {100 * (1 - m['w']):.0f}% of their season margin."
    return s


def availability_sentence(abbr: str, mi: dict) -> str | None:
    if not mi or not mi.get("ok") or not mi.get("players"):
        return None
    ps = mi["players"]
    if len(ps) == 1:
        return f"{ps[0]['n']} ({ps[0]['ppg']:.0f} ppg) is out."
    names = ", ".join(p["n"] for p in ps[:-1]) + f" and {ps[-1]['n']}"
    return f"{names} are out ({mi['v']:.0f} rotation points a game combined)."


def team_story(abbr: str, s: dict, lg: dict) -> list[str]:
    out = []
    r = results_sentence(abbr, s.get("elo"), s.get("margin"))
    if r:
        out.append(r)
    out += unit_sentences("off", s.get("off"), lg)
    out += unit_sentences("def", s.get("def"), lg)
    a = availability_sentence(abbr, s.get("missing"))
    if a:
        out.append(a)
    rest = s.get("rest") or {}
    if rest.get("ok") and rest.get("v") == 1:
        out.append("They're on the second night of a back-to-back.")
    return out
