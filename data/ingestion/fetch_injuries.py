import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import requests

ESPN_TO_NBA = {
    'GS':   'GSW',
    'NO':   'NOP',
    'SA':   'SAS',
    'UTAH': 'UTA',
    'WSH':  'WAS',
}

OUT_PHRASES = [
    "ruled out",
    "won't play",
    "will not play",
    "won't be available",
    "will not be available",
    "unavailable",
    "inactive",
    "sidelined",
    "remains out",
    "has been ruled out",
]

AVAILABLE_PHRASES = [
    "available",
    "expected to play",
    "will play",
    "good to go",
    "not on the injury report",
]


def normalize_status(status, detail="", type_="", short="", long=""):
    text = " ".join([
        str(status or ""),
        str(detail or ""),
        str(type_ or ""),
        str(short or ""),
        str(long or ""),
    ]).lower()

    # Comment text should override broad ESPN status
    if any(phrase in text for phrase in OUT_PHRASES):
        return "Out"

    if any(phrase in text for phrase in AVAILABLE_PHRASES):
        return "Available"

    return status


def fetch_injury_report():
    """
    Fetch current NBA injury report from ESPN.
    Returns dict: {team_abbr: [{'name': str, 'status': str, 'detail': str}]}
    No API key required.
    """
    url = (
        "https://site.api.espn.com/apis/site/v2/"
        "sports/basketball/nba/injuries"
    )

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"  ⚠️  ESPN API error: {response.status_code}")
            return {}

        data = response.json()
        injuries = {}

        for team_entry in data.get("injuries", []):
            for injury in team_entry.get("injuries", []):
                athlete = injury.get("athlete", {})
                team_info = athlete.get("team", {})
                abbr = team_info.get("abbreviation", "")

                abbr = ESPN_TO_NBA.get(abbr, abbr)
                if not abbr:
                    continue

                name = athlete.get("displayName", "Unknown")
                raw_status = injury.get("status", "")
                detail = injury.get("details", {}).get("detail", "")
                type_ = injury.get("details", {}).get("type", "")
                short = injury.get("shortComment", "")
                long = injury.get("longComment", "")

                status = normalize_status(
                    raw_status,
                    detail=detail,
                    type_=type_,
                    short=short,
                    long=long,
                )

                if status == "Available":
                    continue

                if status in ["Out", "Doubtful", "Questionable", "Day-To-Day"]:
                    if abbr not in injuries:
                        injuries[abbr] = []

                    existing = [
                        p for p in injuries[abbr]
                        if p["name"] == name
                    ]

                    if not existing:
                        injuries[abbr].append({
                            "name": name,
                            "status": status,
                            "raw_status": raw_status,
                            "detail": detail,
                            "type": type_,
                            "note": short[:160] if short else "",
                        })

        return injuries

    except Exception as e:
        print(f"  ⚠️  Failed to fetch injuries: {e}")
        return {}


def print_injury_report(injuries):
    """Print formatted injury report."""
    if not injuries:
        print("No injuries found")
        return

    print(f"\n{'='*50}")
    print("NBA INJURY REPORT")
    print(f"{'='*50}")

    for team, players in sorted(injuries.items()):
        print(f"\n{team}:")
        for p in players:
            status_emoji = {
                "Out":          "❌",
                "Doubtful":     "🔴",
                "Questionable": "🟡",
                "Day-To-Day":   "🟠",
            }.get(p["status"], "⚪")

            detail = f" — {p['type']}" if p.get("type") else ""

            raw = ""
            if p.get("raw_status") and p["raw_status"] != p["status"]:
                raw = f" [ESPN: {p['raw_status']}]"

            print(
                f"  {status_emoji} {p['name']} "
                f"({p['status']}){raw}{detail}"
            )


if __name__ == "__main__":
    print("Fetching NBA injury report...")
    injuries = fetch_injury_report()
    print_injury_report(injuries)
    print(f"\nTotal teams with injuries: {len(injuries)}")
    print(f"Total injured players: {sum(len(v) for v in injuries.values())}")