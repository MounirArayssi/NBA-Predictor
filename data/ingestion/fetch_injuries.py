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

        data     = response.json()
        injuries = {}

        for team_entry in data.get('injuries', []):
            # Team name at top level
            team_name = team_entry.get('displayName', '')

            for injury in team_entry.get('injuries', []):
                # Team abbreviation is inside athlete.team
                athlete   = injury.get('athlete', {})
                team_info = athlete.get('team', {})
                abbr      = team_info.get('abbreviation', '')

                abbr = ESPN_TO_NBA.get(abbr, abbr)
                if not abbr:
                    continue

                name   = athlete.get('displayName', 'Unknown')
                status = injury.get('status', '')
                detail = injury.get('details', {}).get('detail', '')
                type_  = injury.get('details', {}).get('type', '')
                short  = injury.get('shortComment', '')

                if status in [
                    'Out', 'Doubtful',
                    'Questionable', 'Day-To-Day'
                ]:
                    if abbr not in injuries:
                        injuries[abbr] = []

                    # Avoid duplicates
                    existing = [
                        p for p in injuries[abbr]
                        if p['name'] == name
                    ]
                    if not existing:
                        injuries[abbr].append({
                            'name':   name,
                            'status': status,
                            'detail': detail,
                            'type':   type_,
                            'note':   short[:100] if short else ''
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
                'Out':          '❌',
                'Doubtful':     '🔴',
                'Questionable': '🟡',
                'Day-To-Day':   '🟠',
            }.get(p['status'], '⚪')

            detail = f" — {p['type']}" if p['type'] else ""
            print(f"  {status_emoji} {p['name']} "
                  f"({p['status']}){detail}")


if __name__ == "__main__":
    print("Fetching NBA injury report...")
    injuries = fetch_injury_report()
    print_injury_report(injuries)
    print(f"\nTotal teams with injuries: {len(injuries)}")
    print(f"Total injured players: "
          f"{sum(len(v) for v in injuries.values())}")