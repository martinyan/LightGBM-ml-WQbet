#!/usr/bin/env python3
"""Fetch Quinella (QIN) pool odds via HKJC GraphQL (info.cld.hkjc.com).

We reuse the existing whitelisted query string from hkjc_scrape_racecard_and_win_odds.py
so the API accepts it.

Output JSON to stdout:
{date, venue, raceNo, ts, odds: {"1-7": 8.5, ...}}

Notes:
- odds keys are normalized as "a-b" with a<b (horse numbers as ints)
- ts is ISO-8601 in UTC with Z.
"""

import argparse, json, gzip, os, sys
from datetime import datetime, timezone
import urllib.request

# allow importing from workspace root when executed from anywhere
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from hkjc_scrape_racecard_and_win_odds import WHITELISTED_QUERY

UA = 'openclaw-qindrift/1.0'


def fetch_qin(date_dash: str, venue: str, race_no: int):
    url = 'https://info.cld.hkjc.com/graphql/base/'
    vars = {'date': date_dash, 'venueCode': venue.upper(), 'oddsTypes': ['QIN'], 'raceNo': int(race_no)}
    body = json.dumps({'query': WHITELISTED_QUERY, 'variables': vars}).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=body,
        headers={'Content-Type': 'application/json', 'User-Agent': UA, 'Accept-Encoding': 'gzip'},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
        if resp.headers.get('Content-Encoding') == 'gzip':
            data = gzip.decompress(data)
    obj = json.loads(data.decode('utf-8', 'ignore'))
    if obj.get('errors'):
        raise SystemExit('GraphQL errors: ' + json.dumps(obj['errors'], ensure_ascii=False))

    pools = (((obj.get('data') or {}).get('raceMeetings') or [None])[0] or {}).get('pmPools') or []
    qin = next((p for p in pools if p.get('oddsType') == 'QIN'), None)
    nodes = (qin or {}).get('oddsNodes') or []

    odds = {}
    for n in nodes:
        cs = (n.get('combString') or '').strip()
        ov = (n.get('oddsValue') or '').strip()
        if not cs or not ov:
            continue
        # combString is like "01,07"
        parts = [p.strip() for p in cs.split(',') if p.strip()]
        if len(parts) != 2:
            continue
        try:
            a = int(parts[0])
            b = int(parts[1])
            if a == b:
                continue
            lo, hi = (a, b) if a < b else (b, a)
            key = f'{lo}-{hi}'
            odds[key] = float(ov)
        except Exception:
            continue

    return odds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--raceNo', required=True, type=int)
    args = ap.parse_args()

    odds = fetch_qin(args.date, args.venue, args.raceNo)
    out = {
        'date': args.date,
        'venue': args.venue.upper(),
        'raceNo': int(args.raceNo),
        'ts': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'odds': odds,
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
