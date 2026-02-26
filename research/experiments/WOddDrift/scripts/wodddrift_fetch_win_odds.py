#!/usr/bin/env python3
"""Fetch WIN odds for a single race from HKJC GraphQL.

Outputs one JSON snapshot to stdout.
"""

import argparse, json, gzip, urllib.request
from datetime import datetime, timezone

UA = 'openclaw-wodddrift/1.0'

# Must match whitelisted template used by bet.hkjc.com
WHITELISTED_QUERY = """query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
          raceMeetings(date: $date, venueCode: $venueCode)
          {
            pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
              id
              status
              sellStatus
              oddsType
              lastUpdateTime
              guarantee
              minTicketCost
              name_en
              name_ch
              leg {
                number
                races
              }
              cWinSelections {
                composite
                name_ch
                name_en
                starters
              }
              oddsNodes {
                combString
                oddsValue
                hotFavourite
                oddsDropValue
                bankerOdds {
                  combString
                  oddsValue
                }
              }
            }
          }
      }"""


def gql(date: str, venue: str, race_no: int):
    url = 'https://info.cld.hkjc.com/graphql/base/'
    variables = {'date': date, 'venueCode': venue, 'oddsTypes': ['WIN'], 'raceNo': int(race_no)}
    body = json.dumps({'query': WHITELISTED_QUERY, 'variables': variables}).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            'Content-Type': 'application/json',
            'User-Agent': UA,
            'Accept-Encoding': 'gzip',
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
        if resp.headers.get('Content-Encoding') == 'gzip':
            data = gzip.decompress(data)
        return json.loads(data.decode('utf-8', 'ignore'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--raceNo', required=True, type=int)
    args = ap.parse_args()

    obj = gql(args.date, args.venue.upper(), args.raceNo)
    mtg = (((obj.get('data') or {}).get('raceMeetings') or [None])[0] or {})
    pools = mtg.get('pmPools') or []
    win = next((p for p in pools if p.get('oddsType') == 'WIN'), None)

    snap = {
        'ts': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'date': args.date,
        'venue': args.venue.upper(),
        'raceNo': int(args.raceNo),
        'pool': 'WIN',
        'status': (win or {}).get('status'),
        'sellStatus': (win or {}).get('sellStatus'),
        'lastUpdateTime': (win or {}).get('lastUpdateTime'),
        'odds': {}
    }

    if win and win.get('oddsNodes'):
        for n in win['oddsNodes']:
            cs = n.get('combString')
            ov = n.get('oddsValue')
            if cs is None or ov is None:
                continue
            try:
                hn = int(cs)
                snap['odds'][str(hn)] = float(ov)
            except Exception:
                pass

    print(json.dumps(snap, ensure_ascii=False))


if __name__ == '__main__':
    main()
