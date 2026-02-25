#!/usr/bin/env python3
"""Fetch Quinella (QIN) odds for a single race from HKJC GraphQL.

Writes one JSON snapshot to stdout or to --out (JSON).

Example:
  python3 qoddchange_fetch_qin_odds.py --date 2026-02-25 --venue HV --raceNo 1

Note:
- GraphQL responses are gzip-compressed; we handle decompression.
"""

import argparse, json, gzip, urllib.request

UA = 'openclaw-qoddchange/1.0'

# IMPORTANT: HKJC GraphQL enforces a query whitelist.
# This query string must match the whitelisted template used by bet.hkjc.com.
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


def gql(date: str, venue: str, race_no: int, odds_type: str):
    url = 'https://info.cld.hkjc.com/graphql/base/'
    variables = {'date': date, 'venueCode': venue, 'oddsTypes': [odds_type], 'raceNo': int(race_no)}
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
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    obj = gql(args.date, args.venue.upper(), args.raceNo, 'QIN')

    mtg = (((obj.get('data') or {}).get('raceMeetings') or [None])[0] or {})
    pools = mtg.get('pmPools') or []
    qin = next((p for p in pools if p.get('oddsType') == 'QIN'), None)

    snap = {
        'ts': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat().replace('+00:00','Z'),
        'date': args.date,
        'venue': args.venue.upper(),
        'raceNo': int(args.raceNo),
        'pool': 'QIN',
        'status': (qin or {}).get('status'),
        'sellStatus': (qin or {}).get('sellStatus'),
        'lastUpdateTime': (qin or {}).get('lastUpdateTime'),
        'odds': []
    }

    if qin and qin.get('oddsNodes'):
        for n in qin['oddsNodes']:
            cs = n.get('combString')
            ov = n.get('oddsValue')
            if cs is None or ov is None:
                continue
            snap['odds'].append({
                'comb': cs,
                'odds': float(ov) if str(ov).replace('.','',1).isdigit() else ov,
                'hot': bool(n.get('hotFavourite')),
                'drop': n.get('oddsDropValue'),
            })

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(snap, ensure_ascii=False))


if __name__ == '__main__':
    main()
