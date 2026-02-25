#!/usr/bin/env python3
"""Scrape HKJC racecard runner meta (racing.hkjc.com) + WIN odds (info.cld.hkjc.com GraphQL).

Outputs a JSON compatible with hkjc_build_feature_rows_from_racecard_sqlite.mjs wrapper format:
{
  betPage: { url, distanceMeters, classNum, surfaceText },
  picks: [{no, horse, code, draw, wt, jockey, trainer, win, place:null}]
}

Notes:
- We intentionally ignore place odds (per user requirement).
- betPage.url is set to the bet.hkjc wp url so the feature builder can infer racedate/venue/raceNo.
"""

import argparse, json, re, urllib.request, urllib.parse, gzip

UA = 'openclaw-hkjc-racecard-scrape/1.0'

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


def fetch(url: str, headers=None) -> bytes:
    if headers is None:
        headers = {}
    h = {'User-Agent': UA, **headers}
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
        if resp.headers.get('Content-Encoding') == 'gzip':
            data = gzip.decompress(data)
        return data


def fetch_text(url: str) -> str:
    return fetch(url, headers={'Accept-Encoding': 'gzip'}).decode('utf-8', 'ignore')


def first_int(s):
    m = re.search(r'(\d+)', s or '')
    return int(m.group(1)) if m else None


def parse_race_meta(text: str):
    # distance like 1200米
    dist = first_int((re.search(r'(\d{3,4})\s*米', text) or [None, None])[1] if re.search(r'(\d{3,4})\s*米', text) else None)
    if dist is None:
        m = re.search(r'(\d{3,4})\s*米', text)
        dist = int(m.group(1)) if m else None

    # class like 第五班
    class_map = {'第一': 1, '第二': 2, '第三': 3, '第四': 4, '第五': 5}
    class_num = None
    m = re.search(r'第([一二三四五])班', text)
    if m:
        class_num = class_map.get(m.group(1))

    # surface
    surface = None
    if '草地' in text:
        surface = 'turf'
    if '全天候' in text or '泥地' in text:
        surface = 'awt'

    return dist, class_num, surface


def parse_racecard(html: str):
    """Parse the server-rendered racecard HTML.

    We locate <tr> rows that contain a horseid link, then extract a few td fields.
    Empirically, the important columns are:
      td[0]=horse no
      td[3]=horse name
      td[4]=horse code (e.g., H436)
      td[5]=carried weight
      td[6]=jockey name
      td[8]=draw
      td[9]=trainer name
    """

    # normalize text for meta parsing
    txt = re.sub(r'\s+', ' ', html.replace('\n', ' '))
    dist, class_num, surface = parse_race_meta(txt)

    rows = []
    seen_no = set()

    for tr in re.findall(r'<tr[^>]*>[\s\S]*?</tr>', html, flags=re.I):
        if 'horse?horseid=' not in tr:
            continue
        # skip standby/backup horses section
        if '後備馬匹' in tr or 'Standby' in tr:
            continue
        tds = re.findall(r'<td[^>]*>([\s\S]*?)</td>', tr, flags=re.I)
        if len(tds) < 10:
            continue

        def clean(s):
            s = re.sub(r'<br\s*/?>', ' ', s, flags=re.I)
            s = re.sub(r'<[^>]+>', ' ', s)
            s = s.replace('&nbsp;', ' ')
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        horse_no = clean(tds[0])
        if not horse_no.isdigit():
            continue
        horse_no = int(horse_no)
        if horse_no in seen_no:
            continue
        seen_no.add(horse_no)

        horse = clean(tds[3])
        code = clean(tds[4])
        wt = clean(tds[5])
        jockey = clean(tds[6])
        draw = clean(tds[8])
        trainer = clean(tds[9])

        try:
            wt = int(re.sub(r'[^0-9]', '', wt)) if wt else None
        except Exception:
            wt = None
        try:
            draw = int(re.sub(r'[^0-9]', '', draw)) if draw else None
        except Exception:
            draw = None

        # if code didn't parse, try from horseid
        if not re.match(r'^[A-Z]\d{3}$', code or ''):
            m = re.search(r'horse\?horseid=HK_\d{4}_([A-Z]\d{3})', tr)
            code = m.group(1) if m else code

        rows.append({'no': horse_no, 'horse': horse, 'code': code, 'draw': draw, 'wt': wt, 'jockey': jockey, 'trainer': trainer})

    return {
        'distanceMeters': dist,
        'classNum': class_num,
        'surfaceText': surface,
        'rows': rows,
        'rowsFound': len(rows)
    }


def fetch_win_odds(date_dash: str, venue: str, race_no: int):
    url = 'https://info.cld.hkjc.com/graphql/base/'
    vars = {'date': date_dash, 'venueCode': venue, 'oddsTypes': ['WIN'], 'raceNo': int(race_no)}
    body = json.dumps({'query': WHITELISTED_QUERY, 'variables': vars}).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            'Content-Type': 'application/json',
            'User-Agent': UA,
            'Accept-Encoding': 'gzip'
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
        if resp.headers.get('Content-Encoding') == 'gzip':
            data = gzip.decompress(data)
        obj = json.loads(data.decode('utf-8', 'ignore'))

    pools = (((obj.get('data') or {}).get('raceMeetings') or [None])[0] or {}).get('pmPools') or []
    win = next((p for p in pools if p.get('oddsType') == 'WIN'), None)
    if not win:
        return {}
    out = {}
    for n in win.get('oddsNodes') or []:
        comb = n.get('combString')
        val = n.get('oddsValue')
        if comb and val:
            try:
                out[int(comb)] = float(val)
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--racedate', required=True, help='YYYY/MM/DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--raceNo', required=True, type=int)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    racedate_slash = args.racedate
    date_dash = racedate_slash.replace('/', '-')
    venue = args.venue.upper()

    racecard_url = f'https://racing.hkjc.com/zh-hk/local/information/racecard?racedate={urllib.parse.quote(racedate_slash)}&Racecourse={venue}&RaceNo={int(args.raceNo)}'
    html = fetch_text(racecard_url)
    rc = parse_racecard(html)

    if rc['rowsFound'] <= 0:
        raise SystemExit(f'No runners parsed from racecard page (rowsFound=0). url={racecard_url}')

    win_odds = fetch_win_odds(date_dash, venue, int(args.raceNo))

    # merge odds into rows; keep only declared runners that have WIN odds
    picks = []
    for r in rc['rows']:
        no = int(r['no'])
        if no not in win_odds:
            continue
        picks.append({
            **r,
            'win': win_odds.get(no),
            'place': None,
        })

    bet_url = f'https://bet.hkjc.com/ch/racing/wp/{date_dash}/{venue}/{int(args.raceNo)}'

    out = {
        'betPage': {
            'url': bet_url,
            'distanceMeters': rc['distanceMeters'],
            'classNum': rc['classNum'],
            'surfaceText': rc['surfaceText'],
            'racecardUrl': racecard_url,
        },
        'picks': picks,
        'meta': {
            'racedate': racedate_slash,
            'venue': venue,
            'raceNo': int(args.raceNo),
            'winOddsCount': len(win_odds),
        }
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({'ok': True, 'out': args.out, 'runners': len(picks), 'winOddsCount': len(win_odds)}, ensure_ascii=False))


if __name__ == '__main__':
    import os
    main()
