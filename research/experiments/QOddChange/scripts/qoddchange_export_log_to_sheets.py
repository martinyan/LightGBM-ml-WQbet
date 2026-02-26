#!/usr/bin/env python3
"""Append QOddChange summary rows into a single Google Sheets tab: QODDCHANGE_LOG.

Designed to run at T-5m per race.

It reads the per-race report JSON produced by qoddchange_report_race_only.py and appends a row.
"""

import argparse, json
from datetime import datetime, timezone
import urllib.request, urllib.parse

from scripts.gsheets_raceday import get_or_create_raceday_sheet


def req_json(url, method='GET', data=None, headers=None):
    if headers is None:
        headers = {}
    req = urllib.request.Request(url, method=method, data=data)
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def ensure_tab(sheet_id, tab_name, auth):
    meta = req_json(f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{sheet_id}', headers=auth)
    existing = {s['properties']['title'] for s in meta.get('sheets', [])}
    if tab_name not in existing:
        body = json.dumps({'requests': [{'addSheet': {'properties': {'title': tab_name}}}]}).encode()
        req_json(f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{sheet_id}:batchUpdate', method='POST', data=body, headers=auth)


def clear_and_put(sheet_id, tab_name, values, auth):
    rng_clear = urllib.parse.quote(f'{tab_name}!A1:ZZ')
    req_json(f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{sheet_id}/values/{rng_clear}:clear', method='POST', data=b'{}', headers=auth)
    rng_put = urllib.parse.quote(f'{tab_name}!A1')
    body = json.dumps({'values': values}, ensure_ascii=False).encode('utf-8')
    req_json(f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{sheet_id}/values/{rng_put}?valueInputOption=USER_ENTERED', method='PUT', data=body, headers=auth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--racedate', required=True, help='YYYY/MM/DD')
    ap.add_argument('--report', required=True, help='Path to qoddchange report json')
    ap.add_argument('--tab', default='QODDCHANGE_LOG')
    args = ap.parse_args()

    import os
    maton = os.environ.get('MATON_API_KEY')
    if not maton:
        raise SystemExit('MATON_API_KEY not set')
    auth = {'Authorization': f'Bearer {maton}', 'Content-Type': 'application/json'}

    date_dash = args.racedate.replace('/', '-')
    sheet_id = get_or_create_raceday_sheet(date_dash)

    rep = json.load(open(args.report, 'r', encoding='utf-8'))

    # Flatten top pairs/horses into compact strings
    top_pairs = rep.get('lead_pairs') or []
    top_horses = rep.get('lead_horses') or []

    pairs_str = '; '.join([f"{x['pair']}({x['drop_ratio']:.3f})" for x in top_pairs[:10] if x.get('drop_ratio') is not None])
    horses_str = '; '.join([f"{x['horse_no']}({x['drop_score']:.3f}|{x.get('pairs_dropping',0)})" for x in top_horses[:10] if x.get('drop_score') is not None])

    ts = datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
    row = {
        'ts': ts,
        'racedate': args.racedate,
        'venue': rep.get('venue'),
        'raceNo': rep.get('raceNo'),
        'total_pairs_seen': rep.get('total_pairs_seen'),
        'top_pairs': pairs_str,
        'top_horses': horses_str,
    }

    ensure_tab(sheet_id, args.tab, auth)

    # header detection
    rng_get = urllib.parse.quote(f'{args.tab}!A1:Z1')
    try:
        head = req_json(f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{sheet_id}/values/{rng_get}', headers=auth)
        has_header = bool((head.get('values') or []) and (head['values'][0] or []))
    except Exception:
        has_header = False

    cols = list(row.keys())
    if not has_header:
        clear_and_put(sheet_id, args.tab, [cols, [row.get(c) for c in cols]], auth)
    else:
        rng_app = urllib.parse.quote(f'{args.tab}!A1')
        body = json.dumps({'values': [[row.get(c) for c in cols]]}, ensure_ascii=False).encode('utf-8')
        req_json(
            f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{sheet_id}/values/{rng_app}:append?valueInputOption=USER_ENTERED',
            method='POST',
            data=body,
            headers=auth,
        )

    print(json.dumps({'ok': True, 'sheetId': sheet_id, 'tab': args.tab, 'raceNo': rep.get('raceNo')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
