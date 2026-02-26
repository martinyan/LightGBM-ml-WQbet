#!/usr/bin/env python3
"""Google Sheets helpers: one spreadsheet per raceday.

Uses Maton gateway (MATON_API_KEY).

- get_or_create_raceday_sheet(date_dash, title=None) -> spreadsheetId
- persist mapping in prod/sheets/raceday_sheets.json (local state; not meant for git)

Date format:
- date_dash: YYYY-MM-DD

Default title:
- YYYYMMDD raceday bet pick
"""

import json, os
from pathlib import Path
import urllib.request

STATE_PATH = Path('prod/sheets/raceday_sheets.json')


def _req_json(url, method='GET', data=None, headers=None, timeout=120):
    if headers is None:
        headers = {}
    req = urllib.request.Request(url, method=method, data=data)
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _auth_headers():
    key = os.environ.get('MATON_API_KEY')
    if not key:
        raise RuntimeError('MATON_API_KEY not set')
    return {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}


def _load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding='utf-8'))
    return {}


def _save_state(obj):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def _default_title(date_dash: str) -> str:
    ymd = date_dash.replace('-', '')
    return f'{ymd} raceday bet pick'


def get_or_create_raceday_sheet(date_dash: str, title: str | None = None) -> str:
    title = title or _default_title(date_dash)
    st = _load_state()
    if date_dash in st and st[date_dash].get('spreadsheetId'):
        return st[date_dash]['spreadsheetId']

    auth = _auth_headers()

    # Create spreadsheet (Sheets API)
    body = json.dumps({'properties': {'title': title}}).encode('utf-8')
    created = _req_json('https://gateway.maton.ai/google-sheets/v4/spreadsheets', method='POST', data=body, headers=auth)
    sid = created.get('spreadsheetId')
    if not sid:
        raise RuntimeError(f'Failed to create spreadsheet: {created}')

    st[date_dash] = {'spreadsheetId': sid, 'title': title}
    _save_state(st)
    return sid


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    args = ap.parse_args()
    print(get_or_create_raceday_sheet(args.date))
