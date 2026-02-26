#!/usr/bin/env python3
"""Production meeting runner using:
- racing.hkjc.com racecard for runner meta (no odds)
- info.cld.hkjc.com GraphQL for WIN odds
Then:
- build features from SQLite
- run GoldenWinBet/GoldenQbet
- export to Google Sheets

This replaces the brittle bet.hkjc HTML scrape when the page is JS-rendered.
"""

import os, json, argparse, subprocess
from datetime import datetime


def sh(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--racedate', required=True, help='YYYY/MM/DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--races', required=True, help='e.g. 1-9')
    ap.add_argument('--prod', default='prod/HKJC_PROD_WQ.json')
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--outDir', default='reports/PROD_PRED')
    ap.add_argument('--sheetId', default=None, help='Optional explicit spreadsheetId (overrides raceday sheet)')
    ap.add_argument('--sheetName', default=None, help='Tab base name (if omitted, uses PROD_PRED_<date>_<venue>_<HHMM>)')
    ap.add_argument('--racedaySheet', action='store_true', help='Write to a dedicated raceday spreadsheet (YYYYMMDD raceday bet pick)')
    ap.add_argument('--noRacedaySheet', action='store_true', help='Disable raceday spreadsheet behavior')
    args = ap.parse_args()

    maton = os.environ.get('MATON_API_KEY')
    if not maton:
        raise SystemExit('MATON_API_KEY not set')

    os.makedirs(args.outDir, exist_ok=True)

    racedate_dash = args.racedate.replace('/', '-')

    # pick spreadsheet: one per raceday by default
    use_raceday = (not args.noRacedaySheet)
    if use_raceday and args.sheetId is None:
        from scripts.gsheets_raceday import get_or_create_raceday_sheet
        args.sheetId = get_or_create_raceday_sheet(racedate_dash)
    if args.sheetId is None:
        # fallback to env
        args.sheetId = os.environ.get('HKJC_SHEET_ID')
    if not args.sheetId:
        raise SystemExit('No sheetId available (set HKJC_SHEET_ID or enable raceday sheet with MATON)')

    # tab base name includes time for each production run
    hhmm = datetime.now().strftime('%H%M')
    sheet = args.sheetName or f"PROD_{racedate_dash}_{args.venue.upper()}_{hhmm}"

    # parse race list
    if '-' in args.races:
        a, b = args.races.split('-')
        race_list = list(range(int(a), int(b) + 1))
    else:
        race_list = [int(x.strip()) for x in args.races.split(',') if x.strip()]

    # per-race: build merged racecard+odds json -> features -> prod predict
    summary_rows = []
    runner_rows = []

    for rn in race_list:
        merged_path = os.path.join(args.outDir, f'merged_{racedate_dash}_{args.venue.upper()}_R{rn}.json')
        sh(['python3', 'hkjc_scrape_racecard_and_win_odds.py', '--racedate', args.racedate, '--venue', args.venue, '--raceNo', str(rn), '--out', merged_path])

        feat_path = os.path.join(args.outDir, f'features_{racedate_dash}_{args.venue.upper()}_R{rn}.json')
        sh(['node', 'hkjc_build_feature_rows_from_racecard_sqlite.mjs', '--db', args.db, '--in', merged_path, '--out', feat_path, '--lastN', '3'])

        pred_path = os.path.join(args.outDir, f'pred_{racedate_dash}_{args.venue.upper()}_R{rn}.json')
        sh(['python3', 'hkjc_prod_predict_single_race_wq.py', '--prod', args.prod, '--in', feat_path, '--out', pred_path])
        pred = json.load(open(pred_path, 'r', encoding='utf-8'))

        wtop1 = pred['W']['top1']
        summary_rows.append({
            'racedate': pred.get('racedate'),
            'venue': pred.get('venue'),
            'raceNo': int(pred.get('raceNo')),
            'W_model': pred['W'].get('name'),
            'Q_model': pred['Q'].get('name'),
            'W_top1_no': wtop1.get('horse_no'),
            'W_top1_horse': wtop1.get('horse'),
            'W_top1_odds': wtop1.get('cur_win_odds'),
            'W_p_mkt_win': wtop1.get('p_mkt_win'),
            'W_score_win': wtop1.get('score_win'),
            'W_overlay': wtop1.get('overlay'),
            'W_pass_thr_0.16': bool(wtop1['overlay'] > 0.16),
            'W_pass_thr_0.18': bool(wtop1['overlay'] > 0.18),
            'W_pass_thr_0.20': bool(wtop1['overlay'] > 0.20),
            'Q_anchor_no': pred['Q'].get('anchor_horse_no'),
            'Q_p2_no': (pred['Q'].get('partner2') or {}).get('horse_no'),
            'Q_p2_horse': (pred['Q'].get('partner2') or {}).get('horse'),
            'Q_p2_ranker_score': (pred['Q'].get('partner2') or {}).get('ranker_score'),
            'Q_p3_no': (pred['Q'].get('partner3') or {}).get('horse_no'),
            'Q_p3_horse': (pred['Q'].get('partner3') or {}).get('horse'),
            'Q_p3_ranker_score': (pred['Q'].get('partner3') or {}).get('ranker_score'),
        })

        # runner-level dump
        rank_map = {int(x['horse_no']): x for x in (pred['Q'].get('ranker_scored_all') or [])}
        for rr in (pred['W'].get('scored_all') or []):
            hn = int(rr['horse_no'])
            rk = rank_map.get(hn) or {}
            runner_rows.append({
                'racedate': pred.get('racedate'),
                'venue': pred.get('venue'),
                'raceNo': int(pred.get('raceNo')),
                'horse_no': hn,
                'horse': rr.get('horse'),
                'cur_win_odds': rr.get('cur_win_odds'),
                'p_mkt_win': rr.get('p_mkt_win'),
                'score_win': rr.get('score_win'),
                'overlay': rr.get('overlay'),
                'ranker_score': rk.get('ranker_score'),
                'is_W_top1': hn == int(wtop1.get('horse_no')),
                'is_Q_anchor': hn == int(pred['Q'].get('anchor_horse_no') or -1),
            })

    # export to sheets using maton gateway (reuse small inline client)
    import urllib.request, urllib.parse

    def req_json(url, method='GET', data=None, headers=None):
        if headers is None:
            headers = {}
        req = urllib.request.Request(url, method=method, data=data)
        for k, v in headers.items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.load(resp)

    auth = {'Authorization': f'Bearer {maton}', 'Content-Type': 'application/json'}

    def ensure_sheet(title):
        meta = req_json(f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{args.sheetId}', headers=auth)
        existing = {s['properties']['title'] for s in meta.get('sheets', [])}
        if title not in existing:
            body = json.dumps({'requests': [{'addSheet': {'properties': {'title': title}}}]}).encode()
            req_json(f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{args.sheetId}:batchUpdate', method='POST', data=body, headers=auth)

    def clear_and_put(title, values):
        rng_clear = urllib.parse.quote(f'{title}!A1:ZZ')
        req_json(f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{args.sheetId}/values/{rng_clear}:clear', method='POST', data=b'{}', headers=auth)
        rng_put = urllib.parse.quote(f'{title}!A1')
        body = json.dumps({'values': values}, ensure_ascii=False).encode('utf-8')
        req_json(f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{args.sheetId}/values/{rng_put}?valueInputOption=USER_ENTERED', method='PUT', data=body, headers=auth)

    tabA = sheet
    tabB = f'{sheet}_RUNNERS'

    ensure_sheet(tabA)
    ensure_sheet(tabB)

    # values
    summary_rows.sort(key=lambda r: r['raceNo'])
    runner_rows.sort(key=lambda r: (r['raceNo'], r['horse_no']))

    colsA = list(summary_rows[0].keys()) if summary_rows else []
    valsA = [colsA] + [[r.get(c) for c in colsA] for r in summary_rows]

    colsB = list(runner_rows[0].keys()) if runner_rows else []
    valsB = [colsB] + [[r.get(c) for c in colsB] for r in runner_rows]

    clear_and_put(tabA, valsA)
    clear_and_put(tabB, valsB)

    print(json.dumps({'ok': True, 'sheetId': args.sheetId, 'tabs': [tabA, tabB], 'races': len(summary_rows), 'runnerRows': len(runner_rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
