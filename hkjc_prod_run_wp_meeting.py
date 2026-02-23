import os, json, argparse, subprocess, glob, urllib.parse
from datetime import datetime

def sh(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    return r.stdout


def maton_req_json(url, method='GET', data=None, headers=None, timeout=60):
    import urllib.request
    if headers is None:
        headers = {}
    req = urllib.request.Request(url, method=method, data=data)
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def ensure_sheet(spreadsheet_id, sheet_name, maton_key):
    auth = {'Authorization': f'Bearer {maton_key}', 'Content-Type': 'application/json'}
    meta = maton_req_json(f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{spreadsheet_id}', headers=auth)
    existing = {s['properties']['title'] for s in meta.get('sheets', [])}
    if sheet_name not in existing:
        body = json.dumps({'requests': [{'addSheet': {'properties': {'title': sheet_name}}}]}).encode()
        maton_req_json(f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{spreadsheet_id}:batchUpdate', method='POST', data=body, headers=auth)


def clear_and_put(spreadsheet_id, sheet_name, values, maton_key):
    auth = {'Authorization': f'Bearer {maton_key}', 'Content-Type': 'application/json'}
    rng_clear = urllib.parse.quote(f'{sheet_name}!A1:ZZ')
    maton_req_json(f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{spreadsheet_id}/values/{rng_clear}:clear', method='POST', data=b'{}', headers=auth)
    rng_put = urllib.parse.quote(f'{sheet_name}!A1')
    body = json.dumps({'values': values}, ensure_ascii=False).encode('utf-8')
    maton_req_json(
        f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{spreadsheet_id}/values/{rng_put}?valueInputOption=USER_ENTERED',
        method='PUT',
        data=body,
        headers=auth,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--racedate', required=True, help='YYYY-MM-DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--races', required=True, help='e.g. 1-9')
    ap.add_argument('--prod', default='prod/HKJC_PROD_WQ.json')
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--outDir', default='reports/PROD_PRED')
    ap.add_argument('--sheetId', default=os.environ.get('HKJC_SHEET_ID', '109I84syg24sJCl9QxP0soUUx63BvmsS8dJCTEs8MrbI'))
    ap.add_argument('--sheetName', default=None)
    ap.add_argument('--thrAlt', type=float, default=0.16, help='also compute pass flag for this alternate overlay threshold')
    args = ap.parse_args()

    os.makedirs(args.outDir, exist_ok=True)
    maton_key = os.environ.get('MATON_API_KEY')
    if not maton_key:
        raise SystemExit('MATON_API_KEY not set')

    sheet_name = args.sheetName or f"PROD_PRED_{args.racedate}_{args.venue}"

    # 1) scrape bet.hkjc wp pages
    scrape_dir = os.path.join(args.outDir, f"scrape_{args.racedate}_{args.venue}")
    os.makedirs(scrape_dir, exist_ok=True)
    sh(['node', 'hkjc_scrape_wp_meeting.mjs', '--racedate', args.racedate, '--venue', args.venue, '--races', args.races, '--outDir', scrape_dir])

    scrape_files = sorted(glob.glob(os.path.join(scrape_dir, f"{args.venue.lower()}_{args.racedate}_race*_bet_scrape.json")))
    if not scrape_files:
        raise SystemExit('no scrape files produced')

    rows_all = []

    # load prod cfg for thresholds
    prod_cfg = json.load(open(args.prod, 'r', encoding='utf-8'))
    thr_main = float(prod_cfg['W']['threshold'])

    for sf in scrape_files:
        card = json.load(open(sf, 'r', encoding='utf-8'))
        race_no = int(card.get('raceNo'))
        # wrap into expected format for feature builder
        wrapped = {
            'betPage': {
                'url': card.get('url'),
                'distanceMeters': card.get('distanceMeters'),
                'classNum': None,
                'surfaceText': None,
            },
            'picks': card.get('rows') or []
        }
        wrapped_path = os.path.join(args.outDir, f"wrapped_{args.racedate}_{args.venue}_R{race_no}.json")
        with open(wrapped_path, 'w', encoding='utf-8') as f:
            json.dump(wrapped, f, ensure_ascii=False, indent=2)

        feat_path = os.path.join(args.outDir, f"features_{args.racedate}_{args.venue}_R{race_no}.json")
        sh(['node', 'hkjc_build_feature_rows_from_racecard_sqlite.mjs', '--db', args.db, '--in', wrapped_path, '--out', feat_path, '--lastN', '3'])

        pred_path = os.path.join(args.outDir, f"pred_{args.racedate}_{args.venue}_R{race_no}.json")
        sh(['python3', 'hkjc_prod_predict_single_race_wq.py', '--prod', args.prod, '--in', feat_path, '--out', pred_path])
        pred = json.load(open(pred_path, 'r', encoding='utf-8'))

        wtop1 = pred['W']['top1']
        pass_main = bool(wtop1['overlay'] > thr_main)
        pass_alt = bool(wtop1['overlay'] > float(args.thrAlt))

        p2 = pred['Q'].get('partner2') or {}
        p3 = pred['Q'].get('partner3') or {}

        rows_all.append({
            'racedate': pred.get('racedate'),
            'venue': pred.get('venue'),
            'raceNo': int(pred.get('raceNo')),
            'W_top1_no': wtop1.get('horse_no'),
            'W_top1_horse': wtop1.get('horse'),
            'W_top1_odds': wtop1.get('cur_win_odds'),
            'W_p_mkt_win': wtop1.get('p_mkt_win'),
            'W_score_win': wtop1.get('score_win'),
            'W_overlay': wtop1.get('overlay'),
            f'W_pass_thr_{thr_main:.2f}': pass_main,
            f'W_pass_thr_{float(args.thrAlt):.2f}': pass_alt,
            'Q_p2_no': p2.get('horse_no'),
            'Q_p2_horse': p2.get('horse'),
            'Q_p2_ranker_score': p2.get('ranker_score'),
            'Q_p3_no': p3.get('horse_no'),
            'Q_p3_horse': p3.get('horse'),
            'Q_p3_ranker_score': p3.get('ranker_score'),
        })

    # sort by race
    rows_all.sort(key=lambda r: r['raceNo'])

    # 2) write a local CSV-like table to json
    out_json = os.path.join(args.outDir, f"meeting_pred_{args.racedate}_{args.venue}.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'generatedAt': datetime.now().isoformat(timespec='seconds'), 'prod': prod_cfg, 'rows': rows_all}, f, ensure_ascii=False, indent=2)

    # 3) push to sheets
    headers = list(rows_all[0].keys()) if rows_all else []
    values = [headers]
    for r in rows_all:
        values.append([r.get(h) for h in headers])

    ensure_sheet(args.sheetId, sheet_name, maton_key)
    clear_and_put(args.sheetId, sheet_name, values, maton_key)

    print(json.dumps({'ok': True, 'sheetId': args.sheetId, 'sheetName': sheet_name, 'outJson': out_json, 'races': len(rows_all)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
