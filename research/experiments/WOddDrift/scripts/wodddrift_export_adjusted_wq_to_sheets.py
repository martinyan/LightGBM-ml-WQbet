#!/usr/bin/env python3
"""Export adjusted GoldenWinBet/GoldenQbet report to Google Sheets (per race).

- At T-2h: export baseline production W/Q + odds snapshot
- At T-5m: export drift-adjusted W top1 + adjusted Q anchor/partners + runner table

This is research-only and does not modify production config/models.

Tabs created:
- {TAB_PREFIX}_{DATE}_{VENUE}_R{raceNo}_{MODE}
- {TAB_PREFIX}_{DATE}_{VENUE}_R{raceNo}_{MODE}_RUNNERS

MODE is T2H or T5M.
"""

import argparse, json, os, math, subprocess
from datetime import datetime, timezone
import urllib.request, urllib.parse


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


def load_jsonl(path):
    out=[]
    if not os.path.exists(path):
        return out
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out


def ln_ratio(a, b):
    try:
        a=float(a); b=float(b)
        if a>0 and b>0:
            return math.log(a/b)
    except Exception:
        pass
    return 0.0


def nearest_before(snaps, target_ts):
    best=None
    for s in snaps:
        ts=s.get('ts')
        if ts and ts <= target_ts:
            best=s
        else:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['T2H','T5M'], required=True)
    ap.add_argument('--racedate', required=True, help='YYYY/MM/DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--raceNo', required=True, type=int)
    ap.add_argument('--snapshots', required=True, help='WIN snapshots JSONL path for this race')
    ap.add_argument('--k', type=float, default=1.0)
    ap.add_argument('--sheetId', default=os.environ.get('HKJC_SHEET_ID', '109I84syg24sJCl9QxP0soUUx63BvmsS8dJCTEs8MrbI'))
    ap.add_argument('--tabPrefix', default='ADJ_WQ')
    ap.add_argument('--prod', default='prod/HKJC_PROD_WQ.json')
    ap.add_argument('--db', default='hkjc.sqlite')
    args = ap.parse_args()

    maton = os.environ.get('MATON_API_KEY')
    if not maton:
        raise SystemExit('MATON_API_KEY not set')

    auth = {'Authorization': f'Bearer {maton}', 'Content-Type': 'application/json'}

    venue=args.venue.upper()
    racedate_dash=args.racedate.replace('/','-')

    # run production predictor using current odds
    merged=f'/tmp/wodddrift_merge_{racedate_dash}_{venue}_R{args.raceNo}.json'
    feat=f'/tmp/wodddrift_feat_{racedate_dash}_{venue}_R{args.raceNo}.json'
    pred_path=f'/tmp/wodddrift_pred_{racedate_dash}_{venue}_R{args.raceNo}.json'

    subprocess.check_call(['python3','hkjc_scrape_racecard_and_win_odds.py','--racedate',args.racedate,'--venue',venue,'--raceNo',str(args.raceNo),'--out',merged])
    subprocess.check_call(['node','hkjc_build_feature_rows_from_racecard_sqlite.mjs','--db',args.db,'--in',merged,'--out',feat,'--lastN','3'])
    subprocess.check_call(['python3','hkjc_prod_predict_single_race_wq.py','--prod',args.prod,'--in',feat,'--out',pred_path])
    pred=json.load(open(pred_path,'r',encoding='utf-8'))

    w_scored=(pred.get('W') or {}).get('scored_all') or []
    q_scored=(pred.get('Q') or {}).get('ranker_scored_all') or []
    w_scored_sorted=sorted(w_scored, key=lambda x: x.get('score_win', -1e9), reverse=True)
    w_top1=pred['W']['top1']

    # Drift adjust
    snaps=load_jsonl(args.snapshots)
    snaps=[s for s in snaps if s.get('date')==racedate_dash and s.get('venue')==venue and int(s.get('raceNo'))==int(args.raceNo)]
    snaps=sorted(snaps, key=lambda s: s.get('ts') or '')

    drift_rows=[]
    drift_top1=None

    if args.mode=='T5M' and snaps:
        latest=snaps[-1]
        odds_now=latest.get('odds') or {}
        latest_ts=latest.get('ts')

        # targets: 10m/30m/2h
        import dateutil.parser
        from datetime import timedelta
        lt=dateutil.parser.isoparse(latest_ts)
        t10=(lt-timedelta(minutes=10)).isoformat().replace('+00:00','Z')
        t30=(lt-timedelta(minutes=30)).isoformat().replace('+00:00','Z')
        t2h=(lt-timedelta(hours=2)).isoformat().replace('+00:00','Z')
        s10=nearest_before(snaps, t10)
        s30=nearest_before(snaps, t30)
        s2h=nearest_before(snaps, t2h)

        for r in w_scored_sorted:
            hn=str(int(r['horse_no']))
            o_now=odds_now.get(hn)
            o10=(s10.get('odds') or {}).get(hn) if s10 else None
            o30=(s30.get('odds') or {}).get(hn) if s30 else None
            o2h=(s2h.get('odds') or {}).get(hn) if s2h else None
            d10=ln_ratio(o10,o_now) if (o10 and o_now) else 0.0
            d30=ln_ratio(o30,o_now) if (o30 and o_now) else 0.0
            d2h=ln_ratio(o2h,o_now) if (o2h and o_now) else 0.0
            combo=0.6*d10+0.3*d30+0.1*d2h
            score=float(r.get('score_win') or 0.0)
            score_star=score+float(args.k)*combo
            drift_rows.append({
                'horse_no': int(r['horse_no']),
                'horse': r.get('horse'),
                'score_win': score,
                'overlay': r.get('overlay'),
                'odds_now': r.get('cur_win_odds'),
                'drift_10m': d10,
                'drift_30m': d30,
                'drift_2h': d2h,
                'drift_combo': combo,
                'score_star': score_star,
            })

        drift_rows.sort(key=lambda x: x['score_star'], reverse=True)
        drift_top1=drift_rows[0] if drift_rows else None

    # Adjusted W/Q picks
    if args.mode=='T5M' and drift_top1:
        adj_anchor=int(drift_top1['horse_no'])
    else:
        adj_anchor=int(w_top1['horse_no'])

    q_sorted=sorted(q_scored, key=lambda x: x.get('ranker_score', -1e9), reverse=True)
    partners=[x for x in q_sorted if int(x['horse_no'])!=adj_anchor]
    p2=partners[0] if len(partners)>0 else None
    p3=partners[1] if len(partners)>1 else None

    # Tabs
    # Write to raceday spreadsheet by default
    from scripts.gsheets_raceday import get_or_create_raceday_sheet
    sheet_id = get_or_create_raceday_sheet(racedate_dash)

    # Append into a single log tab (no new tab every update)
    tabA = f"{args.tabPrefix}_LOG"
    ensure_tab(sheet_id, tabA, auth)

    # Append one row into log tab
    ts = datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
    differs = None
    drift_combo = None
    score_star = None
    drift_top_no = None
    if args.mode=='T5M' and drift_top1:
        drift_top_no = drift_top1['horse_no']
        drift_combo = drift_top1.get('drift_combo')
        score_star = drift_top1.get('score_star')
        differs = (int(drift_top_no) != int(w_top1.get('horse_no')))

    row = {
        'ts': ts,
        'racedate': args.racedate,
        'venue': venue,
        'raceNo': int(args.raceNo),
        'mode': args.mode,
        'k': float(args.k),
        'prod_top1_no': w_top1.get('horse_no'),
        'prod_top1_horse': w_top1.get('horse'),
        'prod_top1_odds': w_top1.get('cur_win_odds'),
        'prod_top1_overlay': w_top1.get('overlay'),
        'drift_top1_no': drift_top_no,
        'drift_top1_score_star': score_star,
        'drift_top1_combo': drift_combo,
        'drift_differs': differs,
        'adj_anchor_no': adj_anchor,
        'adj_q_p2_no': (p2 or {}).get('horse_no'),
        'adj_q_p2_horse': (p2 or {}).get('horse'),
        'adj_q_p3_no': (p3 or {}).get('horse_no'),
        'adj_q_p3_horse': (p3 or {}).get('horse'),
    }

    # Ensure header exists; if empty, write header + row. Else append row.
    # Read first row
    rng_get = urllib.parse.quote(f'{tabA}!A1:Z1')
    try:
        head = req_json(f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{sheet_id}/values/{rng_get}', headers=auth)
        has_header = bool((head.get('values') or []) and (head['values'][0] or []))
    except Exception:
        has_header = False

    cols = list(row.keys())
    if not has_header:
        clear_and_put(sheet_id, tabA, [cols, [row.get(c) for c in cols]], auth)
    else:
        # append
        rng_app = urllib.parse.quote(f'{tabA}!A1')
        body = json.dumps({'values': [[row.get(c) for c in cols]]}, ensure_ascii=False).encode('utf-8')
        req_json(
            f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{sheet_id}/values/{rng_app}:append?valueInputOption=USER_ENTERED',
            method='POST',
            data=body,
            headers=auth,
        )

    print(json.dumps({'ok': True, 'sheetId': sheet_id, 'tab': tabA, 'raceNo': args.raceNo, 'mode': args.mode}, ensure_ascii=False))


if __name__=='__main__':
    main()
