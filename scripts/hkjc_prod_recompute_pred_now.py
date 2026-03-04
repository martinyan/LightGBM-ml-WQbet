#!/usr/bin/env python3
"""Recompute Golden W+Q prediction for a single race *now* using latest WIN odds.

This is a lightweight version of hkjc_prod_run_meeting_racecard_graphql.py without Sheets.
It overwrites the standard pred file in reports/PROD_PRED so downstream jobs can reuse it.

Usage:
  python3 scripts/hkjc_prod_recompute_pred_now.py --racedate 2026/03/04 --venue HV --raceNo 4

Outputs JSON to stdout:
{ ok, predPath, w_top1_no, w_top1_horse, w_top1_odds, w_overlay, w_pass_0p20, q_anchor, q_p2, q_p3 }
"""

import argparse, json, os, subprocess


def sh(cmd):
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--racedate', required=True, help='YYYY/MM/DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--raceNo', required=True, type=int)
    ap.add_argument('--prod', default='prod/HKJC_PROD_WQ.json')
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--outDir', default='reports/PROD_PRED')
    args = ap.parse_args()

    venue = args.venue.upper()
    racedate_dash = args.racedate.replace('/', '-')

    os.makedirs(args.outDir, exist_ok=True)

    merged_path = os.path.join(args.outDir, f'merged_{racedate_dash}_{venue}_R{args.raceNo}.json')
    feat_path = os.path.join(args.outDir, f'features_{racedate_dash}_{venue}_R{args.raceNo}.json')
    pred_path = os.path.join(args.outDir, f'pred_{racedate_dash}_{venue}_R{args.raceNo}.json')

    sh(['python3', 'hkjc_scrape_racecard_and_win_odds.py', '--racedate', args.racedate, '--venue', venue, '--raceNo', str(args.raceNo), '--out', merged_path])
    sh(['node', 'hkjc_build_feature_rows_from_racecard_sqlite.mjs', '--db', args.db, '--in', merged_path, '--out', feat_path, '--lastN', '3'])
    sh(['python3', 'hkjc_prod_predict_single_race_wq.py', '--prod', args.prod, '--in', feat_path, '--out', pred_path])

    pred = json.load(open(pred_path, 'r', encoding='utf-8'))
    w = pred['W']['top1']
    q = pred['Q']
    p2 = q.get('partner2') or {}
    p3 = q.get('partner3') or {}

    out = {
        'ok': True,
        'predPath': pred_path,
        'w_top1_no': w.get('horse_no'),
        'w_top1_horse': w.get('horse'),
        'w_top1_odds': w.get('cur_win_odds'),
        'w_overlay': w.get('overlay'),
        'w_pass_0p20': bool((w.get('overlay') or 0) > 0.2),
        'q_anchor': q.get('anchor_horse_no'),
        'q_p2': {'horse_no': p2.get('horse_no'), 'horse': p2.get('horse')},
        'q_p3': {'horse_no': p3.get('horse_no'), 'horse': p3.get('horse')},
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
