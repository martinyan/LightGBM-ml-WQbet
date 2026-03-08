#!/usr/bin/env python3
"""Print a stable, shell-friendly W/Q summary from a PROD_PRED JSON.

Usage:
  python3 scripts/hkjc_print_pred_wq.py --pred reports/PROD_PRED/pred_YYYY-MM-DD_VV_Rn.json

Outputs lines:
  W gate: PASS|NO
  W top1: #n name odds=... overlay=...
  Q: anchor #n name | p2 #n name | p3 #n name
"""

import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--thr', type=float, default=0.20)
    args = ap.parse_args()

    p = json.load(open(args.pred, 'r', encoding='utf-8'))

    w = (p.get('W') or {}).get('top1') or {}
    overlay = w.get('overlay')
    overlay_f = float(overlay) if overlay is not None else 0.0
    pass_gate = overlay_f > args.thr

    print('W gate: PASS' if pass_gate else 'W gate: NO')
    print('W top1: #{} {} odds={} overlay={:.3f}'.format(
        w.get('horse_no'), w.get('horse'), w.get('cur_win_odds'), overlay_f
    ))

    q = p.get('Q') or {}
    p2 = q.get('partner2') or {}
    p3 = q.get('partner3') or {}

    # Some pred files don't populate anchor_horse; fall back to W top1 horse when numbers match.
    anchor_no = q.get('anchor_horse_no')
    anchor_horse = q.get('anchor_horse')
    if anchor_horse is None and anchor_no is not None and str(anchor_no) == str(w.get('horse_no')):
        anchor_horse = w.get('horse')

    print('Q: anchor #{} {} | p2 #{} {} | p3 #{} {}'.format(
        anchor_no, anchor_horse,
        p2.get('horse_no'), p2.get('horse'),
        p3.get('horse_no'), p3.get('horse')
    ))


if __name__ == '__main__':
    main()
