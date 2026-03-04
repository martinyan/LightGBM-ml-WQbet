#!/usr/bin/env python3
"""Extract focus horses from a production pred_<date>_<venue>_R<n>.json.

Focus = { W.top1.horse_no, Q.anchor_horse_no, Q.partner2.horse_no, Q.partner3.horse_no }
Prints comma-separated ints (sorted). Prints empty line if file missing/unreadable.
"""

import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    args = ap.parse_args()

    try:
        pred = json.load(open(args.pred, 'r', encoding='utf-8'))
        w = (pred.get('W') or {}).get('top1') or {}
        q = pred.get('Q') or {}
        xs = set()
        for x in [w.get('horse_no'), q.get('anchor_horse_no'), (q.get('partner2') or {}).get('horse_no'), (q.get('partner3') or {}).get('horse_no')]:
            if x is None:
                continue
            try:
                xs.add(int(x))
            except Exception:
                pass
        print(','.join(str(x) for x in sorted(xs)))
    except Exception:
        print('')


if __name__ == '__main__':
    main()
