#!/usr/bin/env python3
"""T-5m QOddChange report + GoldenWinBet overlay alert.

1) Reads QIN snapshot JSONL and computes top drops.
2) Runs GoldenWinBet+GoldenQbet production predictor for the same race (racecard+GraphQL WIN odds)
   and prints an ALERT line if GoldenWinBet overlay > threshold.

Designed for cron announce output.
"""

import argparse, json, os, subprocess
from collections import defaultdict


def load_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def compute_drop_report(rows, topN=15):
    by_pair = defaultdict(list)
    for r in rows:
        for o in r.get('odds') or []:
            comb = o.get('comb')
            val = o.get('odds')
            if not comb or val is None:
                continue
            try:
                val = float(val)
            except Exception:
                continue
            by_pair[comb].append((r.get('ts'), val))

    stats = []
    for comb, series in by_pair.items():
        series_sorted = sorted(series, key=lambda t: t[0] or '')
        latest = series_sorted[-1][1]
        minv = min(v for _, v in series_sorted)
        drop_ratio = (latest - minv) / latest if latest else None
        stats.append({
            'pair': comb,
            'latest': latest,
            'min': minv,
            'drop_ratio': drop_ratio,
            'snapshots': len(series_sorted)
        })

    stats.sort(key=lambda x: (x['drop_ratio'] is not None, x['drop_ratio']), reverse=True)
    top = stats[:topN]

    horse_score = defaultdict(float)
    for r in top:
        try:
            a, b = r['pair'].split(',')
            a = int(a)
            b = int(b)
            dr = float(r['drop_ratio'] or 0.0)
            horse_score[a] += dr
            horse_score[b] += dr
        except Exception:
            pass

    horses = sorted([{'horse_no': k, 'drop_score': v} for k, v in horse_score.items()], key=lambda x: x['drop_score'], reverse=True)

    return {
        'total_pairs_seen': len(stats),
        'top_pairs': top,
        'top_horses': horses[:10],
    }


def run_prod_predict(racedate_slash, venue, race_no, prod_cfg, db_path):
    racedate_dash = racedate_slash.replace('/', '-')
    merged = f'/tmp/qoddchange_merged_{racedate_dash}_{venue}_R{race_no}.json'
    feat = f'/tmp/qoddchange_feat_{racedate_dash}_{venue}_R{race_no}.json'
    pred = f'/tmp/qoddchange_pred_{racedate_dash}_{venue}_R{race_no}.json'

    subprocess.check_call([
        'python3', 'hkjc_scrape_racecard_and_win_odds.py',
        '--racedate', racedate_slash,
        '--venue', venue,
        '--raceNo', str(race_no),
        '--out', merged
    ])
    subprocess.check_call([
        'node', 'hkjc_build_feature_rows_from_racecard_sqlite.mjs',
        '--db', db_path,
        '--in', merged,
        '--out', feat,
        '--lastN', '3'
    ])
    subprocess.check_call([
        'python3', 'hkjc_prod_predict_single_race_wq.py',
        '--prod', prod_cfg,
        '--in', feat,
        '--out', pred
    ])

    return json.load(open(pred, 'r', encoding='utf-8'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--racedate', required=True, help='YYYY/MM/DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--raceNo', required=True, type=int)
    ap.add_argument('--snapshots', required=True, help='JSONL snapshots path')
    ap.add_argument('--out', required=True, help='Report JSON output path')
    ap.add_argument('--topN', type=int, default=15)
    ap.add_argument('--prod', default='prod/HKJC_PROD_WQ.json')
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--goldenThr', type=float, default=0.16, help='GoldenWinBet alert threshold (overlay)')
    args = ap.parse_args()

    venue = args.venue.upper()

    rows = load_jsonl(args.snapshots)
    rows = [r for r in rows if r.get('date') == args.racedate.replace('/','-') and r.get('venue') == venue and int(r.get('raceNo')) == int(args.raceNo)]

    drop = compute_drop_report(rows, topN=args.topN)

    # prod prediction
    pred = run_prod_predict(args.racedate, venue, int(args.raceNo), args.prod, args.db)
    wtop1 = (pred.get('W') or {}).get('top1') or {}
    overlay = wtop1.get('overlay')

    rep = {
        'racedate': args.racedate,
        'venue': venue,
        'raceNo': int(args.raceNo),
        'qoddchange': drop,
        'golden': {
            'W_model': (pred.get('W') or {}).get('name'),
            'Q_model': (pred.get('Q') or {}).get('name'),
            'top1': wtop1,
            'alert_thr': float(args.goldenThr),
            'alert_pass': bool(overlay is not None and float(overlay) > float(args.goldenThr)),
        }
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(rep, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    # stdout summary for Telegram
    lines = [f"QOddChange T-5m R{args.raceNo} {args.racedate} {venue}"]

    if overlay is not None and float(overlay) > float(args.goldenThr):
        lines.append(f"ALERT: GoldenWinBet overlay={float(overlay):.3f} > {args.goldenThr:.2f} (W top1 #{wtop1.get('horse_no')} {wtop1.get('horse')}, odds={wtop1.get('cur_win_odds')})")
    else:
        lines.append(f"GoldenWinBet overlay={float(overlay):.3f} (thr {args.goldenThr:.2f})")

    lines.append("Top QIN drops (drop_ratio):")
    for r in (drop.get('top_pairs') or [])[:8]:
        lines.append(f"{r['pair']}: latest={r['latest']:.1f} min={r['min']:.1f} drop_ratio={r['drop_ratio']:.3f}")

    print("\n".join(lines))


if __name__ == '__main__':
    main()
