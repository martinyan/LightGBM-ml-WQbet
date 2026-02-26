#!/usr/bin/env python3
"""Build a T-5m report for a single race using collected JSONL snapshots.

Computes:
- latest odds per pair
- min odds seen per pair
- drop ratio: (latest - min) / latest  (positive means odds dropped vs current)
- top drops and horse involvement

Outputs JSON and prints a short summary (suitable for Telegram).
"""

import argparse, json, os
from collections import defaultdict


def load_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--raceNo', required=True, type=int)
    ap.add_argument('--snapshots', required=True, help='JSONL snapshots path')
    ap.add_argument('--out', required=True, help='Report JSON output path')
    ap.add_argument('--topN', type=int, default=15)
    args = ap.parse_args()

    rows = load_jsonl(args.snapshots)
    rows = [r for r in rows if r.get('date') == args.date and r.get('venue') == args.venue.upper() and int(r.get('raceNo')) == int(args.raceNo)]

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
    top = stats[: args.topN]

    # horse involvement
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

    rep = {
        'date': args.date,
        'venue': args.venue.upper(),
        'raceNo': int(args.raceNo),
        'total_pairs_seen': len(stats),
        'top_pairs': top,
        'top_horses': horses[:10],
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    # concise stdout summary
    lines = [f"QOddChange R{args.raceNo} {args.date} {args.venue.upper()} — top drops (by drop_ratio)"]
    for r in top[:8]:
        lines.append(f"{r['pair']}: latest={r['latest']:.1f} min={r['min']:.1f} drop_ratio={r['drop_ratio']:.3f}")
    print("\n".join(lines))


if __name__ == '__main__':
    main()
