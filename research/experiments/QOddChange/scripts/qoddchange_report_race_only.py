#!/usr/bin/env python3
"""QOddChange T-5m report (QIN only).

No GoldenWinBet run here (per updated spec). Designed for cron announce.
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
        stats.append({'pair': comb, 'latest': latest, 'min': minv, 'drop_ratio': drop_ratio, 'snapshots': len(series_sorted)})

    stats.sort(key=lambda x: (x['drop_ratio'] is not None, x['drop_ratio']), reverse=True)

    # Cap output size: prioritize horse-centric summary + top pairs.
    # We'll compute horse drift scores using the top M pairs (to avoid noise).
    top_pairs = stats[: max(30, args.topN)]

    horse_score = defaultdict(float)
    horse_pair_count = defaultdict(int)
    for r in top_pairs:
        try:
            a, b = r['pair'].split(',')
            a = int(a)
            b = int(b)
            dr = float(r['drop_ratio'] or 0.0)
            horse_score[a] += dr
            horse_score[b] += dr
            horse_pair_count[a] += 1
            horse_pair_count[b] += 1
        except Exception:
            pass

    horses = sorted(
        [{'horse_no': k, 'drop_score': v, 'pairs_dropping': horse_pair_count.get(k, 0)} for k, v in horse_score.items()],
        key=lambda x: x['drop_score'],
        reverse=True,
    )

    # Output caps
    lead_horses = horses[:10]
    lead_pairs = top_pairs[:15]

    rep = {
        'date': args.date,
        'venue': args.venue.upper(),
        'raceNo': int(args.raceNo),
        'total_pairs_seen': len(stats),
        'lead_horses': lead_horses,
        'lead_pairs': lead_pairs,
        'caps': {'max_horses': 10, 'max_pairs': 15, 'max_entries_total': 30}
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    lines = [f"QOddChange T-5m R{args.raceNo} {args.date} {args.venue.upper()} — QIN drift leaders"]

    # Horse-centric leaders
    lines.append('Lead horses (drop_score, pairs_dropping):')
    for h in lead_horses[:10]:
        lines.append(f"#{h['horse_no']}: score={h['drop_score']:.3f}, pairs={h['pairs_dropping']}")

    # Pair-centric leaders
    lines.append('Top drifting pairs (drop_ratio):')
    for r in lead_pairs[:10]:
        lines.append(f"{r['pair']}: latest={r['latest']:.1f} min={r['min']:.1f} drop_ratio={r['drop_ratio']:.3f}")

    # hard cap (keep <=30 lines)
    print("\n".join(lines[:30]))


if __name__ == '__main__':
    main()
