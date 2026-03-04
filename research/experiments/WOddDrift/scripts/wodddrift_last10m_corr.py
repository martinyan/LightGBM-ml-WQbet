#!/usr/bin/env python3
"""Compute last-10-minute WIN-odds drift correlation between horses.

Input snapshots are JSONL produced by wodddrift_append_snapshot.py:
{date, venue, raceNo, ts, odds: {"1": 5.2, ...}}

We:
- take the latest snapshot timestamp as "now"
- find snapshots within [now-10m, now]
- build per-horse log-odds series and delta series (per-minute changes)
- compute Pearson correlation of delta series between focus horses and every other horse

This is meant for raceday reporting (lightweight, no heavy deps).
"""

import argparse, json, math
from datetime import timedelta


def load_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def pearson(x, y):
    n = min(len(x), len(y))
    if n < 3:
        return None
    x = x[:n]
    y = y[:n]
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    return cov / math.sqrt(vx * vy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshots', required=True)
    ap.add_argument('--racedate', required=True, help='YYYY-MM-DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--raceNo', required=True, type=int)
    ap.add_argument('--focus', default='', help='comma list of focus horse nos, e.g. "1,6"')
    ap.add_argument('--top', type=int, default=5)
    args = ap.parse_args()

    venue = args.venue.upper()
    snaps = load_jsonl(args.snapshots)
    snaps = [s for s in snaps if (s.get('date') == args.racedate and s.get('venue') == venue and int(s.get('raceNo')) == int(args.raceNo))]
    snaps = sorted(snaps, key=lambda s: s.get('ts') or '')
    if not snaps:
        raise SystemExit('no snapshots')

    # Use ISO timestamps; keep simple by using dateutil if available, else fallback to fromisoformat.
    try:
        import dateutil.parser
        parse_ts = dateutil.parser.isoparse
    except Exception:
        from datetime import datetime
        def parse_ts(s):
            s = s.replace('Z', '+00:00')
            return datetime.fromisoformat(s)

    t_now = parse_ts(snaps[-1]['ts'])
    t_cut = t_now - timedelta(minutes=10)
    snaps10 = [s for s in snaps if s.get('ts') and parse_ts(s['ts']) >= t_cut]
    if len(snaps10) < 3:
        # not enough points; still output movers using what we have
        snaps10 = snaps[-min(len(snaps), 11):]

    # Collect all horse numbers present
    horses = set()
    for s in snaps10:
        for k in (s.get('odds') or {}).keys():
            horses.add(str(int(k)))
    horses = sorted(horses, key=lambda x: int(x))

    # Build log-odds series per horse aligned by snapshot index; None if missing
    series = {h: [] for h in horses}
    for s in snaps10:
        od = s.get('odds') or {}
        for h in horses:
            v = od.get(h)
            if v is None:
                series[h].append(None)
            else:
                try:
                    fv = float(v)
                    series[h].append(math.log(fv) if fv > 0 else None)
                except Exception:
                    series[h].append(None)

    # Delta series: consecutive differences where both points exist
    deltas = {}
    abs_drift = {}
    for h, xs in series.items():
        ds = []
        for i in range(1, len(xs)):
            a = xs[i-1]
            b = xs[i]
            if a is None or b is None:
                ds.append(0.0)
            else:
                ds.append(a - b)  # positive means odds dropped
        deltas[h] = ds
        abs_drift[h] = sum(abs(x) for x in ds)

    # Focus set
    focus = []
    if args.focus.strip():
        for x in args.focus.split(','):
            x = x.strip()
            if not x:
                continue
            focus.append(str(int(x)))
    focus = [h for h in focus if h in deltas]

    movers = sorted(horses, key=lambda h: abs_drift.get(h, 0.0), reverse=True)
    top_movers = movers[: max(args.top, 5)]

    out = {
        'ok': True,
        'racedate': args.racedate,
        'venue': venue,
        'raceNo': int(args.raceNo),
        'windowPoints': len(snaps10),
        'windowStartTs': snaps10[0].get('ts'),
        'windowEndTs': snaps10[-1].get('ts'),
        'topMovers': [{'horse_no': int(h), 'abs_drift': abs_drift[h]} for h in top_movers],
        'focus': [int(h) for h in focus],
        'correlations': {},
    }

    for fh in focus:
        corrs = []
        for h in horses:
            if h == fh:
                continue
            r = pearson(deltas[fh], deltas[h])
            if r is None:
                continue
            corrs.append({'horse_no': int(h), 'corr': r, 'abs_drift': abs_drift[h]})
        corrs.sort(key=lambda x: abs(x['corr']), reverse=True)
        out['correlations'][str(int(fh))] = corrs[: max(args.top, 5)]

    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
