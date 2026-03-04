#!/usr/bin/env python3
"""Last-10-minute drift + correlation report using QIN (quinella) pool odds.

We convert combo odds into a per-horse indicator:
  horse_value(h, t) = median(log(odds(h-x, t)) for all x != h)

Then we compute per-minute deltas of horse_value and Pearson correlations.

Output JSON to stdout:
{
  ok, racedate, venue, raceNo,
  windowStartTs, windowEndTs, windowPoints,
  topMovers: [{horse_no, abs_drift}],
  focus: [...],
  correlations: { "1": [{horse_no, corr, abs_drift}, ...], ... }
}
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


def median(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


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


def parse_ts_fns():
    try:
        import dateutil.parser
        return dateutil.parser.isoparse
    except Exception:
        from datetime import datetime
        def parse_ts(s):
            return datetime.fromisoformat((s or '').replace('Z', '+00:00'))
        return parse_ts


def combos_for_horse(odds_map, h):
    # odds_map: {"1-7": 8.5}
    out = []
    for k, v in (odds_map or {}).items():
        try:
            a, b = k.split('-')
            a = int(a); b = int(b)
            if a == h or b == h:
                fv = float(v)
                if fv > 0:
                    out.append(math.log(fv))
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshots', required=True)
    ap.add_argument('--racedate', required=True, help='YYYY-MM-DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--raceNo', required=True, type=int)
    ap.add_argument('--focus', default='', help='comma list of focus horse nos')
    ap.add_argument('--top', type=int, default=5)
    args = ap.parse_args()

    venue = args.venue.upper()
    snaps = load_jsonl(args.snapshots)
    snaps = [s for s in snaps if (s.get('date') == args.racedate and s.get('venue') == venue and int(s.get('raceNo')) == int(args.raceNo))]
    snaps = sorted(snaps, key=lambda s: s.get('ts') or '')
    if not snaps:
        raise SystemExit('no snapshots')

    parse_ts = parse_ts_fns()
    t_now = parse_ts(snaps[-1]['ts'])
    t_cut = t_now - timedelta(minutes=10)
    snaps10 = [s for s in snaps if s.get('ts') and parse_ts(s['ts']) >= t_cut]
    if len(snaps10) < 3:
        snaps10 = snaps[-min(len(snaps), 11):]

    # horse universe: infer from odds keys seen in window
    horses = set()
    for s in snaps10:
        for k in (s.get('odds') or {}).keys():
            try:
                a, b = k.split('-')
                horses.add(int(a)); horses.add(int(b))
            except Exception:
                pass
    horses = sorted(horses)

    # build per-horse value series
    series = {h: [] for h in horses}
    for s in snaps10:
        om = s.get('odds') or {}
        for h in horses:
            vals = combos_for_horse(om, h)
            series[h].append(median(vals))

    # deltas + abs drift
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
                ds.append(a - b)  # positive means odds dropped (support increased)
        deltas[h] = ds
        abs_drift[h] = sum(abs(x) for x in ds)

    # focus
    focus = []
    if args.focus.strip():
        for x in args.focus.split(','):
            x = x.strip()
            if x:
                try:
                    focus.append(int(x))
                except Exception:
                    pass
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
