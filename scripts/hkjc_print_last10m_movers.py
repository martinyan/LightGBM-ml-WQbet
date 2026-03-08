#!/usr/bin/env python3
"""Print last-10m top movers for WIN or QIN from snapshot jsonl.

This avoids here-doc blocks in cron payloads.

Usage:
  python3 scripts/hkjc_print_last10m_movers.py --kind WIN --snap <path> --racedate YYYY-MM-DD --venue ST --raceNo 6 --top 5
  python3 scripts/hkjc_print_last10m_movers.py --kind QIN --snap <path> --racedate YYYY-MM-DD --venue ST --raceNo 6 --top 5

Output:
  WIN top movers (abs_drift): #n(name)(0.123) ...
"""

import argparse, json, subprocess, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['WIN', 'QIN'], required=True)
    ap.add_argument('--snap', required=True)
    ap.add_argument('--racedate', required=True)  # YYYY-MM-DD
    ap.add_argument('--venue', required=True)
    ap.add_argument('--raceNo', required=True)
    ap.add_argument('--top', type=int, default=5)
    ap.add_argument('--pred', default='', help='Optional pred JSON path for names (passed to formatter)')
    args = ap.parse_args()

    if args.kind == 'WIN':
        cmd = [
            'python3','research/experiments/WOddDrift/scripts/wodddrift_last10m_corr.py',
            '--snapshots', args.snap,
            '--racedate', args.racedate,
            '--venue', args.venue,
            '--raceNo', str(args.raceNo),
            '--top', str(args.top),
        ]
    else:
        cmd = [
            'python3','research/experiments/WOddDrift/scripts/qindrift_last10m_corr.py',
            '--snapshots', args.snap,
            '--racedate', args.racedate,
            '--venue', args.venue,
            '--raceNo', str(args.raceNo),
            '--top', str(args.top),
        ]

    out = subprocess.check_output(cmd, text=True)
    obj = json.loads(out or '{}')
    movers = obj.get('topMovers') or []

    # Try to add names from pred file (if provided) using the qindrift formatter mapping logic.
    name_map = {}
    pred_path = args.pred.strip()
    if pred_path and os.path.exists(pred_path):
        try:
            p = json.load(open(pred_path, 'r', encoding='utf-8'))
            for rr in (p.get('W') or {}).get('scored_all') or []:
                hn = int(rr.get('horse_no'))
                if rr.get('horse'):
                    name_map[hn] = rr.get('horse')
        except Exception:
            name_map = {}

    parts = []
    for m in movers[: args.top]:
        hn = m.get('horse_no')
        try:
            hn_i = int(hn)
        except Exception:
            hn_i = None
        nm = name_map.get(hn_i) if hn_i is not None else None
        drift = m.get('abs_drift')
        if drift is None:
            continue
        if nm:
            parts.append(f"#{hn_i} {nm}({float(drift):.3f})")
        else:
            parts.append(f"#{hn}({float(drift):.3f})")

    prefix = 'WIN' if args.kind == 'WIN' else 'QIN'
    print(f"{prefix} top movers (abs_drift): " + ' '.join(parts))


if __name__ == '__main__':
    main()
