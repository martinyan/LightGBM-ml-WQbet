#!/usr/bin/env python3
"""Poll QIN odds snapshots for a meeting and append to a JSONL file.

This is intended to be run by cron during the selling window.

Example:
  python3 qoddchange_poll_meeting.py \
    --date 2026-02-25 --venue HV --races 1-9 \
    --intervalSec 60 \
    --out research/experiments/QOddChange/data/snapshots/2026-02-25_HV.jsonl

Notes:
- This just captures snapshots; analysis is a separate step.
"""

import argparse, json, time, os, subprocess


def parse_races(s: str):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--races', required=True, help='1-9 or comma list')
    ap.add_argument('--intervalSec', type=int, default=60)
    ap.add_argument('--iterations', type=int, default=1, help='How many polls to run (1 = single snapshot run)')
    ap.add_argument('--out', required=True, help='JSONL output path')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    races = parse_races(args.races)

    for it in range(args.iterations):
        for rn in races:
            cmd = ['python3', os.path.join(os.path.dirname(__file__), 'qoddchange_fetch_qin_odds.py'),
                   '--date', args.date, '--venue', args.venue, '--raceNo', str(rn)]
            snap = subprocess.check_output(cmd, text=True).strip()
            with open(args.out, 'a', encoding='utf-8') as f:
                f.write(snap + '\n')
        if it < args.iterations - 1:
            time.sleep(args.intervalSec)


if __name__ == '__main__':
    main()
