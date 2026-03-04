#!/usr/bin/env python3
"""Append one QIN odds snapshot to JSONL."""

import argparse, os, subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--raceNo', required=True, type=int)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    cmd = [
        'python3', os.path.join(os.path.dirname(__file__), 'qindrift_fetch_qin_odds.py'),
        '--date', args.date, '--venue', args.venue, '--raceNo', str(args.raceNo)
    ]
    snap = subprocess.check_output(cmd, text=True).strip()

    with open(args.out, 'a', encoding='utf-8') as f:
        f.write(snap + '\n')


if __name__ == '__main__':
    main()
