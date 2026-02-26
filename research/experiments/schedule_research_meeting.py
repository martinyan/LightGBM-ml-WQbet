#!/usr/bin/env python3
"""Schedule research projects for a meeting.

This is a *research-only* orchestrator. It schedules:
- QOddChange (QIN odds drift)
- WOddDrift (WIN odds drift)

It does NOT change production models/config.

Usage:
  python3 research/experiments/schedule_research_meeting.py \
    --racedate 2026/02/25 --venue HV --races 1-9 --k 1.0
"""

import argparse, subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--racedate', required=True, help='YYYY/MM/DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--races', required=True, help='1-9')
    ap.add_argument('--k', type=float, default=1.0, help='WOddDrift drift weight k')
    args = ap.parse_args()

    # Schedule QOddChange
    subprocess.check_call([
        'python3', 'research/experiments/QOddChange/scripts/qoddchange_schedule_meeting.py',
        '--racedate', args.racedate,
        '--venue', args.venue,
        '--races', args.races,
    ])

    # Schedule WOddDrift
    subprocess.check_call([
        'python3', 'research/experiments/WOddDrift/scripts/wodddrift_schedule_meeting.py',
        '--racedate', args.racedate,
        '--venue', args.venue,
        '--races', args.races,
        '--k', str(args.k),
    ])

    print('OK: scheduled research jobs (QOddChange + WOddDrift)')


if __name__ == '__main__':
    main()
