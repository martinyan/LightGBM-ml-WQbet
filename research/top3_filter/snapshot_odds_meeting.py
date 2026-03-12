#!/usr/bin/env python3
"""Research-only odds snapshot logger.

This uses the existing scraper `hkjc_scrape_racecard_and_win_odds.py` (racecard + GraphQL WIN odds)
and writes timestamped merged JSON files.

Rationale:
- Final odds are stored historically in SQLite.
- Early odds are *not* stored historically, but are required to train/validate
  jackpot-mode and T-30 models.

This script creates an audit trail of early odds snapshots without touching production.
"""

import os
import json
import argparse
import subprocess
from datetime import datetime


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    return r.stdout


def parse_races(s: str):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--racedate', required=True, help='YYYY/MM/DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--races', required=True, help='e.g. 1-9')
    ap.add_argument('--outDir', default='research/top3_filter/odds_snapshots')
    args = ap.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_base = os.path.join(args.outDir, args.racedate.replace('/', '-'), args.venue.upper(), ts)
    os.makedirs(out_base, exist_ok=True)

    race_list = parse_races(args.races)
    manifest = {
        'racedate': args.racedate,
        'venue': args.venue.upper(),
        'snapshot_ts': ts,
        'files': [],
    }

    for rn in race_list:
        out_path = os.path.join(out_base, f'merged_R{rn}.json')
        sh(['python3', 'hkjc_scrape_racecard_and_win_odds.py', '--racedate', args.racedate, '--venue', args.venue, '--raceNo', str(rn), '--out', out_path])
        manifest['files'].append({'raceNo': rn, 'path': out_path})

    mani_path = os.path.join(out_base, 'manifest.json')
    with open(mani_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps({'ok': True, 'outDir': out_base, 'manifest': mani_path, 'races': len(race_list)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
