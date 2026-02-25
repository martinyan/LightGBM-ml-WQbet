#!/usr/bin/env python3
"""Schedule a one-shot HKJC ingestion job at (last_race_scheduled_time + 8h).

Inputs:
- A directory produced by hkjc_scrape_wp_meeting.mjs (per-race json files), OR
- A single bet scrape json containing scheduledTime.

This avoids brittle fixed-time ingestion.
"""

import argparse, json, os, glob
from datetime import datetime, timedelta


def parse_hhmm(hhmm: str):
    if not hhmm:
        return None
    hh, mm = hhmm.split(':')
    return int(hh), int(mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--racedate', required=True, help='YYYY-MM-DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--scrapeDir', required=True, help='Dir containing *_bet_scrape.json from hkjc_scrape_wp_meeting.mjs')
    ap.add_argument('--offsetHours', type=int, default=8)
    ap.add_argument('--jobName', default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.scrapeDir, '*_bet_scrape.json')))
    if not paths:
        raise SystemExit(f'no bet_scrape.json files found in {args.scrapeDir}')

    times = []
    for p in paths:
        obj = json.load(open(p, 'r', encoding='utf-8'))
        t = obj.get('scheduledTime')
        if t:
            times.append(t)

    if not times:
        raise SystemExit('no scheduledTime found in scrape files')

    # pick max time
    last = sorted(times)[-1]
    hh, mm = parse_hhmm(last)

    # HK timezone is UTC+8; schedule using explicit +08:00 offset
    dt = datetime.fromisoformat(args.racedate).replace(hour=hh, minute=mm, second=0, microsecond=0)
    dt2 = dt + timedelta(hours=int(args.offsetHours))
    at_iso = dt2.strftime('%Y-%m-%dT%H:%M:00+08:00')

    name = args.jobName or f"HKJC ingest {args.venue} results+sectionals ({args.racedate} +{args.offsetHours}h)"

    # message for isolated agent job
    racedate_slash = args.racedate.replace('-', '/')
    msg = (
        f"HKJC ingestion (scheduled at last race time +{args.offsetHours}h).\n"
        f"racedate={racedate_slash} venue={args.venue}.\n"
        f"Probe localresults RaceNo=1; if no runner table, skip. If exists, determine max raceNo by incrementing RaceNo until missing (cap 12). Then run:\n"
        f"node /data/.openclaw/workspace/hkjc_ingest_meeting_into_sqlite.mjs --db /data/.openclaw/workspace/hkjc.sqlite --racedate {racedate_slash} --venue {args.venue} --races 1-<maxRaceNo> --throttleMs 200\n"
        f"Report summary (races, runners, splits ingested) back to main chat."
    )

    # schedule via openclaw CLI
    import subprocess
    cmd = [
        'openclaw', 'cron', 'add',
        '--name', name,
        '--at', at_iso,
        '--session', 'isolated',
        '--agent', 'main',
        '--announce',
        '--channel', 'telegram',
        '--to', '27381797',
        '--expect-final',
        '--delete-after-run',
        '--description', f'Auto-ingest after meeting ends (+{args.offsetHours}h) based on racecard scheduledTime',
        '--message', msg,
    ]

    out = subprocess.check_output(cmd, text=True)
    print(out)


if __name__ == '__main__':
    main()
