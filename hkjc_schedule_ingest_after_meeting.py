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
    ap.add_argument('--racedate', default=None, help='YYYY-MM-DD (optional; inferred from bet.hkjc url if omitted)')
    ap.add_argument('--venue', default=None, help='HV|ST (optional; inferred from bet.hkjc url if omitted)')
    ap.add_argument('--scrapeDir', required=True, help='Dir containing *_bet_scrape.json from hkjc_scrape_wp_meeting.mjs')
    ap.add_argument('--offsetHours', type=int, default=8)
    ap.add_argument('--jobName', default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.scrapeDir, '*_bet_scrape.json')))
    if not paths:
        raise SystemExit(f'no bet_scrape.json files found in {args.scrapeDir}')

    times = []
    inferred_venue = None
    inferred_racedate = None

    for p in paths:
        obj = json.load(open(p, 'r', encoding='utf-8'))
        t = obj.get('scheduledTime')
        if t:
            times.append(t)

        # infer from bet.hkjc url: /ch/racing/wp/YYYY-MM-DD/HV/1
        url = obj.get('url')
        if url and (inferred_venue is None or inferred_racedate is None):
            try:
                from urllib.parse import urlparse
                parts = [x for x in urlparse(url).path.split('/') if x]
                # find date token
                for i, token in enumerate(parts):
                    if len(token) == 10 and token[4] == '-' and token[7] == '-':
                        inferred_racedate = token
                        inferred_venue = parts[i + 1] if i + 1 < len(parts) else None
                        break
            except Exception:
                pass

    if not times:
        raise SystemExit('no scheduledTime found in scrape files')

    racedate = args.racedate or inferred_racedate
    venue = (args.venue or inferred_venue or '').upper() or None

    if not racedate or not venue:
        raise SystemExit('Unable to infer racedate/venue (pass --racedate and --venue explicitly)')

    # pick max time
    last = sorted(times)[-1]
    hh, mm = parse_hhmm(last)

    # HK timezone is UTC+8; schedule using explicit +08:00 offset
    dt = datetime.fromisoformat(racedate).replace(hour=hh, minute=mm, second=0, microsecond=0)
    dt2 = dt + timedelta(hours=int(args.offsetHours))
    at_iso = dt2.strftime('%Y-%m-%dT%H:%M:00+08:00')

    name = args.jobName or f"HKJC ingest {venue} results+sectionals ({racedate} +{args.offsetHours}h)"

    # message for isolated agent job
    racedate_slash = racedate.replace('-', '/')
    msg = (
        f"HKJC ingestion (scheduled at last race time +{args.offsetHours}h).\n"
        f"racedate={racedate_slash} venue={venue}.\n"
        f"Probe localresults RaceNo=1; if no runner table, skip. If exists, determine max raceNo by incrementing RaceNo until missing (cap 12). Then run:\n"
        f"node /data/.openclaw/workspace/hkjc_ingest_meeting_into_sqlite.mjs --db /data/.openclaw/workspace/hkjc.sqlite --racedate {racedate_slash} --venue {venue} --races 1-<maxRaceNo> --throttleMs 200\n"
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
