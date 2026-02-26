#!/usr/bin/env python3
"""Schedule QOddChange polling + T-5m report jobs for a meeting.

Policy (per user):
- Start polling at T-12h
- Every 30m until T-2h
- Every 10m until T-1h
- Every 2m until race start
- Generate a report at T-5m

This scheduler uses race start times from racing.hkjc.com racecard pages.

It creates one-shot cron jobs (delete-after-run) so nothing keeps running forever.
"""

import argparse, json, os, re, subprocess
from datetime import datetime, timedelta
import urllib.request, urllib.parse


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={'User-Agent': 'openclaw-qoddchange/1.0', 'Accept-Language': 'zh-HK,zh;q=0.9,en;q=0.8'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode('utf-8', 'ignore')


def parse_race_time(html: str):
    # look for HH:MM in the header area
    m = re.search(r'\b(\d{1,2}:\d{2})\b', html)
    return m.group(1) if m else None


def parse_races(s: str):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def at_iso(dt: datetime) -> str:
    # assume dt is naive HK time
    return dt.strftime('%Y-%m-%dT%H:%M:00+08:00')


def add_cron_at(name: str, when_iso: str, message: str):
    cmd = [
        'openclaw', 'cron', 'add',
        '--name', name,
        '--at', when_iso,
        '--session', 'isolated',
        '--agent', 'main',
        '--announce',
        '--channel', 'telegram',
        '--to', '27381797',
        '--expect-final',
        '--delete-after-run',
        '--description', 'QOddChange snapshot/report job',
        '--message', message,
    ]
    subprocess.check_output(cmd, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--racedate', required=True, help='YYYY/MM/DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--races', required=True, help='1-9 or comma list')
    ap.add_argument('--outBase', default='research/experiments/QOddChange/data/snapshots', help='snapshot base folder (date subfolders will be created)')
    args = ap.parse_args()

    racedate_slash = args.racedate
    racedate_dash = racedate_slash.replace('/', '-')
    venue = args.venue.upper()

    races = parse_races(args.races)

    for rn in races:
        racecard_url = f'https://racing.hkjc.com/zh-hk/local/information/racecard?racedate={urllib.parse.quote(racedate_slash)}&Racecourse={venue}&RaceNo={rn}'
        html = fetch_text(racecard_url)
        hhmm = parse_race_time(html)
        if not hhmm:
            raise SystemExit(f'Cannot parse race time from {racecard_url}')
        hh, mm = [int(x) for x in hhmm.split(':')]

        # naive HK time
        start_dt = datetime.fromisoformat(racedate_dash).replace(hour=hh, minute=mm, second=0, microsecond=0)

        # snapshot file (per race)
        snap_dir = os.path.join(args.outBase, racedate_dash)
        snap_path = os.path.join(snap_dir, f'{venue}_R{rn}.jsonl')

        def sched_snap(dt_run: datetime):
            when = at_iso(dt_run)
            jname = f'QOddChange snapshot {racedate_dash} {venue} R{rn} {dt_run.strftime("%H%M")}'
            msg = f"Run: python3 /data/.openclaw/workspace/research/experiments/QOddChange/scripts/qoddchange_append_snapshot.py --date {racedate_dash} --venue {venue} --raceNo {rn} --out /data/.openclaw/workspace/{snap_path}"
            add_cron_at(jname, when, msg)

        # phase 1: T-12h .. T-2h every 2 hours
        t = start_dt - timedelta(hours=12)
        end1 = start_dt - timedelta(hours=2)
        while t <= end1:
            sched_snap(t)
            t += timedelta(hours=2)

        # phase 2: T-2h .. T-1h every 15m
        t = start_dt - timedelta(hours=2)
        end2 = start_dt - timedelta(hours=1)
        while t <= end2:
            sched_snap(t)
            t += timedelta(minutes=15)

        # phase 3: T-1h .. start every 5m
        t = start_dt - timedelta(hours=1)
        end3 = start_dt
        while t <= end3:
            sched_snap(t)
            t += timedelta(minutes=5)

        # report at T-5m
        rep_dt = start_dt - timedelta(minutes=5)
        rep_when = at_iso(rep_dt)
        rep_out = f'research/experiments/QOddChange/reports/{racedate_dash}_{venue}_R{rn}_Tminus5.json'
        rep_name = f'QOddChange report {racedate_dash} {venue} R{rn} T-5m'
        rep_msg = (
            f"Run: python3 /data/.openclaw/workspace/research/experiments/QOddChange/scripts/qoddchange_report_race_with_golden_alert.py "
            f"--racedate {racedate_slash} --venue {venue} --raceNo {rn} "
            f"--snapshots /data/.openclaw/workspace/{snap_path} "
            f"--out /data/.openclaw/workspace/{rep_out} "
            f"--goldenThr 0.16"
        )
        add_cron_at(rep_name, rep_when, rep_msg)

    print(json.dumps({'ok': True, 'races': races, 'venue': venue, 'racedate': racedate_slash}, ensure_ascii=False))


if __name__ == '__main__':
    main()
