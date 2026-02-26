#!/usr/bin/env python3
"""Schedule WOddDrift snapshot + report jobs for a meeting.

Aligned schedule:
- First report at T-2h (and first GoldenWinBet prediction)
- After that:
  - T-2h -> T-1h: snapshots every 15m
  - T-1h -> T: snapshots every 5m
- Final report at T-5m

All jobs are one-shot delete-after-run.
"""

import argparse, json, os, re, subprocess
from datetime import datetime, timedelta
import urllib.request, urllib.parse


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={'User-Agent': 'openclaw-wodddrift/1.0', 'Accept-Language': 'zh-HK,zh;q=0.9,en;q=0.8'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode('utf-8', 'ignore')


def parse_race_time(html: str):
    m = re.search(r'\b(\d{1,2}:\d{2})\b', html)
    return m.group(1) if m else None


def parse_races(s: str):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def at_iso(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%dT%H:%M:00+08:00')


def add_job(name: str, when_iso: str, message: str):
    cmd = [
        'openclaw','cron','add',
        '--name', name,
        '--at', when_iso,
        '--session','isolated',
        '--agent','main',
        '--announce',
        '--channel','telegram',
        '--to','27381797',
        '--expect-final',
        '--delete-after-run',
        '--description','WOddDrift snapshot/report job',
        '--message', message,
    ]
    subprocess.check_output(cmd, text=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--racedate', required=True, help='YYYY/MM/DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--races', required=True, help='1-9')
    ap.add_argument('--k', type=float, default=1.0)
    ap.add_argument('--snapBase', default='research/experiments/WOddDrift/data/snapshots')
    ap.add_argument('--repBase', default='research/experiments/WOddDrift/reports')
    args=ap.parse_args()

    racedate_slash=args.racedate
    racedate_dash=racedate_slash.replace('/','-')
    venue=args.venue.upper()

    races=parse_races(args.races)

    for rn in races:
        racecard_url = f'https://racing.hkjc.com/zh-hk/local/information/racecard?racedate={urllib.parse.quote(racedate_slash)}&Racecourse={venue}&RaceNo={rn}'
        html = fetch_text(racecard_url)
        hhmm = parse_race_time(html)
        if not hhmm:
            raise SystemExit(f'Cannot parse race time from {racecard_url}')
        hh,mm=[int(x) for x in hhmm.split(':')]
        start_dt = datetime.fromisoformat(racedate_dash).replace(hour=hh, minute=mm, second=0, microsecond=0)

        snap_dir=os.path.join(args.snapBase, racedate_dash)
        snap_path=os.path.join(snap_dir, f'{venue}_R{rn}.jsonl')

        rep_dir=os.path.join(args.repBase, racedate_dash)
        rep_t2=os.path.join(rep_dir, f'{venue}_R{rn}_T-2h.json')
        rep_t5=os.path.join(rep_dir, f'{venue}_R{rn}_T-5m.json')

        # T-2h: snapshot + report
        t2 = start_dt - timedelta(hours=2)
        add_job(
            f'WOddDrift snapshot {racedate_dash} {venue} R{rn} T-2h',
            at_iso(t2),
            f"Run: python3 /data/.openclaw/workspace/research/experiments/WOddDrift/scripts/wodddrift_append_snapshot.py --date {racedate_dash} --venue {venue} --raceNo {rn} --out /data/.openclaw/workspace/{snap_path}"
        )
        add_job(
            f'WOddDrift report {racedate_dash} {venue} R{rn} T-2h',
            at_iso(t2),
            f"Run: python3 /data/.openclaw/workspace/research/experiments/WOddDrift/scripts/wodddrift_report_race.py --mode T2H --racedate {racedate_slash} --venue {venue} --raceNo {rn} --snapshots /data/.openclaw/workspace/{snap_path} --out /data/.openclaw/workspace/{rep_t2} --k {args.k}"
        )
        add_job(
            f'WOddDrift Sheets {racedate_dash} {venue} R{rn} T-2h',
            at_iso(t2),
            f"Run: python3 /data/.openclaw/workspace/research/experiments/WOddDrift/scripts/wodddrift_export_adjusted_wq_to_sheets.py --mode T2H --racedate {racedate_slash} --venue {venue} --raceNo {rn} --snapshots /data/.openclaw/workspace/{snap_path} --k {args.k}"
        )

        # T-2h -> T-1h snapshots every 15m (exclude exact T-2h since already scheduled)
        t = t2 + timedelta(minutes=15)
        end1 = start_dt - timedelta(hours=1)
        while t <= end1:
            add_job(
                f'WOddDrift snapshot {racedate_dash} {venue} R{rn} {t.strftime("%H%M")}',
                at_iso(t),
                f"Run: python3 /data/.openclaw/workspace/research/experiments/WOddDrift/scripts/wodddrift_append_snapshot.py --date {racedate_dash} --venue {venue} --raceNo {rn} --out /data/.openclaw/workspace/{snap_path}"
            )
            t += timedelta(minutes=15)

        # T-1h -> T snapshots every 5m
        t = start_dt - timedelta(hours=1)
        end2 = start_dt
        while t <= end2:
            add_job(
                f'WOddDrift snapshot {racedate_dash} {venue} R{rn} {t.strftime("%H%M")}',
                at_iso(t),
                f"Run: python3 /data/.openclaw/workspace/research/experiments/WOddDrift/scripts/wodddrift_append_snapshot.py --date {racedate_dash} --venue {venue} --raceNo {rn} --out /data/.openclaw/workspace/{snap_path}"
            )
            t += timedelta(minutes=5)

        # T-5m report (stdout) + export adjusted report to Sheets
        t5 = start_dt - timedelta(minutes=5)
        add_job(
            f'WOddDrift report {racedate_dash} {venue} R{rn} T-5m',
            at_iso(t5),
            f"Run: python3 /data/.openclaw/workspace/research/experiments/WOddDrift/scripts/wodddrift_report_race.py --mode T5M --racedate {racedate_slash} --venue {venue} --raceNo {rn} --snapshots /data/.openclaw/workspace/{snap_path} --out /data/.openclaw/workspace/{rep_t5} --k {args.k}"
        )
        add_job(
            f'WOddDrift Sheets {racedate_dash} {venue} R{rn} T-5m',
            at_iso(t5),
            f"Run: python3 /data/.openclaw/workspace/research/experiments/WOddDrift/scripts/wodddrift_export_adjusted_wq_to_sheets.py --mode T5M --racedate {racedate_slash} --venue {venue} --raceNo {rn} --snapshots /data/.openclaw/workspace/{snap_path} --k {args.k}"
        )

    print(json.dumps({'ok': True, 'races': races, 'venue': venue, 'racedate': racedate_slash, 'k': args.k}, ensure_ascii=False))


if __name__=='__main__':
    main()
