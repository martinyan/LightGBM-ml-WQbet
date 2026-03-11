#!/usr/bin/env python3
"""Schedule HKJC HV 2026-03-11 (9 races) Golden W+Q + WIN/QIN drift updates.

- Creates one-shot cron jobs (delete-after-run) in isolated session.
- Uses standalone scripts (no heredoc inline Python) to avoid shell parsing failures.

Cadence (per race):
- T-30: run Golden W+Q, write to Google Sheet tab, print W/Q summary
- T-10..T-1: QIN drift update each minute (recompute pred w/ latest WIN odds)
- T-10..T-1: WIN snapshot append each minute (no delivery)
- T-5: run Golden W+Q, write sheet tab, print summary
- FINAL T-1: run Golden W+Q, write sheet tab, print summary + WIN/QIN top movers

Timezone: Asia/Hong_Kong (+08:00)
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta

TZ_OFFSET = "+08:00"
DATE_DASH = "2026-03-11"
DATE_SLASH = "2026/03/11"
VENUE = "HV"
RACES = list(range(1, 10))

# New sheet created for this meeting (created by HV_ALL_NOW run)
SHEET_ID = "1wRhM0k97le1zYBLjVPY3FeDYC7sV1t-_Ia-xTcGDn-0"

# Delivery target
CHANNEL = "telegram"
TO = "telegram:27381797"

MODEL = "openai/gpt-5-mini"


@dataclass(frozen=True)
class Job:
    name: str
    at_iso: str
    message: str
    timeout_seconds: int = 540
    announce: bool = True


def dt_local(hhmm: str) -> datetime:
    return datetime.fromisoformat(f"{DATE_DASH}T{hhmm}:00")


def iso_local(dt: datetime) -> str:
    return dt.strftime(f"%Y-%m-%dT%H:%M:%S{TZ_OFFSET}")


def sh(cmd: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print("DRY:", " ".join(cmd))
        return
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )


def add_job(job: Job, *, dry_run: bool) -> None:
    cmd = [
        "openclaw",
        "cron",
        "add",
        "--name",
        job.name,
        "--at",
        job.at_iso,
        "--session",
        "isolated",
        "--agent",
        "main",
        "--model",
        MODEL,
        "--timeout-seconds",
        str(job.timeout_seconds),
        "--message",
        job.message,
        "--delete-after-run",
        "--wake",
        "now",
    ]
    if job.announce:
        cmd += ["--announce", "--channel", CHANNEL, "--to", TO]
    else:
        cmd += ["--no-deliver"]
    sh(cmd, dry_run=dry_run)


def payload_wq(rn: int, tab_name: str) -> str:
    pred = f"reports/PROD_PRED/pred_{DATE_DASH}_{VENUE}_R{rn}.json"
    return "\n".join(
        [
            "cd /data/.openclaw/workspace",
            f"python3 hkjc_prod_run_meeting_racecard_graphql.py --racedate {DATE_SLASH} --venue {VENUE} --races {rn} --racedaySheet --sheetId {SHEET_ID} --sheetName {tab_name}",
            f"python3 scripts/hkjc_print_pred_wq.py --pred {pred}",
        ]
    )


def payload_final(rn: int, tab_name: str) -> str:
    pred = f"reports/PROD_PRED/pred_{DATE_DASH}_{VENUE}_R{rn}.json"
    win_snap = f"research/experiments/WOddDrift/data/snapshots/{DATE_DASH}/HV_R{rn}.jsonl"
    qin_snap = f"research/experiments/WOddDrift/data/qin_snapshots/{DATE_DASH}/HV_R{rn}.jsonl"

    return "\n".join(
        [
            "cd /data/.openclaw/workspace",
            f"python3 hkjc_prod_run_meeting_racecard_graphql.py --racedate {DATE_SLASH} --venue {VENUE} --races {rn} --racedaySheet --sheetId {SHEET_ID} --sheetName {tab_name}",
            f"python3 scripts/hkjc_print_pred_wq.py --pred {pred}",
            "",
            f"python3 research/experiments/WOddDrift/scripts/wodddrift_append_snapshot.py --date {DATE_DASH} --venue {VENUE} --raceNo {rn} --out {win_snap} || true",
            f"python3 scripts/hkjc_print_last10m_movers.py --kind WIN --snap {win_snap} --racedate {DATE_DASH} --venue {VENUE} --raceNo {rn} --top 5 --pred {pred}",
            "",
            f"python3 research/experiments/WOddDrift/scripts/qindrift_append_snapshot.py --date {DATE_DASH} --venue {VENUE} --raceNo {rn} --out {qin_snap} || true",
            f"python3 scripts/hkjc_print_last10m_movers.py --kind QIN --snap {qin_snap} --racedate {DATE_DASH} --venue {VENUE} --raceNo {rn} --top 5 --pred {pred}",
        ]
    )


def payload_qin_drift(rn: int, title: str) -> str:
    snap = f"/data/.openclaw/workspace/research/experiments/WOddDrift/data/qin_snapshots/{DATE_DASH}/HV_R{rn}.jsonl"
    pred = f"/data/.openclaw/workspace/reports/PROD_PRED/pred_{DATE_DASH}_{VENUE}_R{rn}.json"
    return "\n".join(
        [
            "cd /data/.openclaw/workspace",
            f"python3 scripts/hkjc_prod_recompute_pred_now.py --racedate {DATE_SLASH} --venue {VENUE} --raceNo {rn} | tail -n 1",
            f"SNAP={snap}",
            "mkdir -p $(dirname $SNAP)",
            f"python3 research/experiments/WOddDrift/scripts/qindrift_append_snapshot.py --date {DATE_DASH} --venue {VENUE} --raceNo {rn} --out $SNAP || true",
            f"FOCUS=$(python3 scripts/hkjc_focus_from_pred.py --pred {pred})",
            "python3 research/experiments/WOddDrift/scripts/qindrift_last10m_corr.py --snapshots $SNAP --racedate {DATE_DASH} --venue {VENUE} --raceNo {rn} --focus \"$FOCUS\" --top 3 \\",
            f"  | python3 research/experiments/WOddDrift/scripts/qindrift_format_short.py --title \"{title}\" --top 3 --focus \"$FOCUS\"",
        ]
    )


def payload_win_snapshot(rn: int) -> str:
    out = f"research/experiments/WOddDrift/data/snapshots/{DATE_DASH}/HV_R{rn}.jsonl"
    return "\n".join(
        [
            "cd /data/.openclaw/workspace",
            f"python3 research/experiments/WOddDrift/scripts/wodddrift_append_snapshot.py --date {DATE_DASH} --venue {VENUE} --raceNo {rn} --out {out} || true",
        ]
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # From racing.hkjc.com racecard headers (local +08:00)
    starts = {
        1: "18:40",
        2: "19:10",
        3: "19:40",
        4: "20:10",
        5: "20:40",
        6: "21:10",
        7: "21:45",
        8: "22:15",
        9: "22:50",
    }

    jobs: list[Job] = []

    for rn in RACES:
        st = dt_local(starts[rn])

        # W+Q
        t30 = st - timedelta(minutes=30)
        t5 = st - timedelta(minutes=5)
        t1 = st - timedelta(minutes=1)

        jobs.append(
            Job(
                name=f"HKJC HV 2026-03-11 R{rn} T-30 (Golden W+Q)",
                at_iso=iso_local(t30),
                message=payload_wq(rn, tab_name=f"HV_R{rn}_T-30_{t30.strftime('%H%M')}"),
                timeout_seconds=720,
                announce=True,
            )
        )
        jobs.append(
            Job(
                name=f"HKJC HV 2026-03-11 R{rn} T-5 (Golden W+Q)",
                at_iso=iso_local(t5),
                message=payload_wq(rn, tab_name=f"HV_R{rn}_T-5_{t5.strftime('%H%M')}"),
                timeout_seconds=720,
                announce=True,
            )
        )
        jobs.append(
            Job(
                name=f"HKJC HV 2026-03-11 R{rn} T-1 FINAL (Golden W+Q)",
                at_iso=iso_local(t1),
                message=payload_final(rn, tab_name=f"HV_R{rn}_FINAL_{t1.strftime('%H%M')}"),
                timeout_seconds=900,
                announce=True,
            )
        )

        # WIN snapshots every minute in last 10m (no delivery)
        for m in range(10, 0, -1):
            at = st - timedelta(minutes=m)
            jobs.append(
                Job(
                    name=f"WIN Snapshot {DATE_DASH} {VENUE} R{rn} T-{m:02d}",
                    at_iso=iso_local(at),
                    message=payload_win_snapshot(rn),
                    timeout_seconds=120,
                    announce=False,
                )
            )

        # QIN drift every minute in last 10m (announce)
        for m in range(10, 0, -1):
            at = st - timedelta(minutes=m)
            title = f"QIN Drift {DATE_DASH} {VENUE} R{rn} T-{m:02d} ({at.strftime('%H%M')})"
            jobs.append(
                Job(
                    name=title,
                    at_iso=iso_local(at),
                    message=payload_qin_drift(rn, title=title),
                    timeout_seconds=540,
                    announce=True,
                )
            )

    # Add all jobs
    for j in jobs:
        add_job(j, dry_run=args.dry_run)

    print(f"Added {len(jobs)} jobs.")


if __name__ == "__main__":
    main()
