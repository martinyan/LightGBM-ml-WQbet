#!/usr/bin/env python3
"""Backfill runners.win_odds in SQLite using *final odds from HKJC localresults page*.

This enforces the user-defined truth-source for cur_win_odds in the v7 dataset.

For each race in [start, end] from SQLite meetings/races, fetch:
  https://racing.hkjc.com/zh-hk/local/information/localresults?racedate=YYYY/MM/DD&Racecourse=ST|HV&RaceNo=N

Parse the runner table and extract horse_no + 獨贏賠率 (final odds), then update:
  runners.win_odds where race_id matches and horse_no matches.

Notes:
- We only update when we successfully parse a numeric odds.
- We do not attempt to fill scratches (SCR).

"""

import argparse
import re
import sqlite3
import time
import urllib.request

UA = "openclaw-hkjc-odds-backfill/1.0"


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "ignore")


def norm(s: str) -> str:
    return (s or "").replace("\u00a0", " ").replace("\u3000", " ").strip()


def strip_tags(s: str) -> str:
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", s).strip()


def parse_runner_odds(html: str):
    """Return dict horse_no(int)->odds(float)."""
    # Find the runner table by headers containing 馬號 + 馬名 + 獨贏賠率
    tables = [m.group(0) for m in re.finditer(r"<table[\s\S]*?</table>", html, flags=re.I)]
    target = None
    for t in tables:
        if "馬號" in t and "馬名" in t and "獨贏" in t and "賠率" in t:
            target = t
            break
    if not target:
        return {}

    trs = list(re.finditer(r"<tr[^>]*>([\s\S]*?)</tr>", target, flags=re.I))
    rows = []
    for trm in trs:
        tr = trm.group(1)
        cells = [strip_tags(x) for x in re.findall(r"<t[dh][^>]*>([\s\S]*?)</t[dh]>", tr, flags=re.I)]
        if cells:
            rows.append(cells)

    # header row
    header_idx = None
    for i, r in enumerate(rows):
        if "馬號" in r and any("獨贏" in x and "賠率" in x for x in r):
            header_idx = i
            break
    if header_idx is None:
        return {}

    header = rows[header_idx]
    idx_no = header.index("馬號")
    # find the column containing '獨贏 賠率'
    idx_odds = None
    for j, col in enumerate(header):
        if "獨贏" in col and "賠率" in col:
            idx_odds = j
            break
    if idx_odds is None:
        return {}

    out = {}
    for r in rows[header_idx + 1 :]:
        if len(r) <= max(idx_no, idx_odds):
            continue
        horse_no_s = norm(r[idx_no])
        odds_s = norm(r[idx_odds])
        if not re.fullmatch(r"\d+", horse_no_s):
            continue
        if odds_s.upper() == "SCR":
            continue
        # odds can be like '10' or '10.5'
        try:
            odds = float(odds_s)
        except Exception:
            continue
        out[int(horse_no_s)] = odds

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="hkjc.sqlite")
    ap.add_argument("--start", required=True, help="YYYY/MM/DD")
    ap.add_argument("--end", required=True, help="YYYY/MM/DD")
    ap.add_argument("--sleepMs", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    c = conn.cursor()

    races = c.execute(
        """
        SELECT m.racedate, m.venue, r.race_id, r.race_no
        FROM races r
        JOIN meetings m ON m.meeting_id=r.meeting_id
        WHERE m.racedate BETWEEN ? AND ?
        ORDER BY m.racedate ASC, m.venue ASC, r.race_no ASC
        """,
        (args.start, args.end),
    ).fetchall()

    if args.limit and args.limit > 0:
        races = races[: args.limit]

    updated = 0
    fetched = 0
    errors = 0

    for racedate, venue, race_id, race_no in races:
        url = (
            "https://racing.hkjc.com/zh-hk/local/information/localresults"
            f"?racedate={racedate}&Racecourse={venue}&RaceNo={int(race_no)}"
        )
        try:
            html = fetch(url)
            mp = parse_runner_odds(html)
            fetched += 1
            if not mp:
                continue
            for hn, odds in mp.items():
                c.execute(
                    "UPDATE runners SET win_odds=? WHERE race_id=? AND horse_no=?",
                    (odds, int(race_id), int(hn)),
                )
                updated += c.rowcount
            conn.commit()
        except Exception:
            errors += 1

        time.sleep(max(0, args.sleepMs) / 1000.0)

    conn.close()
    print({"ok": True, "races": len(races), "fetched": fetched, "updated_rows": updated, "errors": errors})


if __name__ == "__main__":
    main()
