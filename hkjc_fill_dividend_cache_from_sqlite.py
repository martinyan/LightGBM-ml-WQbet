#!/usr/bin/env python3
"""Fill dividend caches (WIN + Quinella) for a date range using race list from SQLite.

Writes/updates:
- hkjc_dividend_cache_WINPLACE_byurl.json  (we only populate the 'win' map here)
- hkjc_dividend_cache_Qmap_byurl.json      (quinella map: pair -> dividend)

Keys are sha1(url) where url is HKJC localresults page:
  https://racing.hkjc.com/zh-hk/local/information/localresults?racedate=YYYY/MM/DD&Racecourse=ST|HV&RaceNo=N

This matches the lookup logic in hkjc_backtest_prod_tiered_wq.py.

Parsing approach:
- Fetch HTML via urllib
- Extract the '派彩' table and parse rows for pools:
  - 獨贏 (WIN)
  - 連贏 (Quinella)

"""

import argparse
import hashlib
import json
import re
import sqlite3
import time
import urllib.request


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "ignore")


def strip_tags(s: str) -> str:
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&nbsp;", " ").replace("\u3000", " ")
    return re.sub(r"\s+", " ", s).strip()


def parse_win_q(html: str):
    m = re.search(r'<td[^>]*colspan="3"[^>]*>\s*派彩\s*</td>', html)
    if not m:
        return {}, {}
    chunk = html[m.start() :]
    end = chunk.find("</table>")
    if end != -1:
        chunk = chunk[:end]

    win = {}
    q = {}
    cur = None
    for tr in re.findall(r"<tr[^>]*>.*?</tr>", chunk, flags=re.S | re.I):
        tds = re.findall(r"<td[^>]*>.*?</td>", tr, flags=re.S | re.I)
        cells = [strip_tags(td) for td in tds]
        if not cells:
            continue

        # Pool column is rowspanned. We treat any non-combination string in col0 as pool name.
        # Example pools: 獨贏, 位置, 連贏, 位置Q, 二重彩, 三重彩, 單T, 四連環, 四重彩, etc.
        # We only care about 獨贏 (WIN) and 連贏 (Quinella).
        pool0 = cells[0]
        looks_like_pair = bool(re.fullmatch(r"\d+\s*[,\-]\s*\d+", pool0))
        looks_like_horse_no = bool(re.fullmatch(r"\d+", pool0))

        if (not looks_like_pair) and (not looks_like_horse_no):
            cur = pool0
            rest = cells[1:]
        else:
            rest = cells

        if cur == "獨贏" and len(rest) >= 2:
            hn = rest[0]
            div = rest[1].replace(",", "")
            if re.fullmatch(r"\d+", hn):
                try:
                    win[str(int(hn))] = float(div)
                except Exception:
                    pass

        if cur == "連贏" and len(rest) >= 2:
            comb = rest[0]
            div = rest[1].replace(",", "")
            m2 = re.fullmatch(r"(\d+)\s*[,\-]\s*(\d+)", comb)
            if m2:
                a = int(m2.group(1))
                b = int(m2.group(2))
                key = f"{min(a,b)}-{max(a,b)}"
                try:
                    q[key] = float(div)
                except Exception:
                    pass

    return win, q


def localresults_url(racedate: str, venue: str, race_no: int) -> str:
    return (
        "https://racing.hkjc.com/zh-hk/local/information/localresults"
        f"?racedate={racedate}&Racecourse={venue}&RaceNo={race_no}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="hkjc.sqlite")
    ap.add_argument("--start", required=True, help="YYYY/MM/DD")
    ap.add_argument("--end", required=True, help="YYYY/MM/DD")
    ap.add_argument("--sleepMs", type=int, default=200, help="Politeness delay between fetches")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit number of races (0 = all)")
    ap.add_argument("--winplaceCache", default="hkjc_dividend_cache_WINPLACE_byurl.json")
    ap.add_argument("--qCache", default="hkjc_dividend_cache_Qmap_byurl.json")
    args = ap.parse_args()

    # load caches
    try:
        wp = json.load(open(args.winplaceCache, "r", encoding="utf-8"))
    except Exception:
        wp = {}
    try:
        qc = json.load(open(args.qCache, "r", encoding="utf-8"))
    except Exception:
        qc = {}

    conn = sqlite3.connect(args.db)
    c = conn.cursor()

    rows = c.execute(
        """
        SELECT m.racedate, m.venue, r.race_no
        FROM races r
        JOIN meetings m ON r.meeting_id=m.meeting_id
        WHERE m.racedate BETWEEN ? AND ?
        ORDER BY m.racedate ASC, m.venue ASC, r.race_no ASC
        """,
        (args.start, args.end),
    ).fetchall()

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    fetched = 0
    skipped = 0
    updated = 0
    errors = 0

    for racedate, venue, race_no in rows:
        url = localresults_url(racedate, venue, int(race_no))
        k = sha1(url)
        have_win = k in wp and isinstance(wp.get(k), dict) and "win" in wp.get(k)
        have_q = k in qc
        if have_win and have_q:
            skipped += 1
            continue

        try:
            html = fetch(url)
            win, q = parse_win_q(html)
            if not have_win:
                wp[k] = dict(wp.get(k, {}))
                wp[k]["win"] = win
                # leave place untouched if it exists
                updated += 1
            if not have_q:
                qc[k] = q
                updated += 1
            fetched += 1
        except Exception as e:
            errors += 1

        time.sleep(max(0, args.sleepMs) / 1000.0)

    conn.close()

    json.dump(wp, open(args.winplaceCache, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(qc, open(args.qCache, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "ok": True,
                "races": len(rows),
                "fetched": fetched,
                "skipped": skipped,
                "updated": updated,
                "errors": errors,
                "range": {"start": args.start, "end": args.end},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
