#!/usr/bin/env python3
"""Build a v7-like dataset JSONL from hkjc_ml_dataset.jsonl.

Purpose:
- Our overlay + Q ranker bundles only need a subset of v7 features.
- hkjc_ml_build_dataset_sqlite.mjs can generate all required base features directly from SQLite,
  but it does not include the per-race field-rank features used by the Q ranker:
    - field_rank_prev_kick_delta
    - field_rank_prev_split_last_time
    - field_rank_prev_split_time_std

This script:
- Reads an input JSONL (default: hkjc_ml_dataset.jsonl)
- Groups rows by (racedate, venue, race_no)
- Computes field ranks for the 3 features (dense rank, 1=best/highest)
- Writes an output JSONL.

Ranking convention:
- kick_delta: higher is better => rank descending
- split_last_time, split_time_std: lower is better => rank ascending

Missing values:
- If value is missing/non-numeric, rank is None.

"""

import argparse
import json
import math
from collections import defaultdict


def safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            if math.isnan(x):
                return None
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        return float(s)
    except Exception:
        return None


def dense_rank(values, reverse=False):
    """Return mapping original_index -> dense rank (1..k) for non-None values.

    reverse=False: smaller is better (rank 1 = min)
    reverse=True: larger is better (rank 1 = max)
    """
    idx_vals = [(i, v) for i, v in enumerate(values) if v is not None]
    if not idx_vals:
        return {}
    # sort by value
    idx_vals.sort(key=lambda t: t[1], reverse=reverse)
    ranks = {}
    rank = 0
    last = object()
    for i, v in idx_vals:
        if v != last:
            rank += 1
            last = v
        ranks[i] = rank
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="hkjc_ml_dataset.jsonl")
    ap.add_argument("--out", dest="out", default="hkjc_dataset_v7_from_sqlite.jsonl")
    ap.add_argument("--start", default=None, help="Optional filter start racedate YYYY/MM/DD")
    ap.add_argument("--end", default=None, help="Optional filter end racedate YYYY/MM/DD")
    args = ap.parse_args()

    # 1) read + group
    races = defaultdict(list)  # key -> list of dict rows
    total = 0
    kept = 0

    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            r = json.loads(line)
            rd = r.get("racedate")
            if args.start and (rd is None or rd < args.start):
                continue
            if args.end and (rd is None or rd > args.end):
                continue
            key = (r.get("racedate"), r.get("venue"), int(r.get("race_no")))
            races[key].append(r)
            kept += 1

    # 2) compute ranks + write
    out_rows = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for key in sorted(races.keys()):
            rr = races[key]
            kick = [safe_float(x.get("prev_kick_delta")) for x in rr]
            split_last = [safe_float(x.get("prev_split_last_time")) for x in rr]
            split_std = [safe_float(x.get("prev_split_time_std")) for x in rr]

            rank_kick = dense_rank(kick, reverse=True)
            rank_split_last = dense_rank(split_last, reverse=False)
            rank_split_std = dense_rank(split_std, reverse=False)

            for i, row in enumerate(rr):
                row["field_rank_prev_kick_delta"] = rank_kick.get(i)
                row["field_rank_prev_split_last_time"] = rank_split_last.get(i)
                row["field_rank_prev_split_time_std"] = rank_split_std.get(i)
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_rows += 1

    print(
        json.dumps(
            {
                "ok": True,
                "in": args.inp,
                "out": args.out,
                "read_rows": total,
                "kept_rows": kept,
                "wrote_rows": out_rows,
                "races": len(races),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
