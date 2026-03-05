# HKJC Key Races Runbook (Option 1: paste URL)

Goal: Given a race-day table URL (e.g. trainers-entries), produce a JSON file listing all horses in the table and their **key races**.

**Key race rule (current):**
- horse finished **01 or 02**, and
- **頭馬距離 <= 0.5** (鼻位 / 短馬頭位 / 頭位 / 頸位 / 1/4 / 1/2 etc.)

## Files
- `hkjc_browser_extract_horse_links.js` — browser evaluate snippet to extract horse links from the table (columns 3+)
- `hkjc_process_links_to_keyraces.mjs` — CLI script: links.json -> key_races.json

## How to run (manual steps)

### Step 1) Extract horse links (requires browser tool)
1. Open the race-day table URL in a browser session.
2. Run the evaluate snippet in `hkjc_browser_extract_horse_links.js`.
3. Save the returned JSON as `hkjc_links_<YYYY-MM-DD>.json`.

### Step 2) Process all horse pages into key races JSON
Run:
```bash
node hkjc_process_links_to_keyraces.mjs \
  --links hkjc_links_<YYYY-MM-DD>.json \
  --out hkjc_key_races_<YYYY-MM-DD>.json
```

## Notes
- Step 1 is needed because the trainers-entries table is JS-rendered; plain HTTP fetch may not contain the table.
- Step 2 uses plain HTTP fetch and parses the horse race-history table by locating the header cell `頭馬<br />距離`.

---

# HKJC SQLite ML (DB-only) Runbook

This pipeline uses `hkjc.sqlite` (built from HKJC `localresults` + sectional times) and **does not** scrape individual horse pages.

## Prereqs
- `hkjc.sqlite` present in workspace (see `hkjc_build_sqlite_from_localresults.mjs` + `hkjc_backfill_sectional_splits.mjs`).

## 1) Build a training dataset (JSONL)
Creates one row per runner-start, using **previous run dynamic sectional_splits (variable K splits)** + previous result deltas, plus current draw/weight/odds and jockey/trainer rolling 365d win/place rates.

Note: The legacy `sectionals` table (fixed 3 segments) is deprecated for modeling; prefer `sectional_splits`.

```bash
node hkjc_ml_build_dataset_sqlite.mjs \
  --db hkjc.sqlite \
  --out hkjc_ml_dataset.jsonl \
  --prevRuns 1
```

Key feature blocks emitted:
- `prev_split_last_time`, `prev_split_penult_time`, `prev_kick_delta`
- `prev_pos_early`, `prev_pos_mid`, `prev_pos_late`, `prev_pos_change_*`
- `prev_time_delta_sec`, `prev_margin_len`, `prev_days_since`
- `jockey_365d_win_rate`, `jockey_365d_place_rate`, same for trainer
- `prevN_avg_*` aggregations across the last N runs (when `--prevRuns > 1`)

Note: `prev_rating` is currently `null` because rating is not yet stored in the SQLite schema.

## 2) Predict top-5 from a race card scrape (odds)
Input example: `hkjc_ml_demo_race2_odds_2026-02-11_HV2.json`

```bash
node hkjc_ml_predict_from_racecard_sqlite.mjs \
  --db hkjc.sqlite \
  --in hkjc_ml_demo_race2_odds_2026-02-11_HV2.json \
  --lastN 3 \
  --out pred.json
```

Outputs:
- `top5`: ranked list
- `all`: all runners with computed features + last N runs summary

This is currently a **heuristic scorer** (not a trained ML model), but uses the exact same DB-only feature construction path.
