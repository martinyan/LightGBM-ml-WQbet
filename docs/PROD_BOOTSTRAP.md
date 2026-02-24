# HKJC Production Bootstrap (GoldenWinBet + GoldenQbet)

This repo packages the **production W+Q betting pipeline**:
- Scrape bet.hkjc racecards (wp pages)
- Build per-runner features from `hkjc.sqlite`
- Run production models:
  - **GoldenWinBet** (W overlay residual)
  - **GoldenQbet** (Q partner ranker)

> Note: `hkjc.sqlite` is **not committed**. Rebuild it using the DB scripts.

## 1) Clone + verify artifacts

```bash
git clone <your-repo>
cd <your-repo>
python3 scripts/verify_artifacts.py
```

## 2) Install deps (non-docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Node is required for the scraper + feature builder
node -v
```

## 3) Provide environment variables

```bash
export MATON_API_KEY=...          # for Google Sheets export
export HKJC_SHEET_ID=...          # target spreadsheet id
```

## 4) Rebuild the SQLite DB (data not included)

Typical flow (adjust to your data sources):
- `node hkjc_build_sqlite_from_localresults.mjs ...`
- `node hkjc_backfill_sectional_splits.mjs ...`

Sanity check:

```bash
sqlite3 hkjc.sqlite "select max(racedate) from meetings;"
```

## 5) Run production predictions for a meeting

```bash
make verify
make prod-predict-meeting RACEDATE=2026-02-25 VENUE=HV RACES=1-9 SHEETNAME=PROD_PRED_2026-02-25_HV
```

Outputs:
- Google Sheet tab `<SHEETNAME>` (race summaries)
- `<SHEETNAME>_RUNNERS` (all runners)

## Docker usage (optional)

```bash
docker compose build
docker compose run --rm hkjc-prod bash
```

Inside container:
```bash
make verify
make prod-predict-meeting RACEDATE=... VENUE=... RACES=... SHEETNAME=...
```
