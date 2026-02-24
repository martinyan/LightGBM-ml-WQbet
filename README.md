# HKJC Production (GoldenWinBet + GoldenQbet)

This repository contains a reproducible production pipeline for HKJC betting signals:

- **GoldenWinBet** (W): overlay residual model (WIN-only) with thresholding
- **GoldenQbet** (Q): ranker that selects 2 partners excluding the W anchor

## Production config
- `prod/HKJC_PROD_WQ.json`

## Verify pinned artifacts
- `prod/manifests/artifacts.lock.json`
- `scripts/verify_artifacts.py`

## Key entrypoints
- Scrape racecards: `hkjc_scrape_wp_meeting.mjs`
- Build features from SQLite: `hkjc_build_feature_rows_from_racecard_sqlite.mjs`
- Production predictor (single race): `hkjc_prod_predict_single_race_wq.py`
- Production predictor (meeting): `hkjc_prod_run_wp_meeting.py`

## Bootstrap
See `docs/PROD_BOOTSTRAP.md`.

## Notes
- `hkjc.sqlite` is not tracked; rebuild it with scripts.
- Do not commit secrets (`MATON_API_KEY`).
