# Research vs Production (HKJC)

## Goal
Run experiments without accidentally changing production predictions.

## Production = pinned + reproducible
Production predictions must only use:
- `prod/HKJC_PROD_WQ.json`
- `models/OVERLAY_RESIDUAL_LGBM_v1_PROD_22bets_thr0p2.pkl` (GoldenWinBet)
- `models/Q_RANKER_v7_PROD_FEB22_111ROI/*` (GoldenQbet)
- production entrypoints:
  - `hkjc_prod_run_meeting_racecard_graphql.py` (racecard + GraphQL WIN odds)
  - `hkjc_prod_predict_single_race_wq.py`

## Research schedulers (safe)
- `research/experiments/schedule_research_meeting.py` (orchestrates research-only jobs)

## Research rules
- Put new models under `research/models/`
- Put outputs under `research/reports/`
- Do not edit production bundles in-place.
- Do not change `prod/HKJC_PROD_WQ.json` unless promoting.

## Promotion flow
1) Train + backtest in research paths
2) Review results + sanity checks
3) Copy artifacts into production paths with `scripts/promote_to_prod.sh`
4) Update `prod/HKJC_PROD_WQ.json`
5) Update `prod/manifests/artifacts.lock.json`
6) Run `python3 scripts/verify_artifacts.py`
7) Commit + push

## Optional extra safety
Make production artifacts read-only:

```bash
chmod -R a-w prod/HKJC_PROD_WQ.json models/Q_RANKER_v7_PROD_FEB22_111ROI
chmod a-w models/OVERLAY_RESIDUAL_LGBM_v1_PROD_22bets_thr0p2.pkl
```
