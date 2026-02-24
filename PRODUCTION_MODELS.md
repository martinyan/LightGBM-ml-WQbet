# HKJC Production Models (Pinned)

This repo has lots of research scripts. **Production predictions must use the pinned W+Q models below**.

## Production config

- `prod/HKJC_PROD_WQ.json`

## W (WIN) — production

- **Name:** `GoldenWinBet`
- Model: `OVERLAY_RESIDUAL_LGBM_v1`
- Bundle: `models/OVERLAY_RESIDUAL_LGBM_v1_PROD_22bets_thr0p2.pkl`
- Params: `alpha=1.0`, `beta=0.0`, `threshold=0.2`
- Decision rule: bet WIN if **top1 overlay > 0.2**

## Q (Quinella) — production

- **Name:** `GoldenQbet`
- Model: `LGBMRanker_lambdarank_winnerOnly`
- Bundle: `models/Q_RANKER_v7_PROD_FEB22_111ROI/bundle.json`
- Model file: `models/Q_RANKER_v7_PROD_FEB22_111ROI/ranker.txt`
- Strategy: **2 combos** per selected race:
  - anchor = W top1
  - partners = top ranker horses excluding anchor (2 horses)

## Inference (single race)

1) Build feature pack from racecard/SQLite (must include `rows[*].features` and `rows[*].cur_win_odds`)
2) Run production predictor:

```bash
python3 hkjc_prod_predict_single_race_wq.py \
  --prod prod/HKJC_PROD_WQ.json \
  --in <race_features.json> \
  --out pred_wq.json
```

The output includes:
- W top1 + threshold pass flag
- Q anchor + partner2/partner3

> Note: Q bets are usually placed only when W passes threshold.

## Backtest provenance

The threshold sweep and the thr=0.2 “good run” for the window **2025/09/01..2026/02/22** are in:
- `reports/HKJC-ML_SWEEP_WQ_THRESH_FEB22/sweep_wq_thresholds_overlay_based_0p02_0p20.json`
- `reports/HKJC-ML_OVERLAY_WIN_ONLY_THR0p2_FEB22/overlay_win10_place30_top1_threshold.json`
- `reports/HKJC-ML_OVERLAY_WIN_ONLY_THR0p2_FEB22/q_two_pairs_anchor_ranker_partners_report.json`
