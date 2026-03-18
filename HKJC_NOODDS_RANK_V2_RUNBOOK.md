# HKJC NO-ODDS Ranker v2 (Research)

**Purpose:** a *research-only* model to rank Top5 horses **without using odds**, using:
- improved features (recency + deltas + draw-bias)
- learning-to-rank (within-race LambdaRank)

⚠️ **Safety:** This does **NOT** modify or overwrite any production model/config.

## Files

- Training script: `hkjc_noodds_rank_v2_train.py`
- Inference script: `hkjc_noodds_rank_v2_predict.py`

Model artifacts:
- `models/HKJC-ML_NOODDS_RANK_LGBM_v2.txt`
- `models/HKJC-ML_NOODDS_RANK_LGBM_v2.infermeta.json`

## Training

Train on last 3 years available in `hkjc.sqlite` (time-split test = last 180 days):

```bash
cd /data/.openclaw/workspace
python3 hkjc_noodds_rank_v2_train.py --db hkjc.sqlite --years 3 --testDays 180
```

## Prediction (single race)

Input JSON should be the standard wrapper used by `hkjc_scrape_racecard_and_win_odds.py`:

```bash
python3 hkjc_noodds_rank_v2_predict.py \
  --db hkjc.sqlite \
  --in artifacts/.../merged_YYYY-MM-DD_HV_R1.json \
  --out artifacts/.../pred_v2.json
```

Output contains:
- `top5`: ranking results (by model score)
- `scored_all`: all runners sorted

## Features included (v2)

All features are **odds-free** and computed as-of raceday:

- Horse performance rates:
  - last 365d: starts, win rate, top3 rate
  - last 60d: starts, top3 rate
  - last 365d at same venue
  - last 365d at similar distance (±200m)
- Jockey/Trainer rates:
  - last 365d + last 60d top3 rates
- Last-run deltas:
  - days since last run
  - last finish position
  - distance change vs last run
  - weight change vs last run
  - last-run same venue flag
- Draw bias:
  - historical top3 rate for same venue and similar distance (±200m) by draw

## Label / objective

- Objective: **LambdaRank** (`lambdarank`)
- Relevance label per runner: `rel = max(0, 6 - finish_pos)`
  - winner=5, 2nd=4, 3rd=3, 4th=2, 5th=1, else 0

This directly optimizes within-race ranking quality (Top5).
