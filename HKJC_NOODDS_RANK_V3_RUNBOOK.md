# HKJC NO-ODDS Ranker v3 (Research) — Sectional + Pace (Top1-Optimized)

**Purpose:** research-only model to rank horses **without using odds**, now including:
- sectional-based features from SQLite (`sectionals`, `sectional_splits`)
- a simple pace proxy (field front-runner fraction)
- learning-to-rank (LambdaRank) with winner-heavy labels to optimize Top1

⚠️ Does **not** change any production model/config.

## Files
- Train: `hkjc_noodds_rank_v3_train.py`
- Predict: `hkjc_noodds_rank_v3_predict.py`

Artifacts:
- `models/HKJC-ML_NOODDS_RANK_LGBM_v3_TOP1.txt`
- `models/HKJC-ML_NOODDS_RANK_LGBM_v3_TOP1.infermeta.json`

## Train
```bash
cd /data/.openclaw/workspace
python3 hkjc_noodds_rank_v3_train.py --db hkjc.sqlite --years 3 --testDays 180
```

## Predict (single race)
```bash
python3 hkjc_noodds_rank_v3_predict.py \
  --db hkjc.sqlite \
  --in artifacts/.../merged_YYYY-MM-DD_HV_R1.json \
  --out artifacts/.../pred_v3.json
```

## Sectional features used
From `sectionals` (per runner):
- `pos1`, `pos3`
- `seg2_time`, `seg3_time`, `kick_time`

We use:
- last-run sectionals (as-of raceday)
- prevN averages (N=3)

## Pace proxy
We classify a simple running style from prevN sectionals:
- `style_front`: avg pos1 <= 3
- `style_closer`: avg pos1 >= 8 and avg pos3 improves by >=2

Then per race:
- `pace_index_front_frac` = (#front runners) / (field size)
- interactions: `closer_x_pace`, `front_x_slow`

## Label
Winner-heavy relevance for ranking:
- 1st=10, 2nd=4, 3rd=2, 4th=1, 5th=1, else 0
