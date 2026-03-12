# Top-3 Filter (Research)

Goal: build a **research-only** ML model that helps you *remove non-contenders*.

For each race, we score every runner with `p_top3` and then **hide the bottom 50%**.

This is intentionally **separate from Golden W/Q production**.

## Data

We reuse the existing dataset JSONL built from `hkjc.sqlite`:
- `hkjc_ml_dataset.jsonl` (365d jockey/trainer rates; includes `y_finish_pos`)

Label:
- `y_top3 = 1` if `y_finish_pos <= 3`, else `0`

## Train (NO-ODDS model)

```bash
python3 research/top3_filter/train_top3_filter_lgbm.py \
  --data hkjc_ml_dataset.jsonl \
  --outDir research/top3_filter/models \
  --mode no_odds
```

This trains a LightGBM binary classifier using all numeric/categorical runner features **excluding odds**.

## Predict / produce “hide bottom 50%” list for a race

1) Create per-runner features with the existing (prod) feature builder (does not modify prod models):

```bash
node hkjc_build_feature_rows_from_racecard_sqlite.mjs \
  --db hkjc.sqlite \
  --in <merged_racecard_odds.json> \
  --out /tmp/features.json \
  --lastN 3
```

2) Run predictor:

```bash
python3 research/top3_filter/predict_top3_filter.py \
  --model research/top3_filter/models/top3_filter_no_odds.pkl \
  --featuresJson /tmp/features.json \
  --out /tmp/top3_filter_out.json
```

Output includes:
- all runners with `p_top3`
- `hide` list (bottom 50%)
- `keep` list (top 50%)

## Odds snapshots (for future “T-30” / jackpot-horizon models)

Final odds exist in SQLite (`runners.win_odds`), but **early odds** are not historically stored.

Going forward, log snapshots using:

```bash
python3 research/top3_filter/snapshot_odds_meeting.py --racedate YYYY/MM/DD --venue HV --races 1-9 --outDir research/top3_filter/odds_snapshots
```

This uses `hkjc_scrape_racecard_and_win_odds.py` and writes timestamped merged JSON for each race.
