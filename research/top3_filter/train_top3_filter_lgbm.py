#!/usr/bin/env python3
"""Train a research-only model to predict Top-3 probability per runner.

Default: NO-ODDS model (odds excluded) to avoid last-moment dependency.

Input dataset: JSONL rows like `hkjc_ml_dataset.jsonl`.
Label:
  y_top3 = 1 if y_finish_pos <= 3 else 0

Outputs:
- model pickle
- feature column list
- basic eval metrics

NOTE: This does NOT touch Golden W/Q production code or configs.
"""

import os
import json
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import joblib

import lightgbm as lgb


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def parse_racedate_to_int(s: str) -> int:
    # 'YYYY/MM/DD' -> YYYYMMDD int
    try:
        y, m, d = s.split('/')
        return int(y) * 10000 + int(m) * 100 + int(d)
    except Exception:
        return -1


@dataclass
class TrainResult:
    model: object
    feature_cols: list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outDir', required=True)
    ap.add_argument('--mode', choices=['no_odds', 'with_final_odds'], default='no_odds')
    ap.add_argument('--testDays', type=int, default=120, help='Hold out last N days by racedate for evaluation')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outDir, exist_ok=True)

    df = load_jsonl(args.data)
    if 'y_finish_pos' not in df.columns:
        raise SystemExit('Dataset missing y_finish_pos')

    # label
    df['y_top3'] = (pd.to_numeric(df['y_finish_pos'], errors='coerce') <= 3).astype(int)

    # time split
    df['racedate_int'] = df['racedate'].map(parse_racedate_to_int)
    maxd = int(df['racedate_int'].max())

    # crude last-N-days holdout: compute threshold by unique date ordering
    dates = sorted(d for d in df['racedate_int'].unique() if d > 0)
    if len(dates) < 10:
        raise SystemExit('Not enough dates in dataset')
    holdout_dates = set(dates[-args.testDays:]) if args.testDays < len(dates) else set(dates[-max(1, len(dates)//5):])

    is_test = df['racedate_int'].isin(holdout_dates)
    dtrain = df[~is_test].copy()
    dtest = df[is_test].copy()

    # feature selection: numeric + categoricals (strings) supported by LGBM
    drop_cols = {
        'y_finish_pos', 'y_win', 'y_place', 'y_top3',
        'runner_id',  # id
        'racedate_int',
    }
    if args.mode == 'no_odds':
        drop_cols.add('cur_win_odds')

    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Encode categoricals explicitly to avoid pandas-category mismatch at inference.
    Xtr_df = dtrain[feature_cols].copy()
    Xte_df = dtest[feature_cols].copy()

    cat_cols = [c for c in feature_cols if Xtr_df[c].dtype == 'object']
    num_cols = [c for c in feature_cols if c not in cat_cols]

    from sklearn.preprocessing import OrdinalEncoder

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if cat_cols:
        Xtr_cat = enc.fit_transform(Xtr_df[cat_cols].astype(str))
        Xte_cat = enc.transform(Xte_df[cat_cols].astype(str))
    else:
        Xtr_cat = np.zeros((len(Xtr_df), 0), dtype=float)
        Xte_cat = np.zeros((len(Xte_df), 0), dtype=float)

    Xtr_num = Xtr_df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=float)
    Xte_num = Xte_df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=float)

    Xtr = np.hstack([Xtr_num, Xtr_cat])
    Xte = np.hstack([Xte_num, Xte_cat])

    ytr = dtrain['y_top3'].values
    yte = dtest['y_top3'].values

    clf = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_samples=50,
        random_state=args.seed,
        n_jobs=-1,
    )

    clf.fit(Xtr, ytr)

    # calibrate probabilities (helps bottom-50% cutoff stability)
    cal = CalibratedClassifierCV(clf, method='isotonic', cv=3)
    cal.fit(Xtr, ytr)

    p = cal.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    ll = log_loss(yte, p)

    # bottom-half contamination metric (race-level)
    # For each race: hide bottom 50% p_top3; measure how many top3 got hidden.
    contam = []
    for (rd, venue, race_no), g in dtest.assign(p=p).groupby(['racedate', 'venue', 'race_no']):
        g2 = g.sort_values('p', ascending=True)
        k = max(1, len(g2)//2)
        hidden = g2.head(k)
        # top3 true in hidden
        contam.append(float((hidden['y_top3'] == 1).mean()))
    contam_rate = float(np.mean(contam)) if contam else None

    out_name = f"top3_filter_{args.mode}.pkl" if args.mode != 'no_odds' else 'top3_filter_no_odds.pkl'
    out_path = os.path.join(args.outDir, out_name)

    payload = {
        'model': cal,
        'feature_cols': feature_cols,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'encoder': enc,
        'mode': args.mode,
        'metrics': {
            'test_auc': float(auc),
            'test_logloss': float(ll),
            'test_hidden_bottom_half_contam_rate': contam_rate,
            'n_train_rows': int(len(dtrain)),
            'n_test_rows': int(len(dtest)),
            'test_unique_dates': int(len(set(dtest['racedate_int'].tolist()))),
        },
    }
    joblib.dump(payload, out_path)

    meta_path = out_path + '.meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in payload.items() if k not in ('model', 'encoder')}, f, ensure_ascii=False, indent=2)

    print(json.dumps({'ok': True, 'out': out_path, 'meta': meta_path, 'metrics': payload['metrics']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
