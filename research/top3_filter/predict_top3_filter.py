#!/usr/bin/env python3
"""Predict per-runner Top-3 probability for a single race feature JSON.

Input is the existing feature export produced by:
  node hkjc_build_feature_rows_from_racecard_sqlite.mjs --out features.json

This script:
- loads a trained model payload (joblib) from train_top3_filter_lgbm.py
- aligns feature columns
- outputs sorted runners + suggested hide/keep sets (bottom 50% hide)

Research-only.
"""

import json
import argparse
import joblib
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--featuresJson', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--hideFrac', type=float, default=0.5)
    args = ap.parse_args()

    payload = joblib.load(args.model)
    model = payload['model']
    feature_cols = payload['feature_cols']
    cat_cols = payload.get('cat_cols') or []
    num_cols = payload.get('num_cols') or [c for c in feature_cols if c not in cat_cols]
    enc = payload.get('encoder')

    feat = json.load(open(args.featuresJson, 'r', encoding='utf-8'))
    rows = feat.get('rows') or []

    # build dataframe
    recs = []
    for r in rows:
        f = (r.get('features') or {}).copy()
        recs.append({
            'horse_no': r.get('horse_no'),
            'horse': r.get('horse'),
            'horse_code': r.get('horse_code'),
            **f,
        })

    df = pd.DataFrame(recs)

    # ensure all required columns exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = None

    # Align columns
    for c in feature_cols:
        if c not in df.columns:
            df[c] = None

    Xdf = df[feature_cols].copy()

    # numeric
    X_num = Xdf[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=float)

    # categorical
    if cat_cols:
        if enc is None:
            raise SystemExit('Model payload missing encoder for categorical columns')
        X_cat = enc.transform(Xdf[cat_cols].astype(str))
    else:
        import numpy as np
        X_cat = np.zeros((len(Xdf), 0), dtype=float)

    import numpy as np
    X = np.hstack([X_num, X_cat])

    p = model.predict_proba(X)[:, 1]
    df['p_top3'] = p

    df2 = df.sort_values('p_top3', ascending=False)

    n = len(df2)
    k_hide = max(1, int(n * args.hideFrac))
    hide = df2.sort_values('p_top3', ascending=True).head(k_hide)
    keep = df2.sort_values('p_top3', ascending=False).head(n - k_hide)

    out = {
        'racedate': feat.get('racedate'),
        'venue': feat.get('venue'),
        'raceNo': feat.get('raceNo'),
        'model_mode': payload.get('mode'),
        'hideFrac': args.hideFrac,
        'hide': hide[['horse_no', 'horse', 'horse_code', 'p_top3']].to_dict(orient='records'),
        'keep': keep[['horse_no', 'horse', 'horse_code', 'p_top3']].to_dict(orient='records'),
        'all_sorted': df2[['horse_no', 'horse', 'horse_code', 'p_top3']].to_dict(orient='records'),
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({'ok': True, 'out': args.out, 'n': n, 'k_hide': k_hide}, ensure_ascii=False))


if __name__ == '__main__':
    main()
