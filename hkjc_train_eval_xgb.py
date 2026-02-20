import json, math, argparse
from collections import defaultdict

import numpy as np
import xgboost as xgb


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def top2_hit_rate(rows):
    # rows: list of dict with keys racedate, venue, race_no, horse_no, y_place, p
    by_race = defaultdict(list)
    for r in rows:
        k = (r['racedate'], r['venue'], int(r['race_no']))
        by_race[k].append(r)

    races = 0
    hits = 0
    for k, arr in by_race.items():
        arr_sorted = sorted(arr, key=lambda x: x['p'], reverse=True)
        top2 = arr_sorted[:2]
        races += 1
        if any(int(x['y']) == 1 for x in top2):
            hits += 1
    return races, hits, hits / races if races else None


def both_top2_in_top3(rows):
    by_race = defaultdict(list)
    for r in rows:
        k = (r['racedate'], r['venue'], int(r['race_no']))
        by_race[k].append(r)

    races = 0
    both = 0
    for k, arr in by_race.items():
        arr_sorted = sorted(arr, key=lambda x: x['p'], reverse=True)
        top2 = arr_sorted[:2]
        races += 1
        if all(int(x['y']) == 1 for x in top2):
            both += 1
    return races, both, both / races if races else None


def logloss(y_true, p):
    eps = 1e-6
    y = np.asarray(y_true, dtype=np.float64)
    pr = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-(y * np.log(pr) + (1 - y) * np.log(1 - pr)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--holdoutDates', required=True, help='comma-separated YYYY/MM/DD')
    ap.add_argument('--outModel', default='hkjc_xgb_model.json')
    ap.add_argument('--logregModel', default='hkjc_model_v2_logreg.json')
    args = ap.parse_args()

    holdout_dates = [s.strip() for s in args.holdoutDates.split(',') if s.strip()]

    rows = load_jsonl(args.data)

    # label
    label_key = 'y_place'

    # feature keys (numeric only; categorical excluded already in dataset generator)
    sample = rows[0]
    meta_keys = {
        'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no',
        'y_finish_pos','y_win','y_place','cur_jockey','cur_trainer','cur_surface'
    }
    feat_keys = sorted([k for k in sample.keys() if k not in meta_keys and k != label_key])

    def vec(r):
        return [float(r.get(k, 0) or 0) for k in feat_keys]

    train = [r for r in rows if r['racedate'] not in holdout_dates]
    test_by_date = {d: [r for r in rows if r['racedate'] == d] for d in holdout_dates}

    X_train = np.asarray([vec(r) for r in train], dtype=np.float32)
    y_train = np.asarray([int(r.get(label_key, 0) or 0) for r in train], dtype=np.int32)

    # XGBoost binary classifier
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,
        'eta': 0.05,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'lambda': 1.0,
        'alpha': 0.0,
        'seed': 42,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_keys)
    booster = xgb.train(params, dtrain, num_boost_round=800)

    # Save model
    booster.save_model(args.outModel)

    # Evaluate XGB and LogReg on the same holdout(s)
    with open(args.logregModel, 'r', encoding='utf-8') as f:
        logm = json.load(f)

    # map feat_keys -> logreg featKeys positions
    log_feat = logm['featKeys']
    means = logm['means']
    stds = logm['stds']
    w = logm['w']

    def logreg_p(r):
        x = [1.0]
        for k in log_feat:
            x.append(float(r.get(k, 0) or 0))
        # standardize
        for j in range(1, len(x)):
            x[j] = (x[j] - float(means[j])) / (float(stds[j]) or 1.0)
        z = 0.0
        for j in range(len(w)):
            z += float(w[j]) * x[j]
        return sigmoid(z)

    report = {
        'trained_on_rows': len(train),
        'feature_count': len(feat_keys),
        'holdoutDates': holdout_dates,
        'xgb': {},
        'logreg': {}
    }

    for d, trows in test_by_date.items():
        X_test = np.asarray([vec(r) for r in trows], dtype=np.float32)
        y_test = [int(r.get(label_key, 0) or 0) for r in trows]

        dtest = xgb.DMatrix(X_test, feature_names=feat_keys)
        p_xgb = booster.predict(dtest)

        rows_xgb = []
        rows_lr = []
        for r, px in zip(trows, p_xgb):
            rows_xgb.append({
                'racedate': r['racedate'],
                'venue': r['venue'],
                'race_no': int(r['race_no']),
                'horse_no': int(r['horse_no']),
                'y': int(r.get(label_key, 0) or 0),
                'p': float(px),
            })
            rows_lr.append({
                'racedate': r['racedate'],
                'venue': r['venue'],
                'race_no': int(r['race_no']),
                'horse_no': int(r['horse_no']),
                'y': int(r.get(label_key, 0) or 0),
                'p': float(logreg_p(r)),
            })

        races, hits, hit_rate = top2_hit_rate(rows_xgb)
        races2, both, both_rate = both_top2_in_top3(rows_xgb)
        report['xgb'][d] = {
            'rows': len(trows),
            'logloss': logloss(y_test, p_xgb),
            'top2_hit_rate': hit_rate,
            'top2_hits': hits,
            'races': races,
            'both_top2_in_top3_rate': both_rate,
            'both_top2_in_top3': both,
        }

        races, hits, hit_rate = top2_hit_rate(rows_lr)
        races2, both, both_rate = both_top2_in_top3(rows_lr)
        report['logreg'][d] = {
            'rows': len(trows),
            'logloss': logloss(y_test, [x['p'] for x in rows_lr]),
            'top2_hit_rate': hit_rate,
            'top2_hits': hits,
            'races': races,
            'both_top2_in_top3_rate': both_rate,
            'both_top2_in_top3': both,
        }

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
