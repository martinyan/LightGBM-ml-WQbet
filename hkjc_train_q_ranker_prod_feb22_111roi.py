import os, json, argparse, datetime
from collections import defaultdict

import numpy as np
import lightgbm as lgb


def safe_float(x, d=0.0):
    try:
        if x is None or x == '':
            return d
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ('turf', 't'):
                return 0.0
            if s in ('awt', 'all weather', 'all-weather', 'a'):
                return 1.0
        return float(x)
    except Exception:
        return d


def load_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def group_by_race(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r['racedate'], r['venue'], int(r['race_no']))].append(r)
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='hkjc_dataset_v7_jt60_prev3_fullcols_fieldranks.jsonl')
    ap.add_argument('--trainStart', default='2023/09/01')
    ap.add_argument('--trainEnd', default='2025/07/31')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outDir', default='models/Q_RANKER_v7_PROD_FEB22_111ROI')
    args = ap.parse_args()

    os.makedirs(args.outDir, exist_ok=True)

    rows = load_jsonl(args.data)
    train_rows = [r for r in rows if args.trainStart <= r['racedate'] <= args.trainEnd and r.get('y_finish_pos') is not None]
    if not train_rows:
        raise SystemExit('no training rows')

    # Feature selection matches the Feb22 sweep logic (hkjc_sweep_thresholds_win_q.py):
    # numeric keys only; include cur_win_odds + jockey/trainer_60d_* if present.
    meta7 = {'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no','y_finish_pos','y_win','y_place','cur_jockey','cur_trainer'}
    cand7 = [k for k in train_rows[0].keys() if k not in meta7 and not k.startswith('y_') and not k.startswith('_')]
    feat_keys = []
    for k in sorted(cand7):
        v = train_rows[0].get(k)
        if isinstance(v, (int, float)) or v is None:
            feat_keys.append(k)

    races = group_by_race(train_rows)

    X = []
    y = []
    group = []
    for _, runners in sorted(races.items()):
        group.append(len(runners))
        for r in runners:
            fp = r.get('y_finish_pos')
            if fp is None:
                continue
            X.append([safe_float(r.get(k), 0.0) for k in feat_keys])
            # winner-only relevance (matches sweep logic)
            y.append(1.0 if int(fp) == 1 else 0.0)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    ranker = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        ndcg_eval_at=[1, 3],
        learning_rate=0.05,
        n_estimators=800,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=args.seed,
    )
    ranker.fit(X, y, group=group)

    model_txt = os.path.join(args.outDir, 'ranker.txt')
    ranker.booster_.save_model(model_txt)

    bundle = {
        'generatedAt': datetime.datetime.now().isoformat(timespec='seconds'),
        'model': 'LGBMRanker_lambdarank_winnerOnly',
        'data': args.data,
        'ranges': {'train': {'start': args.trainStart, 'end': args.trainEnd}},
        'feat_keys': feat_keys,
        'seed': args.seed,
        'note': 'Pinned production Q ranker matching Feb22 W/Q threshold sweep: winner-only label; numeric features (includes cur_win_odds + jockey/trainer_60d_* when available).'
    }
    json.dump(bundle, open(os.path.join(args.outDir, 'bundle.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print(json.dumps({'ok': True, 'outDir': args.outDir, 'feat_count': len(feat_keys), 'model_txt': model_txt}, indent=2))


if __name__ == '__main__':
    main()
