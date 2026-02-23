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
    ap.add_argument('--evalFracGroups', type=float, default=0.2)
    ap.add_argument('--featKeysFromReport', default='reports/HKJC-ML_OVERLAY_WIN_ONLY_THR0p2_FEB22/q_two_pairs_anchor_ranker_partners_report.json')
    ap.add_argument('--outDir', default='models/Q_RANKER_v7_FEB22_JT60_ODDS')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outDir, exist_ok=True)

    rep = json.load(open(args.featKeysFromReport, 'r', encoding='utf-8'))
    feat_keys = rep.get('feat_keys')
    if not feat_keys:
        raise SystemExit('feat_keys not found in report')

    rows = load_jsonl(args.data)

    def in_range(r, a, b):
        return a <= r['racedate'] <= b

    train_rows = [r for r in rows if in_range(r, args.trainStart, args.trainEnd) and r.get('y_finish_pos') is not None]

    races = group_by_race(train_rows)

    blocks = []
    for _, runners in sorted(races.items()):
        if len(runners) < 2:
            continue
        X = np.asarray([[safe_float(r.get(k), 0.0) for k in feat_keys] for r in runners], dtype=np.float32)
        y = []
        for r in runners:
            fp = r.get('y_finish_pos')
            fp = int(fp) if fp is not None else 99
            y.append(2 if fp == 1 else (1 if fp == 2 else 0))
        blocks.append({'X': X, 'y': np.asarray(y, dtype=np.float32)})

    n = len(blocks)
    n_eval = max(1, int(round(n * args.evalFracGroups)))
    fit = blocks[:-n_eval]
    evl = blocks[-n_eval:]

    def flatten(blks):
        X = np.concatenate([b['X'] for b in blks], axis=0)
        y = np.concatenate([b['y'] for b in blks], axis=0)
        g = [b['X'].shape[0] for b in blks]
        return X, y, g

    X_fit, y_fit, g_fit = flatten(fit)
    X_eval, y_eval, g_eval = flatten(evl)

    ranker = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        ndcg_eval_at=[2],
        learning_rate=0.05,
        n_estimators=3000,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=args.seed,
    )
    ranker.fit(X_fit, y_fit, group=g_fit, eval_set=[(X_eval, y_eval)], eval_group=[g_eval])

    model_txt = os.path.join(args.outDir, 'ranker.txt')
    ranker.booster_.save_model(model_txt)

    bundle = {
        'generatedAt': datetime.datetime.now().isoformat(timespec='seconds'),
        'model': 'LGBMRanker_lambdarank_top2rel',
        'data': args.data,
        'ranges': {'train': {'start': args.trainStart, 'end': args.trainEnd}},
        'feat_keys': feat_keys,
        'seed': args.seed,
        'no_odds': False,
        'note': 'Trained to match Feb22 Q two-pairs report feature set (includes cur_win_odds + jockey/trainer_60d_*).'
    }
    json.dump(bundle, open(os.path.join(args.outDir, 'bundle.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print(json.dumps({'ok': True, 'outDir': args.outDir, 'feat_count': len(feat_keys), 'model_txt': model_txt}, indent=2))


if __name__ == '__main__':
    main()
