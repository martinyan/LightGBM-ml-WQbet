import os, json, math, argparse, sqlite3, hashlib, datetime
from collections import defaultdict

import numpy as np

import lightgbm as lgb
from catboost import CatBoostRanker, Pool
from sklearn.isotonic import IsotonicRegression

# ------------------------- utils -------------------------

def load_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def stable_hash01(s: str, mod: int = 10_000_000) -> float:
    if not s:
        return 0.0
    h = hashlib.sha1(s.encode('utf-8')).digest()
    x = int.from_bytes(h[:8], 'big', signed=False)
    return (x % mod) / float(mod)


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    m = np.max(x)
    e = np.exp(x - m)
    s = np.sum(e)
    return e / s if s else np.ones_like(x) / len(x)


# relevance scheme: 1st=3, 2nd=2, 3rd=1, else 0

def relevance_from_finish_pos(pos):
    if pos is None:
        return None
    try:
        p = int(pos)
    except Exception:
        return None
    if p == 1:
        return 3
    if p == 2:
        return 2
    if p == 3:
        return 1
    return 0


def ndcg_at_k_by_group(y_true, y_score, group_sizes, k=5):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = []
    idx = 0
    for g in group_sizes:
        yt = y_true[idx:idx+g]
        ys = y_score[idx:idx+g]
        idx += g
        if g == 0:
            continue
        order = np.argsort(-ys)
        yt_sorted = yt[order]
        # DCG
        kk = min(k, g)
        gains = (2.0 ** yt_sorted[:kk] - 1.0)
        discounts = 1.0 / np.log2(np.arange(2, kk + 2))
        dcg = float(np.sum(gains * discounts))
        # IDCG
        ideal = np.sort(yt)[::-1]
        gains_i = (2.0 ** ideal[:kk] - 1.0)
        idcg = float(np.sum(gains_i * discounts))
        out.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(out)) if out else 0.0


def list_meetings(db_path, start, end):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        'select racedate, venue, meeting_id from meetings where racedate>=? and racedate<=? order by racedate asc, venue asc',
        (start, end)
    )
    rows = [(r, v, mid) for (r, v, mid) in cur.fetchall()]
    con.close()
    return rows


def group_rows_by_meeting(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r.get('racedate'), r.get('venue'))].append(r)
    return by


def group_rows_by_race(rows):
    by = defaultdict(list)
    for r in rows:
        key = (r.get('racedate'), r.get('venue'), int(r.get('race_no')))
        by[key].append(r)
    return by


def build_feature_matrix(rows, feat_keys, cat_cols):
    X = []
    for r in rows:
        row = []
        for k in feat_keys:
            v = r.get(k)
            if k in cat_cols:
                # numeric hash feature for LGBM
                row.append(stable_hash01(str(v) if v is not None else ''))
            else:
                row.append(float(v or 0.0))
        X.append(row)
    return np.asarray(X, dtype=np.float32)


def build_grouped_arrays(rows, feat_keys, cat_cols):
    races = group_rows_by_race(rows)
    group_sizes = []
    Xs = []
    ys = []
    metas = []
    for (rd, v, rn), runners in sorted(races.items()):
        # require at least 2 runners with labels
        rels = []
        for rr in runners:
            rels.append(relevance_from_finish_pos(rr.get('y_finish_pos')))
        if any(x is None for x in rels):
            continue
        if len(runners) < 2:
            continue
        group_sizes.append(len(runners))
        Xs.append(build_feature_matrix(runners, feat_keys, cat_cols))
        ys.append(np.asarray(rels, dtype=np.float32))
        for rr in runners:
            metas.append({
                'racedate': rr.get('racedate'),
                'venue': rr.get('venue'),
                'race_no': int(rr.get('race_no')),
                'horse_no': int(rr.get('horse_no')),
                'runner_id': rr.get('runner_id'),
                'y_finish_pos': rr.get('y_finish_pos')
            })
    if not group_sizes:
        return None
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return X, y, group_sizes, metas


def backtest_q_qp(picks_by_race, dividend_cache_path, stake=10.0):
    # picks_by_race: dict[(racedate,venue,race_no)] -> list of horse_no (sorted desc by score)
    try:
        cache = json.load(open(dividend_cache_path, 'r', encoding='utf-8'))
    except Exception:
        cache = {}

    totals = {
        'Q_top2pair': {'stake': 0.0, 'return': 0.0, 'hit': 0, 'races': 0, 'missing': 0},
        'QP_top2pair': {'stake': 0.0, 'return': 0.0, 'hit': 0, 'races': 0, 'missing': 0},
    }

    for (rd, venue, rn), horses in sorted(picks_by_race.items()):
        if len(horses) < 2:
            continue
        h1, h2 = int(horses[0]), int(horses[1])
        pair = f"{min(h1,h2)}-{max(h1,h2)}"
        url = f'https://racing.hkjc.com/zh-hk/local/information/localresults?racedate={rd}&Racecourse={venue}&RaceNo={rn}'
        k = sha1(url)
        pools = cache.get(k)
        if not pools:
            # no web fetch here; count missing
            totals['Q_top2pair']['missing'] += 1
            totals['QP_top2pair']['missing'] += 1
            continue

        def find_div(pool_name):
            for combo, div in pools.get(pool_name, []):
                nums = [int(x) for x in __import__('re').findall(r'(\d+)', str(combo))]
                if len(nums) >= 2:
                    key = f"{min(nums[0], nums[1])}-{max(nums[0], nums[1])}"
                    if key == pair:
                        return float(div)
            return None

        q_div = find_div('連贏')
        qp_div = find_div('位置Q')

        totals['Q_top2pair']['races'] += 1
        totals['Q_top2pair']['stake'] += stake
        if q_div is not None:
            totals['Q_top2pair']['return'] += q_div
            totals['Q_top2pair']['hit'] += 1

        totals['QP_top2pair']['races'] += 1
        totals['QP_top2pair']['stake'] += stake
        if qp_div is not None:
            totals['QP_top2pair']['return'] += qp_div
            totals['QP_top2pair']['hit'] += 1

    for k, v in totals.items():
        v['profit'] = v['return'] - v['stake']
        v['roi'] = (v['profit'] / v['stake']) if v['stake'] else None
    return totals


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--data', default='hkjc_dataset_v4_code_include_debut.jsonl', help='JSONL built from sqlite')
    ap.add_argument('--trainStart', default='2023/09/01')
    ap.add_argument('--trainEnd', default='2025/07/31')
    ap.add_argument('--testStart', default='2025/09/01')
    ap.add_argument('--testEnd', default='2026/02/19')
    ap.add_argument('--evalFracMeetings', type=float, default=0.2)
    ap.add_argument('--catLoss', default='YetiRank', help='CatBoost loss: YetiRank or PairLogitPairwise')
    ap.add_argument('--stake', type=float, default=10.0)
    ap.add_argument('--dividendCache', default='hkjc_dividend_cache_v2.json')
    ap.add_argument('--outDirModel', default='models/HKJC-ML_RANK_BLEND')
    ap.add_argument('--outDirReport', default='reports/HKJC-ML_RANK_BLEND')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outDirModel, exist_ok=True)
    os.makedirs(args.outDirReport, exist_ok=True)

    rows = load_jsonl(args.data)
    by_meeting = group_rows_by_meeting(rows)

    train_meetings = list_meetings(args.db, args.trainStart, args.trainEnd)
    test_meetings = list_meetings(args.db, args.testStart, args.testEnd)
    if not train_meetings or not test_meetings:
        raise SystemExit('No meetings in range')

    n_train = len(train_meetings)
    n_eval = max(1, int(round(n_train * args.evalFracMeetings)))
    fit_meetings = train_meetings[:-n_eval]
    eval_meetings = train_meetings[-n_eval:]

    def build_meeting_rows(meeting_list):
        s = {(r, v) for (r, v, _) in meeting_list}
        out = []
        for key in s:
            out.extend(by_meeting.get(key, []))
        return out

    fit_rows = build_meeting_rows(fit_meetings)
    eval_rows = build_meeting_rows(eval_meetings)
    train_rows = build_meeting_rows(train_meetings)
    test_rows = build_meeting_rows(test_meetings)

    # feature selection
    meta = {
        'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no',
        'y_finish_pos','y_win','y_place',
    }
    # Cat features (strings) used as hashed numeric in LGBM; CatBoost will take raw strings
    cat_cols = ['cur_jockey', 'cur_trainer', 'cur_surface', 'venue']

    # choose feat keys: numeric + explicit categorical cols
    all_keys = set(rows[0].keys())
    feat_keys = sorted([k for k in all_keys if k not in meta and not k.startswith('y_')])
    # ensure cats present
    for c in cat_cols:
        if c not in feat_keys and c in all_keys:
            feat_keys.append(c)

    # build grouped arrays
    fit_pack = build_grouped_arrays(fit_rows, feat_keys, cat_cols)
    eval_pack = build_grouped_arrays(eval_rows, feat_keys, cat_cols)
    train_pack = build_grouped_arrays(train_rows, feat_keys, cat_cols)
    test_pack = build_grouped_arrays(test_rows, feat_keys, cat_cols)

    if not all([fit_pack, eval_pack, train_pack, test_pack]):
        raise SystemExit('Insufficient grouped data after filtering')

    X_fit, y_fit, g_fit, _ = fit_pack
    X_eval, y_eval, g_eval, _ = eval_pack
    X_train, y_train, g_train, _ = train_pack
    X_test, y_test, g_test, meta_test = test_pack

    # ---------------- LightGBM ranker ----------------
    lgb_ranker = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        ndcg_eval_at=[5],
        learning_rate=0.05,
        n_estimators=2000,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        reg_lambda=1.0,
    )

    lgb_ranker.fit(
        X_fit, y_fit,
        group=g_fit,
        eval_set=[(X_eval, y_eval)],
        eval_group=[g_eval],
        eval_at=[5],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    p_lgb_eval = lgb_ranker.predict(X_eval)
    ndcg_lgb = ndcg_at_k_by_group(y_eval, p_lgb_eval, g_eval, k=5)

    # ---------------- CatBoost ranker ----------------
    # For CatBoost, build a Pool with group_id and raw categorical features.
    # We'll provide the same feature vector, but for cat columns we will pass raw strings in a separate matrix.

    def build_catboost_matrix(rows_for_pack):
        # returns X (object array) and group_id per row and label
        races = group_rows_by_race(rows_for_pack)
        X_list = []
        y_list = []
        group_id = []
        for (rd, venue, rn), runners in sorted(races.items()):
            rels = [relevance_from_finish_pos(rr.get('y_finish_pos')) for rr in runners]
            if any(x is None for x in rels) or len(runners) < 2:
                continue
            gid = f"{rd}_{venue}_{rn}"
            for rr, rel in zip(runners, rels):
                row = []
                for k in feat_keys:
                    if k in cat_cols:
                        row.append(str(rr.get(k) or ''))
                    else:
                        row.append(float(rr.get(k) or 0.0))
                X_list.append(row)
                y_list.append(float(rel))
                group_id.append(gid)
        return X_list, y_list, group_id

    Xc_fit, yc_fit, gid_fit = build_catboost_matrix(fit_rows)
    Xc_eval, yc_eval, gid_eval = build_catboost_matrix(eval_rows)

    cat_features = [feat_keys.index(c) for c in cat_cols if c in feat_keys]

    pool_fit = Pool(Xc_fit, label=yc_fit, group_id=gid_fit, cat_features=cat_features)
    pool_eval = Pool(Xc_eval, label=yc_eval, group_id=gid_eval, cat_features=cat_features)

    cb = CatBoostRanker(
        loss_function=args.catLoss,
        eval_metric='NDCG:top=5',
        iterations=4000,
        learning_rate=0.05,
        depth=8,
        random_seed=args.seed,
        l2_leaf_reg=3.0,
        od_type='Iter',
        od_wait=200,
        verbose=False
    )
    cb.fit(pool_fit, eval_set=pool_eval, use_best_model=True)

    p_cb_eval = cb.predict(pool_eval)
    ndcg_cb = ndcg_at_k_by_group(np.asarray(yc_eval), np.asarray(p_cb_eval), [sum(1 for _ in group) for group in []], k=5)  # dummy
    # compute ndcg properly using g_eval (same ordering as build_grouped_arrays for eval_rows)
    # We'll instead recompute on eval_pack with CatBoost predictions by aligning eval_pack ordering.

    # Align CatBoost eval predictions to eval_pack ordering by rebuilding eval_pack metas order.
    X_eval2, y_eval2, g_eval2, meta_eval2 = eval_pack
    # Build a lookup key->pred for catboost (race+horse)
    pred_lookup = {}
    idx0 = 0
    # we need the same order as build_catboost_matrix, which iterates sorted races then runners insertion order
    races_eval = group_rows_by_race(eval_rows)
    for (rd, venue, rn), runners in sorted(races_eval.items()):
        rels = [relevance_from_finish_pos(rr.get('y_finish_pos')) for rr in runners]
        if any(x is None for x in rels) or len(runners) < 2:
            continue
        for rr in runners:
            key = (rr.get('racedate'), rr.get('venue'), int(rr.get('race_no')), int(rr.get('horse_no')))
            pred_lookup[key] = float(p_cb_eval[idx0])
            idx0 += 1

    p_cb_eval_aligned = []
    for m in meta_eval2:
        key = (m['racedate'], m['venue'], m['race_no'], m['horse_no'])
        p_cb_eval_aligned.append(pred_lookup.get(key, 0.0))
    p_cb_eval_aligned = np.asarray(p_cb_eval_aligned, dtype=np.float64)
    ndcg_cb = ndcg_at_k_by_group(y_eval2, p_cb_eval_aligned, g_eval2, k=5)

    # ---------------- Blend tuning ----------------
    p_lgb_eval = np.asarray(p_lgb_eval, dtype=np.float64)
    best = {'w_lgb': None, 'ndcg@5': -1}
    for w in np.linspace(0, 1, 21):
        p = w * p_lgb_eval + (1 - w) * p_cb_eval_aligned
        nd = ndcg_at_k_by_group(y_eval2, p, g_eval2, k=5)
        if nd > best['ndcg@5']:
            best = {'w_lgb': float(w), 'ndcg@5': float(nd)}

    w_lgb = best['w_lgb']

    # ---------------- Retrain on full train ----------------
    lgb_ranker_final = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        ndcg_eval_at=[5],
        learning_rate=0.05,
        n_estimators=int(lgb_ranker.best_iteration_ or lgb_ranker.n_estimators),
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        reg_lambda=1.0,
    )
    lgb_ranker_final.fit(X_train, y_train, group=g_train)

    # CatBoost final train
    Xc_train, yc_train, gid_train = build_catboost_matrix(train_rows)
    pool_train = Pool(Xc_train, label=yc_train, group_id=gid_train, cat_features=cat_features)
    cb_final = CatBoostRanker(
        loss_function=args.catLoss,
        iterations=cb.get_best_iteration() + 1 if cb.get_best_iteration() else cb.tree_count_,
        learning_rate=0.05,
        depth=8,
        random_seed=args.seed,
        l2_leaf_reg=3.0,
        verbose=False
    )
    cb_final.fit(pool_train)

    # ---------------- Predict test + output picks ----------------
    p_lgb_test = np.asarray(lgb_ranker_final.predict(X_test), dtype=np.float64)

    # catboost test predict aligned like meta_test
    Xc_test, yc_test, gid_test = build_catboost_matrix(test_rows)
    pool_test = Pool(Xc_test, label=yc_test, group_id=gid_test, cat_features=cat_features)
    p_cb_test = np.asarray(cb_final.predict(pool_test), dtype=np.float64)

    # align cb test predictions to meta_test ordering
    pred_lookup_test = {}
    idx0 = 0
    races_test = group_rows_by_race(test_rows)
    for (rd, venue, rn), runners in sorted(races_test.items()):
        rels = [relevance_from_finish_pos(rr.get('y_finish_pos')) for rr in runners]
        if any(x is None for x in rels) or len(runners) < 2:
            continue
        for rr in runners:
            key = (rr.get('racedate'), rr.get('venue'), int(rr.get('race_no')), int(rr.get('horse_no')))
            pred_lookup_test[key] = float(p_cb_test[idx0])
            idx0 += 1

    p_cb_test_aligned = np.asarray([
        pred_lookup_test.get((m['racedate'], m['venue'], m['race_no'], m['horse_no']), 0.0)
        for m in meta_test
    ], dtype=np.float64)

    p_blend_test = w_lgb * p_lgb_test + (1 - w_lgb) * p_cb_test_aligned

    # ---------------- Isotonic calibration: score -> P(top3) ----------------
    # Fit on eval (meeting-held-out) to avoid leakage.
    y_eval_place = np.asarray([1 if (m.get('y_finish_pos') is not None and int(m.get('y_finish_pos')) <= 3) else 0 for m in meta_eval2], dtype=np.int32)
    p_blend_eval = w_lgb * p_lgb_eval + (1 - w_lgb) * p_cb_eval_aligned

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p_blend_eval, y_eval_place)
    p_top3_test = iso.predict(p_blend_test)

    # per-race picks
    picks_by_race = defaultdict(list)
    # store runner score + calibrated prob
    for m, s, pt in zip(meta_test, p_blend_test, p_top3_test):
        key = (m['racedate'], m['venue'], m['race_no'])
        picks_by_race[key].append((m['horse_no'], float(s), float(pt), m.get('y_finish_pos')))

    race_outputs = []
    for (rd, venue, rn), arr in sorted(picks_by_race.items()):
        arr_sorted = sorted(arr, key=lambda t: t[1], reverse=True)
        top5 = arr_sorted[:5]
        # p12_prob = p(top3)_1 + p(top3)_2 (from calibrated probs)
        p12_prob = float((arr_sorted[0][2] if len(arr_sorted) > 0 else 0.0) + (arr_sorted[1][2] if len(arr_sorted) > 1 else 0.0))

        # keep p12_like too (softmax concentration) for reference
        scores = np.asarray([x[1] for x in arr_sorted], dtype=np.float64)
        probs = softmax(scores)
        p12_like = float(probs[0] + (probs[1] if len(probs) > 1 else 0.0))

        race_outputs.append({
            'racedate': rd,
            'venue': venue,
            'race_no': rn,
            'top5': [{'horse_no': int(h), 'score': float(sc), 'p_top3': float(pt), 'finish_pos': fp} for (h, sc, pt, fp) in top5],
            'p12_like': p12_like,
            'p12_prob': p12_prob,
            'n_runners': len(arr_sorted)
        })

    # backtests
    bt = backtest_q_qp({k: [x[0] for x in sorted(v, key=lambda t: t[1], reverse=True)] for k, v in picks_by_race.items()},
                      dividend_cache_path=args.dividendCache,
                      stake=args.stake)

    # metrics test
    ndcg_test = ndcg_at_k_by_group(y_test, p_blend_test, g_test, k=5)

    report = {
        'generatedAt': datetime.datetime.now().isoformat(timespec='seconds'),
        'data': args.data,
        'ranges': {
            'train': {'start': args.trainStart, 'end': args.trainEnd},
            'test': {'start': args.testStart, 'end': args.testEnd},
        },
        'split': {
            'train_meetings': len(train_meetings),
            'fit_meetings': len(fit_meetings),
            'eval_meetings': len(eval_meetings),
            'test_meetings': len(test_meetings),
        },
        'features': {
            'n_features': len(feat_keys),
            'cat_cols': cat_cols,
            'feat_keys': feat_keys,
        },
        'eval': {
            'ndcg@5_lgb': float(ndcg_lgb),
            'ndcg@5_cb': float(ndcg_cb),
            'blend_best': best,
        },
        'test': {
            'ndcg@5_blend': float(ndcg_test),
        },
        'calibration': {
            'method': 'isotonic',
            'target': 'P(top3)',
            'fit': 'eval meetings (held-out from train)',
        },
        'backtest': bt,
        'outputs': {
            'n_test_races': len(race_outputs),
        }
    }

    # save artifacts
    path_lgb = os.path.join(args.outDirModel, 'lgbm_ranker.txt')
    lgb_ranker_final.booster_.save_model(path_lgb)

    path_cb = os.path.join(args.outDirModel, 'catboost_ranker.cbm')
    cb_final.save_model(path_cb)

    path_blend = os.path.join(args.outDirModel, 'blend.json')
    with open(path_blend, 'w', encoding='utf-8') as f:
        json.dump({'w_lgb': w_lgb, 'catLoss': args.catLoss}, f, indent=2)

    path_races = os.path.join(args.outDirReport, 'test_race_picks.json')
    with open(path_races, 'w', encoding='utf-8') as f:
        json.dump(race_outputs, f, ensure_ascii=False, indent=2)

    path_report = os.path.join(args.outDirReport, 'rank_blend_report.json')
    with open(path_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # quick stdout
    print(json.dumps({
        'modelDir': args.outDirModel,
        'reportDir': args.outDirReport,
        'eval_ndcg@5_lgb': ndcg_lgb,
        'eval_ndcg@5_cb': ndcg_cb,
        'blend_w_lgb': w_lgb,
        'test_ndcg@5_blend': ndcg_test,
        'backtest': bt,
        'report': path_report,
        'picks': path_races,
    }, indent=2))


if __name__ == '__main__':
    main()
