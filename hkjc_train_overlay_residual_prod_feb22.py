import os, json, math, argparse, pickle, hashlib
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


def group_by_meeting(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r['racedate'], r['venue'])].append(r)
    return by


def list_meetings_sqlite(db_path, start, end):
    import sqlite3
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        'select racedate, venue, meeting_id from meetings where racedate>=? and racedate<=? order by racedate asc, venue asc',
        (start, end),
    )
    rows = [(r, v, mid) for (r, v, mid) in cur.fetchall()]
    con.close()
    return rows


def market_probs_for_race(runners):
    p_raw = []
    for rr in runners:
        o = safe_float(rr.get('cur_win_odds'), None)
        p_raw.append(0.0 if (o is None or o <= 0) else 1.0 / o)
    s = sum(p_raw)
    if s <= 0:
        return [1.0 / len(runners)] * len(runners)
    return [x / s for x in p_raw]


def train_overlay_win_model(rows_fit, feat_keys):
    races = group_by_race(rows_fit)
    X = []
    y = []
    for _, runners in races.items():
        p_mkt = market_probs_for_race(runners)
        for rr, pm in zip(runners, p_mkt):
            fp = rr.get('y_finish_pos')
            if fp is None:
                continue
            y_win = 1.0 if int(fp) == 1 else 0.0
            res_win = y_win - float(pm)
            X.append([safe_float(rr.get(k), 0.0) for k in feat_keys])
            y.append(res_win)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    mdl = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
    )
    mdl.fit(X, y)
    return mdl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--dataset', default='hkjc_dataset_v6b_code_include_debut_jt60_prev3_fullcols.jsonl')
    ap.add_argument('--trainStart', default='2023/09/01')
    ap.add_argument('--trainEnd', default='2025/07/31')
    ap.add_argument('--evalFracMeetings', type=float, default=0.2)
    ap.add_argument('--featKeysFromReport', default=None, help='Path to a JSON report containing features.feat_keys to force exact feature set (for reproducibility)')
    ap.add_argument('--outPkl', default='models/OVERLAY_RESIDUAL_LGBM_v1_FEB22.pkl')
    ap.add_argument('--outJson', default='models/OVERLAY_RESIDUAL_LGBM_v1_FEB22.bundle.json')
    args = ap.parse_args()

    rows = load_jsonl(args.dataset)
    by_meeting = group_by_meeting(rows)

    train_meetings = list_meetings_sqlite(args.db, args.trainStart, args.trainEnd)
    n_train = len(train_meetings)
    n_eval = max(1, int(round(n_train * args.evalFracMeetings)))
    fit_meetings = train_meetings[:-n_eval]

    def build_rows(meeting_list):
        s = {(r, v) for (r, v, _) in meeting_list}
        out = []
        for k in s:
            out.extend(by_meeting.get(k, []))
        return out

    fit_rows = build_rows(fit_meetings)
    if not fit_rows:
        raise SystemExit('no fit rows')

    if args.featKeysFromReport:
        rep = json.load(open(args.featKeysFromReport, 'r', encoding='utf-8'))
        feat_keys = rep.get('features', {}).get('feat_keys')
        if not feat_keys:
            raise SystemExit('featKeysFromReport provided but features.feat_keys not found')
    else:
        meta = {'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no','y_finish_pos','y_win','y_place','cur_jockey','cur_trainer'}
        cand = [k for k in fit_rows[0].keys() if k not in meta and not k.startswith('y_') and not k.startswith('_')]
        feat_keys = sorted([k for k in cand if k not in ('cur_win_odds',)])

    mdl_win = train_overlay_win_model(fit_rows, feat_keys)

    os.makedirs(os.path.dirname(args.outPkl) or '.', exist_ok=True)
    bundle = {
        'generatedAt': __import__('datetime').datetime.now().isoformat(timespec='seconds'),
        'model': 'OVERLAY_RESIDUAL_LGBM_v1',
        'dataset': args.dataset,
        'train': {'start': args.trainStart, 'end': args.trainEnd, 'evalFracMeetings': args.evalFracMeetings, 'fitMeetings': len(fit_meetings), 'totalMeetings': n_train},
        'feat_keys': feat_keys,
        'mdl_win': mdl_win,
        'alpha': 1.0,
        'beta': 0.0,
        'overlay_definition': 'overlay = alpha * res_hat_win; res_hat_win trained on (y_win - p_mkt_win), where p_mkt_win = normalized 1/odds',
        'random_state': 42,
    }

    with open(args.outPkl, 'wb') as f:
        pickle.dump(bundle, f)

    # small json metadata (no binary model)
    meta_out = {k: v for k, v in bundle.items() if k not in ('mdl_win',)}
    with open(args.outJson, 'w', encoding='utf-8') as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    print(json.dumps({'ok': True, 'outPkl': args.outPkl, 'outJson': args.outJson, 'feat_count': len(feat_keys)}, indent=2))


if __name__ == '__main__':
    main()
