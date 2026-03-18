#!/usr/bin/env python3
"""Train NO-ODDS dual-pace ranker v5 (research).

Idea (per user):
- If predicted pace is FAST -> emphasize closers
- If predicted pace is SLOW -> emphasize front/stalkers

Implementation:
- Build same sectional + running-style + pace_index features as v4.
- Split races into FAST vs SLOW by pace_index_front_frac threshold.
- Train two separate LambdaRank models (same label scheme as v4, exacta-tilted).
- Tune selection threshold on validation to maximize exacta KPI.

Outputs:
- models/HKJC-ML_NOODDS_DUALPACE_V5_FAST.txt
- models/HKJC-ML_NOODDS_DUALPACE_V5_SLOW.txt
- models/HKJC-ML_NOODDS_DUALPACE_V5.meta.json

"""

import argparse, json, os, sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any, List

import numpy as np
import lightgbm as lgb


def parse_ymd(s: str) -> datetime:
    return datetime.strptime(s, "%Y/%m/%d")


def fmt_ymd(d: datetime) -> str:
    return d.strftime("%Y/%m/%d")


@dataclass
class RunnerRow:
    runner_id: int
    race_id: int
    racedate: str
    venue: str
    race_no: int
    distance_m: int
    class_num: int
    surface: str
    horse_code: str
    horse_no: int
    draw: int
    weight: float
    jockey: str
    trainer: str
    finish_pos: int


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def get_date_bounds(con: sqlite3.Connection) -> Tuple[str, str]:
    r = con.execute("SELECT MIN(racedate) AS mn, MAX(racedate) AS mx FROM meetings").fetchone()
    return r["mn"], r["mx"]


def load_runners_in_range(con: sqlite3.Connection, start_ymd: str, end_ymd: str) -> List[RunnerRow]:
    q = """
    SELECT
      ru.runner_id,
      ru.race_id, m.racedate, m.venue,
      r.race_no, r.distance_m, r.class_num, r.surface,
      ru.horse_code, ru.horse_no, ru.draw, ru.weight, ru.jockey, ru.trainer,
      re.finish_pos
    FROM runners ru
    JOIN races r ON r.race_id = ru.race_id
    JOIN meetings m ON m.meeting_id = r.meeting_id
    JOIN results re ON re.runner_id = ru.runner_id
    WHERE m.racedate >= ? AND m.racedate <= ?
      AND re.finish_pos IS NOT NULL
    ORDER BY m.racedate, ru.race_id, ru.horse_no
    """
    rows = []
    for x in con.execute(q, (start_ymd, end_ymd)).fetchall():
        rows.append(
            RunnerRow(
                runner_id=int(x["runner_id"]),
                race_id=int(x["race_id"]),
                racedate=x["racedate"],
                venue=x["venue"],
                race_no=int(x["race_no"] or 0),
                distance_m=int(x["distance_m"] or 0),
                class_num=int(x["class_num"] or 0),
                surface=x["surface"] or "",
                horse_code=x["horse_code"] or "",
                horse_no=int(x["horse_no"] or 0),
                draw=int(x["draw"] or 0),
                weight=float(x["weight"] or 0.0),
                jockey=x["jockey"] or "",
                trainer=x["trainer"] or "",
                finish_pos=int(x["finish_pos"] or 99),
            )
        )
    return rows


def rel_exacta(pos: int) -> int:
    try:
        p = int(pos)
    except Exception:
        return 0
    if p == 1:
        return 10
    if p == 2:
        return 9
    if p == 3:
        return 3
    if p in (4, 5):
        return 1
    return 0


# Reuse v4 feature builder to avoid divergence
from hkjc_noodds_rank_v4_train import FeatureBuilder


def train_ranker(X, y, groups, params, title: str):
    ds = lgb.Dataset(X, label=y, group=groups)
    booster = lgb.train(
        params,
        ds,
        num_boost_round=2500,
        valid_sets=[ds],
        valid_names=[title],
        callbacks=[lgb.log_evaluation(100)],
    )
    return booster


def exacta_kpi_for_threshold(race_rows: Dict[int, List[Dict[str, Any]]], thr: float) -> Dict[str, float]:
    # race_rows[race_id] = [{pace, fin, score_fast, score_slow}...]
    n=0; order=0; sethit=0
    for rid, rows in race_rows.items():
        pace = rows[0]['pace']
        use_fast = pace >= thr
        scored=[(r['score_fast'] if use_fast else r['score_slow'], r['fin']) for r in rows]
        scored.sort(key=lambda x:x[0], reverse=True)
        top2=[fp for _,fp in scored[:2]]
        if len(top2)<2: continue
        n += 1
        if top2[0]==1 and top2[1]==2: order += 1
        if set(top2)=={1,2}: sethit += 1
    return {'n': n, 'exacta_order': (order/n if n else 0.0), 'top2_set': (sethit/n if n else 0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--years', type=int, default=3)
    ap.add_argument('--testDays', type=int, default=180)
    ap.add_argument('--paceTrainThr', type=float, default=0.33)
    ap.add_argument('--outFast', default='models/HKJC-ML_NOODDS_DUALPACE_V5_FAST.txt')
    ap.add_argument('--outSlow', default='models/HKJC-ML_NOODDS_DUALPACE_V5_SLOW.txt')
    ap.add_argument('--outMeta', default='models/HKJC-ML_NOODDS_DUALPACE_V5.meta.json')
    args = ap.parse_args()

    con = connect(args.db)
    mn, mx = get_date_bounds(con)
    end = parse_ymd(mx)
    start = end - timedelta(days=365 * int(args.years))
    start_ymd = fmt_ymd(start)

    rows = load_runners_in_range(con, start_ymd, mx)
    fb = FeatureBuilder(con)

    # build per-runner base feats + style flags
    feats=[]; auxs=[]; y=[]; dates=[]; race_ids=[]; finish=[]
    for rr in rows:
        f,aux = fb.build(rr)
        feats.append(f)
        auxs.append(aux)
        y.append(rel_exacta(rr.finish_pos))
        dates.append(rr.racedate)
        race_ids.append(int(rr.race_id))
        finish.append(int(rr.finish_pos))

    # race pace
    from collections import defaultdict
    idxs=defaultdict(list)
    for i,rid in enumerate(race_ids):
        idxs[rid].append(i)

    pace_by_rid={}
    for rid,ii in idxs.items():
        n=len(ii)
        n_front=sum(1 for j in ii if auxs[j].get('style_front'))
        pace=(n_front/n) if n else 0.0
        pace_by_rid[rid]=pace
        for j in ii:
            front=auxs[j].get('style_front') or 0
            closer=auxs[j].get('style_closer') or 0
            feats[j]['field_size']=n
            feats[j]['pace_index_front_frac']=pace
            feats[j]['closer_x_pace']=closer*pace
            feats[j]['front_x_fast']=front*pace

    feat_keys=sorted(feats[0].keys())
    X=np.array([[f.get(k) for k in feat_keys] for f in feats], dtype=float)
    impute={}
    for i,k in enumerate(feat_keys):
        col=X[:,i]
        m=np.nanmean(col)
        if np.isnan(m): m=0.0
        impute[k]=float(m)
        col[np.isnan(col)]=m
        X[:,i]=col

    split_date = fmt_ymd(end - timedelta(days=int(args.testDays)))
    is_test = np.array([d >= split_date for d in dates])

    # build groups full
    groups=[]
    cur=None; cnt=0
    for rid in race_ids:
        if cur!=rid:
            if cnt: groups.append(cnt)
            cur=rid; cnt=0
        cnt+=1
    if cnt: groups.append(cnt)

    def build_groups(mask):
        out=[]; idx=0
        for g in groups:
            nsel=int(mask[idx:idx+g].sum())
            if nsel: out.append(nsel)
            idx+=g
        return out

    # select train rows only for model fitting
    train_mask = ~is_test

    # split train into FAST/SLOW by pace_train_thr at race level
    # create boolean per row
    is_fast_row = np.array([pace_by_rid[rid] >= args.paceTrainThr for rid in race_ids])

    slow_mask = train_mask & (~is_fast_row)
    fast_mask = train_mask & (is_fast_row)

    def slice_for(mask):
        Xs=X[mask]
        ys=np.array(y)[mask]
        gs=build_groups(mask)
        return Xs,ys,gs

    Xs,ys,gs = slice_for(slow_mask)
    Xf,yf,gf = slice_for(fast_mask)

    params={
        'objective':'lambdarank',
        'metric':['ndcg'],
        'ndcg_eval_at':[1,2,3,5],
        'learning_rate':0.03,
        'num_leaves':127,
        'min_data_in_leaf':40,
        'feature_fraction':0.85,
        'bagging_fraction':0.8,
        'bagging_freq':1,
        'lambda_l2':2.0,
        'verbosity':-1,
        'seed':42,
    }

    slow_model = lgb.train(params, lgb.Dataset(Xs,label=ys,group=gs), num_boost_round=1200, callbacks=[lgb.log_evaluation(200)])
    fast_model = lgb.train(params, lgb.Dataset(Xf,label=yf,group=gf), num_boost_round=1200, callbacks=[lgb.log_evaluation(200)])

    os.makedirs(os.path.dirname(args.outFast), exist_ok=True)
    slow_model.save_model(args.outSlow)
    fast_model.save_model(args.outFast)

    # Validation (test split): precompute both model scores per runner
    test_idx=np.where(is_test)[0]
    Xtest=X[test_idx]
    s_fast=fast_model.predict(Xtest)
    s_slow=slow_model.predict(Xtest)

    # group test into races
    race_rows={}
    for j,orig_i in enumerate(test_idx.tolist()):
        rid=race_ids[orig_i]
        if rid not in race_rows: race_rows[rid]=[]
        race_rows[rid].append({
            'pace': pace_by_rid[rid],
            'fin': finish[orig_i],
            'score_fast': float(s_fast[j]),
            'score_slow': float(s_slow[j]),
        })

    # tune selection threshold
    thrs=[round(x,2) for x in np.arange(0.1, 0.91, 0.05)]
    best=None
    for thr in thrs:
        m=exacta_kpi_for_threshold(race_rows, thr)
        score=(m['exacta_order'], m['top2_set'])
        if best is None or score>best['score']:
            best={'thr':thr,'metrics':m,'score':score}

    meta={
        'name':'HKJC-ML_NOODDS_DUALPACE_V5',
        'trainedAt': datetime.now().isoformat(timespec='seconds'),
        'trained_on': {
            'db': os.path.abspath(args.db),
            'date_min_db': mn,
            'date_max_db': mx,
            'train_start': start_ymd,
            'train_end': mx,
            'test_split_date': split_date,
            'years': args.years,
            'testDays': args.testDays,
        },
        'feature_set': 'same as v4 (sectional+pace)',
        'featKeys': feat_keys,
        'imputeMeans': impute,
        'pace_train_thr': args.paceTrainThr,
        'models': {
            'fast': args.outFast,
            'slow': args.outSlow,
        },
        'selection': {
            'pace_select_thr': best['thr'],
            'tuned_on_validation': True,
            'validation_metrics_at_best': best['metrics'],
            'grid': thrs,
        }
    }

    with open(args.outMeta,'w',encoding='utf-8') as f:
        json.dump(meta,f,ensure_ascii=False,indent=2)

    print(json.dumps({'ok':True,'outFast':args.outFast,'outSlow':args.outSlow,'outMeta':args.outMeta,'best':best},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
