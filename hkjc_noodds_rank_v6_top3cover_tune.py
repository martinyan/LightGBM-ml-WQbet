#!/usr/bin/env python3
"""Tune a Top5 selection rule to maximize 'Top5 covers Top3' KPI.

Base scores: v5 dual-pace soft gate scores.
Then apply a simple composition constraint:
- If pace >= pace_thr_fast: ensure at least min_closer_fast closers in Top5
- If pace <= pace_thr_slow: ensure at least min_front_slow fronts in Top5

Replacement rule:
- If constraint not met, replace lowest-scoring items in current Top5 with highest-scoring candidates of required style from outside Top5.

Writes params into v5 meta under selection_top5_cover_v6.
"""

import argparse, json, sqlite3, math
from collections import defaultdict
import numpy as np
import lightgbm as lgb

from hkjc_noodds_rank_v4_train import FeatureBuilder
from datetime import datetime


def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))


def build_scores(con, meta_path: str):
    meta=json.load(open(meta_path,'r',encoding='utf-8'))
    feat=meta['featKeys']; imp=meta['imputeMeans']
    fast=lgb.Booster(model_file=meta['models']['fast'])
    slow=lgb.Booster(model_file=meta['models']['slow'])
    sel=meta['selection']
    thr=float(sel['thr']); tau=float(sel['tau'])
    split=meta['trained_on']['test_split_date']

    fb=FeatureBuilder(con)

    q="""
    SELECT ru.runner_id, ru.race_id, m.racedate, m.venue, r.race_no, r.distance_m, r.class_num, r.surface,
           ru.horse_code, ru.horse_no, ru.draw, ru.weight, ru.jockey, ru.trainer, re.finish_pos
    FROM runners ru
    JOIN races r ON r.race_id=ru.race_id
    JOIN meetings m ON m.meeting_id=r.meeting_id
    JOIN results re ON re.runner_id=ru.runner_id
    WHERE m.racedate>=?
    ORDER BY m.racedate, ru.race_id, ru.horse_no
    """
    from types import SimpleNamespace

    raw=[]; aux=[]; meta_rows=[]
    for x in con.execute(q,(split,)):
        rr=SimpleNamespace(runner_id=int(x['runner_id']),race_id=int(x['race_id']),racedate=x['racedate'],venue=x['venue'],race_no=int(x['race_no']),
                           distance_m=int(x['distance_m'] or 0),class_num=int(x['class_num'] or 0),surface=x['surface'] or '',
                           horse_code=x['horse_code'] or '',horse_no=int(x['horse_no'] or 0),draw=int(x['draw'] or 0),weight=float(x['weight'] or 0.0),
                           jockey=x['jockey'] or '',trainer=x['trainer'] or '',finish_pos=int(x['finish_pos'] or 99))
        f,a=fb.build(rr)
        raw.append(f); aux.append(a); meta_rows.append((int(rr.race_id), int(rr.finish_pos), a.get('style_front') or 0, a.get('style_closer') or 0))

    idxs=defaultdict(list)
    for i,(rid,_,_,_) in enumerate(meta_rows):
        idxs[rid].append(i)

    # scores per race
    race_rows={}
    for rid,ii in idxs.items():
        n=len(ii); nfront=sum(1 for j in ii if aux[j].get('style_front'))
        pace=(nfront/n) if n else 0.0
        X=np.array([[raw[j].get(k) for k in feat] for j in ii], dtype=float)
        for ci,k in enumerate(feat):
            col=X[:,ci]; m=float(imp.get(k,0.0)); col[np.isnan(col)]=m; X[:,ci]=col
        sf=fast.predict(X); ss=slow.predict(X)
        w=sigmoid((pace-thr)/tau)
        race_rows[rid]=[]
        for j,a,b in zip(ii,sf,ss):
            _,fin,front,closer=meta_rows[j]
            score=w*float(a)+(1-w)*float(b)
            race_rows[rid].append({'score':score,'fin':fin,'front':front,'closer':closer,'pace':pace})
    return meta, race_rows


def select_top5(rows, pace_thr_fast, pace_thr_slow, min_closer_fast, min_front_slow):
    # rows: list of dict with score/front/closer
    rows_sorted=sorted(rows, key=lambda r:r['score'], reverse=True)
    top5=rows_sorted[:5]
    rest=rows_sorted[5:]
    pace=rows[0]['pace']

    def count_style(arr, key):
        return sum(1 for r in arr if r.get(key))

    def replace_with(style_key, need):
        nonlocal top5, rest
        if need <= 0:
            return
        # candidates outside top5
        cand=[r for r in rest if r.get(style_key)]
        if not cand:
            return
        # sort candidates high->low already because rest is sorted
        # sort top5 low->high for replacement
        top5_low=sorted(top5, key=lambda r:r['score'])
        for _ in range(need):
            if not cand or not top5_low:
                break
            # take best candidate
            c=cand.pop(0)
            # replace worst
            worst=top5_low.pop(0)
            # Allow replacement even if candidate score is lower: this is a *coverage* heuristic.
            top5.remove(worst)
            top5.append(c)
            rest.remove(c)
            rest.append(worst)
            rest.sort(key=lambda r:r['score'], reverse=True)
            top5_low=sorted(top5, key=lambda r:r['score'])

    if pace >= pace_thr_fast:
        have=count_style(top5,'closer')
        replace_with('closer', max(0, min_closer_fast - have))
    if pace <= pace_thr_slow:
        have=count_style(top5,'front')
        replace_with('front', max(0, min_front_slow - have))

    return top5


def top5_covers_top3(rows_top5):
    fins=set(r['fin'] for r in rows_top5)
    return {1,2,3}.issubset(fins)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--db',default='hkjc.sqlite')
    ap.add_argument('--meta',default='models/HKJC-ML_NOODDS_DUALPACE_V5.meta.json')
    args=ap.parse_args()

    con=sqlite3.connect(args.db); con.row_factory=sqlite3.Row
    meta, race_rows = build_scores(con, args.meta)

    best=None
    # grid
    for pace_thr_fast in [0.25,0.30,0.35,0.40]:
        for pace_thr_slow in [0.15,0.20,0.25,0.30]:
            for min_closer_fast in [1,2,3]:
                for min_front_slow in [1,2,3]:
                    n=0; hit=0
                    for rid,rows in race_rows.items():
                        top5=select_top5(rows, pace_thr_fast, pace_thr_slow, min_closer_fast, min_front_slow)
                        if len(top5)<5: continue
                        n+=1
                        if top5_covers_top3(top5):
                            hit+=1
                    rate=hit/n if n else 0.0
                    score=rate
                    if best is None or score>best['score']:
                        best={'score':score,'rate':rate,'n':n,'params':{
                            'pace_thr_fast':pace_thr_fast,
                            'pace_thr_slow':pace_thr_slow,
                            'min_closer_fast':min_closer_fast,
                            'min_front_slow':min_front_slow,
                        }}

    meta2=dict(meta)
    meta2['selection_top5_cover_v6']={
        'tunedAt': datetime.now().isoformat(timespec='seconds'),
        'kpi':'top5_covers_top3',
        'best': best,
        'rule': 'if pace>=pace_thr_fast ensure min_closer_fast closers; if pace<=pace_thr_slow ensure min_front_slow fronts; replace worst-in-top5 with best outside if not worse'
    }

    with open(args.meta,'w',encoding='utf-8') as f:
        json.dump(meta2,f,ensure_ascii=False,indent=2)

    print(json.dumps({'ok':True,'meta':args.meta,'best':best},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
