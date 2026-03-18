#!/usr/bin/env python3
"""Tune a soft-gating blend for dual-pace v5 models to improve exacta KPI.

Given existing FAST/SLOW models and feature pipeline, tune:
- thr (pace center)
- tau (temperature)
Blend weight:
  w = sigmoid((pace - thr)/tau)
Score:
  score = w*score_fast + (1-w)*score_slow

Writes updated meta with selection.kind='soft'.
"""

import argparse, json, sqlite3, math
from collections import defaultdict
import numpy as np
import lightgbm as lgb

from hkjc_noodds_rank_v4_train import FeatureBuilder
from datetime import datetime


def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--db',default='hkjc.sqlite')
    ap.add_argument('--meta',default='models/HKJC-ML_NOODDS_DUALPACE_V5.meta.json')
    ap.add_argument('--outMeta',default=None)
    ap.add_argument('--splitDate',default=None,help='override test split date')
    args=ap.parse_args()

    meta=json.load(open(args.meta,'r',encoding='utf-8'))
    split = args.splitDate or meta['trained_on']['test_split_date']

    feat=meta['featKeys']; imp=meta['imputeMeans']
    fast=lgb.Booster(model_file=meta['models']['fast'])
    slow=lgb.Booster(model_file=meta['models']['slow'])

    con=sqlite3.connect(args.db); con.row_factory=sqlite3.Row
    fb=FeatureBuilder(con)

    # build test rows from sqlite
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
        raw.append(f); aux.append(a); meta_rows.append((int(rr.race_id), int(rr.finish_pos)))

    idxs=defaultdict(list)
    for i,(rid,_) in enumerate(meta_rows):
        idxs[rid].append(i)

    # compute pace and model scores per runner
    race_rows={}  # rid -> list of dicts
    for rid,ii in idxs.items():
        n=len(ii); nfront=sum(1 for j in ii if aux[j].get('style_front'))
        pace=(nfront/n) if n else 0.0
        X=np.array([[raw[j].get(k) for k in feat] for j in ii], dtype=float)
        for ci,k in enumerate(feat):
            col=X[:,ci]; m=float(imp.get(k,0.0)); col[np.isnan(col)]=m; X[:,ci]=col
        sf=fast.predict(X); ss=slow.predict(X)
        race_rows[rid]=[]
        for j,a,b in zip(ii,sf,ss):
            race_rows[rid].append({'pace':pace,'fin':meta_rows[j][1],'sf':float(a),'ss':float(b)})

    def exacta_for(thr,tau):
        n=0; order=0; sethit=0
        for rid,rows in race_rows.items():
            pace=rows[0]['pace']
            w=sigmoid((pace-thr)/tau) if tau>0 else (1.0 if pace>=thr else 0.0)
            scored=[(w*r['sf']+(1-w)*r['ss'], r['fin']) for r in rows]
            scored.sort(key=lambda x:x[0], reverse=True)
            top2=[fp for _,fp in scored[:2]]
            if len(top2)<2: continue
            n+=1
            if top2[0]==1 and top2[1]==2: order+=1
            if set(top2)=={1,2}: sethit+=1
        return {'n':n,'exacta_order':order/n if n else 0.0,'top2_set':sethit/n if n else 0.0}

    thrs=[round(x,2) for x in np.arange(0.1,0.91,0.05)]
    taus=[0.05,0.08,0.1,0.15,0.2,0.3]
    best=None
    for thr in thrs:
        for tau in taus:
            m=exacta_for(thr,tau)
            score=(m['exacta_order'], m['top2_set'])
            if best is None or score>best['score']:
                best={'thr':thr,'tau':tau,'metrics':m,'score':score}

    meta2=dict(meta)
    meta2['selection']={
        'kind':'soft',
        'thr': best['thr'],
        'tau': best['tau'],
        'formula':'w=sigmoid((pace-thr)/tau); score=w*fast+(1-w)*slow',
        'validation_metrics_at_best': best['metrics'],
        'grid': {'thrs': thrs, 'taus': taus},
        'tunedAt': datetime.now().isoformat(timespec='seconds'),
    }

    out_path=args.outMeta or args.meta
    with open(out_path,'w',encoding='utf-8') as f:
        json.dump(meta2,f,ensure_ascii=False,indent=2)

    print(json.dumps({'ok':True,'outMeta':out_path,'best':best},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
