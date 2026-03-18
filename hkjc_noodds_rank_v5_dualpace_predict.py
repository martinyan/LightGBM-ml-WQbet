#!/usr/bin/env python3
"""Predict NO-ODDS Top5 using dual-pace model v5.

Loads meta JSON which points to two models:
- FAST pace model
- SLOW pace model

Selection:
- compute pace_index_front_frac from running-style proxy
- if pace_index >= pace_select_thr -> use FAST model else SLOW

Usage:
  python3 hkjc_noodds_rank_v5_dualpace_predict.py --db hkjc.sqlite --in merged.json --out pred.json
"""

import argparse, json, os, re, sqlite3
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import lightgbm as lgb

from hkjc_noodds_rank_v4_predict import FeatureBuilder as FB


def parse_race_key(url: str) -> Dict[str, Any]:
    m = re.search(r"/racing/wp/(\d{4}-\d{2}-\d{2})/(HV|ST)/(\d+)", url or "")
    if not m:
        return {"racedate": None, "venue": None, "raceNo": None}
    return {"racedate": m.group(1).replace('-', '/'), "venue": m.group(2), "raceNo": int(m.group(3))}


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--in', dest='inPath', required=True)
    ap.add_argument('--out', dest='outPath', required=True)
    ap.add_argument('--meta', default='models/HKJC-ML_NOODDS_DUALPACE_V5.meta.json')
    ap.add_argument('--topK', type=int, default=5)
    args = ap.parse_args()

    meta = json.load(open(args.meta,'r',encoding='utf-8'))
    feat_keys = meta['featKeys']
    impute = meta.get('imputeMeans') or {}
    fast_path = meta['models']['fast']
    slow_path = meta['models']['slow']
    sel = meta.get('selection') or {}
    # backward compatible
    pace_thr = float(sel.get('pace_select_thr') or sel.get('thr') or 0.25)
    sel_kind = sel.get('kind') or 'hard'
    sel_tau = float(sel.get('tau') or 0.0)

    card = json.load(open(args.inPath,'r',encoding='utf-8'))
    bet_url = (card.get('betPage') or {}).get('url')
    key = parse_race_key(bet_url)
    racedate, venue, raceNo = key.get('racedate'), key.get('venue'), key.get('raceNo')

    distance_m = int(((card.get('betPage') or {}).get('distanceMeters')) or 0)
    class_num = int(((card.get('betPage') or {}).get('classNum')) or 0)
    surface = ((card.get('betPage') or {}).get('surfaceText')) or ''

    con = connect(args.db)
    fb = FB(con)

    picks = card.get('picks') or []
    feats=[]; auxs=[]
    for p in picks:
        f,aux = fb.build(racedate, venue, distance_m, class_num, surface, p)
        feats.append(f); auxs.append(aux)

    n=len(picks)
    n_front=sum(1 for a in auxs if a.get('style_front'))
    pace=(n_front/n) if n else 0.0

    for f,a in zip(feats,auxs):
        front=a.get('style_front') or 0
        closer=a.get('style_closer') or 0
        f['field_size']=n
        f['pace_index_front_frac']=pace
        f['closer_x_pace']=closer*pace
        f['front_x_fast']=front*pace

    # scoring
    # hard: pick one model; soft: blend two models by w=sigmoid((pace-thr)/tau)
    def sigmoid(x):
        import math
        return 1.0/(1.0+math.exp(-x))

    use_fast = pace >= pace_thr
    booster_fast = lgb.Booster(model_file=fast_path)
    booster_slow = lgb.Booster(model_file=slow_path)

    X=np.array([[f.get(k) for k in feat_keys] for f in feats], dtype=float)
    for i,k in enumerate(feat_keys):
        col=X[:,i]
        m=float(impute.get(k,0.0))
        col[np.isnan(col)]=m
        X[:,i]=col

    sf = booster_fast.predict(X)
    ss = booster_slow.predict(X)

    if sel_kind == 'soft' and sel_tau > 0:
        w = sigmoid((pace - pace_thr) / sel_tau)
    else:
        w = 1.0 if use_fast else 0.0

    scores = [w*float(a) + (1.0-w)*float(b) for a,b in zip(sf,ss)]

    scored_all=[]
    for p,sc,a,b in zip(picks,scores,sf,ss):
        scored_all.append({
            'horse_no': int(p.get('no') or 0),
            'horse': p.get('horse'),
            'draw': int(p.get('draw') or 0),
            'jockey': p.get('jockey'),
            'trainer': p.get('trainer'),
            'score': float(sc),
            'score_fast': float(a),
            'score_slow': float(b),
            'w_fast': float(w),
        })
    scored_all.sort(key=lambda x:x['score'], reverse=True)

    out={
        'model': meta.get('name'),
        'racedate': racedate,
        'venue': venue,
        'raceNo': raceNo,
        'pace': {
            'field_size': n,
            'n_front': n_front,
            'pace_index_front_frac': pace,
            'select': {'kind': sel_kind, 'thr': pace_thr, 'tau': sel_tau, 'w_fast': float(w)},
            'used': 'FAST' if use_fast else 'SLOW'
        },
        'top5': scored_all[: int(args.topK)],
        'scored_all': scored_all,
        'generatedAt': datetime.utcnow().isoformat()+'Z',
        'betPage': card.get('betPage'),
    }

    os.makedirs(os.path.dirname(args.outPath) or '.', exist_ok=True)
    json.dump(out, open(args.outPath,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps({'ok':True,'out':args.outPath,'used': out['pace']['used'], 'pace': pace}, ensure_ascii=False))


if __name__=='__main__':
    main()
