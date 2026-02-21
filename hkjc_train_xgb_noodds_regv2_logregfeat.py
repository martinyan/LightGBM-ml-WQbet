import argparse, json, sqlite3, datetime
from collections import defaultdict

import numpy as np
import xgboost as xgb


def load_jsonl(path):
    out=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--db',default='hkjc.sqlite')
    ap.add_argument('--data',default='hkjc_dataset_v3_code_prev1.jsonl')
    ap.add_argument('--featKeysFrom',default='hkjc_model_v2_logreg.json',help='use featKeys from this model json to ensure consistent inference')
    ap.add_argument('--trainStart',default='2023/09/01')
    ap.add_argument('--trainEnd',default='2025/07/31')
    ap.add_argument('--evalFracMeetings',type=float,default=0.2)
    ap.add_argument('--outModel',default='models/HKJC-ML_XGB_NOODDS_REG_v2.bin')
    ap.add_argument('--outMeta',default='models/HKJC-ML_XGB_NOODDS_REG_v2.infermeta.json')
    args=ap.parse_args()

    feat_src=json.load(open(args.featKeysFrom,'r',encoding='utf-8'))
    feat_keys=feat_src.get('featKeys')
    if not feat_keys:
        raise SystemExit('featKeys missing in featKeysFrom')

    con=sqlite3.connect(args.db)
    cur=con.cursor()
    cur.execute('select racedate, venue, meeting_id from meetings where racedate>=? and racedate<=? order by racedate asc, venue asc', (args.trainStart,args.trainEnd))
    meetings=[(r,v,mid) for r,v,mid in cur.fetchall()]
    if not meetings:
        raise SystemExit('no meetings in train range')

    n_train=len(meetings)
    n_eval=max(1,int(round(n_train*args.evalFracMeetings)))
    fit_meetings=meetings[:-n_eval]
    eval_meetings=meetings[-n_eval:]
    fit_set={(r,v) for r,v,_ in fit_meetings}
    eval_set={(r,v) for r,v,_ in eval_meetings}

    rows=load_jsonl(args.data)
    by_meeting=defaultdict(list)
    for r in rows:
        by_meeting[(r.get('racedate'),r.get('venue'))].append(r)

    def vec(r):
        x=[]
        for k in feat_keys:
            v=r.get(k,0)
            if k=='cur_win_odds':
                x.append(0.0)
                continue
            if v is None or v=='':
                x.append(0.0)
                continue
            try:
                x.append(float(v))
            except Exception:
                x.append(0.0)
        return x

    fit_rows=[r for mk in fit_set for r in by_meeting.get(mk,[])]
    eval_rows=[r for mk in eval_set for r in by_meeting.get(mk,[])]
    if not fit_rows or not eval_rows:
        raise SystemExit('fit/eval rows empty')

    X_fit=np.asarray([vec(r) for r in fit_rows],dtype=np.float32)
    y_fit=np.asarray([int(r.get('y_place',0) or 0) for r in fit_rows],dtype=np.int32)

    X_eval=np.asarray([vec(r) for r in eval_rows],dtype=np.float32)
    y_eval=np.asarray([int(r.get('y_place',0) or 0) for r in eval_rows],dtype=np.int32)

    dfit=xgb.DMatrix(X_fit,label=y_fit,feature_names=feat_keys)
    deval=xgb.DMatrix(X_eval,label=y_eval,feature_names=feat_keys)

    params={
        'objective':'binary:logistic',
        'eval_metric':'logloss',
        'max_depth':4,
        'min_child_weight':10,
        'gamma':1.0,
        'lambda':10.0,
        'alpha':0.5,
        'subsample':0.85,
        'colsample_bytree':0.85,
        'eta':0.05,
        'seed':42,
        'nthread':4,
    }

    model=xgb.train(params, dfit, num_boost_round=2000, evals=[(dfit,'fit'),(deval,'eval')], early_stopping_rounds=80, verbose_eval=False)
    model.save_model(args.outModel)

    meta={
        'project_code':'HKJC-ML_XGB_NOODDS_REG_v2',
        'trainedAt': datetime.datetime.now().isoformat(timespec='seconds'),
        'featKeysFrom': args.featKeysFrom,
        'featKeys': feat_keys,
        'trainRange': {'start': args.trainStart, 'end': args.trainEnd, 'meetings': len(meetings), 'fitMeetings': len(fit_meetings), 'evalMeetings': len(eval_meetings)},
        'params': params,
        'best_iteration': getattr(model,'best_iteration',None),
        'best_score': getattr(model,'best_score',None),
    }
    with open(args.outMeta,'w',encoding='utf-8') as f:
        json.dump(meta,f,indent=2)

    print(json.dumps({'outModel':args.outModel,'outMeta':args.outMeta,'best_iteration':meta['best_iteration'],'best_score':meta['best_score']},indent=2))


if __name__=='__main__':
    main()
