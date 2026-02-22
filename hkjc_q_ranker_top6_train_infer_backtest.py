import os, json, math, argparse, hashlib, datetime
from collections import defaultdict

import numpy as np
import lightgbm as lgb


def load_jsonl(path):
    out=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out


def safe_float(x, d=None):
    try:
        if x is None or x=='':
            return d
        return float(x)
    except Exception:
        return d


def sha1(s:str)->str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def group_by_race(rows):
    by=defaultdict(list)
    for r in rows:
        by[(r['racedate'], r['venue'], int(r['race_no']))].append(r)
    return by


def pl_weights(scores, tau):
    t=float(tau)
    if t<=0:
        t=1.0
    m=float(np.max(scores)) if len(scores) else 0.0
    return np.asarray([math.exp((float(s)-m)/t) for s in scores], dtype=np.float64)


def p_win_from_pl(scores, tau):
    w=pl_weights(scores, tau)
    s=float(np.sum(w))
    if s<=0:
        return np.ones(len(w), dtype=np.float64)/max(1,len(w))
    return w/s


def winner_logloss(meta_block, scores_block, tau):
    p=p_win_from_pl(scores_block, tau)
    y=[1 if (m.get('y_finish_pos') is not None and int(m['y_finish_pos'])==1) else 0 for m in meta_block]
    if sum(y)!=1:
        return None
    idx=y.index(1)
    pi=float(max(min(p[idx],1-1e-9),1e-9))
    return -math.log(pi)


def tune_tau(meta_eval_blocks, score_eval_blocks, grid):
    losses={t:[] for t in grid}
    for meta_block, scores in zip(meta_eval_blocks, score_eval_blocks):
        for t in grid:
            ll=winner_logloss(meta_block, scores, t)
            if ll is not None:
                losses[t].append(ll)
    def mean(x):
        return sum(x)/len(x) if x else 1e9
    best=min(grid, key=lambda t: mean(losses[t]))
    return float(best), {float(t): (sum(losses[t])/len(losses[t]) if losses[t] else None) for t in grid}


def top2_set_prob_from_weights(wi, wj, wsum):
    # Plackett-Luce top2 set probability for pair {i,j}
    if wi<=0 or wj<=0 or wsum<=0:
        return 0.0
    if wsum - wi <= 0 or wsum - wj <= 0:
        return 0.0
    return (wi/wsum) * (wj/(wsum-wi)) + (wj/wsum) * (wi/(wsum-wj))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', default='hkjc_dataset_v7_jt60_prev3_fullcols_fieldranks.jsonl')
    ap.add_argument('--trainStart', default='2023/09/01')
    ap.add_argument('--trainEnd', default='2025/07/31')
    ap.add_argument('--testStart', default='2025/09/01')
    ap.add_argument('--testEnd', default='2026/02/19')
    ap.add_argument('--evalFracGroups', type=float, default=0.2)
    ap.add_argument('--tauGrid', default='0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,4,5')
    ap.add_argument('--topM', type=int, default=6)
    ap.add_argument('--stakeQ', type=float, default=10.0)
    ap.add_argument('--qCache', default='hkjc_dividend_cache_Qmap_byurl.json')
    ap.add_argument('--outDir', default='reports/HKJC-ML_Q_RANKER_v7_NOODDS_top6')
    ap.add_argument('--modelOut', default='models/Q_RANKER_v7_NOODDS_top6')
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()

    os.makedirs(args.outDir, exist_ok=True)
    os.makedirs(args.modelOut, exist_ok=True)

    rows=load_jsonl(args.data)

    def in_range(r,a,b):
        return a <= r['racedate'] <= b

    train_rows=[r for r in rows if in_range(r,args.trainStart,args.trainEnd) and r.get('y_finish_pos') is not None]
    test_rows=[r for r in rows if in_range(r,args.testStart,args.testEnd) and r.get('y_finish_pos') is not None]

    meta_keys={'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no','y_finish_pos','y_win','y_place','cur_jockey','cur_trainer'}
    feat_keys=sorted([k for k in train_rows[0].keys()
                     if k not in meta_keys
                     and not k.startswith('y_')
                     and not k.startswith('_')
                     and not k.startswith('jockey_')
                     and not k.startswith('trainer_')
                     and k!='cur_win_odds'])

    def build_blocks(rows_in):
        races=group_by_race(rows_in)
        blocks=[]
        for (rd,venue,rn), runners in sorted(races.items()):
            if len(runners) < 2:
                continue
            X=np.asarray([[safe_float(r.get(k),0.0) or 0.0 for k in feat_keys] for r in runners], dtype=np.float32)
            y=[]
            for r in runners:
                fp=r.get('y_finish_pos')
                fp=int(fp) if fp is not None else 99
                # relevance for top2
                y.append(2 if fp==1 else (1 if fp==2 else 0))
            meta=[{
                'racedate': rd,
                'venue': venue,
                'race_no': int(rn),
                'horse_no': int(r.get('horse_no')),
                'horse': r.get('horse_name_zh'),
                'y_finish_pos': r.get('y_finish_pos'),
            } for r in runners]
            blocks.append({'key':(rd,venue,int(rn)),'X':X,'y':np.asarray(y,dtype=np.float32),'meta':meta})
        return blocks

    def flatten(blks):
        X=np.concatenate([b['X'] for b in blks], axis=0)
        y=np.concatenate([b['y'] for b in blks], axis=0)
        g=[b['X'].shape[0] for b in blks]
        meta=[]
        for b in blks:
            meta.append(b['meta'])
        return X,y,g,meta

    blocks=build_blocks(train_rows)
    n=len(blocks)
    n_eval=max(1,int(round(n*args.evalFracGroups)))
    fit_blocks=blocks[:-n_eval]
    eval_blocks=blocks[-n_eval:]

    X_fit,y_fit,g_fit,_ = flatten(fit_blocks)
    X_eval,y_eval,g_eval,meta_eval = flatten(eval_blocks)

    ranker=lgb.LGBMRanker(
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
    ranker.fit(X_fit,y_fit,group=g_fit, eval_set=[(X_eval,y_eval)], eval_group=[g_eval])

    tau_grid=[float(x) for x in args.tauGrid.split(',') if x.strip()]
    score_eval_blocks=[ranker.predict(b['X']) for b in eval_blocks]
    tau_best, tau_ll = tune_tau(meta_eval, score_eval_blocks, tau_grid)

    # save model bundle
    model_txt=os.path.join(args.modelOut,'ranker.txt')
    ranker.booster_.save_model(model_txt)
    bundle={
        'generatedAt': datetime.datetime.now().isoformat(timespec='seconds'),
        'model': 'LGBMRanker_lambdarank_top2rel',
        'data': args.data,
        'ranges': {'train':{'start':args.trainStart,'end':args.trainEnd}, 'test':{'start':args.testStart,'end':args.testEnd}},
        'feat_keys': feat_keys,
        'tau_best': tau_best,
        'tau_logloss_by_tau': tau_ll,
        'topM': args.topM,
        'no_odds': True,
        'model_file': model_txt,
    }
    json.dump(bundle, open(os.path.join(args.modelOut,'bundle.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=2)

    # backtest Q
    q_cache=json.load(open(args.qCache,'r',encoding='utf-8')) if os.path.exists(args.qCache) else {}

    test_blocks=build_blocks(test_rows)

    stake=0.0; ret=0.0; bets=0; hits=0
    ledger=[]

    for b in test_blocks:
        scores=np.asarray(ranker.predict(b['X']), dtype=np.float64)
        # choose topM by score
        order=np.argsort(-scores)
        m=min(args.topM, len(order))
        top_idx=list(order[:m])

        w=pl_weights(scores, tau_best)
        wsum=float(np.sum(w))

        best=None
        rank_field={int(idx): int(pos+1) for pos, idx in enumerate(order)}
        # evaluate all pairs within topM
        for ii in range(m):
            for jj in range(ii+1,m):
                i=top_idx[ii]; j=top_idx[jj]
                pset=top2_set_prob_from_weights(float(w[i]), float(w[j]), wsum)
                if (best is None) or (pset > best['p_top2_set']):
                    hi=b['meta'][i]; hj=b['meta'][j]
                    best={
                        'i': int(i), 'j': int(j),
                        'horse_no_i': int(hi['horse_no']), 'horse_i': hi.get('horse'),
                        'horse_no_j': int(hj['horse_no']), 'horse_j': hj.get('horse'),
                        'y_finish_pos_i': int(hi['y_finish_pos']) if hi.get('y_finish_pos') is not None else None,
                        'y_finish_pos_j': int(hj['y_finish_pos']) if hj.get('y_finish_pos') is not None else None,
                        'p_top2_set': float(pset),
                        # model score + rank metadata (for plotting/diagnostics)
                        'score_i': float(scores[i]),
                        'score_j': float(scores[j]),
                        'rank_i_field': int(rank_field.get(int(i), 999)),
                        'rank_j_field': int(rank_field.get(int(j), 999)),
                        'rank_i_topM': int(ii+1),
                        'rank_j_topM': int(jj+1),
                    }

        if best is None:
            continue

        rd,venue,rn=b['key']
        url=f'https://racing.hkjc.com/zh-hk/local/information/localresults?racedate={rd}&Racecourse={venue}&RaceNo={rn}'
        h=sha1(url)
        a=min(best['horse_no_i'], best['horse_no_j'])
        c=max(best['horse_no_i'], best['horse_no_j'])
        key=f'{a}-{c}'
        div = q_cache.get(h, {}).get(key)
        div_f = float(div) if div is not None else None

        stake += float(args.stakeQ)
        rret = float(div or 0.0)
        ret += rret
        bets += 1
        is_hit = bool(div and float(div) > 0)
        if is_hit:
            hits += 1

        def rel(fp):
            if fp is None:
                return 0
            fp=int(fp)
            return 2 if fp==1 else (1 if fp==2 else 0)

        ledger.append({
            'racedate': rd,
            'venue': venue,
            'race_no': int(rn),
            'topM': int(m),
            'pick_pair': key,
            'horse_no_i': best['horse_no_i'],
            'horse_i': best.get('horse_i'),
            'horse_no_j': best['horse_no_j'],
            'horse_j': best.get('horse_j'),
            'score_i': best.get('score_i'),
            'score_j': best.get('score_j'),
            'rank_i_field': best.get('rank_i_field'),
            'rank_j_field': best.get('rank_j_field'),
            'rank_i_topM': best.get('rank_i_topM'),
            'rank_j_topM': best.get('rank_j_topM'),
            'y_finish_pos_i': best.get('y_finish_pos_i'),
            'y_finish_pos_j': best.get('y_finish_pos_j'),
            'rel_i': rel(best.get('y_finish_pos_i')),
            'rel_j': rel(best.get('y_finish_pos_j')),
            'p_top2_set': best['p_top2_set'],
            'dividend': div_f,
            'stake': float(args.stakeQ),
            'return': rret,
            'profit': rret - float(args.stakeQ),
            'is_hit': is_hit,
            'url': url,
        })

    summary={
        'bets': bets,
        'hits': hits,
        'stake': stake,
        'return': ret,
        'profit': ret-stake,
        'roi': (ret-stake)/stake if stake else None,
    }

    out={
        'generatedAt': datetime.datetime.now().isoformat(timespec='seconds'),
        'bundle': bundle,
        'summary_Q': summary,
        'ledger': ledger,
    }

    report_path=os.path.join(args.outDir,'q_ranker_top6_test_report.json')
    json.dump(out, open(report_path,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps({'report': report_path, 'modelOut': args.modelOut, 'tau': tau_best, 'summary_Q': summary}, ensure_ascii=False, indent=2))


if __name__=='__main__':
    main()
