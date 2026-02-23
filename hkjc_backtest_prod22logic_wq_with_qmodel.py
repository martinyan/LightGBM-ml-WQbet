import argparse, json, csv, hashlib
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


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def pair(a: int, b: int) -> str:
    x = min(int(a), int(b))
    y = max(int(a), int(b))
    return f"{x}-{y}"


def load_jsonl(path):
    out=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out


def group_by_race(rows):
    by=defaultdict(list)
    for r in rows:
        by[(r['racedate'],r['venue'],int(r['race_no']))].append(r)
    return by


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--selectedRacesJson', default='reports/HKJC-ML_OVERLAY_WIN_ONLY_THR0p2_FEB22/overlay_win10_place30_top1_threshold.json')
    ap.add_argument('--dataset_v7', default='hkjc_dataset_v7_jt60_prev3_fullcols_fieldranks.jsonl')
    ap.add_argument('--qBundle', default='models/Q_RANKER_v7_FEB22_JT60_ODDS/bundle.json')
    ap.add_argument('--qModelTxt', default='models/Q_RANKER_v7_FEB22_JT60_ODDS/ranker.txt')
    ap.add_argument('--qCache', default='hkjc_dividend_cache_Qmap_byurl.json')
    ap.add_argument('--stakeEach', type=float, default=10.0)
    ap.add_argument('--outDir', default='reports/PROD22LOGIC_QMODEL_BACKTEST')
    args=ap.parse_args()

    import os
    os.makedirs(args.outDir, exist_ok=True)

    sel=json.load(open(args.selectedRacesJson,'r',encoding='utf-8'))
    # selected races are those in test_ledger
    sel_ledger=sel.get('test_ledger') or []
    sel_keys={(r['racedate'],r['venue'],int(r['race_no'])): r for r in sel_ledger}

    rows7=load_jsonl(args.dataset_v7)
    by_race7=group_by_race([r for r in rows7 if (r['racedate'],r['venue'],int(r['race_no'])) in sel_keys])

    qb=json.load(open(args.qBundle,'r',encoding='utf-8'))
    feat=qb['feat_keys']
    ranker=lgb.Booster(model_file=args.qModelTxt)

    q_cache=json.load(open(args.qCache,'r',encoding='utf-8'))

    out_rows=[]
    stake=0.0; ret=0.0; hits_any=0

    for key in sorted(sel_keys.keys()):
        rd,venue,rn=key
        w=sel_keys[key]
        url=w['url']
        h=sha1(url)
        anchor=int(w['top1_horse_no'])
        runners=by_race7.get(key) or []
        if not runners:
            continue
        X=np.asarray([[safe_float(r.get(k),0.0) for k in feat] for r in runners], dtype=np.float32)
        sc=ranker.predict(X)
        scored=sorted([(int(r['horse_no']), float(s)) for r,s in zip(runners,sc)], key=lambda t:t[1], reverse=True)
        partners=[hn for hn,_ in scored if hn!=anchor]
        if len(partners)<2:
            continue
        p2,p3=partners[0],partners[1]
        p12=pair(anchor,p2); p13=pair(anchor,p3)
        d12=q_cache.get(h,{}).get(p12); d13=q_cache.get(h,{}).get(p13)
        d12=float(d12) if d12 is not None else None
        d13=float(d13) if d13 is not None else None

        stake_r=2.0*args.stakeEach
        ret_r=float(d12 or 0.0)+float(d13 or 0.0)
        hit= (d12 is not None and d12>0) or (d13 is not None and d13>0)

        stake+=stake_r; ret+=ret_r
        if hit: hits_any+=1

        out_rows.append({
            'racedate': rd,'venue':venue,'race_no':int(rn),'url':url,
            'anchor':anchor,'p2':p2,'p3':p3,
            'pair12':p12,'q_div_12':d12,
            'pair13':p13,'q_div_13':d13,
            'stake_total':stake_r,'return_total':ret_r,'profit_total':ret_r-stake_r,
            'is_hit_any':hit,
        })

    rep={
        'selected_races': len(out_rows),
        'bets': 2*len(out_rows),
        'hits_any': hits_any,
        'stake': stake,
        'return': ret,
        'profit': ret-stake,
        'roi': (ret-stake)/stake if stake else None,
        'rows': out_rows,
    }

    json.dump(rep, open(os.path.join(args.outDir,'q_report.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    with open(os.path.join(args.outDir,'q_rows.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        if out_rows:
            w.writeheader(); w.writerows(out_rows)

    print(json.dumps({'ok':True,'outDir':args.outDir,'roi':rep['roi'],'hits_any':hits_any,'races':len(out_rows)}, indent=2))


if __name__=='__main__':
    main()
