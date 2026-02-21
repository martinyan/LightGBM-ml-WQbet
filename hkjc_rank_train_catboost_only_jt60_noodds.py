import os, json, argparse, sqlite3, hashlib, datetime, re
from collections import defaultdict
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from catboost import CatBoostRanker, Pool

UA='openclaw-hkjc-catboost-jt60-noodds/1.0'


def load_jsonl(path):
    out=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out


def relevance_from_finish_pos(pos):
    if pos is None:
        return None
    try:
        p=int(pos)
    except Exception:
        return None
    if p==1: return 3
    if p==2: return 2
    if p==3: return 1
    return 0


def list_meetings(db_path, start, end):
    con=sqlite3.connect(db_path)
    cur=con.cursor()
    cur.execute('select racedate, venue, meeting_id from meetings where racedate>=? and racedate<=? order by racedate asc, venue asc', (start,end))
    rows=[(r,v,mid) for (r,v,mid) in cur.fetchall()]
    con.close()
    return rows


def group_rows_by_meeting(rows):
    by=defaultdict(list)
    for r in rows:
        by[(r.get('racedate'), r.get('venue'))].append(r)
    return by


def group_rows_by_race(rows):
    by=defaultdict(list)
    for r in rows:
        by[(r.get('racedate'), r.get('venue'), int(r.get('race_no')))].append(r)
    return by


def fetch_html(url: str) -> str:
    req=Request(url, headers={'User-Agent':UA})
    with urlopen(req, timeout=15) as resp:
        return resp.read().decode('utf-8', errors='ignore')


def clean_cell(html: str) -> str:
    s=re.sub(r'<br\s*/?>',' ',html,flags=re.I)
    s=re.sub(r'<[^>]+>',' ',s)
    s=s.replace('&nbsp;',' ').replace('&amp;','&').replace('&quot;','"').replace('&#39;',"'")
    s=re.sub(r'\s+',' ',s).strip()
    return s


def first_number(s: str):
    m=re.findall(r'\d[\d,]*\.?\d*', s)
    if not m:
        return None
    tok=m[0].replace(',','')
    try:
        return float(tok)
    except Exception:
        return None


def parse_Q_map(html: str):
    tables=re.findall(r'<table[\s\S]*?</table>', html, flags=re.I)
    div_table=None
    for t in tables:
        if '派彩' in t and '彩池' in t and '勝出組合' in t:
            div_table=t; break
    if not div_table:
        return {}
    trs=re.findall(r'<tr[^>]*>([\s\S]*?)</tr>', div_table, flags=re.I)
    out={}
    for tr in trs:
        cells=[clean_cell(x) for x in re.findall(r'<t[dh][^>]*>([\s\S]*?)</t[dh]>', tr, flags=re.I)]
        if len(cells)>=3 and cells[0]=='連贏':
            combo=cells[1]; div=first_number(cells[2])
            if div is None:
                continue
            nums=re.findall(r'(\d+)', combo)
            if len(nums)>=2:
                a,b=int(nums[0]),int(nums[1])
                out[f"{min(a,b)}-{max(a,b)}"]=div
    return out


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--data', default='hkjc_dataset_v5_code_include_debut_jt60.jsonl')
    ap.add_argument('--trainStart', default='2023/09/01')
    ap.add_argument('--trainEnd', default='2025/07/31')
    ap.add_argument('--testStart', default='2025/09/01')
    ap.add_argument('--testEnd', default='2026/02/19')
    ap.add_argument('--evalFracMeetings', type=float, default=0.2)
    ap.add_argument('--stake', type=float, default=10.0)
    ap.add_argument('--outDirModel', default='models/HKJC-ML_CB_ONLY_JT60_NOODDS')
    ap.add_argument('--outDirReport', default='reports/HKJC-ML_CB_ONLY_JT60_NOODDS')
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()

    os.makedirs(args.outDirModel, exist_ok=True)
    os.makedirs(args.outDirReport, exist_ok=True)

    rows=load_jsonl(args.data)
    by_meeting=group_rows_by_meeting(rows)

    train_meetings=list_meetings(args.db, args.trainStart, args.trainEnd)
    test_meetings=list_meetings(args.db, args.testStart, args.testEnd)
    if not train_meetings or not test_meetings:
        raise SystemExit('No meetings in range')

    n_train=len(train_meetings)
    n_eval=max(1,int(round(n_train*args.evalFracMeetings)))
    fit_meetings=train_meetings[:-n_eval]
    eval_meetings=train_meetings[-n_eval:]

    def build_meeting_rows(meeting_list):
        s={(r,v) for (r,v,_) in meeting_list}
        out=[]
        for k in s:
            out.extend(by_meeting.get(k,[]))
        return out

    fit_rows=build_meeting_rows(fit_meetings)
    eval_rows=build_meeting_rows(eval_meetings)
    train_rows=build_meeting_rows(train_meetings)
    test_rows=build_meeting_rows(test_meetings)

    # Feature selection: KEEP cur_jockey/cur_trainer and their 60d rolling stats, but REMOVE odds.
    meta={'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no','y_finish_pos','y_win','y_place'}
    all_keys=set(rows[0].keys())
    feat_keys=sorted([k for k in all_keys if k not in meta and not k.startswith('y_')])

    # remove odds
    feat_keys=[k for k in feat_keys if k!='cur_win_odds']

    # cat features
    cat_cols=[c for c in ['cur_jockey','cur_trainer','cur_surface','venue'] if c in feat_keys]

    def build_cat_pool(rows_in):
        races=group_rows_by_race(rows_in)
        X=[]; y=[]; gid=[]; meta_out=[]
        g=0
        for (rd, venue, rn), runners in sorted(races.items()):
            rels=[relevance_from_finish_pos(rr.get('y_finish_pos')) for rr in runners]
            if any(x is None for x in rels) or len(runners)<2:
                continue
            for rr,rel in zip(runners,rels):
                row=[]
                for k in feat_keys:
                    v=rr.get(k)
                    if k in cat_cols:
                        row.append('' if v is None else str(v))
                    else:
                        try:
                            row.append(float(v or 0.0))
                        except Exception:
                            row.append(0.0)
                X.append(row)
                y.append(float(rel))
                gid.append(g)
                meta_out.append({'racedate':rd,'venue':venue,'race_no':int(rn),'horse_no':int(rr.get('horse_no')),'finish_pos':rr.get('y_finish_pos')})
            g+=1
        cat_idx=[feat_keys.index(c) for c in cat_cols]
        return Pool(X, label=y, group_id=gid, cat_features=cat_idx), meta_out

    pool_fit,_=build_cat_pool(fit_rows)
    pool_eval,_=build_cat_pool(eval_rows)

    cb=CatBoostRanker(
        loss_function='YetiRank',
        eval_metric='NDCG:top=5',
        iterations=5000,
        learning_rate=0.05,
        depth=8,
        random_seed=args.seed,
        l2_leaf_reg=3.0,
        od_type='Iter',
        od_wait=200,
        verbose=False
    )
    cb.fit(pool_fit, eval_set=pool_eval, use_best_model=True)

    pool_train,_=build_cat_pool(train_rows)
    cb_final=CatBoostRanker(
        loss_function='YetiRank',
        iterations=cb.get_best_iteration()+1 if cb.get_best_iteration() else cb.tree_count_,
        learning_rate=0.05,
        depth=8,
        random_seed=args.seed,
        l2_leaf_reg=3.0,
        verbose=False
    )
    cb_final.fit(pool_train)

    model_path=os.path.join(args.outDirModel,'catboost_ranker_only.cbm')
    cb_final.save_model(model_path)

    pool_test, meta_test = build_cat_pool(test_rows)
    scores=np.asarray(cb_final.predict(pool_test), dtype=np.float64)

    by_race=defaultdict(list)
    for m,s in zip(meta_test, scores):
        by_race[(m['racedate'],m['venue'],m['race_no'])].append((m['horse_no'], float(s), m.get('finish_pos')))

    race_outputs=[]
    for (rd, venue, rn), arr in sorted(by_race.items()):
        arr_sorted=sorted(arr, key=lambda t:t[1], reverse=True)
        top5=arr_sorted[:5]
        xs=np.asarray([x[1] for x in arr_sorted], dtype=np.float64)
        ex=np.exp(xs-np.max(xs))
        sm=ex/np.sum(ex) if np.sum(ex) else np.ones_like(ex)/len(ex)
        p12_like=float(sm[0]+(sm[1] if len(sm)>1 else 0.0))
        race_outputs.append({
            'racedate': rd,
            'venue': venue,
            'race_no': rn,
            'top5': [{'horse_no':int(h),'score':float(sc),'finish_pos':fp} for (h,sc,fp) in top5],
            'p12_like': p12_like,
            'n_runners': len(arr_sorted)
        })

    # Q backtest: top2 pair
    def q_job(ro):
        t=ro['top5']
        if len(t)<2:
            return None
        a=t[0]['horse_no']; b=t[1]['horse_no']
        pair=f"{min(a,b)}-{max(a,b)}"
        url=f"https://racing.hkjc.com/zh-hk/local/information/localresults?racedate={ro['racedate']}&Racecourse={ro['venue']}&RaceNo={ro['race_no']}"
        return (url, pair, ro)

    jobs=[q_job(ro) for ro in race_outputs]
    jobs=[j for j in jobs if j]

    # Prefer using the already-built per-URL Q map cache (covers the whole test set)
    # to avoid network timeouts. Fallback to fetching if missing.
    cache_path=os.path.join(args.outDirReport,'div_cache_Q_byurl.json')
    seed_cache_path='hkjc_dividend_cache_Qmap_byurl.json'
    cache={}
    try:
        cache=json.load(open(seed_cache_path,'r',encoding='utf-8'))
    except Exception:
        cache={}
    try:
        # merge any local cache entries (local overrides)
        local=json.load(open(cache_path,'r',encoding='utf-8'))
        cache.update(local)
    except Exception:
        pass

    def work(job):
        url,pair,ro=job
        k=sha1(url)
        qmap=cache.get(k)
        if qmap is None:
            # fallback fetch with a couple retries
            last_err=None
            for _ in range(3):
                try:
                    html=fetch_html(url)
                    qmap=parse_Q_map(html)
                    cache[k]=qmap
                    last_err=None
                    break
                except Exception as e:
                    last_err=e
            if last_err is not None:
                qmap={}
        div=qmap.get(pair) if isinstance(qmap, dict) else None
        return {
            'racedate': ro['racedate'], 'venue': ro['venue'], 'race_no': int(ro['race_no']),
            'pair': pair,
            'q_div': div,
            'stake': float(args.stake),
            'return': float(div or 0.0),
            'profit': float(div or 0.0) - float(args.stake),
            'url': url,
        }

    ledger=[]
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs=[ex.submit(work,j) for j in jobs]
        for fut in as_completed(futs):
            ledger.append(fut.result())

    ledger.sort(key=lambda r:(r['racedate'],r['venue'],r['race_no']))

    with open(cache_path,'w',encoding='utf-8') as f:
        json.dump(cache,f,ensure_ascii=False)

    stake_total=sum(r['stake'] for r in ledger)
    ret_total=sum(r['return'] for r in ledger)
    profit=ret_total-stake_total
    roi=profit/stake_total if stake_total else None
    hits=sum(1 for r in ledger if r['q_div'] is not None)

    report={
        'generatedAt': datetime.datetime.now().isoformat(timespec='seconds'),
        'model':'HKJC-ML_CB_ONLY_JT60_NOODDS',
        'data': args.data,
        'ranges': {
            'train': {'start': args.trainStart,'end': args.trainEnd},
            'test': {'start': args.testStart,'end': args.testEnd},
        },
        'features': {'feat_keys': feat_keys, 'cat_cols': cat_cols},
        'strategy': {'pool':'Q','stake_per_race': args.stake,'selection':'top2 pair', 'note':'CatBoostRanker only; JT60 features; no odds feature'},
        'summary': {'races': len(ledger), 'hits': hits, 'stake': stake_total, 'return': ret_total, 'profit': profit, 'roi': roi},
        'ledger': ledger,
    }

    out_picks=os.path.join(args.outDirReport,'test_race_picks.json')
    with open(out_picks,'w',encoding='utf-8') as f:
        json.dump(race_outputs,f,ensure_ascii=False,indent=2)

    out_report=os.path.join(args.outDirReport,'q_flat10_allraces_full_test.json')
    with open(out_report,'w',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)

    print(json.dumps({'model_path': model_path, 'report': out_report, 'picks': out_picks, **report['summary']}, indent=2))


if __name__=='__main__':
    main()
