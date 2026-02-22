import os, json, math, argparse, sqlite3, hashlib, datetime, re
from collections import defaultdict
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import lightgbm as lgb

UA='openclaw-hkjc-overlay/1.0'


def load_jsonl(path):
    out=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out


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


def safe_float(x, d=0.0):
    try:
        if x is None or x=='':
            return d
        return float(x)
    except Exception:
        return d


def logit(p):
    p=min(max(float(p), 1e-6), 1-1e-6)
    return math.log(p/(1-p))


def sigmoid(z):
    return 1.0/(1.0+math.exp(-z))


def place_terms(field_size: int) -> int:
    # Approx HK place terms: <=4 no place pool usually; 5-7:2; >=8:3
    if field_size is None:
        return 3
    n=int(field_size)
    if n <= 4:
        return 0
    if n <= 7:
        return 2
    return 3


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


def parse_div_table(html):
    tables=re.findall(r'<table[\s\S]*?</table>', html, flags=re.I)
    div_table=None
    for t in tables:
        if '派彩' in t and '彩池' in t and '勝出組合' in t:
            div_table=t; break
    if not div_table:
        return []
    trs=re.findall(r'<tr[^>]*>([\s\S]*?)</tr>', div_table, flags=re.I)
    rows=[]
    for tr in trs:
        cells=[clean_cell(x) for x in re.findall(r'<t[dh][^>]*>([\s\S]*?)</t[dh]>', tr, flags=re.I)]
        if len(cells)>=2:
            rows.append(cells)
    return rows


def parse_win_place_maps(html):
    rows=parse_div_table(html)
    win={}; place={}
    for r in rows:
        if r and r[0]=='彩池':
            continue
        if len(r)>=3 and r[0] in ('獨贏','位置'):
            pool=r[0]; combo=r[1]; div=first_number(r[2])
            if div is None:
                continue
            nums=re.findall(r'(\d+)', combo)
            if not nums:
                continue
            hn=int(nums[0])
            if pool=='獨贏':
                win[hn]=div
            else:
                place[hn]=div
    return win, place


def ck(url):
    return hashlib.sha1(url.encode('utf-8')).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--data', default='hkjc_dataset_v5_code_include_debut_jt60.jsonl')
    ap.add_argument('--trainStart', default='2023/09/01')
    ap.add_argument('--trainEnd', default='2025/07/31')
    ap.add_argument('--testStart', default='2025/09/01')
    ap.add_argument('--testEnd', default='2026/02/19')
    ap.add_argument('--evalFracMeetings', type=float, default=0.2)
    ap.add_argument('--stakeWin', type=float, default=10.0)
    ap.add_argument('--stakePlace', type=float, default=30.0)
    ap.add_argument('--outDir', default='reports/HKJC-ML_OVERLAY_WINPLACE')
    ap.add_argument('--thresholdGrid', default='-0.05,-0.02,0,0.01,0.02,0.03,0.04,0.05')
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=0.0, help='optional place residual weight in overlay signal (single value)')
    ap.add_argument('--betaGrid', default=None, help='comma list; if set, sweep beta values and pick best by eval ROI')
    args=ap.parse_args()

    os.makedirs(args.outDir, exist_ok=True)

    all_rows=load_jsonl(args.data)
    by_meeting=group_rows_by_meeting(all_rows)

    train_meetings=list_meetings(args.db, args.trainStart, args.trainEnd)
    test_meetings=list_meetings(args.db, args.testStart, args.testEnd)
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
    test_rows=build_meeting_rows(test_meetings)

    # Build per-race market implied probs using final odds
    def add_market_probs(rows_in):
        races=group_rows_by_race(rows_in)
        out=[]
        for key, runners in races.items():
            # compute p_raw
            p_raw=[]
            for rr in runners:
                o=safe_float(rr.get('cur_win_odds'), None)
                if o is None or o<=0:
                    p_raw.append(0.0)
                else:
                    p_raw.append(1.0/o)
            s=sum(p_raw)
            if s<=0:
                # fallback uniform
                p_mkt=[1.0/len(runners)]*len(runners)
            else:
                p_mkt=[x/s for x in p_raw]
            n=len(runners)
            K=place_terms(n)
            for rr, pm in zip(runners, p_mkt):
                rr2=dict(rr)
                rr2['_field_size']=n
                rr2['_place_terms']=K
                rr2['p_mkt_win']=pm
                out.append(rr2)
        return out

    fit_rows2=add_market_probs(fit_rows)
    eval_rows2=add_market_probs(eval_rows)
    test_rows2=add_market_probs(test_rows)

    # Baseline place model from market win prob + field size (fit on fit_rows)
    def build_place_baseline(rows_in):
        X=[]; y=[]
        for r in rows_in:
            K=r.get('_place_terms')
            if not K or K<=0:
                continue
            fp=r.get('y_finish_pos')
            if fp is None:
                continue
            y_place=1 if int(fp) <= int(K) else 0
            pm=float(r['p_mkt_win'])
            X.append([logit(pm), math.log(float(r.get('_field_size') or 14))])
            y.append(y_place)
        X=np.asarray(X,dtype=float)
        y=np.asarray(y,dtype=int)
        mdl=LogisticRegression(max_iter=200)
        mdl.fit(X,y)
        return mdl

    place_mdl=build_place_baseline(fit_rows2)

    def add_p_mkt_place(rows_in):
        out=[]
        for r in rows_in:
            pm=float(r['p_mkt_win'])
            n=float(r.get('_field_size') or 14)
            p_place=float(place_mdl.predict_proba([[logit(pm), math.log(n)]])[0,1])
            r2=dict(r)
            r2['p_mkt_place']=p_place
            out.append(r2)
        return out

    fit_rows3=add_p_mkt_place(fit_rows2)
    eval_rows3=add_p_mkt_place(eval_rows2)
    test_rows3=add_p_mkt_place(test_rows2)

    # Residual targets
    def add_targets(rows_in):
        out=[]
        for r in rows_in:
            fp=r.get('y_finish_pos')
            if fp is None:
                continue
            fp=int(fp)
            y_win=1 if fp==1 else 0
            K=int(r.get('_place_terms') or 3)
            y_place=1 if (K>0 and fp<=K) else 0
            pmw=float(r['p_mkt_win']); pmp=float(r['p_mkt_place'])
            r2=dict(r)
            r2['y_win']=y_win
            r2['y_placeK']=y_place
            r2['res_win']=float(y_win - pmw)
            r2['res_place']=float(y_place - pmp)
            out.append(r2)
        return out

    fit_rows4=add_targets(fit_rows3)
    eval_rows4=add_targets(eval_rows3)
    test_rows4=add_targets(test_rows3)

    # Feature keys: drop odds and drop jockey/trainer identity and rolling stats (avoid leaking market)
    meta={'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no','y_finish_pos','y_win','y_place','cur_jockey','cur_trainer'}
    exclude_prefix=('jockey_','trainer_')
    feat_keys=sorted([k for k in fit_rows4[0].keys() if k not in meta and not k.startswith('y_') and not k.startswith('_')])
    feat_keys=[k for k in feat_keys if not k.startswith(exclude_prefix) and k not in ('cur_win_odds','p_mkt_win','p_mkt_place','res_win','res_place')]

    def vec(rows_in):
        X=[]; ywin=[]; ypl=[]; pmw=[]; pmp=[]; meta_out=[]
        races=group_rows_by_race(rows_in)
        group_sizes=[]
        for (rd,venue,rn), runners in sorted(races.items()):
            group_sizes.append(len(runners))
            for r in runners:
                X.append([safe_float(r.get(k),0.0) for k in feat_keys])
                ywin.append(float(r['res_win']))
                ypl.append(float(r['res_place']))
                pmw.append(float(r['p_mkt_win']))
                pmp.append(float(r['p_mkt_place']))
                meta_out.append({'racedate':rd,'venue':venue,'race_no':int(rn),'horse_no':int(r.get('horse_no'))})
        return np.asarray(X,dtype=np.float32), np.asarray(ywin), np.asarray(ypl), np.asarray(pmw), np.asarray(pmp), meta_out, group_sizes

    X_fit, y_fit_win, y_fit_pl, pmw_fit, pmp_fit, meta_fit, g_fit = vec(fit_rows4)
    X_eval, y_eval_win, y_eval_pl, pmw_eval, pmp_eval, meta_eval, g_eval = vec(eval_rows4)
    X_test, y_test_win, y_test_pl, pmw_test, pmp_test, meta_test, g_test = vec(test_rows4)

    # train residual regressors
    def train_lgb_reg(Xtr, ytr):
        mdl=lgb.LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
        )
        mdl.fit(Xtr,ytr)
        return mdl

    mdl_win=train_lgb_reg(X_fit, y_fit_win)
    mdl_pl=train_lgb_reg(X_fit, y_fit_pl)

    pred_eval_win=mdl_win.predict(X_eval)
    pred_eval_pl=mdl_pl.predict(X_eval)

    # We'll sweep beta and threshold; overlay signal = alpha*res_win + beta*res_place

    # select top1 per race by score_win (market+res)
    def choose_top1(meta_rows, pmw, pred_res, overlay, threshold):
        # build per race
        by=defaultdict(list)
        for m, p, r, ov in zip(meta_rows, pmw, pred_res, overlay):
            by[(m['racedate'],m['venue'],m['race_no'])].append((m['horse_no'], float(p), float(r), float(ov)))
        picks=[]
        for key, arr in sorted(by.items()):
            # score = logit(p_mkt_win) + alpha*res_hat_win
            arr2=[(hn, logit(pm)+args.alpha*rh, ov) for hn,pm,rh,ov in arr]
            arr2.sort(key=lambda t:t[1], reverse=True)
            hn,score,ov=arr2[0]
            if ov > threshold:
                picks.append((key, hn, ov, score))
        return picks

    thresholds=[float(x) for x in args.thresholdGrid.split(',') if x.strip()]
    betas = [args.beta]
    if args.betaGrid:
        betas = [float(x) for x in args.betaGrid.split(',') if x.strip()]

    # quick win/place backtest using dividends
    cache_path='hkjc_dividend_cache_WINPLACE_byurl.json'
    try:
        cache=json.load(open(cache_path,'r',encoding='utf-8'))
    except Exception:
        cache={}

    def get_div(rd, venue, rn):
        url=f'https://racing.hkjc.com/zh-hk/local/information/localresults?racedate={rd}&Racecourse={venue}&RaceNo={rn}'
        k=hashlib.sha1(url.encode('utf-8')).hexdigest()
        data=cache.get(k)
        if data is None:
            # fetch + cache (with retries)
            last=None
            for _ in range(3):
                try:
                    html=fetch_html(url)
                    win, place = parse_win_place_maps(html)
                    cache[k]={'win':win,'place':place}
                    data=cache[k]
                    last=None
                    break
                except Exception as e:
                    last=e
            if data is None:
                return url, {}, {}, True
        return url, data.get('win',{}), data.get('place',{}), False

    def bt(picks):
        stakeW=args.stakeWin; stakeP=args.stakePlace
        stake=0.0; ret=0.0; n=0; missing=0
        for (rd,venue,rn), hn, ov, score in picks:
            url, win, place, miss = get_div(rd,venue,rn)
            if miss:
                missing += 1
            def get_hdiv(mapobj, hn):
                if not isinstance(mapobj, dict):
                    return None
                if hn in mapobj:
                    return mapobj.get(hn)
                return mapobj.get(str(hn))
            wdiv = get_hdiv(win, hn)
            pdiv = get_hdiv(place, hn)
            stake += stakeW + stakeP
            ret += (float(wdiv) if wdiv is not None else 0.0)
            ret += (float(pdiv) if pdiv is not None else 0.0) * (stakeP/10.0)
            n += 1
        profit=ret-stake
        roi=profit/stake if stake else None
        return {'bets':n,'missing':missing,'stake':stake,'return':ret,'profit':profit,'roi':roi}

    sweep=[]
    best=None
    for beta in betas:
        overlay_eval=args.alpha*pred_eval_win + beta*pred_eval_pl
        for th in thresholds:
            picks=choose_top1(meta_eval, pmw_eval, pred_eval_win, overlay_eval, th)
            s=bt(picks)
            row={**s,'threshold':th,'beta':beta}
            sweep.append(row)
            if best is None or (row['roi'] is not None and row['roi']>best['roi']):
                best=row

    # test using best (beta, threshold)
    pred_test_win=mdl_win.predict(X_test)
    pred_test_pl=mdl_pl.predict(X_test)
    overlay_test=args.alpha*pred_test_win + best['beta']*pred_test_pl

    picks_test=choose_top1(meta_test, pmw_test, pred_test_win, overlay_test, best['threshold'])
    summary_test=bt(picks_test)

    # persist cache
    with open(cache_path,'w',encoding='utf-8') as f:
        json.dump(cache,f,ensure_ascii=False)

    # Build a detailed test ledger for downstream strategy variants (win-only, place-only, etc.)
    test_ledger=[]
    for (rd,venue,rn), hn, ov, score in picks_test:
        url, win, place, miss = get_div(rd,venue,rn)
        def get_hdiv(mapobj, hn):
            if not isinstance(mapobj, dict):
                return None
            if hn in mapobj:
                return mapobj.get(hn)
            return mapobj.get(str(hn))
        wdiv=get_hdiv(win, hn)
        pdiv=get_hdiv(place, hn)
        test_ledger.append({
            'racedate': rd, 'venue': venue, 'race_no': int(rn),
            'horse_no': int(hn),
            'overlay': float(ov),
            'score_win': float(score),
            'win_div': wdiv,
            'place_div': pdiv,
            'url': url,
        })

    out={
        'generatedAt': datetime.datetime.now().isoformat(timespec='seconds'),
        'model':'OVERLAY_RESIDUAL_LGBM_v1',
        'ranges':{
            'train':{'start':args.trainStart,'end':args.trainEnd},
            'test':{'start':args.testStart,'end':args.testEnd},
        },
        'features':{
            'feat_keys':feat_keys,
            'note':'odds used only in market baseline; residual models exclude odds/jockey/trainer',
        },
        'baseline':{'p_mkt_win':'normalized 1/odds', 'p_mkt_place':'logistic baseline from logit(p_mkt_win)+log(field_size)'},
        'overlay':{'alpha':args.alpha,'signal':'alpha*res_hat_win + beta*res_hat_place'},
        'tuning':{'thresholdGrid':thresholds,'betaGrid':betas,'best':best,'sweep':sweep},
        'test':summary_test,
        'test_ledger': sorted(test_ledger, key=lambda r:(r['racedate'], r['venue'], r['race_no']))
    }

    out_path=os.path.join(args.outDir,'overlay_win10_place30_top1_threshold.json')
    with open(out_path,'w',encoding='utf-8') as f:
        json.dump(out,f,ensure_ascii=False,indent=2)

    print(json.dumps({'out':out_path,'best_threshold':best['threshold'], **summary_test}, indent=2))


if __name__=='__main__':
    main()
