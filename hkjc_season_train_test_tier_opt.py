import re, json, argparse, sqlite3, hashlib, datetime
from collections import defaultdict
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

UA = 'openclaw-hkjc-season-train-test/1.0'


def fetch_html(url: str) -> str:
    req = Request(url, headers={'User-Agent': UA})
    with urlopen(req, timeout=12) as resp:
        return resp.read().decode('utf-8', errors='ignore')


def clean_cell(html: str) -> str:
    s = re.sub(r'<br\s*/?>', ' ', html, flags=re.I)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = s.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&quot;', '"').replace('&#39;', "'")
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def parse_dividends_Q_QP(html: str):
    tables = re.findall(r'<table[\s\S]*?</table>', html, flags=re.I)
    div_table = None
    for t in tables:
        if '派彩' in t and '彩池' in t and '勝出組合' in t:
            div_table = t
            break
    if not div_table:
        return {'Q': {}, 'QP': {}, 'hasQP': False}

    trs = re.findall(r'<tr[^>]*>([\s\S]*?)</tr>', div_table, flags=re.I)
    rows = []
    for tr in trs:
        cells = [clean_cell(x) for x in re.findall(r'<t[dh][^>]*>([\s\S]*?)</t[dh]>', tr, flags=re.I)]
        if len(cells) >= 2:
            rows.append(cells)

    outQ = {}
    outQP = {}
    mode = None
    seenQP = False

    for r in rows:
        if r[0] == '彩池':
            continue

        if len(r) >= 3:
            pool, combo, divtxt = r[0], r[1], r[2]
            mode = None
            # Some dividends contain commas and occasionally multiple numbers; take the first valid numeric token.
            m = re.findall(r'\d[\d,]*\.?\d*', divtxt)
            div = None
            if m:
                tok = m[0].replace(',', '')
                try:
                    div = float(tok)
                except Exception:
                    div = None
            if not div:
                continue
            nums = re.findall(r'(\d+)', combo)
            if len(nums) >= 2:
                a, b = int(nums[0]), int(nums[1])
                key = f"{min(a,b)}-{max(a,b)}"
            else:
                key = None

            if pool == '連贏' and key:
                outQ[key] = div
            elif pool == '位置Q' and key:
                outQP[key] = div
                seenQP = True
                mode = 'QP_CONT'
            continue

        if len(r) == 2 and mode == 'QP_CONT':
            combo, divtxt = r[0], r[1]
            m = re.findall(r'\d[\d,]*\.?\d*', divtxt)
            div = None
            if m:
                tok = m[0].replace(',', '')
                try:
                    div = float(tok)
                except Exception:
                    div = None
            if not div:
                continue
            nums = re.findall(r'(\d+)', combo)
            if len(nums) >= 2:
                a, b = int(nums[0]), int(nums[1])
                key = f"{min(a,b)}-{max(a,b)}"
                outQP[key] = div
            continue

        mode = None

    return {'Q': outQ, 'QP': outQP, 'hasQP': seenQP}


def load_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def finalize(stake, ret):
    profit = ret - stake
    roi = profit / stake if stake else None
    return {'stake': stake, 'return': ret, 'profit': profit, 'roi': roi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--data', default='hkjc_dataset_v3_code_prev1.jsonl')
    ap.add_argument('--trainStart', required=True)
    ap.add_argument('--trainEnd', required=True)
    ap.add_argument('--testStart', required=True)
    ap.add_argument('--testEnd', required=True)
    ap.add_argument('--calibFrac', type=float, default=0.2)
    ap.add_argument('--stake10', type=float, default=10.0)
    ap.add_argument('--stakes', default='50,100,300')
    ap.add_argument('--minBets', type=int, default=50)
    ap.add_argument('--maxWorkers', type=int, default=24)
    ap.add_argument('--cache', default='hkjc_dividend_cache.json')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    stake50, stake100, stake300 = [float(x) for x in args.stakes.split(',')]

    con = sqlite3.connect(args.db)
    cur = con.cursor()

    def list_meetings(start, end):
        cur.execute('select racedate, venue, meeting_id from meetings where racedate>=? and racedate<=? order by racedate asc, venue asc', (start, end))
        return [(r,v,mid) for r,v,mid in cur.fetchall()]

    train_meetings = list_meetings(args.trainStart, args.trainEnd)
    test_meetings = list_meetings(args.testStart, args.testEnd)
    if not train_meetings or not test_meetings:
        raise SystemExit('No meetings in range')

    # race numbers per meeting
    meeting_races = {}
    for r,v,mid in train_meetings + test_meetings:
        if (r,v) in meeting_races:
            continue
        cur.execute('select race_no from races where meeting_id=? order by race_no asc', (mid,))
        meeting_races[(r,v)] = [int(x[0]) for x in cur.fetchall()]

    rows = load_jsonl(args.data)

    label_key = 'y_place'
    meta = {
        'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no',
        'y_finish_pos','y_win','y_place','cur_jockey','cur_trainer','cur_surface'
    }
    feat_keys = sorted([k for k in rows[0].keys() if k not in meta and k != label_key])

    # index rows by meeting
    by_meeting = defaultdict(list)
    for r in rows:
        by_meeting[(r.get('racedate'), r.get('venue'))].append(r)

    def vec(r):
        out = []
        for k in feat_keys:
            if k == 'cur_win_odds':
                out.append(0.0)  # NO-ODDS
            else:
                out.append(float(r.get(k, 0) or 0))
        return out

    # train split into fit vs calib by time
    n_train = len(train_meetings)
    n_calib = max(1, int(round(n_train * args.calibFrac)))
    fit_meetings = train_meetings[:-n_calib]
    calib_meetings = train_meetings[-n_calib:]

    def build_rows(meeting_list):
        s = {(r,v) for r,v,_ in meeting_list}
        return [r for mk in s for r in by_meeting.get(mk, [])]

    fit_rows = build_rows(fit_meetings)
    calib_rows = build_rows(calib_meetings)
    test_rows = build_rows(test_meetings)

    X_fit = np.asarray([vec(r) for r in fit_rows], dtype=np.float32)
    y_fit = np.asarray([int(r.get(label_key, 0) or 0) for r in fit_rows], dtype=np.int32)

    X_cal = np.asarray([vec(r) for r in calib_rows], dtype=np.float32)
    y_cal = np.asarray([int(r.get(label_key, 0) or 0) for r in calib_rows], dtype=np.int32)

    X_test = np.asarray([vec(r) for r in test_rows], dtype=np.float32)

    # dividend cache
    try:
        cache = json.load(open(args.cache, 'r', encoding='utf-8'))
    except Exception:
        cache = {}

    def cache_key(url, pair):
        return hashlib.sha1((url+'|'+pair).encode('utf-8')).hexdigest()

    def get_dividends(url, pair):
        k = cache_key(url, pair)
        if k in cache:
            return cache[k]
        html = fetch_html(url)
        divs = parse_dividends_Q_QP(html)
        out = {
            'q_div': divs['Q'].get(pair),
            'qp_div': divs['QP'].get(pair),
            'qp_offered': divs['hasQP']
        }
        cache[k] = out
        return out

    def platt_fit(p_raw, y):
        X = np.asarray(p_raw, dtype=np.float64).reshape(-1,1)
        y = np.asarray(y, dtype=np.int32)
        clf = LogisticRegression(solver='lbfgs', max_iter=200)
        clf.fit(X, y)
        return clf

    def platt_apply(clf, p_raw):
        X = np.asarray(p_raw, dtype=np.float64).reshape(-1,1)
        return clf.predict_proba(X)[:,1]

    def make_race_picks(base_rows, p_pred):
        by_race = defaultdict(list)
        for r, pp in zip(base_rows, p_pred):
            mk = (r['racedate'], r['venue'])
            if int(r['race_no']) not in meeting_races.get(mk, []):
                continue
            key = (r['racedate'], r['venue'], int(r['race_no']))
            by_race[key].append((int(r['horse_no']), float(pp)))

        picks = []
        for (rd, venue, race_no), arr in by_race.items():
            if len(arr) < 2:
                continue
            arr_sorted = sorted(arr, key=lambda t: t[1], reverse=True)
            h1,p1 = arr_sorted[0]
            h2,p2 = arr_sorted[1]
            pair = f"{min(h1,h2)}-{max(h1,h2)}"
            url = f'https://racing.hkjc.com/zh-hk/local/information/localresults?racedate={rd}&Racecourse={venue}&RaceNo={race_no}'
            picks.append({'racedate': rd, 'venue': venue, 'race_no': race_no, 'pair': pair, 'p12': p1+p2, 'url': url})
        return picks

    def eval_tiered(picks, t50, t100, t300):
        # t50 < t100 < t300
        def stake(p12):
            if p12 > t300:
                return stake300
            if p12 > t100:
                return stake100
            if p12 > t50:
                return stake50
            return 0.0

        items = []
        for p in picks:
            st = stake(float(p['p12']))
            if st > 0:
                items.append((p, st))

        if not items:
            return {'races_bet': 0, 'combined': finalize(0.0, 0.0), 'Q': finalize(0.0, 0.0), 'QP': finalize(0.0, 0.0)}

        # fetch dividends aligned
        stake_q = sum(st for _, st in items)
        stake_qp = 0.0
        ret_q = 0.0
        ret_qp = 0.0

        # parallel fetch
        with ThreadPoolExecutor(max_workers=args.maxWorkers) as ex:
            futs = [ex.submit(get_dividends, p['url'], p['pair']) for p, _ in items]
            _ = [f.result() if not f.exception() else None for f in as_completed(futs)]

        for p, st in items:
            d = get_dividends(p['url'], p['pair'])
            mult = st / args.stake10
            if d.get('q_div') is not None:
                ret_q += float(d['q_div']) * mult
            if d.get('qp_offered'):
                stake_qp += st
                if d.get('qp_div') is not None:
                    ret_qp += float(d['qp_div']) * mult

        return {
            'races_bet': len(items),
            'Q': finalize(stake_q, ret_q),
            'QP': finalize(stake_qp, ret_qp),
            'combined': finalize(stake_q+stake_qp, ret_q+ret_qp)
        }

    def optimize_thresholds(picks_cal, label):
        # fetch all dividends once for calibration picks
        with ThreadPoolExecutor(max_workers=args.maxWorkers) as ex:
            futs = [ex.submit(get_dividends, p['url'], p['pair']) for p in picks_cal]
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception:
                    pass

        p12s = np.asarray([float(p['p12']) for p in picks_cal], dtype=np.float64)
        # candidate thresholds from quantiles
        qs = [0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.92,0.94,0.96,0.98]
        cand = sorted(set(float(np.quantile(p12s, q)) for q in qs))

        best = None
        tried = 0
        for t50 in cand:
            for t100 in cand:
                if t100 <= t50:
                    continue
                for t300 in cand:
                    if t300 <= t100:
                        continue
                    tried += 1
                    res = eval_tiered(picks_cal, t50, t100, t300)
                    if res['races_bet'] < args.minBets:
                        continue
                    roi = res['combined']['roi']
                    if roi is None:
                        continue
                    if (best is None) or (roi > best['roi']):
                        best = {
                            'roi': roi,
                            'races_bet': res['races_bet'],
                            'thresholds': {'t50': t50, 't100': t100, 't300': t300},
                            'result': res
                        }
        return {'label': label, 'candidates': len(cand), 'tried': tried, 'best': best}

    # train base models
    # XGB
    dfit = xgb.DMatrix(X_fit, label=y_fit, feature_names=feat_keys)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,
        'eta': 0.07,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'lambda': 1.0,
        'min_child_weight': 10,
        'seed': 42,
        'nthread': 4,
    }
    xgb_model = xgb.train(params, dfit, num_boost_round=350)
    p_xgb_cal_raw = xgb_model.predict(xgb.DMatrix(X_cal, feature_names=feat_keys))
    p_xgb_test_raw = xgb_model.predict(xgb.DMatrix(X_test, feature_names=feat_keys))

    # LogReg
    lr_model = LogisticRegression(solver='saga', penalty='l2', C=1.0, max_iter=800, n_jobs=4, random_state=42)
    lr_model.fit(X_fit, y_fit)
    p_lr_cal_raw = lr_model.predict_proba(X_cal)[:,1]
    p_lr_test_raw = lr_model.predict_proba(X_test)[:,1]

    # calibrate
    cal_xgb = platt_fit(p_xgb_cal_raw, y_cal)
    cal_lr = platt_fit(p_lr_cal_raw, y_cal)

    p_xgb_cal = platt_apply(cal_xgb, p_xgb_cal_raw)
    p_lr_cal = platt_apply(cal_lr, p_lr_cal_raw)

    p_xgb_test = platt_apply(cal_xgb, p_xgb_test_raw)
    p_lr_test = platt_apply(cal_lr, p_lr_test_raw)

    # make picks
    picks_xgb_cal = make_race_picks(calib_rows, p_xgb_cal)
    picks_lr_cal = make_race_picks(calib_rows, p_lr_cal)

    picks_xgb_test = make_race_picks(test_rows, p_xgb_test)
    picks_lr_test = make_race_picks(test_rows, p_lr_test)

    # optimize on calib
    opt_xgb = optimize_thresholds(picks_xgb_cal, 'xgb')
    opt_lr = optimize_thresholds(picks_lr_cal, 'logreg')

    # evaluate on test using optimized thresholds
    def test_eval(picks_test, opt):
        if not opt['best']:
            return None
        th = opt['best']['thresholds']
        return eval_tiered(picks_test, th['t50'], th['t100'], th['t300'])

    test_xgb = test_eval(picks_xgb_test, opt_xgb)
    test_lr = test_eval(picks_lr_test, opt_lr)

    with open(args.cache, 'w', encoding='utf-8') as f:
        json.dump(cache, f)

    out = {
        'trainRange': {'start': args.trainStart, 'end': args.trainEnd, 'meetings': len(train_meetings), 'fitMeetings': len(fit_meetings), 'calibMeetings': len(calib_meetings)},
        'testRange': {'start': args.testStart, 'end': args.testEnd, 'meetings': len(test_meetings)},
        'noOdds': True,
        'stakes': {'50': stake50, '100': stake100, '300': stake300, 'per': args.stake10},
        'minBets': args.minBets,
        'xgb': {
            'opt': opt_xgb,
            'test': test_xgb
        },
        'logreg': {
            'opt': opt_lr,
            'test': test_lr
        },
        'generatedAt': datetime.datetime.now().isoformat(timespec='seconds')
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(json.dumps({'out': args.out}, indent=2))


if __name__ == '__main__':
    main()
