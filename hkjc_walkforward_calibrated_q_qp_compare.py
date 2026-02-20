import re, json, argparse, sqlite3, hashlib
from collections import defaultdict
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

UA = 'openclaw-hkjc-walkforward-cal/1.0'


def fetch_html(url: str) -> str:
    req = Request(url, headers={'User-Agent': UA})
    with urlopen(req, timeout=10) as resp:
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
            divm = re.sub(r'[^0-9.]', '', divtxt)
            div = float(divm) if divm else None
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
            divm = re.sub(r'[^0-9.]', '', divtxt)
            div = float(divm) if divm else None
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
    ap.add_argument('--start', default='2024/09/01')
    ap.add_argument('--end', default='2026/02/19')
    ap.add_argument('--testMeetings', type=int, default=10)
    ap.add_argument('--minTrainMeetings', type=int, default=40)
    ap.add_argument('--stepMeetings', type=int, default=10)
    ap.add_argument('--calibFrac', type=float, default=0.2, help='fraction of train meetings reserved for calibration (last portion)')
    ap.add_argument('--calibMethod', choices=['platt'], default='platt')
    ap.add_argument('--stake', type=float, default=10.0)
    ap.add_argument('--thresholds', default='1.85,1.9,1.92')
    ap.add_argument('--maxWorkers', type=int, default=24)
    ap.add_argument('--cache', default='hkjc_dividend_cache.json')
    ap.add_argument('--out', default='hkjc_walkforward_calibrated_q_qp_compare.json')
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(',') if x.strip()]

    con = sqlite3.connect(args.db)
    cur = con.cursor()
    cur.execute('select racedate, venue, meeting_id from meetings where racedate>=? and racedate<=? order by racedate asc, venue asc', (args.start, args.end))
    meetings = [(r,v,mid) for r,v,mid in cur.fetchall()]
    if len(meetings) < args.minTrainMeetings + args.testMeetings:
        raise SystemExit('Not enough meetings in range')

    # race numbers per meeting
    meeting_races = {}
    for r,v,mid in meetings:
        cur.execute('select race_no from races where meeting_id=? order by race_no asc', (mid,))
        meeting_races[(r,v)] = [int(x[0]) for x in cur.fetchall()]

    rows = load_jsonl(args.data)

    label_key = 'y_place'
    meta = {
        'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no',
        'y_finish_pos','y_win','y_place','cur_jockey','cur_trainer','cur_surface'
    }
    feat_keys = sorted([k for k in rows[0].keys() if k not in meta and k != label_key])

    # index rows by meeting (racedate, venue)
    by_meeting = defaultdict(list)
    for r in rows:
        mk = (r.get('racedate'), r.get('venue'))
        by_meeting[mk].append(r)

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

    # blocks
    blocks = []
    i = args.minTrainMeetings
    while i + args.testMeetings <= len(meetings):
        train = meetings[:i]
        test = meetings[i:i+args.testMeetings]
        blocks.append({'train': train, 'test': test})
        i += args.stepMeetings

    def vec(r):
        out = []
        for k in feat_keys:
            if k == 'cur_win_odds':
                out.append(0.0)  # NO-ODDS
            else:
                out.append(float(r.get(k, 0) or 0))
        return out

    def build_rows(meeting_list):
        s = {(r,v) for r,v,_ in meeting_list}
        return [r for mk in s for r in by_meeting.get(mk, [])]

    def platt_fit(p_raw, y):
        # Fit a 1D logistic regression on raw probabilities -> calibrated probability
        X = np.asarray(p_raw, dtype=np.float64).reshape(-1,1)
        y = np.asarray(y, dtype=np.int32)
        clf = LogisticRegression(solver='lbfgs', max_iter=200)
        clf.fit(X, y)
        return clf

    def platt_apply(clf, p_raw):
        X = np.asarray(p_raw, dtype=np.float64).reshape(-1,1)
        return clf.predict_proba(X)[:,1]

    def make_race_picks(test_rows, p_test):
        by_race = defaultdict(list)
        for r, pp in zip(test_rows, p_test):
            key = (r['racedate'], r['venue'], int(r['race_no']))
            if int(r['race_no']) not in meeting_races.get((r['racedate'], r['venue']), []):
                continue
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

    def eval_picks(picks, p12_threshold):
        picks = [p for p in picks if p.get('p12', 0.0) > p12_threshold]
        stake_q = len(picks) * args.stake
        ret_q = 0.0
        stake_qp = 0.0
        ret_qp = 0.0

        with ThreadPoolExecutor(max_workers=args.maxWorkers) as ex:
            futs = [ex.submit(get_dividends, p['url'], p['pair']) for p in picks]
            divs = []
            for fut in as_completed(futs):
                try:
                    divs.append(fut.result())
                except Exception:
                    divs.append({'q_div': None, 'qp_div': None, 'qp_offered': False})

        for d in divs:
            if d.get('q_div') is not None:
                ret_q += float(d['q_div'])
            if d.get('qp_offered'):
                stake_qp += args.stake
                if d.get('qp_div') is not None:
                    ret_qp += float(d['qp_div'])

        return {
            'races': len(picks),
            'Q': finalize(stake_q, ret_q),
            'QP': finalize(stake_qp, ret_qp),
            'combined': finalize(stake_q+stake_qp, ret_q+ret_qp)
        }

    out_blocks = []

    for bi, b in enumerate(blocks, 1):
        train = b['train']
        test = b['test']

        # split train into fit vs calib by time (last calibFrac for calibration)
        n_train = len(train)
        n_calib = max(1, int(round(n_train * args.calibFrac)))
        fit_meetings = train[:-n_calib]
        calib_meetings = train[-n_calib:]

        fit_rows = build_rows(fit_meetings)
        calib_rows = build_rows(calib_meetings)
        test_rows = build_rows(test)
        if not fit_rows or not calib_rows or not test_rows:
            continue

        X_fit = np.asarray([vec(r) for r in fit_rows], dtype=np.float32)
        y_fit = np.asarray([int(r.get(label_key, 0) or 0) for r in fit_rows], dtype=np.int32)

        X_cal = np.asarray([vec(r) for r in calib_rows], dtype=np.float32)
        y_cal = np.asarray([int(r.get(label_key, 0) or 0) for r in calib_rows], dtype=np.int32)

        X_test = np.asarray([vec(r) for r in test_rows], dtype=np.float32)

        # Base XGB
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

        # Base LogReg
        lr = LogisticRegression(
            solver='saga',
            penalty='l2',
            C=1.0,
            max_iter=500,
            n_jobs=4,
            random_state=42
        )
        lr.fit(X_fit, y_fit)
        p_lr_cal_raw = lr.predict_proba(X_cal)[:, 1]
        p_lr_test_raw = lr.predict_proba(X_test)[:, 1]

        # Calibrate using Platt (sigmoid) on calibration set
        cal_xgb = platt_fit(p_xgb_cal_raw, y_cal)
        cal_lr = platt_fit(p_lr_cal_raw, y_cal)

        p_xgb_test = platt_apply(cal_xgb, p_xgb_test_raw)
        p_lr_test = platt_apply(cal_lr, p_lr_test_raw)

        picks_xgb = make_race_picks(test_rows, p_xgb_test)
        picks_lr = make_race_picks(test_rows, p_lr_test)

        def p12_stats(picks):
            arr = sorted([float(p.get('p12', 0.0)) for p in picks])
            if not arr:
                return None
            def q(p):
                idx = int(round(p * (len(arr)-1)))
                return arr[idx]
            return {
                'n': len(arr),
                'min': arr[0],
                'p25': q(0.25),
                'p50': q(0.50),
                'p75': q(0.75),
                'p90': q(0.90),
                'max': arr[-1],
            }

        per_th = {}
        for th in thresholds:
            per_th[str(th)] = {
                'xgb': eval_picks(picks_xgb, th),
                'logreg': eval_picks(picks_lr, th),
            }

        out_blocks.append({
            'block': bi,
            'train_meetings': len(train),
            'fit_meetings': len(fit_meetings),
            'calib_meetings': len(calib_meetings),
            'test_meetings': len(test),
            'test_start': test[0][0],
            'test_end': test[-1][0],
            'p12_stats': {
                'xgb': p12_stats(picks_xgb),
                'logreg': p12_stats(picks_lr)
            },
            'thresholds': thresholds,
            'resultsByThreshold': per_th
        })

        if bi % 2 == 0:
            with open(args.cache, 'w', encoding='utf-8') as f:
                json.dump(cache, f)

    with open(args.cache, 'w', encoding='utf-8') as f:
        json.dump(cache, f)

    # overall aggregate by threshold
    def agg(model_name):
        out = {}
        for th in thresholds:
            st=rt=0.0
            races=0
            for b in out_blocks:
                m = b['resultsByThreshold'][str(th)][model_name]
                races += m['races']
                st += m['combined']['stake']
                rt += m['combined']['return']
            out[str(th)] = {'races': races, 'combined': finalize(st, rt)}
        return out

    out = {
        'range': {'start': args.start, 'end': args.end},
        'blockSpec': {
            'minTrainMeetings': args.minTrainMeetings,
            'testMeetings': args.testMeetings,
            'stepMeetings': args.stepMeetings,
            'calibFrac': args.calibFrac,
            'calibMethod': args.calibMethod,
        },
        'stake_unit': args.stake,
        'thresholds': thresholds,
        'blocks': out_blocks,
        'overallByThreshold': {
            'xgb': agg('xgb'),
            'logreg': agg('logreg')
        }
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(json.dumps({'out': args.out, 'blocks': len(out_blocks)}, indent=2))


if __name__ == '__main__':
    main()
