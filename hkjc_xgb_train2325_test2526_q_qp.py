import re, json, argparse, sqlite3, hashlib, datetime
from collections import defaultdict
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xgboost as xgb

UA = 'openclaw-hkjc-xgb-train-test/1.0'


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


def first_number(s: str):
    m = re.findall(r'\d[\d,]*\.?\d*', s)
    if not m:
        return None
    tok = m[0].replace(',', '')
    try:
        return float(tok)
    except Exception:
        return None


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
            div = first_number(divtxt)
            if not div:
                continue
            nums = re.findall(r'(\d+)', combo)
            key = None
            if len(nums) >= 2:
                a, b = int(nums[0]), int(nums[1])
                key = f"{min(a,b)}-{max(a,b)}"
            if pool == '連贏' and key:
                outQ[key] = div
            elif pool == '位置Q' and key:
                outQP[key] = div
                seenQP = True
                mode = 'QP_CONT'
            continue

        if len(r) == 2 and mode == 'QP_CONT':
            combo, divtxt = r[0], r[1]
            div = first_number(divtxt)
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
    ap.add_argument('--trainStart', default='2023/09/01')
    ap.add_argument('--trainEnd', default='2025/07/31')
    ap.add_argument('--testStart', default='2025/09/01')
    ap.add_argument('--testEnd', default='2026/02/19')
    ap.add_argument('--evalFracMeetings', type=float, default=0.2)
    ap.add_argument('--stake', type=float, default=10.0)
    ap.add_argument('--maxWorkers', type=int, default=24)
    ap.add_argument('--cache', default='hkjc_dividend_cache.json')
    ap.add_argument('--out', default='hkjc_xgb_train2325_test2526_q_qp.json')
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    cur = con.cursor()

    def list_meetings(start, end):
        cur.execute('select racedate, venue, meeting_id from meetings where racedate>=? and racedate<=? order by racedate asc, venue asc', (start, end))
        return [(r,v,mid) for r,v,mid in cur.fetchall()]

    train_meetings = list_meetings(args.trainStart, args.trainEnd)
    test_meetings = list_meetings(args.testStart, args.testEnd)
    if not train_meetings or not test_meetings:
        raise SystemExit('No meetings found in range')

    # split train meetings into fit vs eval by time
    n_train = len(train_meetings)
    n_eval = max(1, int(round(n_train * args.evalFracMeetings)))
    fit_meetings = train_meetings[:-n_eval]
    eval_meetings = train_meetings[-n_eval:]

    # race numbers per meeting (for test)
    meeting_races = {}
    for r,v,mid in test_meetings:
        cur.execute('select race_no from races where meeting_id=? order by race_no asc', (mid,))
        meeting_races[(r,v)] = [int(x[0]) for x in cur.fetchall()]

    rows = load_jsonl(args.data)

    label_key = 'y_place'
    meta = {
        'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no',
        'y_finish_pos','y_win','y_place','cur_jockey','cur_trainer','cur_surface'
    }
    feat_keys = sorted([k for k in rows[0].keys() if k not in meta and k != label_key])

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

    def build_rows(meeting_list):
        s = {(r,v) for r,v,_ in meeting_list}
        return [r for mk in s for r in by_meeting.get(mk, [])]

    fit_rows = build_rows(fit_meetings)
    eval_rows = build_rows(eval_meetings)
    test_rows = build_rows(test_meetings)

    X_fit = np.asarray([vec(r) for r in fit_rows], dtype=np.float32)
    y_fit = np.asarray([int(r.get(label_key, 0) or 0) for r in fit_rows], dtype=np.int32)

    X_eval = np.asarray([vec(r) for r in eval_rows], dtype=np.float32)
    y_eval = np.asarray([int(r.get(label_key, 0) or 0) for r in eval_rows], dtype=np.int32)

    X_test = np.asarray([vec(r) for r in test_rows], dtype=np.float32)

    dfit = xgb.DMatrix(X_fit, label=y_fit, feature_names=feat_keys)
    deval = xgb.DMatrix(X_eval, label=y_eval, feature_names=feat_keys)

    # Overfitting prevention knobs
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,
        'min_child_weight': 10,
        'gamma': 1.0,
        'lambda': 10.0,   # L2
        'alpha': 0.5,     # L1
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'eta': 0.05,
        'seed': 42,
        'nthread': 4,
    }

    evals = [(dfit, 'fit'), (deval, 'eval')]
    model = xgb.train(
        params,
        dfit,
        num_boost_round=2000,
        evals=evals,
        early_stopping_rounds=80,
        verbose_eval=False,
    )

    best_iter = getattr(model, 'best_iteration', None)
    best_score = getattr(model, 'best_score', None)

    p_test = model.predict(xgb.DMatrix(X_test, feature_names=feat_keys))

    # predictions by race, choose top2 pair
    by_race = defaultdict(list)
    for r, pp in zip(test_rows, p_test):
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
        picks.append({'racedate': rd, 'venue': venue, 'race_no': race_no, 'pair': pair, 'p1': p1, 'p2': p2, 'p12': p1+p2, 'url': url})

    # dividend cache
    try:
        cache = json.load(open(args.cache, 'r', encoding='utf-8'))
    except Exception:
        cache = {}

    def cache_key(url, pair):
        return hashlib.sha1((url+'|'+pair).encode('utf-8')).hexdigest()

    def get_div(url, pair):
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

    # fetch dividends
    per_race = []
    with ThreadPoolExecutor(max_workers=args.maxWorkers) as ex:
        futs = {ex.submit(get_div, p['url'], p['pair']): p for p in picks}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                d = fut.result()
            except Exception:
                d = {'q_div': None, 'qp_div': None, 'qp_offered': False}
            per_race.append({
                **p,
                'q_div': d.get('q_div'),
                'qp_div': d.get('qp_div'),
                'qp_offered': bool(d.get('qp_offered')),
            })

    with open(args.cache, 'w', encoding='utf-8') as f:
        json.dump(cache, f)

    # ROI
    stake_q = len(per_race) * args.stake
    ret_q = sum((r['q_div'] or 0.0) for r in per_race)

    per_race_qp = [r for r in per_race if r.get('qp_offered')]
    stake_qp = len(per_race_qp) * args.stake
    ret_qp = sum((r['qp_div'] or 0.0) for r in per_race_qp)

    out = {
        'trainRange': {'start': args.trainStart, 'end': args.trainEnd, 'meetings': len(train_meetings), 'fitMeetings': len(fit_meetings), 'evalMeetings': len(eval_meetings)},
        'testRange': {'start': args.testStart, 'end': args.testEnd, 'meetings': len(test_meetings)},
        'noOdds': True,
        'params': params,
        'best_iteration': best_iter,
        'best_score': best_score,
        'stake_unit': args.stake,
        'summary': {
            'Q': finalize(stake_q, ret_q),
            'QP': finalize(stake_qp, ret_qp),
            'combined': finalize(stake_q+stake_qp, ret_q+ret_qp)
        },
        'per_race': sorted(per_race, key=lambda r:(r['racedate'], r['venue'], r['race_no'])),
        'generatedAt': datetime.datetime.now().isoformat(timespec='seconds')
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(json.dumps({'out': args.out, 'races': len(per_race)}, indent=2))


if __name__ == '__main__':
    main()
