import re, json, argparse, sqlite3, hashlib, datetime
from collections import defaultdict
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.ensemble import RandomForestClassifier

UA = 'openclaw-hkjc-rf-train-test/1.0'


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
    ap.add_argument('--stake', type=float, default=10.0)
    ap.add_argument('--maxWorkers', type=int, default=24)
    ap.add_argument('--cache', default='hkjc_dividend_cache.json')
    ap.add_argument('--out', default='hkjc_rf_noodds_train2325_test2526_q_qp.json')

    # RF controls
    ap.add_argument('--nEstimators', type=int, default=1200)
    ap.add_argument('--maxDepth', type=int, default=14)
    ap.add_argument('--minSamplesLeaf', type=int, default=50)
    ap.add_argument('--minSamplesSplit', type=int, default=200)
    ap.add_argument('--maxFeatures', default='sqrt')  # or float
    ap.add_argument('--classWeight', default='balanced')
    ap.add_argument('--nJobs', type=int, default=4)

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

    train_rows = build_rows(train_meetings)
    test_rows = build_rows(test_meetings)

    X_train = np.asarray([vec(r) for r in train_rows], dtype=np.float32)
    y_train = np.asarray([int(r.get(label_key, 0) or 0) for r in train_rows], dtype=np.int32)

    X_test = np.asarray([vec(r) for r in test_rows], dtype=np.float32)

    max_features = args.maxFeatures
    try:
        max_features = float(max_features)
    except Exception:
        pass

    class_weight = None if args.classWeight.lower() == 'none' else args.classWeight

    rf = RandomForestClassifier(
        n_estimators=args.nEstimators,
        max_depth=args.maxDepth,
        min_samples_leaf=args.minSamplesLeaf,
        min_samples_split=args.minSamplesSplit,
        max_features=max_features,
        bootstrap=True,
        oob_score=False,
        n_jobs=args.nJobs,
        random_state=42,
        class_weight=class_weight
    )

    rf.fit(X_train, y_train)
    p_test = rf.predict_proba(X_test)[:, 1]

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
                'qp_offered': bool(d.get('qp_offered'))
            })

    with open(args.cache, 'w', encoding='utf-8') as f:
        json.dump(cache, f)

    stake_q = len(per_race) * args.stake
    ret_q = sum((r['q_div'] or 0.0) for r in per_race)

    per_race_qp = [r for r in per_race if r.get('qp_offered')]
    stake_qp = len(per_race_qp) * args.stake
    ret_qp = sum((r['qp_div'] or 0.0) for r in per_race_qp)

    out = {
        'model': 'RandomForestClassifier',
        'noOdds': True,
        'trainRange': {'start': args.trainStart, 'end': args.trainEnd, 'meetings': len(train_meetings)},
        'testRange': {'start': args.testStart, 'end': args.testEnd, 'meetings': len(test_meetings)},
        'rf_params': {
            'n_estimators': args.nEstimators,
            'max_depth': args.maxDepth,
            'min_samples_leaf': args.minSamplesLeaf,
            'min_samples_split': args.minSamplesSplit,
            'max_features': args.maxFeatures,
            'class_weight': args.classWeight,
            'n_jobs': args.nJobs
        },
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
