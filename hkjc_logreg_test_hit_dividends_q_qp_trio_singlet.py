import re, json, argparse, sqlite3, hashlib, datetime
from collections import defaultdict
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.linear_model import LogisticRegression

UA = 'openclaw-hkjc-hit-divs/1.0'


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


def parse_dividends_table(html: str):
    # returns dict pool -> list of (comboStr, dividendFloat)
    tables = re.findall(r'<table[\s\S]*?</table>', html, flags=re.I)
    div_table = None
    for t in tables:
        if '派彩' in t and '彩池' in t and '勝出組合' in t:
            div_table = t
            break
    if not div_table:
        return {}

    trs = re.findall(r'<tr[^>]*>([\s\S]*?)</tr>', div_table, flags=re.I)
    rows = []
    for tr in trs:
        cells = [clean_cell(x) for x in re.findall(r'<t[dh][^>]*>([\s\S]*?)</t[dh]>', tr, flags=re.I)]
        if len(cells) >= 2:
            rows.append(cells)

    out = defaultdict(list)
    mode = None
    cur_pool = None

    for r in rows:
        if r[0] == '彩池':
            continue

        if len(r) >= 3:
            pool, combo, divtxt = r[0], r[1], r[2]
            cur_pool = pool
            mode = None
            div = first_number(divtxt)
            if div is None:
                continue
            out[pool].append((combo, div))
            if pool in ('位置', '位置Q'):
                mode = 'CONT'
            continue

        if len(r) == 2 and mode == 'CONT' and cur_pool in ('位置', '位置Q'):
            combo, divtxt = r[0], r[1]
            div = first_number(divtxt)
            if div is None:
                continue
            out[cur_pool].append((combo, div))
            continue

        mode = None
        cur_pool = None

    return dict(out)


def load_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def platt_fit(p_raw, y):
    X = np.asarray(p_raw, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.int32)
    clf = LogisticRegression(solver='lbfgs', max_iter=200)
    clf.fit(X, y)
    return clf


def platt_apply(clf, p_raw):
    X = np.asarray(p_raw, dtype=np.float64).reshape(-1, 1)
    return clf.predict_proba(X)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--data', default='hkjc_dataset_v3_code_prev1.jsonl')
    ap.add_argument('--trainStart', default='2024/09/01')
    ap.add_argument('--trainEnd', default='2025/07/31')
    ap.add_argument('--testStart', default='2025/09/01')
    ap.add_argument('--testEnd', default='2026/02/19')
    ap.add_argument('--calibFrac', type=float, default=0.2)
    ap.add_argument('--tiers', default='0.8107407465945993,0.8383387192732921,0.8789906061855338')
    ap.add_argument('--out', default='hkjc_logreg_test_hits_q_qp_trio_singlet.json')
    ap.add_argument('--cache', default='hkjc_dividend_cache_v2.json')
    args = ap.parse_args()

    t50, t100, t300 = [float(x) for x in args.tiers.split(',')]

    con = sqlite3.connect(args.db)
    cur = con.cursor()

    def list_meetings(start, end):
        cur.execute('select racedate, venue, meeting_id from meetings where racedate>=? and racedate<=? order by racedate asc, venue asc', (start, end))
        return [(r,v,mid) for r,v,mid in cur.fetchall()]

    train_meetings = list_meetings(args.trainStart, args.trainEnd)
    test_meetings = list_meetings(args.testStart, args.testEnd)

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

    by_meeting = defaultdict(list)
    for r in rows:
        by_meeting[(r.get('racedate'), r.get('venue'))].append(r)

    def vec(r):
        out = []
        for k in feat_keys:
            if k == 'cur_win_odds':
                out.append(0.0)
            else:
                out.append(float(r.get(k, 0) or 0))
        return out

    # train split fit vs calib
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

    # base logreg
    lr = LogisticRegression(solver='saga', penalty='l2', C=1.0, max_iter=800, n_jobs=4, random_state=42)
    lr.fit(X_fit, y_fit)
    p_cal_raw = lr.predict_proba(X_cal)[:,1]
    p_test_raw = lr.predict_proba(X_test)[:,1]

    cal = platt_fit(p_cal_raw, y_cal)
    p_test = platt_apply(cal, p_test_raw)

    # picks by race
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
        p12 = p1+p2
        # tiered stake just for reference
        tier = None
        if p12 > t300:
            tier = '300'
        elif p12 > t100:
            tier = '100'
        elif p12 > t50:
            tier = '50'
        else:
            tier = '0'
        if tier == '0':
            continue
        pair = f"{min(h1,h2)}-{max(h1,h2)}"
        url = f'https://racing.hkjc.com/zh-hk/local/information/localresults?racedate={rd}&Racecourse={venue}&RaceNo={race_no}'
        picks.append({'racedate': rd, 'venue': venue, 'race_no': race_no, 'pair': pair, 'p12': p12, 'tier': tier, 'url': url})

    # dividend cache
    try:
        cache = json.load(open(args.cache, 'r', encoding='utf-8'))
    except Exception:
        cache = {}

    def cache_key(url):
        return hashlib.sha1(url.encode('utf-8')).hexdigest()

    def get_all_divs(url):
        k = cache_key(url)
        if k in cache:
            return cache[k]
        html = fetch_html(url)
        pools = parse_dividends_table(html)
        cache[k] = pools
        return pools

    # fetch all dividends per picked race
    out_rows = []
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(get_all_divs, p['url']): p for p in picks}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                pools = fut.result()
            except Exception:
                continue

            # our Q/QP dividends
            q_div = None
            qp_div = None
            # Q rows look like ('8,9', 6246.5)
            for combo, div in pools.get('連贏', []):
                nums = re.findall(r'(\d+)', combo)
                if len(nums) >= 2:
                    key = f"{min(int(nums[0]),int(nums[1]))}-{max(int(nums[0]),int(nums[1]))}"
                    if key == p['pair']:
                        q_div = div
                        break
            for combo, div in pools.get('位置Q', []):
                nums = re.findall(r'(\d+)', combo)
                if len(nums) >= 2:
                    key = f"{min(int(nums[0]),int(nums[1]))}-{max(int(nums[0]),int(nums[1]))}"
                    if key == p['pair']:
                        qp_div = div
                        break

            if q_div is None and qp_div is None:
                continue

            # winning dividends for 三重彩 and 單T (first row is the winning combo)
            tri_combo = tri_div = None
            if pools.get('三重彩'):
                tri_combo, tri_div = pools['三重彩'][0]
            t_combo = t_div = None
            if pools.get('單T'):
                t_combo, t_div = pools['單T'][0]

            out_rows.append({
                'racedate': p['racedate'],
                'venue': p['venue'],
                'race_no': p['race_no'],
                'pair': p['pair'],
                'p12': p['p12'],
                'tier': p['tier'],
                'Q_div': q_div,
                'QP_div': qp_div,
                'Tri_combo': tri_combo,
                'Tri_div': tri_div,
                'SingleT_combo': t_combo,
                'SingleT_div': t_div,
                'url': p['url']
            })

    with open(args.cache, 'w', encoding='utf-8') as f:
        json.dump(cache, f)

    out_rows.sort(key=lambda r: (r['racedate'], r['venue'], r['race_no']))

    out = {
        'trainRange': {'start': args.trainStart, 'end': args.trainEnd},
        'testRange': {'start': args.testStart, 'end': args.testEnd},
        'tiers': {'t50': t50, 't100': t100, 't300': t300},
        'hit_races': len(out_rows),
        'rows': out_rows,
        'generatedAt': datetime.datetime.now().isoformat(timespec='seconds')
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(json.dumps({'out': args.out, 'hit_races': len(out_rows)}, indent=2))


if __name__ == '__main__':
    main()
