import re, json, argparse, sqlite3
from collections import defaultdict
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xgboost as xgb

UA = 'openclaw-hkjc-q-qp-backtest/1.0'


def fetch_html(url: str) -> str:
    req = Request(url, headers={'User-Agent': UA})
    with urlopen(req, timeout=8) as resp:
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
        return {'Q': {}, 'QP': {}}

    trs = re.findall(r'<tr[^>]*>([\s\S]*?)</tr>', div_table, flags=re.I)
    rows = []
    for tr in trs:
        cells = [clean_cell(x) for x in re.findall(r'<t[dh][^>]*>([\s\S]*?)</t[dh]>', tr, flags=re.I)]
        if len(cells) >= 2:
            rows.append(cells)

    outQ = {}
    outQP = {}

    for r in rows:
        if r[0] == '彩池':
            continue
        if len(r) < 3:
            continue
        pool, combo, divtxt = r[0], r[1], r[2]
        divm = re.sub(r'[^0-9.]', '', divtxt)
        div = float(divm) if divm else None
        if not div:
            continue
        m = re.findall(r'(\d+)', combo)
        if len(m) < 2:
            continue
        a, b = int(m[0]), int(m[1])
        key = f"{min(a,b)}-{max(a,b)}"

        if pool == '連贏':
            outQ[key] = div
        elif pool == '位置Q':
            outQP[key] = div

    return {'Q': outQ, 'QP': outQP}


def load_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--days', type=int, default=4, help='number of most recent meetings (date+venue)')
    ap.add_argument('--stake', type=float, default=10.0)
    ap.add_argument('--out', default='hkjc_q_qp_roi_last4days_noodds.json')
    args = ap.parse_args()

    # last N meetings
    con = sqlite3.connect(args.db)
    cur = con.cursor()
    cur.execute('select racedate, venue, meeting_id from meetings order by racedate desc, venue desc limit ?', (args.days,))
    meetings = [(r, v, mid) for r, v, mid in cur.fetchall()]
    meeting_set = {(r, v) for r, v, _ in meetings}

    # race list per meeting
    meeting_races = {}
    for r, v, mid in meetings:
        cur.execute('select race_no from races where meeting_id=? order by race_no asc', (mid,))
        meeting_races[(r, v)] = [int(x[0]) for x in cur.fetchall()]

    rows = load_jsonl(args.data)

    label_key = 'y_place'
    meta = {
        'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no',
        'y_finish_pos','y_win','y_place','cur_jockey','cur_trainer','cur_surface'
    }
    feat_keys = sorted([k for k in rows[0].keys() if k not in meta and k != label_key])

    # Build feature matrix for all selected rows, but with cur_win_odds forced to 0
    selected = [r for r in rows if (r.get('racedate'), r.get('venue')) in meeting_set]

    def vec(r):
        out = []
        for k in feat_keys:
            if k == 'cur_win_odds':
                out.append(0.0)
            else:
                out.append(float(r.get(k, 0) or 0))
        return out

    bst = xgb.Booster()
    bst.load_model(args.model)

    X = np.asarray([vec(r) for r in selected], dtype=np.float32)
    p = bst.predict(xgb.DMatrix(X, feature_names=feat_keys))

    by_race = defaultdict(list)
    for r, pp in zip(selected, p):
        key = (r['racedate'], r['venue'], int(r['race_no']))
        by_race[key].append((r, float(pp)))

    # decide top2 pair per race
    race_items = []
    for (rd, venue, race_no), arr in by_race.items():
        # skip races not in sqlite meeting_races list (safety)
        if race_no not in meeting_races.get((rd, venue), []):
            continue
        arr_sorted = sorted(arr, key=lambda t: t[1], reverse=True)
        if len(arr_sorted) < 2:
            continue
        h1 = int(arr_sorted[0][0]['horse_no'])
        h2 = int(arr_sorted[1][0]['horse_no'])
        pair = f"{min(h1,h2)}-{max(h1,h2)}"
        url = f'https://racing.hkjc.com/zh-hk/local/information/localresults?racedate={rd}&Racecourse={venue}&RaceNo={race_no}'
        race_items.append((rd, venue, race_no, pair, url))

    # fetch dividends in parallel
    def fetch_one(item):
        rd, venue, race_no, pair, url = item
        html = fetch_html(url)
        divs = parse_dividends_Q_QP(html)
        return rd, venue, race_no, pair, divs.get('Q', {}).get(pair), divs.get('QP', {}).get(pair), url

    per_race = []
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = [ex.submit(fetch_one, it) for it in race_items]
        done = 0
        for fut in as_completed(futs):
            try:
                per_race.append(fut.result())
            except Exception:
                continue
            done += 1
            if done % 50 == 0:
                print(json.dumps({'progress_fetch': done, 'of': len(futs)}))

    # aggregate
    totals = { 'Q': {'stake': 0.0, 'return': 0.0}, 'QP': {'stake': 0.0, 'return': 0.0} }
    totals_by_meeting = {}

    for rd, venue, race_no, pair, q_div, qp_div, url in per_race:
        mk = (rd, venue)
        if mk not in totals_by_meeting:
            totals_by_meeting[mk] = { 'Q': {'stake': 0.0, 'return': 0.0}, 'QP': {'stake': 0.0, 'return': 0.0} }

        stake = args.stake
        retQ = float(q_div) if q_div is not None else 0.0
        retQP = float(qp_div) if qp_div is not None else 0.0

        for pool, ret in [('Q', retQ), ('QP', retQP)]:
            totals[pool]['stake'] += stake
            totals[pool]['return'] += ret
            totals_by_meeting[mk][pool]['stake'] += stake
            totals_by_meeting[mk][pool]['return'] += ret

    def finalize(t):
        st = t['stake']
        rt = t['return']
        return {'stake': st, 'return': rt, 'profit': rt-st, 'roi': (rt-st)/st if st else None}

    out = {
        'meetings': [{'racedate': r, 'venue': v} for r, v, _ in meetings],
        'stake_unit': args.stake,
        'races_tested': len(per_race),
        'total': {k: finalize(v) for k,v in totals.items()},
        'by_meeting': {
            f'{r}|{v}': {k: finalize(vv) for k, vv in pools.items()}
            for (r, v), pools in totals_by_meeting.items()
        },
        'per_race': [
            {
                'racedate': rd, 'venue': venue, 'race_no': race_no,
                'pair': pair, 'q_div': q_div, 'qp_div': qp_div,
                'url': url
            }
            for rd, venue, race_no, pair, q_div, qp_div, url in sorted(per_race)
        ]
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(json.dumps({'out': args.out, 'meetings': len(meetings), 'races_tested': len(per_race)}, indent=2))


if __name__ == '__main__':
    main()
