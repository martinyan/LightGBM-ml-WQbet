import json, re, math, argparse
from collections import defaultdict
from urllib.request import Request, urlopen
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xgboost as xgb

UA = 'openclaw-hkjc-gbdt-backtest/1.0'


def fetch_html(url: str) -> str:
    req = Request(url, headers={'User-Agent': UA})
    with urlopen(req, timeout=6) as resp:
        return resp.read().decode('utf-8', errors='ignore')


def clean_cell(html: str) -> str:
    s = re.sub(r'<br\s*/?>', ' ', html, flags=re.I)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = s.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&quot;', '"').replace('&#39;', "'")
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def parse_dividends(html: str):
    tables = re.findall(r'<table[\s\S]*?</table>', html, flags=re.I)
    div_table = None
    for t in tables:
        if '派彩' in t and '彩池' in t and '勝出組合' in t:
            div_table = t
            break
    if not div_table:
        return {'win': {}, 'place': {}}

    trs = re.findall(r'<tr[^>]*>([\s\S]*?)</tr>', div_table, flags=re.I)
    rows = []
    for tr in trs:
        cells = [clean_cell(x) for x in re.findall(r'<t[dh][^>]*>([\s\S]*?)</t[dh]>', tr, flags=re.I)]
        if len(cells) >= 2:
            rows.append(cells)

    win = {}
    place = {}
    mode = None
    for r in rows:
        if r[0] == '彩池':
            continue
        if len(r) >= 3:
            pool, combo, divtxt = r[0], r[1], r[2]
            divm = re.sub(r'[^0-9.]', '', divtxt)
            div = float(divm) if divm else None
            mode = None
            if not div:
                continue
            if pool == '獨贏':
                hn = re.search(r'(\d+)', combo)
                if hn:
                    win[hn.group(1)] = div
            elif pool == '位置':
                hn = re.search(r'(\d+)', combo)
                if hn:
                    place[hn.group(1)] = div
                mode = 'PLACE_CONT'
        elif len(r) == 2 and mode == 'PLACE_CONT':
            hn = re.search(r'(\d+)', r[0] or '')
            divm = re.sub(r'[^0-9.]', '', r[1] or '')
            div = float(divm) if divm else None
            if hn and div:
                place[hn.group(1)] = div
        else:
            mode = None

    return {'win': win, 'place': place}


def load_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def group_races(rows):
    by = defaultdict(list)
    for r in rows:
        key = (r['racedate'], r['venue'], int(r['race_no']))
        by[key].append(r)
    return by


def top1_in_top3(race_rows):
    races = 0
    hits = 0
    for key, arr in race_rows.items():
        races += 1
        top1 = max(arr, key=lambda x: x['_p'])
        if int(top1['y_place']) == 1:
            hits += 1
    return races, hits, hits / races if races else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--data', required=True)
    ap.add_argument('--meetings', type=int, default=20)
    ap.add_argument('--stake', type=float, default=10.0)
    ap.add_argument('--out', default='hkjc_gbdt_roi_lastN.json')
    args = ap.parse_args()

    # 1) choose last N meetings from sqlite
    con = sqlite3.connect(args.db)
    cur = con.cursor()
    cur.execute("select racedate, venue from meetings order by racedate desc, venue desc limit ?", (args.meetings,))
    meeting_keys = [(r, v) for r, v in cur.fetchall()]
    meeting_set = set(meeting_keys)

    # 2) load dataset
    rows = load_jsonl(args.data)

    label_key = 'y_place'
    meta = {
        'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no',
        'y_finish_pos','y_win','y_place','cur_jockey','cur_trainer','cur_surface'
    }
    feat_keys = sorted([k for k in rows[0].keys() if k not in meta and k != label_key])

    def vec(r):
        return [float(r.get(k, 0) or 0) for k in feat_keys]

    train = [r for r in rows if (r['racedate'], r['venue']) not in meeting_set]
    test = [r for r in rows if (r['racedate'], r['venue']) in meeting_set]

    X_train = np.asarray([vec(r) for r in train], dtype=np.float32)
    y_train = np.asarray([int(r.get(label_key, 0) or 0) for r in train], dtype=np.int32)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,
        'eta': 0.05,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'lambda': 1.0,
        'alpha': 0.0,
        'seed': 42,
    }

    booster = xgb.train(params, xgb.DMatrix(X_train, label=y_train, feature_names=feat_keys), num_boost_round=800)

    # predict test
    X_test = np.asarray([vec(r) for r in test], dtype=np.float32)
    p = booster.predict(xgb.DMatrix(X_test, feature_names=feat_keys))
    for rr, pp in zip(test, p):
        rr['_p'] = float(pp)

    races = group_races(test)

    # 3) compute ROI strategies with thresholds
    thresholds = [None, 0.40, 0.45, 0.50, 0.55, 0.60]

    totals = {}
    for thr in thresholds:
        key = 'all' if thr is None else f'p>={thr:.2f}'
        totals[key] = {
            'win_top1': {'stake': 0.0, 'return': 0.0, 'bets': 0},
            'place_top1': {'stake': 0.0, 'return': 0.0, 'bets': 0},
            'win1_place3_top1': {'stake': 0.0, 'return': 0.0, 'bets': 0},
        }

    # Precompute top1 picks per race
    race_list = []
    for (racedate, venue, race_no), runners in sorted(races.items()):
        runners_sorted = sorted(runners, key=lambda x: x['_p'], reverse=True)
        top1r = runners_sorted[0]
        top1 = str(int(top1r['horse_no']))
        url = f'https://racing.hkjc.com/zh-hk/local/information/localresults?racedate={racedate}&Racecourse={venue}&RaceNo={race_no}'
        race_list.append((racedate, venue, race_no, top1, float(top1r['_p']), url))

    # Fetch dividends in parallel (this was the bottleneck)
    def fetch_div_for_race(item):
        racedate, venue, race_no, top1, p1, url = item
        html = fetch_html(url)
        divs = parse_dividends(html)
        return (racedate, venue, race_no, top1, p1, divs)

    fetched = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(fetch_div_for_race, it) for it in race_list]
        done = 0
        for fut in as_completed(futs):
            try:
                fetched.append(fut.result())
            except Exception:
                continue
            done += 1
            if done % 50 == 0:
                print(json.dumps({'progress_fetch': done, 'of': len(futs)}))

    # Apply strategies
    for racedate, venue, race_no, top1, p1, divs in fetched:
        win_div = divs['win'].get(top1)
        place_div = divs['place'].get(top1)

        for thr in thresholds:
            if thr is not None and p1 < thr:
                continue
            k = 'all' if thr is None else f'p>={thr:.2f}'

            # WIN top1
            totals[k]['win_top1']['stake'] += args.stake
            totals[k]['win_top1']['return'] += (win_div if win_div is not None else 0.0)
            totals[k]['win_top1']['bets'] += 1

            # PLACE top1
            totals[k]['place_top1']['stake'] += args.stake
            totals[k]['place_top1']['return'] += (place_div if place_div is not None else 0.0)
            totals[k]['place_top1']['bets'] += 1

            # 1x WIN + 3x PLACE top1
            totals[k]['win1_place3_top1']['stake'] += args.stake * 4
            ret = 0.0
            if win_div is not None:
                ret += win_div
            if place_div is not None:
                ret += 3 * place_div
            totals[k]['win1_place3_top1']['return'] += ret
            totals[k]['win1_place3_top1']['bets'] += 1

    # extra: hit metric
    races_count, top1_hits, top1_rate = top1_in_top3(races)

    out = {
        'meetings_used': args.meetings,
        'meeting_keys': meeting_keys,
        'train_rows': len(train),
        'test_rows': len(test),
        'test_races': races_count,
        'top1_in_top3_hits': top1_hits,
        'top1_in_top3_rate': top1_rate,
        'strategies': {}
    }

    for k, strat in totals.items():
        out['strategies'][k] = {}
        for sname, v in strat.items():
            profit = v['return'] - v['stake']
            roi = (profit / v['stake']) if v['stake'] else None
            out['strategies'][k][sname] = {
                'bets': v['bets'],
                'stake': v['stake'],
                'return': v['return'],
                'profit': profit,
                'roi': roi,
            }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(json.dumps({'out': args.out, 'test_races': races_count}, indent=2))


if __name__ == '__main__':
    main()
