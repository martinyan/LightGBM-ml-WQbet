import re, json, argparse
from collections import defaultdict
from urllib.request import Request, urlopen

import numpy as np
import xgboost as xgb

UA = 'openclaw-hkjc-q-qp-backtest/1.0'


def fetch_html(url: str) -> str:
    req = Request(url, headers={'User-Agent': UA})
    with urlopen(req, timeout=15) as resp:
        return resp.read().decode('utf-8', errors='ignore')


def clean_cell(html: str) -> str:
    s = re.sub(r'<br\s*/?>', ' ', html, flags=re.I)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = s.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&quot;', '"').replace('&#39;', "'")
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def parse_dividends_Q_QP(html: str):
    # return dict for Q (quinella) and QP (quinella place)
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

    outQ = {}   # 'a-b' -> dividend
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
        # normalize combo to a-b
        m = re.findall(r'(\d+)', combo)
        if len(m) >= 2:
            a, b = m[0], m[1]
            key = f"{min(int(a),int(b))}-{max(int(a),int(b))}"
        else:
            continue

        if pool in ('連贏', 'Q'):
            outQ[key] = div
        elif pool in ('位置Q', 'QP'):
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
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--racedate', required=True, help='YYYY/MM/DD')
    ap.add_argument('--venue', required=True, help='ST|HV')
    ap.add_argument('--races', required=True, help='e.g. 8-11 or 1,2,3')
    ap.add_argument('--stake', type=float, default=10.0)
    args = ap.parse_args()

    # parse race range
    if '-' in args.races:
        a,b = [int(x) for x in args.races.split('-')]
        race_nos = list(range(a,b+1))
    else:
        race_nos = [int(x) for x in args.races.split(',') if x.strip()]

    rows = load_jsonl(args.data)
    sub = [r for r in rows if r.get('racedate') == args.racedate and r.get('venue') == args.venue and int(r.get('race_no')) in race_nos]
    if not sub:
        raise SystemExit('No rows for requested races')

    label_key = 'y_place'
    meta = {
        'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no',
        'y_finish_pos','y_win','y_place','cur_jockey','cur_trainer','cur_surface'
    }
    feat_keys = sorted([k for k in sub[0].keys() if k not in meta and k != label_key])

    def vec(r):
        return [float(r.get(k, 0) or 0) for k in feat_keys]

    bst = xgb.Booster()
    bst.load_model(args.model)

    X = np.asarray([vec(r) for r in sub], dtype=np.float32)
    p = bst.predict(xgb.DMatrix(X, feature_names=feat_keys))

    # group by race
    by_race = defaultdict(list)
    for r, pp in zip(sub, p):
        by_race[(r['racedate'], r['venue'], int(r['race_no']))].append((r, float(pp)))

    total = {
        'Q': {'stake': 0.0, 'return': 0.0},
        'QP': {'stake': 0.0, 'return': 0.0},
    }
    per_race = []

    for (rd, venue, race_no), arr in sorted(by_race.items()):
        arr_sorted = sorted(arr, key=lambda t: t[1], reverse=True)
        top2 = [str(int(arr_sorted[0][0]['horse_no'])), str(int(arr_sorted[1][0]['horse_no']))]
        a,b = int(top2[0]), int(top2[1])
        pair = f"{min(a,b)}-{max(a,b)}"

        url = f'https://racing.hkjc.com/zh-hk/local/information/localresults?racedate={rd}&Racecourse={venue}&RaceNo={race_no}'
        html = fetch_html(url)
        divs = parse_dividends_Q_QP(html)
        q_div = divs['Q'].get(pair)
        qp_div = divs['QP'].get(pair)

        stake = args.stake
        retQ = q_div if q_div is not None else 0.0
        retQP = qp_div if qp_div is not None else 0.0

        total['Q']['stake'] += stake
        total['Q']['return'] += retQ
        total['QP']['stake'] += stake
        total['QP']['return'] += retQP

        per_race.append({
            'race_no': race_no,
            'pair': pair,
            'q_div': q_div,
            'qp_div': qp_div,
            'stake': stake,
            'retQ': retQ,
            'retQP': retQP,
            'url': url,
        })

    out = {'racedate': args.racedate, 'venue': args.venue, 'races': race_nos, 'stake_unit': args.stake, 'per_race': per_race, 'total': {}}
    for k in ('Q','QP'):
        st = total[k]['stake']
        rt = total[k]['return']
        out['total'][k] = {'stake': st, 'return': rt, 'profit': rt-st, 'roi': (rt-st)/st if st else None}

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
