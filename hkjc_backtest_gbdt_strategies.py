import re, json, math, argparse
from collections import defaultdict
from urllib.request import Request, urlopen

import numpy as np
import xgboost as xgb

UA = 'openclaw-hkjc-gbdt-backtest/1.0'

def fetch_html(url: str) -> str:
    req = Request(url, headers={'User-Agent': UA})
    with urlopen(req, timeout=60) as resp:
        return resp.read().decode('utf-8', errors='ignore')


def clean_cell(html: str) -> str:
    s = re.sub(r'<br\s*/?>', ' ', html, flags=re.I)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = s.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&quot;', '"').replace('&#39;', "'")
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def parse_dividends(html: str):
    # Find dividends table by headers 彩池/勝出組合/派彩
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

    win = {}   # horseNo -> dividend
    place = {} # horseNo -> dividend

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--dates', required=True, help='comma-separated YYYY/MM/DD')
    ap.add_argument('--stake', type=float, default=10.0, help='base stake unit (HK dividend is per $10)')
    args = ap.parse_args()

    dates = [s.strip() for s in args.dates.split(',') if s.strip()]
    all_rows = load_jsonl(args.data)
    hold = [r for r in all_rows if r['racedate'] in dates]

    # feature keys
    label_key = 'y_place'
    meta = {
        'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no',
        'y_finish_pos','y_win','y_place','cur_jockey','cur_trainer','cur_surface'
    }
    feat_keys = sorted([k for k in hold[0].keys() if k not in meta and k != label_key])

    def vec(r):
        return [float(r.get(k, 0) or 0) for k in feat_keys]

    # load model
    bst = xgb.Booster()
    bst.load_model(args.model)

    # predict
    X = np.asarray([vec(r) for r in hold], dtype=np.float32)
    p = bst.predict(xgb.DMatrix(X, feature_names=feat_keys))
    for rr, pp in zip(hold, p):
        rr['_p'] = float(pp)

    races = group_races(hold)

    # strategies
    totals = defaultdict(lambda: {'stake': 0.0, 'return': 0.0})

    for (racedate, venue, race_no), runners in sorted(races.items()):
        # pick top1/top2
        runners_sorted = sorted(runners, key=lambda x: x['_p'], reverse=True)
        top1 = str(int(runners_sorted[0]['horse_no']))
        top2 = [str(int(x['horse_no'])) for x in runners_sorted[:2]]

        racecourse = venue  # ST/HV
        url = f'https://racing.hkjc.com/zh-hk/local/information/localresults?racedate={racedate}&Racecourse={racecourse}&RaceNo={race_no}'
        html = fetch_html(url)
        divs = parse_dividends(html)
        win_div = divs['win'].get(top1)
        place_div = divs['place'].get(top1)

        # S1: $10 WIN on top1
        stake = args.stake
        ret = win_div if win_div is not None else 0.0
        totals['win_top1']['stake'] += stake
        totals['win_top1']['return'] += ret

        # S2: $10 PLACE on top1
        stake = args.stake
        ret = place_div if place_div is not None else 0.0
        totals['place_top1']['stake'] += stake
        totals['place_top1']['return'] += ret

        # S3: $10 PLACE on top2 horses ($20)
        stake = args.stake * len(top2)
        ret = 0.0
        for h in top2:
            d = divs['place'].get(h)
            if d is not None:
                ret += d
        totals['place_top2']['stake'] += stake
        totals['place_top2']['return'] += ret

        # S4: 1x WIN + 3x PLACE on top1 ($40)
        stake = args.stake * 4
        ret = 0.0
        if win_div is not None:
            ret += win_div
        if place_div is not None:
            ret += 3 * place_div
        totals['win1_place3_top1']['stake'] += stake
        totals['win1_place3_top1']['return'] += ret

    out = {}
    for k, v in totals.items():
        profit = v['return'] - v['stake']
        out[k] = {
            'stake': v['stake'],
            'return': v['return'],
            'profit': profit,
            'roi': (profit / v['stake']) if v['stake'] else None,
        }

    print(json.dumps({'dates': dates, 'model': args.model, 'totals': out}, indent=2))


if __name__ == '__main__':
    main()
