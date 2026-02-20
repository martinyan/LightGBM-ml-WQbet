import re, json, argparse, sqlite3, datetime
from collections import defaultdict
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xgboost as xgb

UA = 'openclaw-hkjc-season-sweep/1.0'


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
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--start', required=True, help='YYYY/MM/DD')
    ap.add_argument('--end', required=True, help='YYYY/MM/DD')
    ap.add_argument('--thresholds', default='1.8,1.85,1.9,1.92')
    ap.add_argument('--stake', type=float, default=10.0)
    ap.add_argument('--maxWorkers', type=int, default=24)
    ap.add_argument('--out', default='hkjc_q_qp_season_sweep.json')
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(',') if x.strip()]

    # meetings in range
    con = sqlite3.connect(args.db)
    cur = con.cursor()
    cur.execute('select racedate, venue, meeting_id from meetings where racedate>=? and racedate<=? order by racedate asc, venue asc', (args.start, args.end))
    meetings = [(r,v,mid) for r,v,mid in cur.fetchall()]
    meeting_set = {(r,v) for r,v,_ in meetings}

    # race list for meetings
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

    # Build race-level decisions (top2 pair + p12)
    races = []
    for (rd, venue, race_no), arr in by_race.items():
        if race_no not in meeting_races.get((rd, venue), []):
            continue
        arr_sorted = sorted(arr, key=lambda t: t[1], reverse=True)
        if len(arr_sorted) < 2:
            continue
        h1 = int(arr_sorted[0][0]['horse_no']); p1 = float(arr_sorted[0][1])
        h2 = int(arr_sorted[1][0]['horse_no']); p2 = float(arr_sorted[1][1])
        pair = f"{min(h1,h2)}-{max(h1,h2)}"
        url = f'https://racing.hkjc.com/zh-hk/local/information/localresults?racedate={rd}&Racecourse={venue}&RaceNo={race_no}'
        races.append({'racedate': rd, 'venue': venue, 'race_no': race_no, 'pair': pair, 'p1': p1, 'p2': p2, 'p12': p1+p2, 'url': url})

    # Prefetch dividends ONCE per race (for the picked pair)
    def fetch_one(r):
        html = fetch_html(r['url'])
        divs = parse_dividends_Q_QP(html)
        pair = r['pair']
        return {
            **r,
            'q_div': divs['Q'].get(pair),
            'qp_div': divs['QP'].get(pair),
            'qp_offered': divs['hasQP']
        }

    fetched = []
    with ThreadPoolExecutor(max_workers=args.maxWorkers) as ex:
        futs = [ex.submit(fetch_one, r) for r in races]
        done = 0
        for fut in as_completed(futs):
            try:
                fetched.append(fut.result())
            except Exception:
                continue
            done += 1
            if done % 100 == 0:
                print(json.dumps({'progress_fetch': done, 'of': len(futs)}))

    # Sweep thresholds without more network calls
    results = []
    for th in thresholds:
        sel = [r for r in fetched if r['p12'] > th]
        stake_q = len(sel) * args.stake
        ret_q = sum((r['q_div'] or 0.0) for r in sel)

        # QP: if pool not offered, treat as NO BET for QP
        sel_qp = [r for r in sel if r.get('qp_offered')]
        stake_qp = len(sel_qp) * args.stake
        ret_qp = sum((r['qp_div'] or 0.0) for r in sel_qp)

        hit_q = sum(1 for r in sel if r.get('q_div') is not None)
        hit_qp = sum(1 for r in sel_qp if r.get('qp_div') is not None)

        comb_stake = stake_q + stake_qp
        comb_ret = ret_q + ret_qp

        results.append({
            'threshold': th,
            'races_q_bet': len(sel),
            'races_qp_bet': len(sel_qp),
            'Q': {**finalize(stake_q, ret_q), 'hits': hit_q},
            'QP': {**finalize(stake_qp, ret_qp), 'hits': hit_qp},
            'combined': finalize(comb_stake, comb_ret)
        })

    results_sorted = sorted(results, key=lambda r: (r['combined']['roi'] if r['combined']['roi'] is not None else -1e9), reverse=True)

    out = {
        'range': {'start': args.start, 'end': args.end},
        'meetings': [{'racedate': r, 'venue': v} for r,v,_ in meetings],
        'races_total': len(fetched),
        'stake_unit': args.stake,
        'thresholds': thresholds,
        'results': results,
        'results_sorted_by_combined_roi': results_sorted,
        'generatedAt': datetime.datetime.now().isoformat(timespec='seconds'),
        'note': 'NO-ODDS Top2 inference; QP stake is only counted for races where QP pool is offered on HKJC dividends table.'
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(json.dumps({'out': args.out, 'meetings': len(meetings), 'races': len(fetched)}, indent=2))


if __name__ == '__main__':
    main()
