import os, json, math, pickle, argparse, hashlib, csv
from collections import defaultdict

import numpy as np
import lightgbm as lgb


def safe_float(x, d=0.0):
    try:
        if x is None or x == '':
            return d
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ('turf', 't'):
                return 0.0
            if s in ('awt', 'all weather', 'all-weather', 'a'):
                return 1.0
        return float(x)
    except Exception:
        return d


def logit(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def pair(a: int, b: int) -> str:
    x = min(int(a), int(b))
    y = max(int(a), int(b))
    return f"{x}-{y}"


def load_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def group_by_race(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r['racedate'], r['venue'], int(r['race_no']))].append(r)
    return by


def market_probs_for_race(runners):
    p_raw = []
    for rr in runners:
        o = safe_float(rr.get('cur_win_odds'), None)
        p_raw.append(0.0 if (o is None or o <= 0) else 1.0 / o)
    s = sum(p_raw)
    if s <= 0:
        return [1.0 / len(runners)] * len(runners)
    return [x / s for x in p_raw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_v6b', default='hkjc_dataset_v6b_code_include_debut_jt60_prev3_fullcols.jsonl')
    ap.add_argument('--dataset_v7', default='hkjc_dataset_v7_jt60_prev3_fullcols_fieldranks.jsonl')
    ap.add_argument('--wBundle', default='models/OVERLAY_RESIDUAL_LGBM_v1_PROD_22bets_thr0p2.pkl')
    ap.add_argument('--qBundle', default='models/Q_RANKER_v7_PROD_FEB22_111ROI/bundle.json')
    ap.add_argument('--qModelTxt', default='models/Q_RANKER_v7_PROD_FEB22_111ROI/ranker.txt')
    ap.add_argument('--winplaceCache', default='hkjc_dividend_cache_WINPLACE_byurl.json')
    ap.add_argument('--qCache', default='hkjc_dividend_cache_Qmap_byurl.json')

    ap.add_argument('--testStart', default='2025/09/01')
    ap.add_argument('--testEnd', default='2026/02/22')

    ap.add_argument('--thrStart', type=float, default=0.02)
    ap.add_argument('--thrEnd', type=float, default=0.20)
    ap.add_argument('--thrStep', type=float, default=0.02)

    ap.add_argument('--stakeW', type=float, default=10.0)
    ap.add_argument('--stakeQ', type=float, default=10.0)

    ap.add_argument('--outDir', default='reports/PROD_SWEEP_WQ_THRESH_0p02_0p20')
    args = ap.parse_args()

    os.makedirs(args.outDir, exist_ok=True)

    winplace_cache = json.load(open(args.winplaceCache, 'r', encoding='utf-8')) if args.winplaceCache else {}
    q_cache = json.load(open(args.qCache, 'r', encoding='utf-8')) if args.qCache else {}

    # load models
    w_bundle = pickle.load(open(args.wBundle, 'rb'))
    feat_w = w_bundle['feat_keys']
    mdl_win = w_bundle['mdl_win']

    qb = json.load(open(args.qBundle, 'r', encoding='utf-8'))
    feat_q = qb['feat_keys']
    ranker = lgb.Booster(model_file=args.qModelTxt)

    rows6 = load_jsonl(args.dataset_v6b)
    rows7 = load_jsonl(args.dataset_v7)

    rows6_test = [r for r in rows6 if args.testStart <= r.get('racedate') <= args.testEnd]
    rows7_test = [r for r in rows7 if args.testStart <= r.get('racedate') <= args.testEnd]

    races6 = group_by_race(rows6_test)
    races7 = group_by_race(rows7_test)

    # precompute per-race top1 overlay + score + url + win dividend (for speed)
    race_top = {}
    for key, runners in races6.items():
        rd, venue, rn = key
        url = f'https://racing.hkjc.com/zh-hk/local/information/localresults?racedate={rd}&Racecourse={venue}&RaceNo={int(rn)}'
        h = sha1(url)
        pm = market_probs_for_race(runners)
        X = np.asarray([[safe_float(r.get(k), 0.0) for k in feat_w] for r in runners], dtype=np.float32)
        pred_res = mdl_win.predict(X)
        scored = []
        for r, p, rw in zip(runners, pm, pred_res):
            score_win = logit(p) + float(rw)
            overlay = float(rw)
            scored.append((int(r['horse_no']), score_win, overlay))
        scored.sort(key=lambda t: t[1], reverse=True)
        if not scored:
            continue
        hn1, sc1, ov1 = scored[0]

        win_div = (winplace_cache.get(h, {}).get('win', {}) or {}).get(str(hn1))
        if win_div is None:
            win_div = (winplace_cache.get(h, {}).get('win', {}) or {}).get(hn1)
        win_div = float(win_div) if win_div is not None else None

        race_top[key] = {
            'racedate': rd,
            'venue': venue,
            'race_no': int(rn),
            'url': url,
            'top1_horse_no': int(hn1),
            'top1_score_win': float(sc1),
            'top1_overlay': float(ov1),
            'win_div': win_div,
        }

    # precompute ranker scores per race (for speed)
    race_rank = {}
    for key, runners in races7.items():
        if key not in race_top:
            continue
        Xq = np.asarray([[safe_float(r.get(k), 0.0) for k in feat_q] for r in runners], dtype=np.float32)
        sc = ranker.predict(Xq)
        scored = sorted([(int(r['horse_no']), float(s)) for r, s in zip(runners, sc)], key=lambda t: t[1], reverse=True)
        race_rank[key] = scored

    thr_values = []
    x = args.thrStart
    while x <= args.thrEnd + 1e-12:
        thr_values.append(round(x, 10))
        x += args.thrStep

    sweep = []

    for thr in thr_values:
        ledger = []
        bets_w = wins_w = 0
        stake_w = ret_w = 0.0

        stake_q = ret_q = 0.0
        hits_any = 0
        bets_q = 0

        for key, r in sorted(race_top.items()):
            if float(r['top1_overlay']) <= thr:
                continue

            # W
            bets_w += 1
            stake_w += args.stakeW
            if r['win_div'] is not None:
                wins_w += 1
                ret_w += float(r['win_div'])

            # Q
            ranked = race_rank.get(key) or []
            anchor = int(r['top1_horse_no'])
            partners = [hn for hn, _ in ranked if hn != anchor]
            p2 = partners[0] if len(partners) > 0 else None
            p3 = partners[1] if len(partners) > 1 else None

            q12 = q13 = None
            if p2 is not None and p3 is not None:
                h = sha1(r['url'])
                p12 = pair(anchor, p2)
                p13 = pair(anchor, p3)
                q12 = q_cache.get(h, {}).get(p12)
                q13 = q_cache.get(h, {}).get(p13)
                q12 = float(q12) if q12 is not None else None
                q13 = float(q13) if q13 is not None else None

            # always stake Q if we can form 2 combos
            stake_q_r = 0.0
            ret_q_r = 0.0
            hit = False
            if p2 is not None and p3 is not None:
                bets_q += 2
                stake_q_r = 2.0 * args.stakeQ
                ret_q_r = float(q12 or 0.0) + float(q13 or 0.0)
                hit = (q12 is not None and q12 > 0) or (q13 is not None and q13 > 0)

                stake_q += stake_q_r
                ret_q += ret_q_r
                if hit:
                    hits_any += 1

            ledger.append({
                **r,
                'threshold': thr,
                'stake_w': args.stakeW,
                'return_w': float(r['win_div'] or 0.0),
                'profit_w': float(r['win_div'] or 0.0) - args.stakeW,
                'q_anchor': anchor,
                'q_partner2': p2,
                'q_partner3': p3,
                'q_div_12': q12,
                'q_div_13': q13,
                'stake_q_total': stake_q_r,
                'return_q_total': ret_q_r,
                'profit_q_total': ret_q_r - stake_q_r,
                'is_hit_any': hit,
            })

        rep_w = {
            'bets': bets_w,
            'wins': wins_w,
            'hit_rate': (wins_w / bets_w) if bets_w else None,
            'stake': stake_w,
            'return': ret_w,
            'profit': ret_w - stake_w,
            'roi': ((ret_w - stake_w) / stake_w) if stake_w else None,
        }
        rep_q = {
            'races': bets_w,
            'bets': bets_q,
            'hits_any': hits_any,
            'hit_rate_any': (hits_any / bets_w) if bets_w else None,
            'stake': stake_q,
            'return': ret_q,
            'profit': ret_q - stake_q,
            'roi': ((ret_q - stake_q) / stake_q) if stake_q else None,
        }
        rep_t = {
            'stake': stake_w + stake_q,
            'return': ret_w + ret_q,
            'profit': (ret_w + ret_q) - (stake_w + stake_q),
            'roi': (((ret_w + ret_q) - (stake_w + stake_q)) / (stake_w + stake_q)) if (stake_w + stake_q) else None,
        }

        sweep.append({
            'threshold': thr,
            'w_bets': rep_w['bets'],
            'w_wins': rep_w['wins'],
            'w_hit_rate': rep_w['hit_rate'],
            'w_roi': rep_w['roi'],
            'q_races': rep_q['races'],
            'q_bets': rep_q['bets'],
            'q_hits_any': rep_q['hits_any'],
            'q_hit_rate_any': rep_q['hit_rate_any'],
            'q_roi': rep_q['roi'],
            'total_roi': rep_t['roi'],
            'total_profit': rep_t['profit'],
        })

        # write per-threshold ledger CSV
        led_path = os.path.join(args.outDir, f'ledger_thr_{thr:.2f}.csv'.replace('.', 'p'))
        if ledger:
            cols = list(ledger[0].keys())
            with open(led_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(ledger)

    out = {
        'generatedAt': __import__('datetime').datetime.now().isoformat(timespec='seconds'),
        'ranges': {'test': {'start': args.testStart, 'end': args.testEnd}},
        'models': {
            'W': {'bundle': args.wBundle},
            'Q': {'bundle': args.qBundle, 'modelTxt': args.qModelTxt},
        },
        'params': {
            'stakeW': args.stakeW,
            'stakeQ': args.stakeQ,
            'thrStart': args.thrStart,
            'thrEnd': args.thrEnd,
            'thrStep': args.thrStep,
        },
        'sweep': sweep,
    }

    out_json = os.path.join(args.outDir, 'sweep.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({'ok': True, 'outDir': args.outDir, 'rows': len(sweep), 'outJson': out_json}, indent=2))


if __name__ == '__main__':
    main()
