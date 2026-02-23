import argparse, json, math, os, pickle, hashlib, csv
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
    ap.add_argument('--overlayPkl', default='models/OVERLAY_TRAIN_20230901_20250731_v6b_prev3.pkl')
    ap.add_argument('--rankerBundle', default='models/Q_RANKER_v7_NOODDS_top6/bundle.json')
    ap.add_argument('--rankerTxt', default='models/Q_RANKER_v7_NOODDS_top6/ranker.txt')
    ap.add_argument('--winplaceCache', default='hkjc_dividend_cache_WINPLACE_byurl.json')
    ap.add_argument('--qCache', default='hkjc_dividend_cache_Qmap_byurl.json')

    ap.add_argument('--testStart', default='2025/09/01')
    ap.add_argument('--testEnd', default='2026/02/22')

    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=0.0)

    ap.add_argument('--thr1', type=float, default=0.16)
    ap.add_argument('--stake1', type=float, default=10.0)
    ap.add_argument('--thr2', type=float, default=0.18)
    ap.add_argument('--stake2', type=float, default=20.0)
    ap.add_argument('--thr3', type=float, default=0.20)
    ap.add_argument('--stake3', type=float, default=50.0)

    ap.add_argument('--stakeQ', type=float, default=10.0)

    ap.add_argument('--outDir', default='reports/PROD_BACKTEST_TIERED_WQ_2025-09_to_2026-02-22')
    args = ap.parse_args()

    os.makedirs(args.outDir, exist_ok=True)

    # load dividend caches
    winplace_cache = json.load(open(args.winplaceCache, 'r', encoding='utf-8')) if args.winplaceCache else {}
    q_cache = json.load(open(args.qCache, 'r', encoding='utf-8')) if args.qCache else {}

    # load models
    overlay_bundle = pickle.load(open(args.overlayPkl, 'rb'))
    feat_w = overlay_bundle['feat_keys']
    mdl_win = overlay_bundle['mdl_win']
    mdl_pl = overlay_bundle.get('mdl_pl')

    rank_bundle = json.load(open(args.rankerBundle, 'r', encoding='utf-8'))
    feat_q = rank_bundle['feat_keys']
    ranker = lgb.Booster(model_file=args.rankerTxt)

    rows6 = load_jsonl(args.dataset_v6b)
    rows7 = load_jsonl(args.dataset_v7)

    rows6_test = [r for r in rows6 if args.testStart <= r.get('racedate') <= args.testEnd]
    rows7_test = [r for r in rows7 if args.testStart <= r.get('racedate') <= args.testEnd]

    races6 = group_by_race(rows6_test)
    races7 = group_by_race(rows7_test)

    # per-race selection via overlay model
    race_picks = []
    runner_dump = []

    for key, runners in sorted(races6.items()):
        rd, venue, rn = key
        url = f'https://racing.hkjc.com/zh-hk/local/information/localresults?racedate={rd}&Racecourse={venue}&RaceNo={int(rn)}'
        h = sha1(url)

        p_mkt = market_probs_for_race(runners)
        X = np.asarray([[safe_float(r.get(k), 0.0) for k in feat_w] for r in runners], dtype=np.float32)
        pred_res_w = mdl_win.predict(X)
        pred_res_p = mdl_pl.predict(X) if (mdl_pl is not None and args.beta != 0.0) else np.zeros(len(runners), dtype=np.float32)

        scored = []
        for r, pm, rw, rp in zip(runners, p_mkt, pred_res_w, pred_res_p):
            score_win = logit(pm) + float(args.alpha) * float(rw)
            overlay = float(args.alpha) * float(rw) + float(args.beta) * float(rp)
            scored.append({
                'horse_no': int(r['horse_no']),
                'horse_code': r.get('horse_code'),
                'horse_name_zh': r.get('horse_name_zh'),
                'cur_win_odds': safe_float(r.get('cur_win_odds'), None),
                'p_mkt_win': float(pm),
                'score_win': float(score_win),
                'overlay': float(overlay),
                'y_finish_pos': (int(r['y_finish_pos']) if r.get('y_finish_pos') is not None else None),
            })

        scored.sort(key=lambda x: x['score_win'], reverse=True)
        for i, rr in enumerate(scored, start=1):
            rr['overlay_rank'] = i

        top1 = scored[0]

        # decide tiered stake
        ov = float(top1['overlay'])
        stake_w = 0.0
        tier = None
        if ov > args.thr3:
            stake_w = float(args.stake3)
            tier = f">{args.thr3}"
        elif ov > args.thr2:
            stake_w = float(args.stake2)
            tier = f">{args.thr2}"
        elif ov > args.thr1:
            stake_w = float(args.stake1)
            tier = f">{args.thr1}"

        # win dividend
        win_div = (winplace_cache.get(h, {}).get('win', {}) or {}).get(str(top1['horse_no']))
        if win_div is None:
            win_div = (winplace_cache.get(h, {}).get('win', {}) or {}).get(top1['horse_no'])
        win_div = float(win_div) if win_div is not None else None

        # Q partners from ranker (v7 rows)
        runners7 = races7.get(key) or []
        q_partner2 = q_partner3 = None
        q12 = q13 = None
        if runners7:
            Xq = np.asarray([[safe_float(r.get(k), 0.0) for k in feat_q] for r in runners7], dtype=np.float32)
            q_sc = ranker.predict(Xq)
            q_scored = sorted(
                [{'horse_no': int(r['horse_no']), 'horse_name_zh': r.get('horse_name_zh'), 'ranker_score': float(s)} for r, s in zip(runners7, q_sc)],
                key=lambda x: x['ranker_score'],
                reverse=True,
            )
            # join ranker_score into scored list
            rk_map = {int(x['horse_no']): x['ranker_score'] for x in q_scored}
            for rr in scored:
                rr['ranker_score'] = rk_map.get(int(rr['horse_no']))

            anchor = int(top1['horse_no'])
            partners = [x for x in q_scored if int(x['horse_no']) != anchor]
            if len(partners) >= 2:
                q_partner2 = partners[0]
                q_partner3 = partners[1]
                p12 = pair(anchor, q_partner2['horse_no'])
                p13 = pair(anchor, q_partner3['horse_no'])
                q12 = q_cache.get(h, {}).get(p12)
                q13 = q_cache.get(h, {}).get(p13)
                q12 = float(q12) if q12 is not None else None
                q13 = float(q13) if q13 is not None else None

        # runner dump (all runners)
        for rr in scored:
            runner_dump.append({
                'racedate': rd,
                'venue': venue,
                'race_no': int(rn),
                'url': url,
                'horse_no': rr['horse_no'],
                'horse_name_zh': rr.get('horse_name_zh'),
                'cur_win_odds': rr.get('cur_win_odds'),
                'p_mkt_win': rr.get('p_mkt_win'),
                'score_win': rr.get('score_win'),
                'overlay': rr.get('overlay'),
                'overlay_rank': rr.get('overlay_rank'),
                'ranker_score': rr.get('ranker_score'),
                'y_finish_pos': rr.get('y_finish_pos'),
                'is_top1': rr['horse_no'] == top1['horse_no'],
            })

        # race-level record
        race_picks.append({
            'racedate': rd,
            'venue': venue,
            'race_no': int(rn),
            'url': url,
            'top1_horse_no': top1['horse_no'],
            'top1_overlay': top1['overlay'],
            'top1_score_win': top1['score_win'],
            'top1_p_mkt_win': top1['p_mkt_win'],
            'top1_odds': top1['cur_win_odds'],
            'stake_w': stake_w,
            'stake_w_tier': tier,
            'win_div': win_div,
            'return_w': float(win_div or 0.0) if stake_w > 0 else 0.0,
            'profit_w': (float(win_div or 0.0) - stake_w) if stake_w > 0 else 0.0,
            'did_bet_w': bool(stake_w > 0),

            'q_partner2_no': (q_partner2 or {}).get('horse_no'),
            'q_partner2_ranker_score': (q_partner2 or {}).get('ranker_score'),
            'q_partner3_no': (q_partner3 or {}).get('horse_no'),
            'q_partner3_ranker_score': (q_partner3 or {}).get('ranker_score'),
            'q_div_12': q12,
            'q_div_13': q13,
            'stake_q_total': (2.0 * args.stakeQ) if stake_w > 0 else 0.0,
            'return_q_total': (float(q12 or 0.0) + float(q13 or 0.0)) if stake_w > 0 else 0.0,
            'profit_q_total': ((float(q12 or 0.0) + float(q13 or 0.0)) - (2.0 * args.stakeQ)) if stake_w > 0 else 0.0,
        })

    # summarize
    w_stake = sum(r['stake_w'] for r in race_picks)
    w_return = sum(float(r['return_w']) for r in race_picks)
    q_stake = sum(float(r['stake_q_total']) for r in race_picks)
    q_return = sum(float(r['return_q_total']) for r in race_picks)

    out = {
        'generatedAt': __import__('datetime').datetime.now().isoformat(timespec='seconds'),
        'ranges': {'test': {'start': args.testStart, 'end': args.testEnd}},
        'models': {
            'W': {'model': 'OVERLAY_RESIDUAL_LGBM_v1', 'bundle': args.overlayPkl, 'alpha': args.alpha, 'beta': args.beta},
            'Q': {'model': rank_bundle.get('model'), 'bundle': args.rankerBundle, 'modelTxt': args.rankerTxt},
        },
        'strategy': {
            'W_tiers': [
                {'overlay_gt': args.thr1, 'stake': args.stake1},
                {'overlay_gt': args.thr2, 'stake': args.stake2},
                {'overlay_gt': args.thr3, 'stake': args.stake3},
            ],
            'Q': {'stake_each': args.stakeQ, 'bets_per_race': 2, 'condition': 'only when W bet placed (overlay>thr1)'}
        },
        'summary': {
            'W': {
                'races_bet': sum(1 for r in race_picks if r['did_bet_w']),
                'stake': w_stake,
                'return': w_return,
                'profit': w_return - w_stake,
                'roi': ((w_return - w_stake) / w_stake) if w_stake else None,
            },
            'Q': {
                'races_bet': sum(1 for r in race_picks if r['stake_q_total'] > 0),
                'bets': sum(2 for r in race_picks if r['stake_q_total'] > 0),
                'stake': q_stake,
                'return': q_return,
                'profit': q_return - q_stake,
                'roi': ((q_return - q_stake) / q_stake) if q_stake else None,
            },
            'TOTAL': {
                'stake': w_stake + q_stake,
                'return': w_return + q_return,
                'profit': (w_return + q_return) - (w_stake + q_stake),
                'roi': (((w_return + q_return) - (w_stake + q_stake)) / (w_stake + q_stake)) if (w_stake + q_stake) else None,
                'races_bet': sum(1 for r in race_picks if r['did_bet_w']),
                'races_seen': len(race_picks),
            }
        },
        'races': race_picks,
    }

    out_json = os.path.join(args.outDir, 'report.json')
    out_races_csv = os.path.join(args.outDir, 'races.csv')
    out_runners_csv = os.path.join(args.outDir, 'runners.csv')

    json.dump(out, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    # write races csv
    race_cols = list(race_picks[0].keys()) if race_picks else []
    with open(out_races_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=race_cols)
        w.writeheader()
        w.writerows(race_picks)

    # write runners csv
    runner_cols = list(runner_dump[0].keys()) if runner_dump else []
    with open(out_runners_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=runner_cols)
        w.writeheader()
        w.writerows(runner_dump)

    print(json.dumps({'ok': True, 'outDir': args.outDir, 'report': out_json, 'races': len(race_picks), 'runnerRows': len(runner_dump)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
