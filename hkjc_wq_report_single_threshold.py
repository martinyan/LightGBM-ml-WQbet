import os, json, math, hashlib, argparse
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


def group_by_meeting(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r['racedate'], r['venue'])].append(r)
    return by


def list_meetings_sqlite(db_path, start, end):
    import sqlite3
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        'select racedate, venue, meeting_id from meetings where racedate>=? and racedate<=? order by racedate asc, venue asc',
        (start, end),
    )
    rows = [(r, v, mid) for (r, v, mid) in cur.fetchall()]
    con.close()
    return rows


def market_probs_for_race(runners):
    p_raw = []
    for rr in runners:
        o = safe_float(rr.get('cur_win_odds'), None)
        p_raw.append(0.0 if (o is None or o <= 0) else 1.0 / o)
    s = sum(p_raw)
    if s <= 0:
        return [1.0 / len(runners)] * len(runners)
    return [x / s for x in p_raw]


def train_overlay_win_model(rows_fit, feat_keys):
    races = group_by_race(rows_fit)
    X = []
    y = []
    for _, runners in races.items():
        p_mkt = market_probs_for_race(runners)
        for rr, pm in zip(runners, p_mkt):
            fp = rr.get('y_finish_pos')
            if fp is None:
                continue
            y_win = 1.0 if int(fp) == 1 else 0.0
            res_win = y_win - float(pm)
            X.append([safe_float(rr.get(k), 0.0) for k in feat_keys])
            y.append(res_win)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    mdl = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
    )
    mdl.fit(X, y)
    return mdl


def score_test_races(rows_test, feat_keys, mdl_win, alpha=1.0):
    races = group_by_race(rows_test)
    out = {}
    for key, runners in races.items():
        p_mkt = market_probs_for_race(runners)
        Xr = np.asarray([[safe_float(r.get(k), 0.0) for k in feat_keys] for r in runners], dtype=np.float32)
        pred_res = mdl_win.predict(Xr)
        scored = []
        for r, pm, rw in zip(runners, p_mkt, pred_res):
            score_win = logit(pm) + alpha * float(rw)
            overlay = alpha * float(rw)
            scored.append((int(r['horse_no']), score_win, overlay))
        scored.sort(key=lambda t: t[1], reverse=True)
        if not scored:
            continue
        hn1, sc1, ov1 = scored[0]
        hn2, sc2, ov2 = scored[1] if len(scored) > 1 else (None, None, None)
        rd, venue, rn = key
        url = f'https://racing.hkjc.com/zh-hk/local/information/localresults?racedate={rd}&Racecourse={venue}&RaceNo={int(rn)}'
        out[key] = {
            'racedate': rd,
            'venue': venue,
            'race_no': int(rn),
            'url': url,
            'top1_horse_no': hn1,
            'top1_score_win': float(sc1),
            'top1_overlay': float(ov1),
            'top2_horse_no': (int(hn2) if hn2 is not None else None),
            'top2_score_win': (float(sc2) if sc2 is not None else None),
            'top2_overlay': (float(ov2) if ov2 is not None else None),
        }
    return out


def train_ranker_for_q(rows_train_v7, feat_keys_v7):
    races = group_by_race(rows_train_v7)
    X = []
    y = []
    group = []
    for _, runners in sorted(races.items()):
        group.append(len(runners))
        for r in runners:
            fp = r.get('y_finish_pos')
            if fp is None:
                continue
            X.append([safe_float(r.get(k), 0.0) for k in feat_keys_v7])
            y.append(1.0 if int(fp) == 1 else 0.0)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mdl = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        ndcg_eval_at=[1, 3],
        learning_rate=0.05,
        n_estimators=800,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
    )
    mdl.fit(X, y, group=group)
    return mdl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--dataset_v6b', default='hkjc_dataset_v6b_code_include_debut_jt60_prev3_fullcols.jsonl')
    ap.add_argument('--dataset_v7', default='hkjc_dataset_v7_jt60_prev3_fullcols_fieldranks.jsonl')
    ap.add_argument('--trainStart', default='2023/09/01')
    ap.add_argument('--trainEnd', default='2025/07/31')
    ap.add_argument('--testStart', default='2025/09/01')
    ap.add_argument('--testEnd', default='2026/02/22')
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--stakeWin', type=float, default=10.0)
    ap.add_argument('--stakeQ', type=float, default=10.0)
    ap.add_argument('--threshold', type=float, default=0.14)
    ap.add_argument('--winplaceCache', default='hkjc_dividend_cache_WINPLACE_byurl.json')
    ap.add_argument('--qCache', default='hkjc_dividend_cache_Qmap_byurl.json')
    ap.add_argument('--outJson', default='reports/WQ_REPORT_2025-09_to_2026-02-22/wq_report_thr0p14.json')
    ap.add_argument('--outCsv', default='reports/WQ_REPORT_2025-09_to_2026-02-22/wq_report_thr0p14.csv')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.outJson), exist_ok=True)

    rows6 = load_jsonl(args.dataset_v6b)
    by_meeting6 = group_by_meeting(rows6)

    train_meetings = list_meetings_sqlite(args.db, args.trainStart, args.trainEnd)
    n_train = len(train_meetings)
    n_eval = max(1, int(round(n_train * 0.2)))
    fit_meetings = train_meetings[:-n_eval]

    def build_rows(meeting_list):
        s = {(r, v) for (r, v, _) in meeting_list}
        out = []
        for k in s:
            out.extend(by_meeting6.get(k, []))
        return out

    fit_rows = build_rows(fit_meetings)
    test_rows = [r for r in rows6 if args.testStart <= r.get('racedate') <= args.testEnd]

    meta = {'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no','y_finish_pos','y_win','y_place','cur_jockey','cur_trainer'}
    cand = [k for k in fit_rows[0].keys() if k not in meta and not k.startswith('y_') and not k.startswith('_')]
    feat_keys = sorted([k for k in cand if k not in ('cur_win_odds',)])

    mdl_win = train_overlay_win_model(fit_rows, feat_keys)
    test_scores = score_test_races(test_rows, feat_keys, mdl_win, alpha=args.alpha)

    winplace_cache = json.load(open(args.winplaceCache, 'r', encoding='utf-8')) if args.winplaceCache else {}
    q_cache = json.load(open(args.qCache, 'r', encoding='utf-8')) if args.qCache else {}

    # train ranker once (v7)
    rows7 = load_jsonl(args.dataset_v7)
    train7 = [r for r in rows7 if args.trainStart <= r['racedate'] <= args.trainEnd and r.get('y_finish_pos') is not None]
    meta7 = {'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no','y_finish_pos','y_win','y_place','cur_jockey','cur_trainer'}
    cand7 = [k for k in train7[0].keys() if k not in meta7 and not k.startswith('y_') and not k.startswith('_')]
    feat7 = []
    for k in sorted(cand7):
        v = train7[0].get(k)
        if isinstance(v, (int, float)) or v is None:
            feat7.append(k)

    ranker = train_ranker_for_q(train7, feat7)
    by_race7 = group_by_race([r for r in rows7 if args.testStart <= r['racedate'] <= args.testEnd])

    rows_out = []
    w_bets = w_hits = 0
    w_stake = w_return = 0.0
    q_bets = q_hits_any = 0
    q_stake = q_return = 0.0

    for key, r in sorted(test_scores.items()):
        if float(r['top1_overlay']) <= args.threshold:
            continue

        # W
        url = r['url']
        h = sha1(url)
        anchor = int(r['top1_horse_no'])
        win_div = (winplace_cache.get(h, {}).get('win', {}) or {}).get(str(anchor))
        if win_div is None:
            win_div = (winplace_cache.get(h, {}).get('win', {}) or {}).get(anchor)
        win_div = float(win_div) if win_div is not None else None

        w_bets += 1
        w_stake += args.stakeWin
        if win_div is not None:
            w_hits += 1
            w_return += win_div

        # Q partners (2 combos: anchor with top2/top3 by ranker)
        runners7 = by_race7.get(key) or []
        p2 = p3 = None
        q12 = q13 = None

        if runners7:
            Xr = np.asarray([[safe_float(rr.get(k), 0.0) for k in feat7] for rr in runners7], dtype=np.float32)
            sc = ranker.predict(Xr)
            scored = sorted([(int(rr['horse_no']), float(s)) for rr, s in zip(runners7, sc)], key=lambda t: t[1], reverse=True)
            partners = [hn for hn, _ in scored if hn != anchor]
            if len(partners) >= 2:
                p2, p3 = int(partners[0]), int(partners[1])
                p12 = pair(anchor, p2)
                p13 = pair(anchor, p3)
                q12 = q_cache.get(h, {}).get(p12)
                q13 = q_cache.get(h, {}).get(p13)
                q12 = float(q12) if q12 is not None else None
                q13 = float(q13) if q13 is not None else None

        # we still count Q stake if we have partners
        q_stake_r = 0.0
        q_return_r = 0.0
        hit_any = False
        if p2 is not None and p3 is not None:
            q_bets += 2
            q_stake_r = 2.0 * args.stakeQ
            q_stake += q_stake_r
            q_return_r = float(q12 or 0.0) + float(q13 or 0.0)
            q_return += q_return_r
            hit_any = (q12 is not None and q12 > 0) or (q13 is not None and q13 > 0)
            if hit_any:
                q_hits_any += 1

        rows_out.append({
            'racedate': r['racedate'],
            'venue': r['venue'],
            'race_no': int(r['race_no']),
            'url': url,
            'overlay': float(r['top1_overlay']),
            'anchor_horse_no': anchor,
            'win_div': win_div,
            'q_partner2': p2,
            'q_partner3': p3,
            'q_div_12': q12,
            'q_div_13': q13,
            'stake_win': args.stakeWin,
            'stake_q_total': q_stake_r,
            'stake_total': args.stakeWin + q_stake_r,
            'return_win': float(win_div or 0.0),
            'return_q_total': q_return_r,
            'return_total': float(win_div or 0.0) + q_return_r,
            'profit_total': float(win_div or 0.0) + q_return_r - (args.stakeWin + q_stake_r),
            'q_hit_any': bool(hit_any),
        })

    rep = {
        'ranges': {'train': {'start': args.trainStart, 'end': args.trainEnd}, 'test': {'start': args.testStart, 'end': args.testEnd}},
        'params': {'alpha': args.alpha, 'threshold': args.threshold, 'stakeWin': args.stakeWin, 'stakeQ': args.stakeQ},
        'summary': {
            'w': {
                'bets': w_bets,
                'wins': w_hits,
                'hit_rate': (w_hits / w_bets) if w_bets else None,
                'stake': w_stake,
                'return': w_return,
                'profit': w_return - w_stake,
                'roi': (w_return - w_stake) / w_stake if w_stake else None,
            },
            'q': {
                'q_bets': q_bets,
                'races': q_hits_any + (0 if q_hits_any==0 else 0),
                'hits_any': q_hits_any,
                'stake': q_stake,
                'return': q_return,
                'profit': q_return - q_stake,
                'roi': (q_return - q_stake) / q_stake if q_stake else None,
            },
            'total': {
                'stake': w_stake + q_stake,
                'return': w_return + q_return,
                'profit': (w_return + q_return) - (w_stake + q_stake),
                'roi': ((w_return + q_return) - (w_stake + q_stake)) / (w_stake + q_stake) if (w_stake + q_stake) else None,
                'races': len(rows_out),
            }
        },
        'bets': rows_out,
    }

    with open(args.outJson, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    # CSV
    import csv
    with open(args.outCsv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else [])
        if rows_out:
            w.writeheader()
            w.writerows(rows_out)

    print(json.dumps({'ok': True, 'outJson': args.outJson, 'outCsv': args.outCsv, 'races': len(rows_out)}, indent=2))


if __name__ == '__main__':
    main()
