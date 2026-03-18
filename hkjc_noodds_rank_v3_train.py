#!/usr/bin/env python3
"""Train NO-ODDS ranker v3 (research) with sectional + pace proxies.

Design goals
- Research-only: DO NOT overwrite production models.
- No odds.
- Add sectional features from SQLite tables: sectionals / sectional_splits.
- Add simple race-level pace proxy from running-style indicators.
- Use learning-to-rank within each race.

Output
- models/HKJC-ML_NOODDS_RANK_LGBM_v3_TOP1.txt
- models/HKJC-ML_NOODDS_RANK_LGBM_v3_TOP1.infermeta.json

Usage
  python3 hkjc_noodds_rank_v3_train.py --db hkjc.sqlite --years 3

Label
- Relevance is winner-heavy (optimize Top1) while still rewarding top3/top5:
  1st=10, 2nd=4, 3rd=2, 4th=1, 5th=1, else 0

"""

import argparse, json, os, sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any, List

import numpy as np
import lightgbm as lgb


def parse_ymd(s: str) -> datetime:
    return datetime.strptime(s, "%Y/%m/%d")


def fmt_ymd(d: datetime) -> str:
    return d.strftime("%Y/%m/%d")


@dataclass
class RunnerRow:
    runner_id: int
    race_id: int
    racedate: str
    venue: str
    race_no: int
    distance_m: int
    class_num: int
    surface: str
    horse_code: str
    horse_no: int
    draw: int
    weight: float
    jockey: str
    trainer: str
    finish_pos: int


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def get_date_bounds(con: sqlite3.Connection) -> Tuple[str, str]:
    r = con.execute("SELECT MIN(racedate) AS mn, MAX(racedate) AS mx FROM meetings").fetchone()
    return r["mn"], r["mx"]


def load_runners_in_range(con: sqlite3.Connection, start_ymd: str, end_ymd: str) -> List[RunnerRow]:
    q = """
    SELECT
      ru.runner_id,
      ru.race_id, m.racedate, m.venue,
      r.race_no, r.distance_m, r.class_num, r.surface,
      ru.horse_code, ru.horse_no, ru.draw, ru.weight, ru.jockey, ru.trainer,
      re.finish_pos
    FROM runners ru
    JOIN races r ON r.race_id = ru.race_id
    JOIN meetings m ON m.meeting_id = r.meeting_id
    JOIN results re ON re.runner_id = ru.runner_id
    WHERE m.racedate >= ? AND m.racedate <= ?
      AND re.finish_pos IS NOT NULL
    ORDER BY m.racedate, ru.race_id, ru.horse_no
    """
    rows = []
    for x in con.execute(q, (start_ymd, end_ymd)).fetchall():
        rows.append(
            RunnerRow(
                runner_id=int(x["runner_id"]),
                race_id=int(x["race_id"]),
                racedate=x["racedate"],
                venue=x["venue"],
                race_no=int(x["race_no"] or 0),
                distance_m=int(x["distance_m"] or 0),
                class_num=int(x["class_num"] or 0),
                surface=x["surface"] or "",
                horse_code=x["horse_code"] or "",
                horse_no=int(x["horse_no"] or 0),
                draw=int(x["draw"] or 0),
                weight=float(x["weight"] or 0.0),
                jockey=x["jockey"] or "",
                trainer=x["trainer"] or "",
                finish_pos=int(x["finish_pos"] or 99),
            )
        )
    return rows


def relevance_top1_heavy(pos: int) -> int:
    try:
        p = int(pos)
    except Exception:
        return 0
    if p == 1:
        return 10
    if p == 2:
        return 4
    if p == 3:
        return 2
    if p in (4, 5):
        return 1
    return 0


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


class FeatureBuilder:
    def __init__(self, con: sqlite3.Connection):
        self.con = con
        self._cache_role: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {}
        self._cache_horse: Dict[Tuple[str, int, int, str, int], Dict[str, Any]] = {}
        self._cache_last: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._cache_prevN_sec: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        self._draw_bias_cache: Dict[Tuple[str, int, int, int], float] = {}

    def role_rates(self, role_field: str, name: str, asof_ymd: str, window_days: int) -> Dict[str, float]:
        if not name or not asof_ymd:
            return {"starts": 0, "win_rate": 0.0, "top3_rate": 0.0}
        key = (role_field, name, parse_ymd(asof_ymd).toordinal(), window_days)
        if key in self._cache_role:
            return self._cache_role[key]
        q = f"""
        SELECT
          COUNT(1) AS starts,
          SUM(CASE WHEN re.finish_pos = 1 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN re.finish_pos <= 3 THEN 1 ELSE 0 END) AS top3
        FROM runners ru
        JOIN races r ON r.race_id = ru.race_id
        JOIN meetings m ON m.meeting_id = r.meeting_id
        JOIN results re ON re.runner_id = ru.runner_id
        WHERE ru.{role_field} = ?
          AND m.racedate < ?
          AND m.racedate >= strftime('%Y/%m/%d', date(replace(?, '/', '-') , ?))
        """
        row = self.con.execute(q, (name, asof_ymd, asof_ymd, f"-{int(window_days)} day")).fetchone()
        starts = int(row["starts"] or 0)
        wins = int(row["wins"] or 0)
        top3 = int(row["top3"] or 0)
        out = {"starts": starts, "win_rate": safe_div(wins, starts), "top3_rate": safe_div(top3, starts)}
        self._cache_role[key] = out
        return out

    def horse_rates(self, horse_code: str, asof_ymd: str, window_days: int, venue: str = None, distance_m: int = None) -> Dict[str, float]:
        if not horse_code or not asof_ymd:
            return {"starts": 0, "win_rate": 0.0, "top3_rate": 0.0}
        key = (horse_code, parse_ymd(asof_ymd).toordinal(), window_days, venue or "", int(distance_m or 0))
        if key in self._cache_horse:
            return self._cache_horse[key]

        wh = ["ru.horse_code = ?", "m.racedate < ?", "m.racedate >= strftime('%Y/%m/%d', date(replace(?, '/', '-') , ?))"]
        params: List[Any] = [horse_code, asof_ymd, asof_ymd, f"-{int(window_days)} day"]
        if venue:
            wh.append("m.venue = ?")
            params.append(venue)
        if distance_m:
            wh.append("ABS(r.distance_m - ?) <= 200")
            params.append(int(distance_m))

        q = """
        SELECT COUNT(1) AS starts,
               SUM(CASE WHEN re.finish_pos = 1 THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN re.finish_pos <= 3 THEN 1 ELSE 0 END) AS top3
        FROM runners ru
        JOIN races r ON r.race_id = ru.race_id
        JOIN meetings m ON m.meeting_id = r.meeting_id
        JOIN results re ON re.runner_id = ru.runner_id
        WHERE """ + " AND ".join(wh)

        row = self.con.execute(q, params).fetchone()
        starts = int(row["starts"] or 0)
        wins = int(row["wins"] or 0)
        top3 = int(row["top3"] or 0)
        out = {"starts": starts, "win_rate": safe_div(wins, starts), "top3_rate": safe_div(top3, starts)}
        self._cache_horse[key] = out
        return out

    def last_run_with_sectionals(self, horse_code: str, asof_ymd: str) -> Dict[str, Any]:
        if not horse_code or not asof_ymd:
            return {}
        key = (horse_code, parse_ymd(asof_ymd).toordinal())
        if key in self._cache_last:
            return self._cache_last[key]
        q = """
        SELECT m.racedate, m.venue, r.distance_m, ru.weight, re.finish_pos,
               se.pos1, se.pos3, se.seg2_time, se.seg3_time, se.kick_time
        FROM runners ru
        JOIN races r ON r.race_id = ru.race_id
        JOIN meetings m ON m.meeting_id = r.meeting_id
        JOIN results re ON re.runner_id = ru.runner_id
        LEFT JOIN sectionals se ON se.runner_id = ru.runner_id
        WHERE ru.horse_code = ?
          AND m.racedate < ?
        ORDER BY m.racedate DESC
        LIMIT 1
        """
        row = self.con.execute(q, (horse_code, asof_ymd)).fetchone()
        if not row:
            out = {}
        else:
            days = (parse_ymd(asof_ymd) - parse_ymd(row["racedate"])).days
            out = {
                "days_since": days,
                "last_finish_pos": int(row["finish_pos"] or 99),
                "last_distance_m": int(row["distance_m"] or 0),
                "last_venue": row["venue"],
                "last_weight": float(row["weight"] or 0.0),
                # sectional
                "last_pos1": row["pos1"],
                "last_pos3": row["pos3"],
                "last_seg2": row["seg2_time"],
                "last_seg3": row["seg3_time"],
                "last_kick": row["kick_time"],
            }
        self._cache_last[key] = out
        return out

    def prevN_sectionals(self, horse_code: str, asof_ymd: str, N: int = 3) -> Dict[str, Any]:
        if not horse_code or not asof_ymd:
            return {"n": 0}
        key = (horse_code, parse_ymd(asof_ymd).toordinal(), N)
        if key in self._cache_prevN_sec:
            return self._cache_prevN_sec[key]
        q = """
        SELECT se.pos1, se.pos3, se.seg2_time, se.seg3_time, se.kick_time
        FROM runners ru
        JOIN races r ON r.race_id = ru.race_id
        JOIN meetings m ON m.meeting_id = r.meeting_id
        JOIN results re ON re.runner_id = ru.runner_id
        JOIN sectionals se ON se.runner_id = ru.runner_id
        WHERE ru.horse_code = ?
          AND m.racedate < ?
          AND se.kick_time IS NOT NULL
        ORDER BY m.racedate DESC
        LIMIT ?
        """
        rows = self.con.execute(q, (horse_code, asof_ymd, int(N))).fetchall()
        if not rows:
            out = {"n": 0}
        else:
            def avg(col):
                xs=[float(r[col]) for r in rows if r[col] is not None]
                return float(np.mean(xs)) if xs else None
            out = {
                "n": len(rows),
                "avg_pos1": avg("pos1"),
                "avg_pos3": avg("pos3"),
                "avg_seg2": avg("seg2_time"),
                "avg_seg3": avg("seg3_time"),
                "avg_kick": avg("kick_time"),
            }
        self._cache_prevN_sec[key] = out
        return out

    def draw_bias_top3(self, venue: str, distance_m: int, draw: int, asof_ymd: str, lookback_days: int = 365 * 3) -> float:
        if not venue or not distance_m or not draw or not asof_ymd:
            return 0.0
        key = (venue, int(distance_m), int(draw), parse_ymd(asof_ymd).toordinal())
        if key in self._draw_bias_cache:
            return self._draw_bias_cache[key]
        q = """
        SELECT COUNT(1) AS starts,
               SUM(CASE WHEN re.finish_pos <= 3 THEN 1 ELSE 0 END) AS top3
        FROM runners ru
        JOIN races r ON r.race_id = ru.race_id
        JOIN meetings m ON m.meeting_id = r.meeting_id
        JOIN results re ON re.runner_id = ru.runner_id
        WHERE m.venue = ?
          AND m.racedate < ?
          AND m.racedate >= strftime('%Y/%m/%d', date(replace(?, '/', '-') , ?))
          AND ABS(r.distance_m - ?) <= 200
          AND ru.draw = ?
        """
        row = self.con.execute(q, (venue, asof_ymd, asof_ymd, f"-{int(lookback_days)} day", int(distance_m), int(draw))).fetchone()
        starts = int(row["starts"] or 0)
        top3 = int(row["top3"] or 0)
        rate = safe_div(top3, starts)
        self._draw_bias_cache[key] = rate
        return rate

    def build_base(self, rr: RunnerRow) -> Dict[str, Any]:
        asof = rr.racedate
        # recency windows
        h365 = self.horse_rates(rr.horse_code, asof, 365)
        h60 = self.horse_rates(rr.horse_code, asof, 60)
        h365_v = self.horse_rates(rr.horse_code, asof, 365, venue=rr.venue)
        h365_d = self.horse_rates(rr.horse_code, asof, 365, distance_m=rr.distance_m)

        j365 = self.role_rates("jockey", rr.jockey, asof, 365)
        j60 = self.role_rates("jockey", rr.jockey, asof, 60)
        t365 = self.role_rates("trainer", rr.trainer, asof, 365)
        t60 = self.role_rates("trainer", rr.trainer, asof, 60)

        last = self.last_run_with_sectionals(rr.horse_code, asof)
        prevN = self.prevN_sectionals(rr.horse_code, asof, N=3)

        dist_diff = None
        if last.get("last_distance_m"):
            dist_diff = rr.distance_m - int(last["last_distance_m"])
        wt_diff = None
        if last.get("last_weight") is not None:
            wt_diff = rr.weight - float(last["last_weight"])

        draw_bias = self.draw_bias_top3(rr.venue, rr.distance_m, rr.draw, asof)

        # running style proxies from prevN sectionals
        avg_pos1 = prevN.get('avg_pos1')
        avg_pos3 = prevN.get('avg_pos3')
        style_front = 1 if (avg_pos1 is not None and avg_pos1 <= 3) else 0
        style_closer = 1 if (avg_pos1 is not None and avg_pos1 >= 8 and avg_pos3 is not None and avg_pos3 <= avg_pos1 - 2) else 0

        f = {
            # race context
            "distance_m": rr.distance_m,
            "class_num": rr.class_num,
            "venue_HV": 1 if rr.venue == "HV" else 0,
            "venue_ST": 1 if rr.venue == "ST" else 0,
            "surface_turf": 1 if (rr.surface or "").lower().startswith("t") else 0,
            "surface_awt": 1 if (rr.surface or "").lower().startswith("a") else 0,
            # runner basics
            "draw": rr.draw,
            "weight": rr.weight,
            # horse historical
            "h_starts_365": h365["starts"],
            "h_winrate_365": h365["win_rate"],
            "h_top3rate_365": h365["top3_rate"],
            "h_starts_60": h60["starts"],
            "h_top3rate_60": h60["top3_rate"],
            "h_top3rate_venue_365": h365_v["top3_rate"],
            "h_top3rate_dist_365": h365_d["top3_rate"],
            # jockey/trainer
            "j_starts_365": j365["starts"],
            "j_top3rate_365": j365["top3_rate"],
            "j_top3rate_60": j60["top3_rate"],
            "t_starts_365": t365["starts"],
            "t_top3rate_365": t365["top3_rate"],
            "t_top3rate_60": t60["top3_rate"],
            # last run deltas
            "days_since_last": last.get("days_since"),
            "last_finish_pos": last.get("last_finish_pos"),
            "dist_diff_from_last": dist_diff,
            "wt_diff_from_last": wt_diff,
            "last_same_venue": 1 if (last.get("last_venue") == rr.venue and last.get("last_venue")) else 0,
            # draw bias
            "draw_bias_top3": draw_bias,
            # sectionals (last)
            "last_pos1": last.get('last_pos1'),
            "last_pos3": last.get('last_pos3'),
            "last_seg2": last.get('last_seg2'),
            "last_seg3": last.get('last_seg3'),
            "last_kick": last.get('last_kick'),
            # sectionals (prevN avg)
            "prevN_sec_n": prevN.get('n'),
            "prevN_avg_pos1": avg_pos1,
            "prevN_avg_pos3": avg_pos3,
            "prevN_avg_seg2": prevN.get('avg_seg2'),
            "prevN_avg_seg3": prevN.get('avg_seg3'),
            "prevN_avg_kick": prevN.get('avg_kick'),
            # style flags (for race pace calc)
            "style_front": style_front,
            "style_closer": style_closer,
        }
        return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--years', type=int, default=3)
    ap.add_argument('--testDays', type=int, default=180)
    ap.add_argument('--outModel', default='models/HKJC-ML_NOODDS_RANK_LGBM_v3_TOP1.txt')
    ap.add_argument('--outMeta', default='models/HKJC-ML_NOODDS_RANK_LGBM_v3_TOP1.infermeta.json')
    args = ap.parse_args()

    con = connect(args.db)
    mn, mx = get_date_bounds(con)
    end = parse_ymd(mx)
    start = end - timedelta(days=365 * int(args.years))
    start_ymd = fmt_ymd(start)

    runners = load_runners_in_range(con, start_ymd, mx)
    if not runners:
        raise SystemExit('No training rows found')

    fb = FeatureBuilder(con)

    # build features per runner first
    raw_feats: List[Dict[str, Any]] = []
    y: List[int] = []
    dates: List[str] = []
    race_ids: List[int] = []

    for rr in runners:
        raw_feats.append(fb.build_base(rr))
        y.append(relevance_top1_heavy(rr.finish_pos))
        dates.append(rr.racedate)
        race_ids.append(rr.race_id)

    # add race-level pace features using style flags within each race
    # pace_index = fraction of front-style runners in field
    # and interactions with runner style
    from collections import defaultdict
    by_race_idx = defaultdict(list)
    for i, rid in enumerate(race_ids):
        by_race_idx[int(rid)].append(i)

    for rid, idxs in by_race_idx.items():
        n = len(idxs)
        n_front = sum(1 for i in idxs if raw_feats[i].get('style_front'))
        pace_index = (n_front / n) if n else 0.0
        for i in idxs:
            raw_feats[i]['field_size'] = n
            raw_feats[i]['pace_index_front_frac'] = pace_index
            raw_feats[i]['closer_x_pace'] = (raw_feats[i].get('style_closer') or 0) * pace_index
            raw_feats[i]['front_x_slow'] = (raw_feats[i].get('style_front') or 0) * (1.0 - pace_index)

    # drop style flags from training features (keep interactions + pace)
    for f in raw_feats:
        f.pop('style_front', None)
        f.pop('style_closer', None)

    feat_keys = sorted(raw_feats[0].keys())
    X = np.array([[f.get(k) for k in feat_keys] for f in raw_feats], dtype=float)

    # impute means
    impute = {}
    for i, k in enumerate(feat_keys):
        col = X[:, i]
        m = np.nanmean(col)
        if np.isnan(m):
            m = 0.0
        impute[k] = float(m)
        col[np.isnan(col)] = m
        X[:, i] = col

    split_date = fmt_ymd(end - timedelta(days=int(args.testDays)))
    is_test = np.array([d >= split_date for d in dates])
    X_train, X_test = X[~is_test], X[is_test]
    y_train, y_test = np.array(y)[~is_test], np.array(y)[is_test]

    # group arrays
    # compute group sizes for full set
    groups = []
    cur = None
    cnt = 0
    for rid in race_ids:
        if cur != rid:
            if cnt:
                groups.append(cnt)
            cur = rid
            cnt = 0
        cnt += 1
    if cnt:
        groups.append(cnt)

    def build_groups(mask: np.ndarray) -> List[int]:
        out = []
        idx = 0
        for g in groups:
            nsel = int(mask[idx:idx+g].sum())
            if nsel:
                out.append(nsel)
            idx += g
        return out

    g_train = build_groups(~is_test)
    g_test = build_groups(is_test)

    train_set = lgb.Dataset(X_train, label=y_train, group=g_train)
    test_set = lgb.Dataset(X_test, label=y_test, group=g_test, reference=train_set)

    params = {
        'objective': 'lambdarank',
        'metric': ['ndcg'],
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03,
        'num_leaves': 127,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l2': 2.0,
        'verbosity': -1,
        'seed': 42,
    }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=4000,
        valid_sets=[train_set, test_set],
        valid_names=['train', 'test'],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(50)],
    )

    os.makedirs(os.path.dirname(args.outModel), exist_ok=True)
    booster.save_model(args.outModel)

    meta = {
        'name': 'HKJC-ML_NOODDS_RANK_LGBM_v3_TOP1',
        'kind': 'lgbm_lambdarank',
        'trainedAt': datetime.now().isoformat(timespec='seconds'),
        'trained_on': {
            'db': os.path.abspath(args.db),
            'date_min_db': mn,
            'date_max_db': mx,
            'train_start': start_ymd,
            'train_end': mx,
            'test_split_date': split_date,
            'years': args.years,
            'testDays': args.testDays,
        },
        'featKeys': feat_keys,
        'imputeMeans': impute,
        'params': params,
        'best_iteration': booster.best_iteration,
        'label': 'top1-heavy (1st=10, 2nd=4, 3rd=2, 4-5=1)'
    }
    with open(args.outMeta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps({'ok': True, 'outModel': args.outModel, 'outMeta': args.outMeta, 'best_iteration': booster.best_iteration, 'train_rows': int(X_train.shape[0]), 'test_rows': int(X_test.shape[0])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
