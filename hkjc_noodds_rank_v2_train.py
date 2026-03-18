#!/usr/bin/env python3
"""Train NO-ODDS ranker v2 (research only).

Goals
- Do NOT touch/overwrite any production models.
- Use ONLY pre-race info derivable from SQLite (no odds).
- Improve features (recency windows + deltas) and use learning-to-rank within each race.

Output
- models/HKJC-ML_NOODDS_RANK_LGBM_v2.txt
- models/HKJC-ML_NOODDS_RANK_LGBM_v2.infermeta.json

Usage
  python3 hkjc_noodds_rank_v2_train.py --db hkjc.sqlite --years 3

Notes
- Labels are relevance for LambdaRank: rel = max(0, 6 - finish_pos) (winner=5 ... 5th=1, else 0)
- Train/test is time-split: lastNdays (default 180) for test.
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
                race_id=int(x["race_id"]),
                racedate=x["racedate"],
                venue=x["venue"],
                race_no=int(x["race_no"]),
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


def relevance_from_finish_pos(pos: int) -> int:
    if pos is None:
        return 0
    try:
        pos = int(pos)
    except Exception:
        return 0
    return max(0, 6 - pos)  # 1->5, 2->4, 3->3, 4->2, 5->1, else 0


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


class FeatureBuilder:
    def __init__(self, con: sqlite3.Connection):
        self.con = con
        self._cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        self._cache_role: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        self._cache_last: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._draw_bias_cache: Dict[Tuple[str, int, int], float] = {}

    def role_rates(self, role_field: str, name: str, asof_ymd: str, window_days: int) -> Dict[str, float]:
        # cache key: (role_field, name, asof_ordinal, window_days)
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
        # SQLite date offset string
        offset = f"-{int(window_days)} day"
        row = self.con.execute(q, (name, asof_ymd, asof_ymd, offset)).fetchone()
        starts = int(row["starts"] or 0)
        wins = int(row["wins"] or 0)
        top3 = int(row["top3"] or 0)
        out = {
            "starts": starts,
            "win_rate": safe_div(wins, starts),
            "top3_rate": safe_div(top3, starts),
        }
        self._cache_role[key] = out
        return out

    def horse_rates(self, horse_code: str, asof_ymd: str, window_days: int, venue: str = None, distance_m: int = None) -> Dict[str, float]:
        if not horse_code or not asof_ymd:
            return {"starts": 0, "win_rate": 0.0, "top3_rate": 0.0}
        key = (horse_code, asof_ymd, window_days, venue or "", int(distance_m or 0))
        if key in self._cache:
            return self._cache[key]

        wh = ["ru.horse_code = ?", "m.racedate < ?", "m.racedate >= strftime('%Y/%m/%d', date(replace(?, '/', '-') , ?))"]
        params: List[Any] = [horse_code, asof_ymd, asof_ymd, f"-{int(window_days)} day"]
        if venue:
            wh.append("m.venue = ?")
            params.append(venue)
        if distance_m:
            # bucket distance to reduce sparsity (e.g. within 200m)
            wh.append("ABS(r.distance_m - ?) <= 200")
            params.append(int(distance_m))

        q = """
        SELECT
          COUNT(1) AS starts,
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
        out = {
            "starts": starts,
            "win_rate": safe_div(wins, starts),
            "top3_rate": safe_div(top3, starts),
        }
        self._cache[key] = out
        return out

    def last_run(self, horse_code: str, asof_ymd: str) -> Dict[str, Any]:
        if not horse_code or not asof_ymd:
            return {"days_since": None, "last_finish_pos": None, "last_distance_m": None, "last_venue": None, "last_weight": None}
        key = (horse_code, asof_ymd)
        if key in self._cache_last:
            return self._cache_last[key]
        q = """
        SELECT m.racedate, m.venue, r.distance_m, ru.weight, re.finish_pos
        FROM runners ru
        JOIN races r ON r.race_id = ru.race_id
        JOIN meetings m ON m.meeting_id = r.meeting_id
        JOIN results re ON re.runner_id = ru.runner_id
        WHERE ru.horse_code = ?
          AND m.racedate < ?
        ORDER BY m.racedate DESC
        LIMIT 1
        """
        row = self.con.execute(q, (horse_code, asof_ymd)).fetchone()
        if not row:
            out = {"days_since": None, "last_finish_pos": None, "last_distance_m": None, "last_venue": None, "last_weight": None}
        else:
            days = (parse_ymd(asof_ymd) - parse_ymd(row["racedate"])).days
            out = {
                "days_since": days,
                "last_finish_pos": int(row["finish_pos"] or 99),
                "last_distance_m": int(row["distance_m"] or 0),
                "last_venue": row["venue"],
                "last_weight": float(row["weight"] or 0.0),
            }
        self._cache_last[key] = out
        return out

    def draw_bias_top3(self, venue: str, distance_m: int, draw: int, asof_ymd: str, lookback_days: int = 365 * 3) -> float:
        # historical top3 rate for same venue + within 200m distance, by draw
        if not venue or not distance_m or not draw or not asof_ymd:
            return 0.0
        key = (venue, int(distance_m), int(draw))
        if key in self._draw_bias_cache:
            return self._draw_bias_cache[key]
        q = """
        SELECT
          COUNT(1) AS starts,
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

    def build(self, rr: RunnerRow) -> Dict[str, Any]:
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

        last = self.last_run(rr.horse_code, asof)
        dist_diff = None
        if last.get("last_distance_m"):
            dist_diff = rr.distance_m - int(last["last_distance_m"]) 
        wt_diff = None
        if last.get("last_weight") is not None:
            wt_diff = rr.weight - float(last["last_weight"])

        draw_bias = self.draw_bias_top3(rr.venue, rr.distance_m, rr.draw, asof)

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
            # bias
            "draw_bias_top3": draw_bias,
        }
        return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--years', type=int, default=3)
    ap.add_argument('--testDays', type=int, default=180)
    ap.add_argument('--outModel', default='models/HKJC-ML_NOODDS_RANK_LGBM_v2.txt')
    ap.add_argument('--outMeta', default='models/HKJC-ML_NOODDS_RANK_LGBM_v2.infermeta.json')
    args = ap.parse_args()

    con = connect(args.db)
    mn, mx = get_date_bounds(con)
    end = parse_ymd(mx)
    start = end - timedelta(days=365 * int(args.years))
    start_ymd = fmt_ymd(start)

    rows = load_runners_in_range(con, start_ymd, mx)
    if not rows:
        raise SystemExit('No training rows found')

    fb = FeatureBuilder(con)
    # build X/y/group + keep race date for time split
    feats = []
    y = []
    groups = []
    dates = []

    cur_race = None
    cur_group = 0

    for rr in rows:
        if cur_race != rr.race_id:
            if cur_group:
                groups.append(cur_group)
            cur_race = rr.race_id
            cur_group = 0
        f = fb.build(rr)
        feats.append(f)
        y.append(relevance_from_finish_pos(rr.finish_pos))
        dates.append(rr.racedate)
        cur_group += 1
    if cur_group:
        groups.append(cur_group)

    feat_keys = sorted(feats[0].keys())
    X = np.array([[f.get(k) for k in feat_keys] for f in feats], dtype=float)

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

    # time split: last testDays
    split_date = fmt_ymd(end - timedelta(days=int(args.testDays)))
    is_test = np.array([d >= split_date for d in dates])
    X_train, X_test = X[~is_test], X[is_test]
    y_train, y_test = np.array(y)[~is_test], np.array(y)[is_test]

    # rebuild group arrays for train/test (need per-race)
    # We can reconstruct by iterating again with race boundaries.
    def build_groups(mask: np.ndarray) -> List[int]:
        out = []
        idx = 0
        gi = 0
        for g in groups:
            sel = mask[idx:idx+g]
            n = int(sel.sum())
            if n:
                out.append(n)
            idx += g
            gi += 1
        return out

    g_train = build_groups(~is_test)
    g_test = build_groups(is_test)

    train_set = lgb.Dataset(X_train, label=y_train, group=g_train)
    test_set = lgb.Dataset(X_test, label=y_test, group=g_test, reference=train_set)

    params = {
        'objective': 'lambdarank',
        'metric': ['ndcg'],
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.05,
        'num_leaves': 63,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l2': 1.0,
        'verbosity': -1,
        'seed': 42,
    }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, test_set],
        valid_names=['train', 'test'],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )

    os.makedirs(os.path.dirname(args.outModel), exist_ok=True)
    booster.save_model(args.outModel)

    meta = {
        'name': 'HKJC-ML_NOODDS_RANK_LGBM_v2',
        'kind': 'lgbm_lambdarank',
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
    }

    with open(args.outMeta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps({'ok': True, 'outModel': args.outModel, 'outMeta': args.outMeta, 'best_iteration': booster.best_iteration, 'train_rows': int(X_train.shape[0]), 'test_rows': int(X_test.shape[0])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
