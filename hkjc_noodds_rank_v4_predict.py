#!/usr/bin/env python3
"""Predict NO-ODDS Top5 using ranker v4 (exacta-tuned pace tilt).

- Base model: LGBM LambdaRank
- Post-adjustment using tuned alpha/beta:
    adj = raw + alpha*closer_x_pace - beta*front_x_fast

Usage:
  python3 hkjc_noodds_rank_v4_predict.py --db hkjc.sqlite --in merged.json --out pred.json
"""

import argparse, json, re, os, sqlite3
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import lightgbm as lgb


def parse_race_key(url: str) -> Dict[str, Any]:
    m = re.search(r"/racing/wp/(\d{4}-\d{2}-\d{2})/(HV|ST)/(\d+)", url or "")
    if not m:
        return {"racedate": None, "venue": None, "raceNo": None}
    return {"racedate": m.group(1).replace('-', '/'), "venue": m.group(2), "raceNo": int(m.group(3))}


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


class FeatureBuilder:
    # same as v3 predict, but outputs style flags for pace tilt
    def __init__(self, con: sqlite3.Connection):
        self.con = con
        self._cache_role = {}
        self._cache_horse = {}
        self._cache_last = {}
        self._cache_prevN_sec = {}
        self._draw_bias_cache = {}

    def role_rates(self, role_field: str, name: str, asof_ymd: str, window_days: int) -> Dict[str, float]:
        if not name or not asof_ymd:
            return {"starts": 0, "win_rate": 0.0, "top3_rate": 0.0}
        key = (role_field, name, asof_ymd, window_days)
        if key in self._cache_role:
            return self._cache_role[key]
        q = f"""
        SELECT COUNT(1) AS starts,
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
        starts = int(row['starts'] or 0)
        wins = int(row['wins'] or 0)
        top3 = int(row['top3'] or 0)
        out = {"starts": starts, "win_rate": safe_div(wins, starts), "top3_rate": safe_div(top3, starts)}
        self._cache_role[key] = out
        return out

    def horse_rates(self, horse_code: str, asof_ymd: str, window_days: int, venue: str = None, distance_m: int = None) -> Dict[str, float]:
        if not horse_code or not asof_ymd:
            return {"starts": 0, "win_rate": 0.0, "top3_rate": 0.0}
        key = (horse_code, asof_ymd, window_days, venue or '', int(distance_m or 0))
        if key in self._cache_horse:
            return self._cache_horse[key]
        wh = ["ru.horse_code = ?", "m.racedate < ?", "m.racedate >= strftime('%Y/%m/%d', date(replace(?, '/', '-') , ?))"]
        params: List[Any] = [horse_code, asof_ymd, asof_ymd, f"-{int(window_days)} day"]
        if venue:
            wh.append('m.venue = ?'); params.append(venue)
        if distance_m:
            wh.append('ABS(r.distance_m - ?) <= 200'); params.append(int(distance_m))
        q = """SELECT COUNT(1) AS starts,
                      SUM(CASE WHEN re.finish_pos = 1 THEN 1 ELSE 0 END) AS wins,
                      SUM(CASE WHEN re.finish_pos <= 3 THEN 1 ELSE 0 END) AS top3
               FROM runners ru
               JOIN races r ON r.race_id = ru.race_id
               JOIN meetings m ON m.meeting_id = r.meeting_id
               JOIN results re ON re.runner_id = ru.runner_id
               WHERE """ + ' AND '.join(wh)
        row = self.con.execute(q, params).fetchone()
        starts = int(row['starts'] or 0)
        wins = int(row['wins'] or 0)
        top3 = int(row['top3'] or 0)
        out = {"starts": starts, "win_rate": safe_div(wins, starts), "top3_rate": safe_div(top3, starts)}
        self._cache_horse[key] = out
        return out

    def last_run_with_sectionals(self, horse_code: str, asof_ymd: str) -> Dict[str, Any]:
        if not horse_code or not asof_ymd:
            return {}
        key = (horse_code, asof_ymd)
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
        WHERE ru.horse_code = ? AND m.racedate < ?
        ORDER BY m.racedate DESC
        LIMIT 1
        """
        row = self.con.execute(q, (horse_code, asof_ymd)).fetchone()
        if not row:
            out = {}
        else:
            days = (datetime.strptime(asof_ymd, '%Y/%m/%d') - datetime.strptime(row['racedate'], '%Y/%m/%d')).days
            out = {
                'days_since': days,
                'last_finish_pos': int(row['finish_pos'] or 99),
                'last_distance_m': int(row['distance_m'] or 0),
                'last_venue': row['venue'],
                'last_weight': float(row['weight'] or 0.0),
                'last_pos1': row['pos1'],
                'last_pos3': row['pos3'],
                'last_seg2': row['seg2_time'],
                'last_seg3': row['seg3_time'],
                'last_kick': row['kick_time'],
            }
        self._cache_last[key] = out
        return out

    def prevN_sectionals(self, horse_code: str, asof_ymd: str, N: int = 3) -> Dict[str, Any]:
        if not horse_code or not asof_ymd:
            return {'n': 0}
        key = (horse_code, asof_ymd, N)
        if key in self._cache_prevN_sec:
            return self._cache_prevN_sec[key]
        q = """
        SELECT se.pos1, se.pos3, se.seg2_time, se.seg3_time, se.kick_time
        FROM runners ru
        JOIN races r ON r.race_id = ru.race_id
        JOIN meetings m ON m.meeting_id = r.meeting_id
        JOIN results re ON re.runner_id = ru.runner_id
        JOIN sectionals se ON se.runner_id = ru.runner_id
        WHERE ru.horse_code = ? AND m.racedate < ? AND se.kick_time IS NOT NULL
        ORDER BY m.racedate DESC
        LIMIT ?
        """
        rows = self.con.execute(q, (horse_code, asof_ymd, int(N))).fetchall()
        if not rows:
            out = {'n': 0}
        else:
            def avg(col):
                xs=[float(r[col]) for r in rows if r[col] is not None]
                return float(np.mean(xs)) if xs else None
            out = {
                'n': len(rows),
                'avg_pos1': avg('pos1'),
                'avg_pos3': avg('pos3'),
                'avg_seg2': avg('seg2_time'),
                'avg_seg3': avg('seg3_time'),
                'avg_kick': avg('kick_time'),
            }
        self._cache_prevN_sec[key] = out
        return out

    def draw_bias_top3(self, venue: str, distance_m: int, draw: int, asof_ymd: str, lookback_days: int = 365 * 3) -> float:
        if not venue or not distance_m or not draw or not asof_ymd:
            return 0.0
        key = (venue, int(distance_m), int(draw), asof_ymd)
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
        starts = int(row['starts'] or 0)
        top3 = int(row['top3'] or 0)
        rate = safe_div(top3, starts)
        self._draw_bias_cache[key] = rate
        return rate

    def build(self, racedate: str, venue: str, distance_m: int, class_num: int, surface: str, pick: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
        horse_code = pick.get('code') or ''
        jockey = pick.get('jockey') or ''
        trainer = pick.get('trainer') or ''
        draw = int(pick.get('draw') or 0)
        weight = float(pick.get('wt') or 0.0)

        h365 = self.horse_rates(horse_code, racedate, 365)
        h60 = self.horse_rates(horse_code, racedate, 60)
        h365_v = self.horse_rates(horse_code, racedate, 365, venue=venue)
        h365_d = self.horse_rates(horse_code, racedate, 365, distance_m=distance_m)

        j365 = self.role_rates('jockey', jockey, racedate, 365)
        j60 = self.role_rates('jockey', jockey, racedate, 60)
        t365 = self.role_rates('trainer', trainer, racedate, 365)
        t60 = self.role_rates('trainer', trainer, racedate, 60)

        last = self.last_run_with_sectionals(horse_code, racedate)
        prevN = self.prevN_sectionals(horse_code, racedate, N=3)

        dist_diff = (distance_m - int(last.get('last_distance_m') or 0)) if last.get('last_distance_m') else None
        wt_diff = (weight - float(last.get('last_weight') or 0.0)) if last.get('last_weight') is not None else None
        draw_bias = self.draw_bias_top3(venue, distance_m, draw, racedate)

        avg_pos1 = prevN.get('avg_pos1'); avg_pos3 = prevN.get('avg_pos3')
        style_front = 1 if (avg_pos1 is not None and avg_pos1 <= 3) else 0
        style_closer = 1 if (avg_pos1 is not None and avg_pos1 >= 8 and avg_pos3 is not None and avg_pos3 <= avg_pos1 - 2) else 0

        f = {
            'distance_m': distance_m,
            'class_num': class_num or 0,
            'venue_HV': 1 if venue == 'HV' else 0,
            'venue_ST': 1 if venue == 'ST' else 0,
            'surface_turf': 1 if (surface or '').lower().startswith('t') else 0,
            'surface_awt': 1 if (surface or '').lower().startswith('a') else 0,
            'draw': draw,
            'weight': weight,
            'h_starts_365': h365['starts'],
            'h_winrate_365': h365['win_rate'],
            'h_top3rate_365': h365['top3_rate'],
            'h_starts_60': h60['starts'],
            'h_top3rate_60': h60['top3_rate'],
            'h_top3rate_venue_365': h365_v['top3_rate'],
            'h_top3rate_dist_365': h365_d['top3_rate'],
            'j_starts_365': j365['starts'],
            'j_top3rate_365': j365['top3_rate'],
            'j_top3rate_60': j60['top3_rate'],
            't_starts_365': t365['starts'],
            't_top3rate_365': t365['top3_rate'],
            't_top3rate_60': t60['top3_rate'],
            'days_since_last': last.get('days_since'),
            'last_finish_pos': last.get('last_finish_pos'),
            'dist_diff_from_last': dist_diff,
            'wt_diff_from_last': wt_diff,
            'last_same_venue': 1 if (last.get('last_venue') == venue and last.get('last_venue')) else 0,
            'draw_bias_top3': draw_bias,
            'last_pos1': last.get('last_pos1'),
            'last_pos3': last.get('last_pos3'),
            'last_seg2': last.get('last_seg2'),
            'last_seg3': last.get('last_seg3'),
            'last_kick': last.get('last_kick'),
            'prevN_sec_n': prevN.get('n'),
            'prevN_avg_pos1': avg_pos1,
            'prevN_avg_pos3': avg_pos3,
            'prevN_avg_seg2': prevN.get('avg_seg2'),
            'prevN_avg_seg3': prevN.get('avg_seg3'),
            'prevN_avg_kick': prevN.get('avg_kick'),
            # race-level
            'field_size': None,
            'pace_index_front_frac': None,
            'closer_x_pace': None,
            'front_x_fast': None,
        }
        aux={'style_front': style_front, 'style_closer': style_closer}
        return f, aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--in', dest='inPath', required=True)
    ap.add_argument('--model', default='models/HKJC-ML_NOODDS_RANK_LGBM_v4_EXACTA.txt')
    ap.add_argument('--meta', default='models/HKJC-ML_NOODDS_RANK_LGBM_v4_EXACTA.infermeta.json')
    ap.add_argument('--out', dest='outPath', required=True)
    ap.add_argument('--topK', type=int, default=5)
    args = ap.parse_args()

    meta = json.load(open(args.meta,'r',encoding='utf-8'))
    feat_keys = meta['featKeys']
    impute = meta.get('imputeMeans') or {}
    tilt = (meta.get('pace_tilt') or {})
    alpha = float(tilt.get('alpha') or 0.0)
    beta = float(tilt.get('beta') or 0.0)

    card = json.load(open(args.inPath,'r',encoding='utf-8'))
    bet_url = (card.get('betPage') or {}).get('url')
    key = parse_race_key(bet_url)
    racedate, venue, raceNo = key.get('racedate'), key.get('venue'), key.get('raceNo')

    distance_m = int(((card.get('betPage') or {}).get('distanceMeters')) or 0)
    class_num = int(((card.get('betPage') or {}).get('classNum')) or 0)
    surface = ((card.get('betPage') or {}).get('surfaceText')) or ''

    con = connect(args.db)
    fb = FeatureBuilder(con)

    picks = card.get('picks') or []
    feats=[]
    auxs=[]
    for p in picks:
        f,aux=fb.build(racedate, venue, distance_m, class_num, surface, p)
        feats.append(f); auxs.append(aux)

    n=len(picks)
    n_front=sum(1 for a in auxs if a.get('style_front'))
    pace=(n_front/n) if n else 0.0
    for f,a in zip(feats,auxs):
        front=a.get('style_front') or 0
        closer=a.get('style_closer') or 0
        f['field_size']=n
        f['pace_index_front_frac']=pace
        f['closer_x_pace']=closer*pace
        f['front_x_fast']=front*pace

    X=np.array([[f.get(k) for k in feat_keys] for f in feats], dtype=float)
    for i,k in enumerate(feat_keys):
        col=X[:,i]
        m=float(impute.get(k,0.0))
        col[np.isnan(col)]=m
        X[:,i]=col

    booster=lgb.Booster(model_file=args.model)
    raw=booster.predict(X)

    scored_all=[]
    for p,f,rw in zip(picks,feats,raw):
        adj = float(rw) + alpha*float(f.get('closer_x_pace') or 0.0) - beta*float(f.get('front_x_fast') or 0.0)
        scored_all.append({
            'horse_no': int(p.get('no') or 0),
            'horse': p.get('horse'),
            'draw': int(p.get('draw') or 0),
            'jockey': p.get('jockey'),
            'trainer': p.get('trainer'),
            'raw_score': float(rw),
            'adj_score': float(adj),
        })

    scored_all.sort(key=lambda x:x['adj_score'], reverse=True)

    out={
        'model': meta.get('name'),
        'racedate': racedate,
        'venue': venue,
        'raceNo': raceNo,
        'top5': scored_all[: int(args.topK)],
        'scored_all': scored_all,
        'generatedAt': datetime.utcnow().isoformat()+'Z',
        'betPage': card.get('betPage'),
        'pace': {'field_size': n, 'n_front': n_front, 'pace_index_front_frac': pace},
        'pace_tilt': {'alpha': alpha, 'beta': beta, 'formula': tilt.get('formula')},
    }

    os.makedirs(os.path.dirname(args.outPath) or '.', exist_ok=True)
    json.dump(out, open(args.outPath,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps({'ok':True,'out':args.outPath,'alpha':alpha,'beta':beta}, ensure_ascii=False))


if __name__=='__main__':
    main()
