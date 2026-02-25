import argparse, json, math, os, pickle
from dataclasses import dataclass

import numpy as np
import lightgbm as lgb


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


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


def normalize_market_probs(rows):
    p_raw = []
    for r in rows:
        o = safe_float(r.get('cur_win_odds'), None)
        p_raw.append(0.0 if (o is None or o <= 0) else 1.0 / o)
    s = sum(p_raw)
    if s <= 0:
        return [1.0 / len(rows)] * len(rows)
    return [x / s for x in p_raw]


def load_prod_cfg(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_overlay_bundle(pkl_path: str):
    with open(pkl_path, 'rb') as f:
        b = pickle.load(f)
    # Minimum required keys for WIN-only overlay bundles
    for k in ['feat_keys', 'mdl_win']:
        if k not in b:
            raise ValueError(f"overlay bundle missing key: {k}")
    # Optional keys for place-aware bundles
    b.setdefault('place_mdl', None)
    b.setdefault('mdl_pl', None)
    return b


def load_ranker(bundle_json: str, model_txt: str):
    bundle = json.load(open(bundle_json, 'r', encoding='utf-8'))
    feat_keys = bundle['feat_keys']
    booster = lgb.Booster(model_file=model_txt)
    return bundle, feat_keys, booster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prod', default='prod/HKJC_PROD_WQ.json')
    ap.add_argument('--in', dest='in_path', required=True, help='Single-race features JSON (see hkjc_build_feature_rows_from_racecard_sqlite.mjs)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = load_prod_cfg(args.prod)

    pack = json.load(open(args.in_path, 'r', encoding='utf-8'))
    rows = pack.get('rows') or []
    if not rows:
        raise SystemExit('input has no rows')

    # --- W (overlay residual) ---
    wcfg = cfg['W']
    bundle = load_overlay_bundle(wcfg['model_bundle_pkl'])
    feat_keys_w = bundle['feat_keys']
    mdl_win = bundle['mdl_win']

    p_mkt = normalize_market_probs(rows)

    scored_w = []
    Xw = np.asarray([[safe_float(r.get('features', {}).get(k), 0.0) for k in feat_keys_w] for r in rows], dtype=np.float32)
    pred_res_win = mdl_win.predict(Xw)

    for r, pm, rw in zip(rows, p_mkt, pred_res_win):
        score_win = logit(pm) + float(wcfg.get('alpha', 1.0)) * float(rw)
        overlay = float(wcfg.get('alpha', 1.0)) * float(rw)  # beta=0 in production
        scored_w.append({
            'horse_no': int(r['horse_no']),
            'horse': r.get('horse'),
            'cur_win_odds': r.get('cur_win_odds'),
            'p_mkt_win': float(pm),
            'score_win': float(score_win),
            'overlay': float(overlay),
        })

    scored_w.sort(key=lambda x: x['score_win'], reverse=True)
    top1 = scored_w[0]
    passes = bool(top1['overlay'] > float(wcfg['threshold']))

    # --- Q (ranker partners) ---
    qcfg = cfg['Q']
    q_bundle, feat_keys_q, ranker = load_ranker(qcfg['bundle_json'], qcfg['model_txt'])

    Xq = np.asarray([[safe_float(r.get('features', {}).get(k), 0.0) for k in feat_keys_q] for r in rows], dtype=np.float32)
    q_scores = ranker.predict(Xq)
    scored_q = sorted(
        [{'horse_no': int(r['horse_no']), 'horse': r.get('horse'), 'ranker_score': float(s)} for r, s in zip(rows, q_scores)],
        key=lambda x: x['ranker_score'],
        reverse=True,
    )

    anchor = int(top1['horse_no'])
    partners = [x for x in scored_q if int(x['horse_no']) != anchor]
    partner2 = partners[0] if len(partners) > 0 else None
    partner3 = partners[1] if len(partners) > 1 else None

    out = {
        'prod': cfg['name'],
        'racedate': pack.get('racedate'),
        'venue': pack.get('venue'),
        'raceNo': pack.get('raceNo') or pack.get('race_no'),
        'W': {
            'name': wcfg.get('name'),
            'model': wcfg['model'],
            'threshold': wcfg['threshold'],
            'top1': top1,
            'passes_threshold': passes,
            'scored_all': scored_w,
            'scored_top5': scored_w[:5]
        },
        'Q': {
            'name': qcfg.get('name'),
            'model': qcfg['model'],
            'anchor_horse_no': anchor,
            'partner2': partner2,
            'partner3': partner3,
            'ranker_scored_all': scored_q,
            'ranker_scored_top6': scored_q[:6],
            'note': 'Place Q bets only if W passes_threshold (by convention); override at caller if desired.'
        }
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({'ok': True, 'out': args.out, 'passes': passes, 'anchor': anchor}, ensure_ascii=False))


if __name__ == '__main__':
    main()
