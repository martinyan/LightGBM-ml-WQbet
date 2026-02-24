#!/usr/bin/env python3
"""Smoke test for HKJC production packaging.

This test is designed to run in CI without needing hkjc.sqlite.
It verifies:
- artifact checksums
- model bundles load
- feature key counts are stable
"""

import json
import pickle
import subprocess


def main():
    # 1) verify artifacts lock
    subprocess.check_call(['python3', 'scripts/verify_artifacts.py'])

    # 2) load GoldenWinBet bundle
    w = pickle.load(open('models/OVERLAY_RESIDUAL_LGBM_v1_PROD_22bets_thr0p2.pkl', 'rb'))
    assert 'feat_keys' in w and isinstance(w['feat_keys'], list)
    assert len(w['feat_keys']) == 33, f"unexpected W feat_keys len: {len(w['feat_keys'])}"
    assert 'mdl_win' in w, 'missing mdl_win'

    # 3) load GoldenQbet bundle
    q = json.load(open('models/Q_RANKER_v7_PROD_FEB22_111ROI/bundle.json', 'r', encoding='utf-8'))
    assert 'feat_keys' in q and isinstance(q['feat_keys'], list)
    assert len(q['feat_keys']) == 42, f"unexpected Q feat_keys len: {len(q['feat_keys'])}"

    # 4) quick import of lightgbm and model load
    import lightgbm as lgb  # noqa

    booster = lgb.Booster(model_file='models/Q_RANKER_v7_PROD_FEB22_111ROI/ranker.txt')
    assert booster.num_model_per_iteration() >= 1

    print('SMOKE_TEST_OK')


if __name__ == '__main__':
    main()
