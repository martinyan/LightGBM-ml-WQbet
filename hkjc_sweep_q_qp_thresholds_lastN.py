import json, argparse, subprocess, os

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stdout)
    return p.stdout


def load_summary(out_json):
    j = json.load(open(out_json, 'r', encoding='utf-8'))
    stake_q = j['total']['Q']['stake']; ret_q = j['total']['Q']['return']
    stake_qp = j['total']['QP']['stake']; ret_qp = j['total']['QP']['return']
    comb_stake = stake_q + stake_qp
    comb_ret = ret_q + ret_qp
    comb_profit = comb_ret - comb_stake
    comb_roi = comb_profit / comb_stake if comb_stake else None
    hit_q = sum(1 for r in j['per_race'] if r.get('q_div') is not None)
    hit_qp = sum(1 for r in j['per_race'] if r.get('qp_div') is not None)
    return {
        'threshold': j['threshold'],
        'races_bet': j['races_bet'],
        'Q_roi': j['total']['Q']['roi'],
        'Q_profit': j['total']['Q']['profit'],
        'Q_hits': hit_q,
        'QP_roi': j['total']['QP']['roi'],
        'QP_profit': j['total']['QP']['profit'],
        'QP_hits': hit_qp,
        'combined_roi': comb_roi,
        'combined_profit': comb_profit,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meetings', type=int, default=60)
    ap.add_argument('--thresholds', default='1.8,1.85,1.9,1.92')
    ap.add_argument('--db', default='hkjc.sqlite')
    ap.add_argument('--data', default='hkjc_dataset_v3_code_prev1.jsonl')
    ap.add_argument('--model', default='hkjc_xgb_model_v1.bin')
    ap.add_argument('--script', default='hkjc_backtest_q_qp_lastNmeetings_noodds_threshold.py')
    ap.add_argument('--outPrefix', default='hkjc_q_qp_roi_last60meetings_noodds')
    ap.add_argument('--reportOut', default='hkjc_q_qp_sweep_last60meetings_noodds.json')
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(',') if x.strip()]

    results = []
    out_files = []
    for th in thresholds:
        out_json = f"{args.outPrefix}_p12gt{str(th).replace('.','p')}.json"
        cmd = [
            'python3','-u',args.script,
            '--db',args.db,
            '--data',args.data,
            '--model',args.model,
            '--meetings',str(args.meetings),
            '--threshold',str(th),
            '--out',out_json
        ]
        print(f"RUN threshold={th} -> {out_json}")
        run(cmd)
        out_files.append(out_json)
        results.append(load_summary(out_json))

    results_sorted = sorted(results, key=lambda r: (r['combined_roi'] if r['combined_roi'] is not None else -1e9), reverse=True)

    report = {
        'meetings': args.meetings,
        'thresholds': thresholds,
        'results': results,
        'results_sorted_by_combined_roi': results_sorted,
        'out_files': out_files,
        'note': 'NO-ODDS Top2 Q/QP backtest; combined metrics assume betting both pools each qualifying race.'
    }

    with open(args.reportOut, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print('WROTE', args.reportOut)

if __name__ == '__main__':
    main()
