#!/usr/bin/env python3
"""Generate WOddDrift report for a race.

Modes:
- T-2h report: run production prediction (GoldenWinBet/GoldenQbet) using odds at that time; store baseline odds.
- T-5m report: compare current odds vs baseline odds; compute drift metrics; compute research score_star.

This script does not modify production. It prints a concise summary (for cron announce).

Drift definition:
- drift_10m, drift_30m, drift_2h: ln(odds_prev / odds_now)  (positive = odds dropped)
- drift_combo = 0.6*drift_10m + 0.3*drift_30m + 0.1*drift_2h
- score_star = score_win + k*drift_combo

Note: if historical odds points aren't available (missing snapshots), that component is treated as 0.
"""

import argparse, json, math, os, subprocess
from datetime import datetime, timezone


def load_jsonl(path):
    out=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out


def nearest_before(snaps, target_ts):
    # snaps sorted by ts
    best=None
    for s in snaps:
        ts=s.get('ts')
        if ts and ts <= target_ts:
            best=s
        else:
            break
    return best


def ln_ratio(a, b):
    # ln(a/b)
    try:
        a=float(a); b=float(b)
        if a>0 and b>0:
            return math.log(a/b)
    except Exception:
        pass
    return 0.0


def run_prod_predict(racedate_slash, venue, race_no, prod_cfg, db_path, merged_out):
    # use current odds via racecard+graphql (WIN odds)
    subprocess.check_call([
        'python3','hkjc_scrape_racecard_and_win_odds.py',
        '--racedate', racedate_slash,
        '--venue', venue,
        '--raceNo', str(race_no),
        '--out', merged_out
    ])
    feat = merged_out.replace('merged_', 'feat_')
    pred = merged_out.replace('merged_', 'pred_')

    subprocess.check_call(['node','hkjc_build_feature_rows_from_racecard_sqlite.mjs','--db',db_path,'--in',merged_out,'--out',feat,'--lastN','3'])
    subprocess.check_call(['python3','hkjc_prod_predict_single_race_wq.py','--prod',prod_cfg,'--in',feat,'--out',pred])
    return json.load(open(pred,'r',encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['T2H','T5M'], required=True)
    ap.add_argument('--racedate', required=True, help='YYYY/MM/DD')
    ap.add_argument('--venue', required=True, help='HV|ST')
    ap.add_argument('--raceNo', required=True, type=int)
    ap.add_argument('--snapshots', required=True, help='JSONL snapshots path')
    ap.add_argument('--out', required=True, help='report json path')
    ap.add_argument('--k', type=float, default=1.0)
    ap.add_argument('--prod', default='prod/HKJC_PROD_WQ.json')
    ap.add_argument('--db', default='hkjc.sqlite')
    args=ap.parse_args()

    venue=args.venue.upper()
    racedate_dash=args.racedate.replace('/','-')

    snaps=load_jsonl(args.snapshots)
    snaps=[s for s in snaps if s.get('date')==racedate_dash and s.get('venue')==venue and int(s.get('raceNo'))==int(args.raceNo)]
    snaps=sorted(snaps, key=lambda s: s.get('ts') or '')

    now_ts=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')

    rep={'mode':args.mode,'racedate':args.racedate,'venue':venue,'raceNo':int(args.raceNo),'generatedAt':now_ts,'k':args.k}

    # production prediction snapshot (uses current odds at runtime)
    merged_out=f'/tmp/wodddrift_merged_{racedate_dash}_{venue}_R{args.raceNo}.json'
    pred=run_prod_predict(args.racedate, venue, int(args.raceNo), args.prod, args.db, merged_out)
    rep['prod']=pred

    # odds baselines
    if args.mode=='T2H':
        # store baseline odds from latest snapshot
        latest=snaps[-1] if snaps else None
        rep['baseline']={'ts': latest.get('ts') if latest else None, 'odds': (latest.get('odds') if latest else {})}
    else:
        # load baseline from previous report if exists
        baseline_path=args.out.replace('_T-5m.json','_T-2h.json')
        baseline={}
        if os.path.exists(baseline_path):
            baseline=json.load(open(baseline_path,'r',encoding='utf-8')).get('baseline',{})
        rep['baseline']=baseline

        odds_now=(snaps[-1].get('odds') if snaps else {})
        rep['odds_now']={'ts': snaps[-1].get('ts') if snaps else None, 'odds': odds_now}

        # compute drift for relevant horses (top 5 by production W score)
        w_scored = (pred.get('W') or {}).get('scored_all') or []
        w_scored_sorted = sorted(w_scored, key=lambda x: x.get('score_win', -1e9), reverse=True)
        focus = w_scored_sorted[:6]

        # build snapshot lookup for -10m/-30m/-2h relative to latest snapshot ts
        latest_ts = snaps[-1].get('ts') if snaps else None
        # if no ts, skip
        comp=[]
        if latest_ts:
            # approximate by string ordering; timestamps are ISO Z
            # define targets
            from datetime import timedelta
            import dateutil.parser
            lt=dateutil.parser.isoparse(latest_ts)
            t10=(lt - timedelta(minutes=10)).isoformat().replace('+00:00','Z')
            t30=(lt - timedelta(minutes=30)).isoformat().replace('+00:00','Z')
            t2h=(lt - timedelta(hours=2)).isoformat().replace('+00:00','Z')

            s10=nearest_before(snaps, t10)
            s30=nearest_before(snaps, t30)
            s2h=nearest_before(snaps, t2h)

            for r in focus:
                hn=str(int(r['horse_no']))
                o_now = odds_now.get(hn)
                o10 = (s10.get('odds') or {}).get(hn) if s10 else None
                o30 = (s30.get('odds') or {}).get(hn) if s30 else None
                o2h = (s2h.get('odds') or {}).get(hn) if s2h else None

                d10 = ln_ratio(o10, o_now) if (o10 and o_now) else 0.0
                d30 = ln_ratio(o30, o_now) if (o30 and o_now) else 0.0
                d2h = ln_ratio(o2h, o_now) if (o2h and o_now) else 0.0
                combo = 0.6*d10 + 0.3*d30 + 0.1*d2h
                score_win = float(r.get('score_win') or 0.0)
                score_star = score_win + float(args.k)*combo

                comp.append({
                    'horse_no': int(r['horse_no']),
                    'horse': r.get('horse'),
                    'odds_now': o_now,
                    'drift_10m': d10,
                    'drift_30m': d30,
                    'drift_2h': d2h,
                    'drift_combo': combo,
                    'score_win': score_win,
                    'score_star': score_star,
                })

        rep['drift_comparison']=sorted(comp, key=lambda x: x['score_star'], reverse=True)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(rep, open(args.out,'w',encoding='utf-8'), ensure_ascii=False, indent=2)

    # stdout summary (Telegram)
    wtop1=(pred.get('W') or {}).get('top1') or {}
    lines=[f"WOddDrift {args.mode} R{args.raceNo} {args.racedate} {venue}",
           f"Prod top1: #{wtop1.get('horse_no')} {wtop1.get('horse')} odds={wtop1.get('cur_win_odds')} overlay={wtop1.get('overlay'):.3f}"]
    if args.mode=='T5M':
        best=(rep.get('drift_comparison') or [None])[0]
        if best:
            lines.append(f"Drift-adjusted top1: #{best['horse_no']} score*={best['score_star']:.3f} (k={args.k}) drift={best['drift_combo']:.3f} odds_now={best['odds_now']}")
    print("\n".join(lines))


if __name__=='__main__':
    main()
