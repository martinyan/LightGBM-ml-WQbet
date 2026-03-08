#!/usr/bin/env python3
"""Format a short Telegram-friendly summary from qindrift_last10m_corr JSON.

Input: JSON from stdin.
Output:
QIN Drift HV R1 T-10
focus: 1,7,12
movers: 1(0.12) 7(0.09) 4(0.07)
correl:
  focus 1: 7(+0.81) 12(+0.62) 4(-0.58)
  ...

By default: top 3 movers and top 3 correlations per focus horse.
"""

import argparse, json, sys, os, glob, re


def fmt(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def infer_pred_path_from_title(title: str) -> str | None:
    """Best-effort: infer pred_YYYY-MM-DD_<VENUE>_R<race>.json from title and pick newest by mtime."""
    t = title or ''
    m = re.search(r'\b(HV|ST)\b.*?\bR\s*(\d{1,2})\b', t)
    if not m:
        # also support e.g. "QIN Drift ST R5" format
        m = re.search(r'\b(HV|ST)\b\s*R\s*(\d{1,2})\b', t)
    if not m:
        return None
    venue = m.group(1)
    race_no = int(m.group(2))
    patt = os.path.join('reports', 'PROD_PRED', f'pred_*_{venue}_R{race_no}.json')
    cand = glob.glob(patt)
    if not cand:
        return None
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]


def load_name_map(pred_path: str | None) -> dict[int, str]:
    if not pred_path or not os.path.exists(pred_path):
        return {}
    try:
        pred = json.load(open(pred_path, 'r', encoding='utf-8'))
    except Exception:
        return {}

    mp: dict[int, str] = {}

    # Prefer W scored_all (contains horse + horse_no)
    for rr in (pred.get('W') or {}).get('scored_all') or []:
        try:
            hn = int(rr.get('horse_no'))
        except Exception:
            continue
        name = rr.get('horse')
        if name and hn not in mp:
            mp[hn] = str(name)

    # Also add Q ranker list
    for rr in (pred.get('Q') or {}).get('ranker_scored_all') or []:
        try:
            hn = int(rr.get('horse_no'))
        except Exception:
            continue
        name = rr.get('horse')
        if name and hn not in mp:
            mp[hn] = str(name)

    return mp


def hn_with_name(hn, name_map: dict[int, str]) -> str:
    try:
        i = int(hn)
    except Exception:
        return str(hn)
    nm = name_map.get(i)
    return f"#{i} {nm}" if nm else f"#{i}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--title', default='')
    ap.add_argument('--top', type=int, default=3)
    ap.add_argument('--focus', default='')
    ap.add_argument('--pred', default='', help='Optional pred JSON path to attach horse names')
    args = ap.parse_args()

    obj = json.load(sys.stdin)

    title = args.title.strip() or f"QIN Drift {obj.get('venue','')} R{obj.get('raceNo','')}"

    pred_path = args.pred.strip() or infer_pred_path_from_title(title)
    name_map = load_name_map(pred_path)

    focus = args.focus.strip()
    if not focus:
        # if the corr json already contains focus list, use it
        fl = obj.get('focus') or []
        focus = ','.join(str(int(x)) for x in fl)

    print(title)
    if focus:
        print(f"focus: {focus}")

    movers = (obj.get('topMovers') or [])[: args.top]
    if movers:
        mline = ' '.join([f"{hn_with_name(m.get('horse_no'), name_map)}({fmt(m.get('abs_drift'),3)})" for m in movers])
        print(f"movers: {mline}")

    corr = obj.get('correlations') or {}
    if corr:
        print("corr:")
        # only show correlations for provided focus list if any
        focus_set = None
        if focus:
            try:
                focus_set = {str(int(x.strip())) for x in focus.split(',') if x.strip()}
            except Exception:
                focus_set = None
        keys = list(corr.keys())
        if focus_set is not None:
            keys = [k for k in keys if k in focus_set]

        for k in keys:
            lst = (corr.get(k) or [])[: args.top]
            if not lst:
                continue
            items = ' '.join([
                f"{hn_with_name(it.get('horse_no'), name_map)}({('+' if (it.get('corr') or 0)>=0 else '')}{fmt(it.get('corr'),2)})"
                for it in lst
            ])
            print(f"  {hn_with_name(k, name_map)}: {items}")


if __name__ == '__main__':
    main()
