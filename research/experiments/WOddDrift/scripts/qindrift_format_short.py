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

import argparse, json, sys


def fmt(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--title', default='')
    ap.add_argument('--top', type=int, default=3)
    ap.add_argument('--focus', default='')
    args = ap.parse_args()

    obj = json.load(sys.stdin)

    title = args.title.strip() or f"QIN Drift {obj.get('venue','')} R{obj.get('raceNo','')}"
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
        mline = ' '.join([f"{m.get('horse_no')}({fmt(m.get('abs_drift'),3)})" for m in movers])
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
            items = ' '.join([f"{it.get('horse_no')}({('+' if (it.get('corr') or 0)>=0 else '')}{fmt(it.get('corr'),2)})" for it in lst])
            print(f"  {k}: {items}")


if __name__ == '__main__':
    main()
