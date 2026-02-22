import os, json, urllib.request, urllib.parse

SPREADSHEET_ID = os.environ.get('HKJC_SHEET_ID', '109I84syg24sJCl9QxP0soUUx63BvmsS8dJCTEs8MrbI')
REPORT_PATH = os.environ.get('HKJC_Q_REPORT', 'reports/HKJC-ML_Q_RANKER_v7_NOODDS_top6/q_ranker_top6_test_report.json')
SHEET_NAME = os.environ.get('HKJC_Q_SHEET', 'Q_RANKER_v7_NOODDS_top6_LEDGER')


def req_json(url, method='GET', data=None, headers=None):
    if headers is None:
        headers = {}
    req = urllib.request.Request(url, method=method, data=data)
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def main():
    maton_key = os.environ.get('MATON_API_KEY')
    if not maton_key:
        raise SystemExit('MATON_API_KEY is required')

    report = json.load(open(REPORT_PATH, 'r', encoding='utf-8'))
    ledger = report['ledger']

    # deterministic column order
    cols = [
        'racedate','venue','race_no','topM','pick_pair',
        'horse_no_i','horse_i','score_i','rank_i_field','rank_i_topM','y_finish_pos_i','rel_i',
        'horse_no_j','horse_j','score_j','rank_j_field','rank_j_topM','y_finish_pos_j','rel_j',
        'p_top2_set','dividend','stake','return','profit','is_hit','url'
    ]

    values = [cols]
    for r in ledger:
        row = []
        for c in cols:
            v = r.get(c)
            if v is True:
                v = 'TRUE'
            elif v is False:
                v = 'FALSE'
            row.append(v)
        values.append(row)

    auth = {'Authorization': f'Bearer {maton_key}', 'Content-Type': 'application/json'}

    # ensure sheet exists
    meta = req_json(
        f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{SPREADSHEET_ID}',
        headers=auth,
    )
    existing = {s['properties']['title'] for s in meta.get('sheets', [])}
    if SHEET_NAME not in existing:
        body = json.dumps({'requests': [{'addSheet': {'properties': {'title': SHEET_NAME}}}]}).encode()
        req_json(
            f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{SPREADSHEET_ID}:batchUpdate',
            method='POST',
            data=body,
            headers=auth,
        )

    # clear then write
    rng = urllib.parse.quote(f'{SHEET_NAME}!A1:Z')
    req_json(
        f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{SPREADSHEET_ID}/values/{rng}:clear',
        method='POST',
        data=b'{}',
        headers=auth,
    )

    put_body = json.dumps({'values': values}, ensure_ascii=False).encode('utf-8')
    rng2 = urllib.parse.quote(f'{SHEET_NAME}!A1')
    out = req_json(
        f'https://gateway.maton.ai/google-sheets/v4/spreadsheets/{SPREADSHEET_ID}/values/{rng2}?valueInputOption=USER_ENTERED',
        method='PUT',
        data=put_body,
        headers=auth,
    )

    print(json.dumps({
        'ok': True,
        'sheet': SHEET_NAME,
        'updatedRange': out.get('updatedRange'),
        'rows': len(values)-1,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
