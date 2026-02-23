# Scheduling: HKJC Production W/Q predictions (bet.hkjc wp)

## One-shot run (example)

```bash
export MATON_API_KEY=...
export HKJC_SHEET_ID=109I84syg24sJCl9QxP0soUUx63BvmsS8dJCTEs8MrbI

python3 hkjc_prod_run_wp_meeting.py \
  --racedate 2026-02-25 \
  --venue HV \
  --races 1-9 \
  --sheetName PROD_PRED_2026-02-25_HV
```

This will:
1) scrape https://bet.hkjc.com/ch/racing/wp/<date>/<venue>/<race>
2) build runner features from `hkjc.sqlite`
3) run pinned production W/Q models (`prod/HKJC_PROD_WQ.json`)
4) export a table to Google Sheets

## Telegram notification

OpenClaw-side cron can send a Telegram message after the run.
Recommended: send a short summary + link to the sheet tab.

## Notes

- Odds may be null until markets open; production W overlay requires odds.
- We compute two pass flags:
  - production threshold (from prod config, currently 0.20)
  - alternate threshold (default 0.16) for race-day discretion
