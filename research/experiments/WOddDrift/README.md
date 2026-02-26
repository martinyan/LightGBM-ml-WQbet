# WOddDrift (research)

Purpose: capture **WIN odds drift** from the HKJC GraphQL feed and compute a drift-adjusted research score alongside production GoldenWinBet.

Policy (aligned):
- **First report at T-2h** (also first time we run GoldenWinBet prediction for that race).
- After T-2h prediction, collect WIN odds snapshots:
  - T-2h → T-1h: every 15 minutes
  - T-1h → T: every 5 minutes
- **Final report at T-5m**.

Data source:
- GraphQL: https://info.cld.hkjc.com/graphql/base/ (whitelisted query)
- Race start time: racing.hkjc.com racecard page

Outputs:
- snapshots archived under: `data/snapshots/YYYY-MM-DD/<VENUE>_R<race>.jsonl`
- reports under: `reports/YYYY-MM-DD/<VENUE>_R<race>_T-2h.json`, `..._T-5m.json`

Notes:
- This is research-only; does not change production p_mkt_win or production models.
