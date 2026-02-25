# QOddChange

Goal: capture **live Quinella (QIN) odds changes** before races, to detect significant odds drops associated with specific horses/pairs.

Key constraints:
- This is a forward-only project. We **cannot** reconstruct historical intra-race odds paths reliably.
- We capture snapshots at a regular interval for future races, then do after-race analysis.

Data sources (current plan):
- HKJC GraphQL: `https://info.cld.hkjc.com/graphql/base/`
- Runner meta + schedule time: `racing.hkjc.com` racecard page

Outputs (local):
- `data/snapshots/*.jsonl` : raw odds snapshots over time
- `reports/*.json` : derived alerts / drops / summaries

Next steps:
1) Decide polling interval (e.g., every 1m / 2m / 5m)
2) Decide time window (e.g., start when sellStatus=START_SELL; stop at race start)
3) Define “significant drop” metrics (absolute and/or % drop; rank change; concentrated on horses)
