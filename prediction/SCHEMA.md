# Prediction JSON Output Schemas

## Production (WQ)

**Current (v2 picks-only):**
- `prediction/<dd-mm-yyyy>/productionWQ-result.json` (schema_version: `productionWQ-picks.v2`)

**Legacy (v1 full):**
- `prediction/<dd-mm-yyyy>/productionWQ-result.full.v1.json` (schema_version: `productionWQ-result.v1`)

## Research

- `prediction/<dd-mm-yyyy>/research-result.json`

Where `<dd-mm-yyyy>` matches the same date string used inside the JSON (`racedate`).

## Date convention

- `racedate`: **dd-mm-yyyy** (e.g. `22-03-2026`)
- `generated_at`: ISO-8601 with timezone (e.g. `2026-03-21T13:05:00+08:00`)

## Common rules

- Each race object MUST include `ranked_runners_top6` (Top6 only).
- ProductionWQ MUST include (for at least W top1/top2 and Top6 runners):
  - `cur_win_odds`, `p_mkt_win`, `score_win`, `overlay`

## Schema files

- `prediction/schema_research-result.v1.json`
- `prediction/schema_productionWQ-result.v1.json`

These are JSON Schemas intended for validation.
