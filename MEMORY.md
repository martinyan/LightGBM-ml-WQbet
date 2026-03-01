# MEMORY.md

## Model Preferences

- **Default for main agent (persistent):** ChatGPT **5 Mini** (`openai/gpt-5-mini`). This preference should be used as the default LLM for the main agent across restarts.
- **HKJC Project (ML/DB/Pipeline):** Use **ChatGPT** (GPT-4 or higher) for anything related to ML model building, pipeline development, debugging, and SQLite operations.
- **Lightweight Tasks:** Use **Gemini Flash** for simple reasoning and Google Sheets tab updates.

## Projects

### HKJC-ML (default prediction model)

**Default model for HKJC predictions (as of 2026-02-21):** `HKJC-ML_XGB_NOODDS_REG_v2`

**Pinned production W+Q models for betting (as of 2026-02-24):**
- **W (WIN overlay residual):** `OVERLAY_RESIDUAL_LGBM_v1` bundle `models/OVERLAY_TRAIN_20230901_20250731_v6b_prev3.pkl` with `threshold=0.2` (alpha=1, beta=0)
- **Q (Quinella partners ranker):** `models/Q_RANKER_v7_NOODDS_top6` (ranker.txt + bundle.json), strategy: 2-combo Q with anchor=W top1 and partners=ranker top2 excluding anchor
- Config: `prod/HKJC_PROD_WQ.json`
- Regularized XGBoost, **NO-ODDS** inference (`cur_win_odds=0`)
- Saved/tagged in git: tag `HKJC-ML_XGB_NOODDS_REG_v2`
- Signature file: `/data/.openclaw/workspace/models/HKJC-ML_XGB_NOODDS_REG_v2.signature.json`
- Report copy: `/data/.openclaw/workspace/models/HKJC-ML_XGB_NOODDS_REG_v2_train2325_test2526_2026-02-21.report.json`


### Project Stock Pick

**Project code:** STOCK-PICK

**Status:** Requirements defined (free-data v1) (2026-02-10)

**Momentum stock definition (v1):**
- Pre-market gap up **>= 6%**
- Price **>= $5**
- Volume **>= (5-day avg volume) * 1.3**
- Daily trend: **5 EMA > 12 EMA**
- Market cap **>= $100M**

**Data constraint:** use **free** sources only for now.

**Next steps:**
- Confirm universe + exact volume definition/time window
- Pick free data sources (likely semi-manual for pre-market in v1)
- Create runbook + initial scripts

### HKJC-KEYRACES (HKJC key races scanner)

**Project code:** HKJC-KEYRACES

**What it does:**
- Input: HKJC trainers-entries table URL (race day)
- Extract horse profile links from table columns 3+
- For each horse profile page, scan race history and flag **key races** where:
  - finishing position is 01 or 02
  - 頭馬距離 <= 0.5 (鼻位 / 短馬頭位 / 頭位 / 頸位 / 1/4 / 1/2 etc.)

**Reusable files (workspace):**
- Runbook: `/data/.openclaw/workspace/HKJC_RUNBOOK.md`
- Browser snippet: `/data/.openclaw/workspace/hkjc_browser_extract_horse_links.js`
- Processor script: `/data/.openclaw/workspace/hkjc_process_links_to_keyraces.mjs`

**Example outputs (2026-02-08):**
- Links JSON: `/data/.openclaw/workspace/hkjc_trainers_entries_links_2026-02-08.full.json`
- Key races JSON: `/data/.openclaw/workspace/hkjc_key_races_2026-02-08.json`
