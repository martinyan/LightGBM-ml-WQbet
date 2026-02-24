#!/usr/bin/env bash
set -euo pipefail

# HKJC DB rebuild skeleton
# This repo does NOT ship hkjc.sqlite or raw data sources.
# Fill in the paths/URLs for your localresults + sectional sources.

DB_PATH=${DB_PATH:-hkjc.sqlite}

echo "[db_rebuild] target db: ${DB_PATH}"

# 1) Build base DB from localresults (race results)
# Example:
# node hkjc_build_sqlite_from_localresults.mjs --out "$DB_PATH" --in /path/to/localresults

echo "TODO: implement: build sqlite from localresults"

# 2) Backfill/ingest any missing meetings if you run incrementally
# Example:
# node hkjc_ingest_meeting_into_sqlite.mjs --db "$DB_PATH" --racedate YYYY/MM/DD --venue HV

echo "TODO: implement: ingest meetings incrementally (optional)"

# 3) Backfill sectional splits
# Example:
# node hkjc_backfill_sectional_splits.mjs --db "$DB_PATH" --from YYYY/MM/DD --to YYYY/MM/DD

echo "TODO: implement: backfill sectional splits"

# 4) Sanity checks
python3 - <<'PY'
import sqlite3
con=sqlite3.connect('hkjc.sqlite')
cur=con.cursor()
cur.execute('select count(1) from meetings')
print('meetings:', cur.fetchone()[0])
cur.execute('select max(racedate) from meetings')
print('max racedate:', cur.fetchone()[0])
con.close()
PY

echo "[db_rebuild] done (skeleton)"
