#!/usr/bin/env bash
set -euo pipefail

# Promotion helper: move a vetted research artifact into production.
# This script is intentionally conservative.

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <SRC_PATH> <DEST_PATH>" >&2
  echo "Example: $0 research/models/my_new_win_model.pkl models/" >&2
  exit 2
fi

SRC="$1"
DEST="$2"

if [[ ! -f "$SRC" && ! -d "$SRC" ]]; then
  echo "Source not found: $SRC" >&2
  exit 2
fi

echo "About to PROMOTE artifact to production:"
echo "  SRC:  $SRC"
echo "  DEST: $DEST"
echo

echo "Checklist (manual):"
echo "  [ ] Backtest vs current production on Sept2025-now"
echo "  [ ] Update prod/HKJC_PROD_WQ.json to point to new artifact (if applicable)"
echo "  [ ] Run: python3 scripts/verify_artifacts.py (and update lockfile if prod artifacts changed)"
echo "  [ ] Commit on a PR branch; merge to main only after review"
echo

read -p "Type YES to continue: " ans
if [[ "$ans" != "YES" ]]; then
  echo "Aborted."
  exit 1
fi

mkdir -p "$DEST"
cp -a "$SRC" "$DEST"

echo "Copied. Now run tests + update config + commit."
