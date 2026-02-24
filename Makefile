.PHONY: verify prod-predict-meeting db-rebuild

verify:
	python3 scripts/verify_artifacts.py

# Example production meeting run (bet.hkjc wp)
# Usage: make prod-predict-meeting RACEDATE=2026-02-25 VENUE=HV RACES=1-9 SHEETNAME=PROD_PRED_2026-02-25_HV
prod-predict-meeting:
	python3 hkjc_prod_run_wp_meeting.py --racedate $(RACEDATE) --venue $(VENUE) --races $(RACES) --sheetName $(SHEETNAME)

# DB rebuild placeholder (wire to your preferred localresults/sectionals sources)
# This repo intentionally does NOT ship hkjc.sqlite.
# You can add your raw data sources and implement the exact steps here.
db-rebuild:
	@echo "TODO: implement db rebuild pipeline (localresults + sectionals)"
