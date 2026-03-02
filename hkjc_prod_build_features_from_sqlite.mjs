#!/usr/bin/env node
/* Build production feature rows for a specific race from SQLite (truth-first).

Goal (per user decision A):
- Use SQLite as source of truth for historical runs + sectional splits.
- Use SQLite race context (distance/class/surface/draw/weight/jockey/trainer).
- Use FINAL odds from HKJC localresults when available (assumed already backfilled into runners.win_odds),
  otherwise fall back to whatever odds are present in SQLite.

Output JSON matches reports/PROD_PRED/features_YYYY-MM-DD_VENUE_RN.json shape:
{
  racedate, venue, raceNo, builtAt, source:'sqlite', rows:[
    { horse_no, horse, horse_code, cur_win_odds, features:{...} }
  ]
}

Usage:
  node hkjc_prod_build_features_from_sqlite.mjs --db hkjc.sqlite --racedate YYYY/MM/DD --venue ST|HV --raceNo 11 --out reports/PROD_PRED/features_YYYY-MM-DD_VENUE_R11.json

Notes:
- This script intentionally does not scrape. It expects SQLite has been ingested/backfilled.
*/

import fs from 'node:fs/promises';
import Database from 'better-sqlite3';
import {
  ymdToUtcDate,
  daysBetweenUtc,
  getHorsePreviousRuns,
  computeSectionalFeaturesFromSplits,
  getSectionalSplitsForRunner,
  computeAggregatedRunFeatures,
  safeNum
} from './hkjc_ml_features_sqlite.mjs';

function arg(name, dflt = null) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : dflt;
}

const dbPath = arg('--db', 'hkjc.sqlite');
const racedate = arg('--racedate');
const venue = arg('--venue');
const raceNo = Number(arg('--raceNo'));
const outPath = arg('--out');
const prevRuns = Math.max(1, Number(arg('--prevRuns', '3')));
const jtDays = Math.max(1, Number(arg('--jtDays', '60')));

if (!racedate || !venue || !raceNo || !outPath) {
  console.error('Missing required args: --racedate YYYY/MM/DD --venue ST|HV --raceNo N --out path');
  process.exit(2);
}

const db = new Database(dbPath, { readonly: true });

function getRaceContext() {
  const row = db.prepare(`
    SELECT
      m.meeting_id,
      m.racedate,
      m.venue,
      r.race_id,
      r.race_no,
      r.distance_m,
      r.class_num,
      r.surface
    FROM races r
    JOIN meetings m ON m.meeting_id = r.meeting_id
    WHERE m.racedate = ? AND m.venue = ? AND r.race_no = ?
  `).get(racedate, venue, raceNo);
  if (!row) throw new Error(`Race not found in sqlite: ${racedate} ${venue} R${raceNo}`);
  return row;
}

function getRunnersForRace(race_id) {
  return db.prepare(`
    SELECT
      ru.runner_id,
      ru.horse_code,
      ru.horse_name_zh,
      ru.horse_no,
      ru.draw,
      ru.weight,
      ru.jockey,
      ru.trainer,
      ru.win_odds
    FROM runners ru
    WHERE ru.race_id = ?
    ORDER BY ru.horse_no ASC
  `).all(race_id);
}

function rollingStatsByKey({ key, keyCol, windowDays, curDate, curVenue, curRaceNo }) {
  // Compute starts/wins/places for jockey/trainer in last windowDays before curDate.
  // Excludes the current race key by strict (racedate, venue, race_no) ordering.
  if (!key) return { starts: 0, win_rate: 0, place_rate: 0 };
  const threshold = new Date(curDate.getTime() - windowDays * 24 * 3600 * 1000);
  const thY = threshold.getUTCFullYear();
  const thM = String(threshold.getUTCMonth() + 1).padStart(2, '0');
  const thD = String(threshold.getUTCDate()).padStart(2, '0');
  const thStr = `${thY}/${thM}/${thD}`;

  const row = db.prepare(`
    SELECT
      COUNT(*) AS starts,
      SUM(CASE WHEN re.finish_pos = 1 THEN 1 ELSE 0 END) AS wins,
      SUM(CASE WHEN re.finish_pos IS NOT NULL AND re.finish_pos <= 3 THEN 1 ELSE 0 END) AS places
    FROM runners ru
    JOIN races r ON r.race_id = ru.race_id
    JOIN meetings m ON m.meeting_id = r.meeting_id
    LEFT JOIN results re ON re.runner_id = ru.runner_id
    WHERE ru.${keyCol} = ?
      AND m.racedate >= ?
      AND (
        m.racedate < ?
        OR (m.racedate = ? AND (m.venue < ? OR (m.venue = ? AND r.race_no < ?)))
      )
  `).get(key, thStr, racedate, racedate, curVenue, curVenue, curRaceNo);

  const starts = Number(row?.starts ?? 0);
  const wins = Number(row?.wins ?? 0);
  const places = Number(row?.places ?? 0);
  return {
    starts,
    win_rate: starts ? wins / starts : 0,
    place_rate: starts ? places / starts : 0,
  };
}

function buildRunnerFeatures(rctx, ru) {
  const raceDate = ymdToUtcDate(rctx.racedate);
  const beforeKey = {
    beforeRaceDate: rctx.racedate,
    beforeVenue: rctx.venue,
    beforeRaceNo: rctx.race_no,
    limit: prevRuns,
    keyType: 'code'
  };

  const horseKey = ru.horse_code;
  const prev = horseKey ? getHorsePreviousRuns(db, horseKey, beforeKey) : [];
  const lastRun = prev[0] || null;

  let lastRunSectional = computeSectionalFeaturesFromSplits([]);
  let daysSinceLast = null;
  if (lastRun) {
    const splits = getSectionalSplitsForRunner(db, lastRun.runner_id);
    lastRunSectional = computeSectionalFeaturesFromSplits(splits);
    const lastDate = ymdToUtcDate(lastRun.racedate);
    if (lastDate && raceDate) daysSinceLast = daysBetweenUtc(lastDate, raceDate);
  }

  const aggKeys = [
    'prevN_avg_split_last_time',
    'prevN_avg_split_penult_time',
    'prevN_avg_kick_delta',
    'prevN_avg_pos_early',
    'prevN_avg_pos_mid',
    'prevN_avg_pos_late',
    'prevN_avg_pos_change_early_to_late',
    'prevN_avg_pos_change_mid_to_late',
    'prevN_avg_time_delta_sec',
    'prevN_avg_margin_len'
  ];
  const agg = (prev.length
    ? computeAggregatedRunFeatures(db, prev).agg
    : Object.fromEntries(aggKeys.map(k => [k, null])));

  const j = rollingStatsByKey({ key: ru.jockey, keyCol: 'jockey', windowDays: jtDays, curDate: raceDate, curVenue: rctx.venue, curRaceNo: rctx.race_no });
  const t = rollingStatsByKey({ key: ru.trainer, keyCol: 'trainer', windowDays: jtDays, curDate: raceDate, curVenue: rctx.venue, curRaceNo: rctx.race_no });

  return {
    // current race basics
    cur_distance_m: safeNum(rctx.distance_m),
    cur_class_num: safeNum(rctx.class_num),
    cur_surface: rctx.surface ?? null,
    cur_draw: safeNum(ru.draw),
    cur_weight: safeNum(ru.weight),
    cur_win_odds: safeNum(ru.win_odds),
    cur_jockey: ru.jockey ?? null,
    cur_trainer: ru.trainer ?? null,
    venue: rctx.venue,

    // rolling stats
    [`jockey_${jtDays}d_starts`]: j.starts,
    [`jockey_${jtDays}d_win_rate`]: j.win_rate,
    [`jockey_${jtDays}d_place_rate`]: j.place_rate,
    [`trainer_${jtDays}d_starts`]: t.starts,
    [`trainer_${jtDays}d_win_rate`]: t.win_rate,
    [`trainer_${jtDays}d_place_rate`]: t.place_rate,

    // debut flags
    is_debut: prev.length ? 0 : 1,
    prev_run_count: prev.length,

    // previous run deltas
    prev_days_since: daysSinceLast,
    prev_finish_pos: safeNum(lastRun?.finish_pos),
    prev_time_delta_sec: safeNum(lastRun?.time_delta_sec),
    prev_margin_len: safeNum(lastRun?.margin_len),
    prev_rating: null,

    // sectional features of last run
    prev_split_count: safeNum(lastRunSectional.split_count),
    prev_split_last_time: safeNum(lastRunSectional.split_last_time),
    prev_split_penult_time: safeNum(lastRunSectional.split_penult_time),
    prev_kick_delta: safeNum(lastRunSectional.kick_delta),
    prev_split_time_std: safeNum(lastRunSectional.split_time_std),
    prev_split_time_range: safeNum(lastRunSectional.split_time_range),
    prev_pos_early: safeNum(lastRunSectional.pos_early),
    prev_pos_mid: safeNum(lastRunSectional.pos_mid),
    prev_pos_late: safeNum(lastRunSectional.pos_late),
    prev_pos_change_early_to_late: safeNum(lastRunSectional.pos_change_early_to_late),
    prev_pos_change_mid_to_late: safeNum(lastRunSectional.pos_change_mid_to_late),

    // aggregated prevN features
    ...agg,
  };
}

const rctx = getRaceContext();
const runners = getRunnersForRace(rctx.race_id);

const out = {
  racedate,
  venue,
  raceNo,
  builtAt: new Date().toISOString(),
  source: 'sqlite',
  rows: runners.map(ru => ({
    horse_no: Number(ru.horse_no),
    horse: ru.horse_name_zh,
    horse_code: ru.horse_code,
    cur_win_odds: safeNum(ru.win_odds),
    features: buildRunnerFeatures(rctx, ru),
  }))
};

await fs.mkdir(outPath.split('/').slice(0, -1).join('/') || '.', { recursive: true });
await fs.writeFile(outPath, JSON.stringify(out, null, 2), 'utf-8');
console.log(JSON.stringify({ ok: true, out: outPath, rows: out.rows.length }, null, 0));
