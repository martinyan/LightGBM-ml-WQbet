#!/usr/bin/env node
/* Build per-runner ML training rows from SQLite (hkjc.sqlite).

   Each row corresponds to a runner start (the label is that start's result),
   with features computed from the horse's previous run sectional_splits + result deltas,
   plus current-race basics (draw/weight/odds) and jockey/trainer rolling 365d stats
   as-of the race (date+race_no order).

   Output: JSONL (one JSON object per line).

   Usage:
     node hkjc_ml_build_dataset_sqlite.mjs --db hkjc.sqlite --out dataset.jsonl --prevRuns 1

   Notes:
   - rating is not currently available in the DB schema; emitted as null.
*/

import fs from 'node:fs/promises';
import {
  openDb,
  ymdToUtcDate,
  daysBetweenUtc,
  getRunnerContextRows,
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
const outPath = arg('--out', 'hkjc_ml_dataset.jsonl');
const prevRuns = Math.max(1, Number(arg('--prevRuns', '1')));
const minPrevRuns = Math.max(0, Number(arg('--minPrevRuns', String(prevRuns))));

const db = openDb(dbPath, { readonly: true });

// Rolling window helpers
function makeRollingStatsWindow(windowDays = 365) {
  const byKey = new Map();

  function ensure(key) {
    if (!byKey.has(key)) byKey.set(key, { events: [], starts: 0, wins: 0, places: 0 });
    return byKey.get(key);
  }

  function prune(state, curDate) {
    // remove events with date < curDate - windowDays
    const threshold = new Date(curDate.getTime() - windowDays * 24 * 3600 * 1000);
    let idx = 0;
    while (idx < state.events.length && state.events[idx].date < threshold) {
      const ev = state.events[idx];
      state.starts -= 1;
      state.wins -= ev.win;
      state.places -= ev.place;
      idx++;
    }
    if (idx > 0) state.events = state.events.slice(idx);
  }

  function getRates(key, curDate) {
    if (!key) return { starts: 0, win_rate: 0, place_rate: 0 };
    const state = ensure(key);
    prune(state, curDate);
    const starts = state.starts;
    return {
      starts,
      win_rate: starts ? state.wins / starts : 0,
      place_rate: starts ? state.places / starts : 0
    };
  }

  function addEvent(key, date, finishPos) {
    if (!key) return;
    const state = ensure(key);
    const win = finishPos === 1 ? 1 : 0;
    const place = (finishPos != null && finishPos <= 3) ? 1 : 0;
    state.events.push({ date, win, place });
    state.starts += 1;
    state.wins += win;
    state.places += place;
  }

  return { getRates, addEvent };
}

const jockeyStats = makeRollingStatsWindow(365);
const trainerStats = makeRollingStatsWindow(365);

const rows = getRunnerContextRows(db);
let outLines = 0;
let skippedNoPrev = 0;

const out = await fs.open(outPath, 'w');

for (const r of rows) {
  const raceDate = ymdToUtcDate(r.racedate);
  if (!raceDate) continue;

  // compute as-of stats BEFORE adding this runner's outcome
  const jockey = r.jockey;
  const trainer = r.trainer;

  const j = jockeyStats.getRates(jockey, raceDate);
  const t = trainerStats.getRates(trainer, raceDate);

  // previous runs for the horse
  const horseKey = r.horse_code || r.horse_name_zh;
  const prev = horseKey ? getHorsePreviousRuns(db, horseKey, {
    beforeRaceDate: r.racedate,
    beforeVenue: r.venue,
    beforeRaceNo: r.race_no,
    limit: prevRuns,
    keyType: r.horse_code ? 'code' : 'name'
  }) : [];

  if (prev.length < minPrevRuns) {
    skippedNoPrev++;
  } else {
    const lastRun = prev[0] || null;
    let lastRunSectional = {};
    let daysSinceLast = null;

    if (lastRun) {
      const splits = getSectionalSplitsForRunner(db, lastRun.runner_id);
      lastRunSectional = computeSectionalFeaturesFromSplits(splits);

      const lastDate = ymdToUtcDate(lastRun.racedate);
      if (lastDate) daysSinceLast = daysBetweenUtc(lastDate, raceDate);
    }

    const agg = computeAggregatedRunFeatures(db, prev).agg;

    const finishPos = safeNum(r.finish_pos);

    const row = {
      // identity
      racedate: r.racedate,
      venue: r.venue,
      race_no: r.race_no,
      runner_id: r.runner_id,
      horse_code: r.horse_code,
      horse_name_zh: r.horse_name_zh,
      horse_no: r.horse_no,

      // label(s)
      y_finish_pos: finishPos,
      y_win: finishPos === 1 ? 1 : 0,
      y_place: (finishPos != null && finishPos <= 3) ? 1 : 0,

      // current race basics (known before start)
      cur_distance_m: safeNum(r.distance_m),
      cur_class_num: safeNum(r.class_num),
      cur_surface: r.surface ?? null,
      cur_draw: safeNum(r.draw),
      cur_weight: safeNum(r.weight),
      cur_win_odds: safeNum(r.win_odds),
      cur_jockey: jockey ?? null,
      cur_trainer: trainer ?? null,

      // rolling stats as-of this start
      jockey_365d_starts: j.starts,
      jockey_365d_win_rate: j.win_rate,
      jockey_365d_place_rate: j.place_rate,
      trainer_365d_starts: t.starts,
      trainer_365d_win_rate: t.win_rate,
      trainer_365d_place_rate: t.place_rate,

      // previous run result deltas
      prev_days_since: daysSinceLast,
      prev_finish_pos: safeNum(lastRun?.finish_pos),
      prev_time_delta_sec: safeNum(lastRun?.time_delta_sec),
      prev_margin_len: safeNum(lastRun?.margin_len),

      // rating placeholder (not in DB yet)
      prev_rating: null,

      // previous run sectionals-derived
      ...Object.fromEntries(Object.entries(lastRunSectional).map(([k, v]) => [`prev_${k}`, v])),

      // previous N aggregation
      ...agg
    };

    await out.write(JSON.stringify(row) + '\n');
    outLines++;
  }

  // After emitting (or skipping), add this runner's outcome to rolling windows
  // so later races (same day with higher race_no) can use it.
  if (r.finish_pos != null) {
    jockeyStats.addEvent(jockey, raceDate, Number(r.finish_pos));
    trainerStats.addEvent(trainer, raceDate, Number(r.finish_pos));
  }
}

await out.close();

console.error(JSON.stringify({
  db: dbPath,
  out: outPath,
  total_starts: rows.length,
  wrote_rows: outLines,
  skipped_no_prev: skippedNoPrev,
  prevRuns,
  minPrevRuns
}, null, 2));
