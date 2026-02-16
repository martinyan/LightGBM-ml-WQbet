#!/usr/bin/env node
/* Predict top-5 per race using ONLY SQLite history (no horse-page scraping).

   Input: race card scrape JSON with odds, e.g.:
     { betPage:{url,scrapedAt,scheduledTime,distanceMeters}, picks:[{no,horse,code,draw,wt,jockey,trainer,win,place}] }

   The predictor:
   - looks up each horse's last N runs (before racedate inferred from betPage.url if possible)
   - computes sectional_splits derived features from those runs
   - adds jockey/trainer rolling 365d win/place rates as-of that date
   - produces a simple composite score and outputs top-5

   Usage:
     node hkjc_ml_predict_from_racecard_sqlite.mjs --db hkjc.sqlite --in racecard.json --lastN 3 --out pred.json

   NOTE: This is a heuristic scorer (not a trained model) but provides a DB-only pipeline.
*/

import fs from 'node:fs/promises';
import {
  openDb,
  ymdToUtcDate,
  getHorsePreviousRuns,
  computeAggregatedRunFeatures,
  safeNum
} from './hkjc_ml_features_sqlite.mjs';

function arg(name, dflt = null) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : dflt;
}

const dbPath = arg('--db', 'hkjc.sqlite');
const inPath = arg('--in');
const outPath = arg('--out', null);
const lastN = Math.max(1, Number(arg('--lastN', '3')));

if (!inPath) {
  console.error('Usage: node hkjc_ml_predict_from_racecard_sqlite.mjs --db hkjc.sqlite --in racecard.json [--lastN 3] [--out pred.json]');
  process.exit(2);
}

function inferRaceKeyFromBetUrl(url) {
  // bet url example: https://bet.hkjc.com/ch/racing/wp/2026-02-11/HV/2
  try {
    const u = new URL(url);
    const parts = u.pathname.split('/').filter(Boolean);
    const idx = parts.findIndex(p => /^\d{4}-\d{2}-\d{2}$/.test(p));
    if (idx < 0) return null;
    const dateDash = parts[idx];
    const venue = parts[idx + 1];
    const raceNo = Number(parts[idx + 2]);
    const [Y, M, D] = dateDash.split('-');
    const racedate = `${Y}/${M}/${D}`;
    return { racedate, venue, race_no: Number.isFinite(raceNo) ? raceNo : null };
  } catch {
    return null;
  }
}

function clamp(x, a, b) {
  return Math.max(a, Math.min(b, x));
}

function zLike(x, scale = 1) {
  // crude normalization: squish into [-1, 1]
  if (x == null || !Number.isFinite(x)) return 0;
  const v = x / scale;
  return clamp(v, -1, 1);
}

const db = openDb(dbPath, { readonly: true });

const card = JSON.parse(await fs.readFile(inPath, 'utf8'));
const betUrl = card?.betPage?.url ?? null;
const key = betUrl ? inferRaceKeyFromBetUrl(betUrl) : null;

const racedate = key?.racedate ?? null;
const venue = key?.venue ?? 'HV';
const raceNo = key?.race_no ?? 99;

// Rolling rates (simple SQL approach for just this prediction call)
function rollingRates(roleField, name, asOfRacedate) {
  if (!name || !asOfRacedate) return { starts: 0, win_rate: 0, place_rate: 0 };
  // use DATE comparisons; racedate stored as YYYY/MM/DD
  const starts = db.prepare(
    `SELECT COUNT(1) AS n
     FROM runners ru
     JOIN races r ON r.race_id = ru.race_id
     JOIN meetings m ON m.meeting_id = r.meeting_id
     JOIN results re ON re.runner_id = ru.runner_id
     WHERE ru.${roleField} = ?
       AND m.racedate < ?
       AND m.racedate >= strftime('%Y/%m/%d', date(replace(?, '/', '-') , '-365 day'))`
  ).get(name, asOfRacedate, asOfRacedate).n;

  const wins = db.prepare(
    `SELECT COUNT(1) AS n
     FROM runners ru
     JOIN races r ON r.race_id = ru.race_id
     JOIN meetings m ON m.meeting_id = r.meeting_id
     JOIN results re ON re.runner_id = ru.runner_id
     WHERE ru.${roleField} = ?
       AND m.racedate < ?
       AND m.racedate >= strftime('%Y/%m/%d', date(replace(?, '/', '-') , '-365 day'))
       AND re.finish_pos = 1`
  ).get(name, asOfRacedate, asOfRacedate).n;

  const places = db.prepare(
    `SELECT COUNT(1) AS n
     FROM runners ru
     JOIN races r ON r.race_id = ru.race_id
     JOIN meetings m ON m.meeting_id = r.meeting_id
     JOIN results re ON re.runner_id = ru.runner_id
     WHERE ru.${roleField} = ?
       AND m.racedate < ?
       AND m.racedate >= strftime('%Y/%m/%d', date(replace(?, '/', '-') , '-365 day'))
       AND re.finish_pos <= 3`
  ).get(name, asOfRacedate, asOfRacedate).n;

  return {
    starts,
    win_rate: starts ? wins / starts : 0,
    place_rate: starts ? places / starts : 0
  };
}

function scoreRunner({ winOdds, agg, jockeyRate, trainerRate }) {
  // Higher is better.
  // Components:
  // - market signal: lower odds => higher score
  // - performance signal: negative time_delta/margin better
  // - sectional kick: larger kick_delta better
  // - positional gain: larger pos_change better
  // - jockey/trainer win/place rates

  const oddsScore = winOdds ? (1 / winOdds) : 0;
  const timeScore = agg.prevN_avg_time_delta_sec != null ? -agg.prevN_avg_time_delta_sec : 0;
  const marginScore = agg.prevN_avg_margin_len != null ? -agg.prevN_avg_margin_len : 0;
  const kickScore = agg.prevN_avg_kick_delta != null ? agg.prevN_avg_kick_delta : 0;
  const posScore = agg.prevN_avg_pos_change_early_to_late != null ? agg.prevN_avg_pos_change_early_to_late : 0;

  const jt = 0.6 * jockeyRate.win_rate + 0.4 * jockeyRate.place_rate;
  const tt = 0.6 * trainerRate.win_rate + 0.4 * trainerRate.place_rate;

  // crude normalization so terms are commensurate
  const S =
    3.0 * zLike(oddsScore, 0.25) +
    1.5 * zLike(timeScore, 2.0) +
    1.0 * zLike(marginScore, 2.0) +
    1.0 * zLike(kickScore, 1.0) +
    0.5 * zLike(posScore, 5.0) +
    1.0 * zLike(jt, 0.25) +
    0.8 * zLike(tt, 0.25);

  return S;
}

const picks = Array.isArray(card?.picks) ? card.picks : [];

const enriched = [];
for (const p of picks) {
  const horseCode = p.code;
  const horseName = p.horse;
  const winOdds = safeNum(p.win);

  // DB built from localresults may not have horse_code populated; fall back to horse_name_zh matching.
  const prev = (horseName && racedate)
    ? getHorsePreviousRuns(db, horseName, { beforeRaceDate: racedate, beforeVenue: venue, beforeRaceNo: raceNo, limit: lastN, keyType: 'name' })
    : [];

  const { agg, prev_runs } = computeAggregatedRunFeatures(db, prev);

  const jockeyRate = rollingRates('jockey', p.jockey, racedate);
  const trainerRate = rollingRates('trainer', p.trainer, racedate);

  const score = scoreRunner({ winOdds, agg, jockeyRate, trainerRate });

  enriched.push({
    no: p.no,
    horse: p.horse,
    code: p.code,
    draw: safeNum(p.draw),
    wt: safeNum(p.wt),
    jockey: p.jockey,
    trainer: p.trainer,
    win: winOdds,
    place: safeNum(p.place),
    racedate,
    venue,
    race_no: raceNo,
    lastN,
    jockey_365d: jockeyRate,
    trainer_365d: trainerRate,
    features: {
      ...agg
    },
    prev_runs: prev_runs.map(r => ({
      racedate: r.racedate,
      venue: r.venue,
      race_no: r.race_no,
      finish_pos: r.finish_pos,
      time_delta_sec: r.time_delta_sec,
      margin_len: r.margin_len,
      split_last_time: r.split_last_time,
      split_penult_time: r.split_penult_time,
      kick_delta: r.kick_delta,
      pos_early: r.pos_early,
      pos_mid: r.pos_mid,
      pos_late: r.pos_late,
      pos_change_early_to_late: r.pos_change_early_to_late
    })),
    score
  });
}

enriched.sort((a, b) => b.score - a.score);

const outObj = {
  source: 'hkjc_ml_predict_from_racecard_sqlite.mjs',
  generatedAt: new Date().toISOString(),
  betUrl,
  racedate,
  venue,
  race_no: raceNo,
  lastN,
  top5: enriched.slice(0, 5),
  all: enriched
};

const outJson = JSON.stringify(outObj, null, 2);
if (outPath) {
  await fs.writeFile(outPath, outJson, 'utf8');
} else {
  process.stdout.write(outJson + '\n');
}
