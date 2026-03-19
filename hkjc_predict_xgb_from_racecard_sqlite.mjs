#!/usr/bin/env node
/* Predict top5 + p12 using XGB NO-ODDS REG v2 (trained in python) from a racecard scrape.

   Pipeline:
   1) Use same feature generation as LogReg predictor (hkjc_model_v2_logreg featKeys)
   2) Call python helper to load XGB model and score runners

   Usage:
     node hkjc_predict_xgb_from_racecard_sqlite.mjs \
       --db hkjc.sqlite \
       --in bet_scrape.json \
       --xgbModel models/HKJC-ML_XGB_NOODDS_REG_v2.bin \
       --xgbMeta models/HKJC-ML_XGB_NOODDS_REG_v2.infermeta.json \
       --out pred.json
*/

import fs from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import {
  openDb,
  getHorsePreviousRuns,
  getSectionalSplitsForRunner,
  computeSectionalFeaturesFromSplits,
  computeAggregatedRunFeatures,
  ymdToUtcDate,
  safeNum
} from './hkjc_ml_features_sqlite.mjs';

function arg(name, dflt = null) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : dflt;
}

const dbPath = arg('--db', 'hkjc.sqlite');
const inPath = arg('--in');
const outPath = arg('--out', null);
const xgbModelPath = arg('--xgbModel', 'models/HKJC-ML_XGB_NOODDS_REG_v2.bin');
const xgbMetaPath = arg('--xgbMeta', 'models/HKJC-ML_XGB_NOODDS_REG_v2.infermeta.json');
const lastN = Math.max(1, Number(arg('--lastN', '3')));
const topK = Math.max(1, Number(arg('--topK', '6')));
const noOdds = true;

if (!inPath) {
  console.error('Missing --in');
  process.exit(2);
}

function inferRaceKeyFromBetUrl(url) {
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

function classTextToNum(t) {
  const s = String(t || '').trim();
  if (!s) return null;
  const map = { '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6 };
  if (map[s] != null) return map[s];
  const m = s.match(/(\d+)/);
  return m ? Number(m[1]) : null;
}

function daysBetween(racedateA, racedateB) {
  const a = ymdToUtcDate(racedateA);
  const b = ymdToUtcDate(racedateB);
  if (!a || !b) return null;
  return Math.round((a.getTime() - b.getTime()) / (24 * 3600 * 1000));
}

const meta = JSON.parse(await fs.readFile(xgbMetaPath, 'utf8'));
const featKeys = meta.featKeys;
const imputeMeans = meta.imputeMeans || {};
if (!Array.isArray(featKeys) || featKeys.length < 10) throw new Error('xgb meta missing featKeys');

const card = JSON.parse(await fs.readFile(inPath, 'utf8'));
const betUrl = card?.betPage?.url ?? null;
const key = betUrl ? inferRaceKeyFromBetUrl(betUrl) : null;
const racedate = key?.racedate ?? null;
const venue = key?.venue ?? null;
const raceNo = key?.race_no ?? null;

const cur_distance_m = safeNum(card?.betPage?.distanceMeters) ?? null;
const cur_class_num = classTextToNum(card?.betPage?.classText) ?? null;

const db = openDb(dbPath, { readonly: true });

function rollingRates(roleField, name, asOfRacedate) {
  if (!name || !asOfRacedate) return { starts: 0, win_rate: 0, place_rate: 0 };

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

  const s = Number(starts) || 0;
  return {
    starts: s,
    win_rate: s ? wins / s : 0,
    place_rate: s ? places / s : 0
  };
}

function buildFeatureRow(pick) {
  const horseNo = safeNum(pick?.no) ?? null;
  const horse_code = pick?.code ?? null;
  const cur_draw = safeNum(pick?.draw) ?? null;
  const cur_weight = safeNum(pick?.wt) ?? null;
  const cur_jockey = pick?.jockey ?? null;
  const cur_trainer = pick?.trainer ?? null;
  const cur_win_odds = noOdds ? 0 : (safeNum(pick?.win) ?? null);

  const jockeyRates = rollingRates('jockey', cur_jockey, racedate);
  const trainerRates = rollingRates('trainer', cur_trainer, racedate);

  const prevRuns = (horse_code && racedate)
    ? getHorsePreviousRuns(db, horse_code, { beforeRaceDate: racedate, beforeVenue: venue, beforeRaceNo: raceNo, limit: lastN, keyType: 'code' })
    : [];
  const is_debut = prevRuns.length ? 0 : 1;
  const prev_run_count = prevRuns.length;

  const aggPack = computeAggregatedRunFeatures(db, prevRuns);
  const agg = aggPack?.agg ?? {};

  const lastRun = prevRuns?.[0] ?? null;

  // IMPORTANT: for debutants, "days since last run" is missing/undefined.
  // If we leave it null and the infermeta has no mean for this key, it will be imputed to 0,
  // which incorrectly looks like "raced very recently" and can inflate debut runners.
  // Use a conservative large value for debutants.
  const prev_days_since = is_debut
    ? 365
    : ((racedate && lastRun?.racedate) ? daysBetween(racedate, lastRun.racedate) : null);

  let prevSplit = { split_count: null, split_last_time: null, split_penult_time: null, kick_delta: null, pos_early: null, pos_mid: null, pos_late: null, pos_change_early_to_late: null, pos_change_mid_to_late: null };
  if (lastRun?.runner_id != null) {
    const splits = getSectionalSplitsForRunner(db, lastRun.runner_id);
    prevSplit = computeSectionalFeaturesFromSplits(splits);
  }

  // Keys aligned to hkjc_model_v2_logreg.json featKeys
  const row = {
    cur_class_num,
    cur_distance_m,
    cur_draw,
    cur_weight,
    cur_win_odds,

    jockey_365d_starts: jockeyRates.starts,
    jockey_365d_win_rate: jockeyRates.win_rate,
    jockey_365d_place_rate: jockeyRates.place_rate,
    trainer_365d_starts: trainerRates.starts,
    trainer_365d_win_rate: trainerRates.win_rate,
    trainer_365d_place_rate: trainerRates.place_rate,

    is_debut,
    prev_run_count,

    ...agg,

    prev_days_since,
    prev_finish_pos: safeNum(lastRun?.finish_pos),
    prev_time_delta_sec: safeNum(lastRun?.time_delta_sec),
    prev_margin_len: safeNum(lastRun?.margin_len),
    prev_rating: null,

    prev_split_count: safeNum(prevSplit?.split_count),
    prev_split_last_time: safeNum(prevSplit?.split_last_time),
    prev_split_penult_time: safeNum(prevSplit?.split_penult_time),
    prev_kick_delta: safeNum(prevSplit?.kick_delta),
    prev_pos_early: safeNum(prevSplit?.pos_early),
    prev_pos_mid: safeNum(prevSplit?.pos_mid),
    prev_pos_late: safeNum(prevSplit?.pos_late),
    prev_pos_change_early_to_late: safeNum(prevSplit?.pos_change_early_to_late),
    prev_pos_change_mid_to_late: safeNum(prevSplit?.pos_change_mid_to_late)
  };

  // vectorize
  const x = featKeys.map(k => {
    const v = row[k];
    if (k === 'cur_win_odds') return 0;
    if (v == null || !Number.isFinite(Number(v))) {
      const m = imputeMeans[k];
      return (m == null || !Number.isFinite(Number(m))) ? 0 : Number(m);
    }
    return Number(v);
  });

  return {
    horse_no: horseNo,
    horse: pick?.horse ?? null,
    horse_code,
    draw: cur_draw,
    jockey: cur_jockey,
    trainer: cur_trainer,
    x
  };
}

const picks = Array.isArray(card?.picks) ? card.picks : (Array.isArray(card?.rows) ? card.rows : []);
if (!racedate || !venue || !raceNo) {
  throw new Error(`Cannot infer racedate/venue/raceNo from betPage.url: ${betUrl}`);
}

const featRows = picks.map(buildFeatureRow).filter(r => Number.isFinite(r.horse_no));

// call python scorer
const payload = { featKeys, rows: featRows, xgbModelPath };
const tmp = `${inPath}.xgb_features.json`;
await fs.writeFile(tmp, JSON.stringify(payload));

const py = spawnSync('python3', ['-u', '-c', `
import json, sys
import numpy as np
import xgboost as xgb
p=json.load(open(sys.argv[1],'r',encoding='utf-8'))
model=xgb.Booster()
model.load_model(p['xgbModelPath'])
X=np.asarray([r['x'] for r in p['rows']],dtype=np.float32)
d=xgb.DMatrix(X, feature_names=p['featKeys'])
probs=model.predict(d)
for r,pp in zip(p['rows'], probs):
    r['p_top3']=float(pp)
print(json.dumps({'rows':p['rows']}))
` , tmp], { encoding: 'utf-8', maxBuffer: 10*1024*1024 });

if (py.status !== 0) {
  console.error(py.stdout);
  console.error(py.stderr);
  throw new Error(`python scoring failed: ${py.status}`);
}

const scored = JSON.parse(py.stdout);
const runners = scored.rows.sort((a,b) => b.p_top3 - a.p_top3);
const top = runners.slice(0, topK);
const top5 = top.slice(0, 5);
const top6 = top.slice(0, 6);
const p12 = (runners[0]?.p_top3 ?? 0) + (runners[1]?.p_top3 ?? 0);

const out = {
  model: meta.project_code,
  racedate,
  venue,
  raceNo,
  p12,
  topK,
  // Backward-compatible fields
  top5: top5.map(r => ({ horse_no: r.horse_no, horse: r.horse, p_top3: r.p_top3, draw: r.draw, jockey: r.jockey, trainer: r.trainer })),
  // Preferred going forward
  top6: top6.map(r => ({ horse_no: r.horse_no, horse: r.horse, p_top3: r.p_top3, draw: r.draw, jockey: r.jockey, trainer: r.trainer })),
  scored_all: runners.map(r => ({ horse_no: r.horse_no, horse: r.horse, p_top3: r.p_top3, draw: r.draw, jockey: r.jockey, trainer: r.trainer })),
  generatedAt: new Date().toISOString(),
  betPage: card?.betPage ?? null
};

if (outPath) await fs.writeFile(outPath, JSON.stringify(out,null,2));
console.log(JSON.stringify(out,null,2));
