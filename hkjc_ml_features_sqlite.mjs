// Shared helpers for HKJC SQLite ML features
// Uses: better-sqlite3

import Database from 'better-sqlite3';

export function openDb(dbPath, { readonly = true } = {}) {
  return new Database(dbPath, { readonly });
}

export function ymdToUtcDate(ymd) {
  // ymd: YYYY/MM/DD
  const [Y, M, D] = String(ymd).split('/').map(Number);
  if (!Y || !M || !D) return null;
  return new Date(Date.UTC(Y, M - 1, D));
}

export function daysBetweenUtc(a, b) {
  // b-a in days
  const ms = b.getTime() - a.getTime();
  return ms / (24 * 3600 * 1000);
}

export function safeNum(x) {
  if (x == null) return null;
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

export function mean(nums) {
  const xs = (nums || []).filter(n => Number.isFinite(n));
  if (!xs.length) return null;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

export function computeSectionalFeaturesFromSplits(splits) {
  // splits: [{split_idx,pos,split_time}...] ordered by split_idx
  const K = splits?.length || 0;
  if (!K) {
    return {
      split_count: 0,
      split_last_time: null,
      split_penult_time: null,
      kick_delta: null,
      pos_early: null,
      pos_mid: null,
      pos_late: null,
      pos_change_early_to_late: null,
      pos_change_mid_to_late: null
    };
  }

  const last = splits[K - 1];
  const pen = K >= 2 ? splits[K - 2] : null;

  const early = splits[0];
  const mid = splits[Math.floor((K - 1) / 2)];
  const late = last;

  const split_last_time = safeNum(last.split_time);
  const split_penult_time = safeNum(pen?.split_time);

  // Kick: how much faster the last split is vs the previous split (positive is a stronger kick)
  const kick_delta = (split_penult_time != null && split_last_time != null)
    ? (split_penult_time - split_last_time)
    : null;

  const pos_early = safeNum(early.pos);
  const pos_mid = safeNum(mid.pos);
  const pos_late = safeNum(late.pos);

  const pos_change_early_to_late = (pos_early != null && pos_late != null) ? (pos_early - pos_late) : null;
  const pos_change_mid_to_late = (pos_mid != null && pos_late != null) ? (pos_mid - pos_late) : null;

  return {
    split_count: K,
    split_last_time,
    split_penult_time,
    kick_delta,
    pos_early,
    pos_mid,
    pos_late,
    pos_change_early_to_late,
    pos_change_mid_to_late
  };
}

export function getSectionalSplitsForRunner(db, runnerId) {
  return db.prepare(
    `SELECT split_idx, split_label, pos, split_time
     FROM sectional_splits
     WHERE runner_id=?
     ORDER BY split_idx ASC`
  ).all(runnerId);
}

export function getRunnerContextRows(db) {
  // One row per runner start with meeting/race context.
  // After horse_code backfill, use it as the primary key.
  return db.prepare(
    `SELECT
       m.racedate AS racedate,
       m.venue AS venue,
       r.race_no AS race_no,
       r.distance_m AS distance_m,
       r.class_num AS class_num,
       r.surface AS surface,
       ru.runner_id AS runner_id,
       ru.horse_code AS horse_code,
       ru.horse_name_zh AS horse_name_zh,
       ru.horse_no AS horse_no,
       ru.draw AS draw,
       ru.weight AS weight,
       ru.jockey AS jockey,
       ru.trainer AS trainer,
       ru.win_odds AS win_odds,
       re.finish_pos AS finish_pos,
       re.margin_len AS margin_len,
       re.time_delta_sec AS time_delta_sec
     FROM runners ru
     JOIN races r ON r.race_id = ru.race_id
     JOIN meetings m ON m.meeting_id = r.meeting_id
     LEFT JOIN results re ON re.runner_id = ru.runner_id
     WHERE ru.horse_code IS NOT NULL AND ru.horse_code != ''
     ORDER BY m.racedate ASC, m.venue ASC, r.race_no ASC, ru.horse_no ASC`
  ).all();
}

export function getHorsePreviousRuns(db, horseKey, { beforeRaceDate, beforeVenue, beforeRaceNo, limit = 5, keyType = 'code' } = {}) {
  // Fetch previous runs strictly before the (date, venue, race_no) key.
  // horseKey can be horse_code or horse_name_zh depending on keyType.
  const whereClause = (keyType === 'code')
    ? 'ru.horse_code = ?'
    : (keyType === 'name')
      ? 'ru.horse_name_zh = ?'
      : '(ru.horse_code = ? OR ru.horse_name_zh = ?)';

  if(keyType === 'auto') throw new Error('keyType=auto no longer supported after horse_code backfill; use keyType=code');

  const sql =
    `SELECT
       m.racedate AS racedate,
       m.venue AS venue,
       r.race_no AS race_no,
       r.distance_m AS distance_m,
       r.class_num AS class_num,
       r.surface AS surface,
       ru.runner_id AS runner_id,
       ru.horse_code AS horse_code,
       ru.horse_name_zh AS horse_name_zh,
       ru.draw AS draw,
       ru.weight AS weight,
       ru.jockey AS jockey,
       ru.trainer AS trainer,
       ru.win_odds AS win_odds,
       re.finish_pos AS finish_pos,
       re.margin_len AS margin_len,
       re.time_delta_sec AS time_delta_sec
     FROM runners ru
     JOIN races r ON r.race_id = ru.race_id
     JOIN meetings m ON m.meeting_id = r.meeting_id
     LEFT JOIN results re ON re.runner_id = ru.runner_id
     WHERE ${whereClause}
       AND (
         m.racedate < ?
         OR (m.racedate = ? AND (m.venue < ? OR (m.venue = ? AND r.race_no < ?)))
       )
     ORDER BY m.racedate DESC, m.venue DESC, r.race_no DESC
     LIMIT ?`;

  if (keyType === 'auto') {
    return db.prepare(sql).all(horseKey, horseKey, beforeRaceDate, beforeRaceDate, beforeVenue, beforeVenue, beforeRaceNo, limit);
  }
  return db.prepare(sql).all(horseKey, beforeRaceDate, beforeRaceDate, beforeVenue, beforeVenue, beforeRaceNo, limit);
}

export function computeAggregatedRunFeatures(db, runs) {
  // Aggregate across previous runs (runs[0] is most recent)
  const perRun = runs.map(run => {
    const splits = getSectionalSplitsForRunner(db, run.runner_id);
    const sfeat = computeSectionalFeaturesFromSplits(splits);
    return {
      ...run,
      ...sfeat
    };
  });

  const f = (k) => mean(perRun.map(r => r[k]));
  return {
    prev_runs: perRun,
    agg: {
      prevN_avg_split_last_time: f('split_last_time'),
      prevN_avg_split_penult_time: f('split_penult_time'),
      prevN_avg_kick_delta: f('kick_delta'),
      prevN_avg_pos_early: f('pos_early'),
      prevN_avg_pos_mid: f('pos_mid'),
      prevN_avg_pos_late: f('pos_late'),
      prevN_avg_pos_change_early_to_late: f('pos_change_early_to_late'),
      prevN_avg_pos_change_mid_to_late: f('pos_change_mid_to_late'),
      prevN_avg_time_delta_sec: f('time_delta_sec'),
      prevN_avg_margin_len: f('margin_len')
    }
  };
}
