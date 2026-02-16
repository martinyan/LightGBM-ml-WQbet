#!/usr/bin/env node
// Pilot model for HKJC HV 2026-02-11 Race 4 using:
// - bet.hkjc.com odds snapshot (manually inlined below)
// - horse history table rows (all columns) with raceUrl extraction
// - sectional times from displaysectionaltime pages
// - standard times (from screenshot JSON)
//
// Model: regularized logistic regression with feature hashing for categorical fields.

import fs from 'node:fs/promises';

const UA = 'openclaw-hkjc-r4pilot-full/0.1';

function norm(s) {
  return (s ?? '').replace(/\s+/g, ' ').replace(/\u00a0/g, ' ').trim();
}
function decodeEntities(s) {
  return (s ?? '')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)));
}
function cleanCell(html) {
  return norm(
    decodeEntities(
      (html ?? '')
        .replace(/<br\s*\/?\s*>/gi, ' ')
        .replace(/<[^>]+>/g, ' ')
    )
  );
}

async function fetchHtml(url) {
  const res = await fetch(url, { headers: { 'user-agent': UA } });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText} for ${url}`);
  return await res.text();
}

function parseDateDDMMYY(s) {
  const m = String(s || '').match(/^(\d{2})\/(\d{2})\/(\d{2})$/);
  if (!m) return null;
  const dd = Number(m[1]), mm = Number(m[2]), yy = Number(m[3]);
  const yyyy = yy + (yy >= 70 ? 1900 : 2000);
  return new Date(Date.UTC(yyyy, mm - 1, dd));
}

function parseFinishTimeToSec(s) {
  // formats: 1.10.30 (from history) or 1:10.30 (from sectionals)
  const t = String(s || '').trim();
  if (!t) return null;
  let m = t.match(/^(\d+)\.(\d{2})\.(\d{2})$/);
  if (m) return Number(m[1]) * 60 + Number(m[2]) + Number(m[3]) / 100;
  m = t.match(/^(\d+):(\d{2})\.(\d{2})$/);
  if (m) return Number(m[1]) * 60 + Number(m[2]) + Number(m[3]) / 100;
  return null;
}

function toNum(x) {
  if (x == null) return null;
  const t = String(x).replace(/[^0-9.]/g, '');
  if (!t) return null;
  return Number(t);
}

function parseMarginLen(marginText) {
  const t = String(marginText || '').trim();
  if (!t) return null;
  if (t.includes('鼻位')) return 0.05;
  if (t.includes('短馬頭位')) return 0.10;
  if (t === '頭位' || t.includes('馬頭位')) return 0.20;
  if (t.includes('頸位')) return 0.30;
  if (/^\d+(?:-\d+\/\d+|\/\d+)?$/.test(t)) {
    if (t.includes('-')) {
      const [a, frac] = t.split('-');
      const [n, d] = frac.split('/');
      return Number(a) + Number(n) / Number(d);
    }
    if (t.includes('/')) {
      const [n, d] = t.split('/');
      return Number(n) / Number(d);
    }
    return Number(t);
  }
  return null;
}

function isTop3(pos) {
  const p = String(pos || '').trim();
  return p === '01' || p === '02' || p === '03' || p === '1' || p === '2' || p === '3';
}

function parseCurrentRating(html) {
  const m = html.match(/現時評分[\s\S]{0,400}?>(\d+)<\/td>/i) || html.match(/現時評分[\s\S]{0,400}?(\d{1,3})/i);
  return m ? Number(m[1]) : null;
}

function parseVenueSurface(courseStr) {
  const s = String(courseStr || '');
  const venue = s.includes('跑馬地') ? 'HV' : (s.includes('沙田') ? 'ST' : null);
  const surface = s.includes('全天候') ? 'awt' : (s.includes('草地') ? 'turf' : null);
  return { venue, surface };
}

function parseClassNum(clsStr) {
  const t = String(clsStr || '').trim();
  if (!t) return null;
  const m = t.match(/^(\d)$/) || t.match(/第(\d)班/);
  return m ? Number(m[1]) : null;
}

function toSectionalUrlFromRaceUrl(raceUrl) {
  const u = new URL(raceUrl);
  const racedate = u.searchParams.get('racedate');
  const [yyyy, mm, dd] = racedate.split('/');
  const raceNo = u.searchParams.get('RaceNo');
  return `https://racing.hkjc.com/zh-hk/local/information/displaysectionaltime?racedate=${dd}/${mm}/${yyyy}&RaceNo=${raceNo}`;
}

// ----- Parse horse history table with raceUrl per row -----
function extractHistoryTableHtml(html) {
  const tables = [...html.matchAll(/<table[\s\S]*?<\/table>/gi)].map(m => m[0]);
  const candidates = tables.filter(t => t.includes('場次') && t.includes('名次') && t.includes('日期') && t.includes('獨贏'));
  if (!candidates.length) return null;
  return candidates.sort((a, b) => b.length - a.length)[0];
}

function parseHistoryRowsWithRaceUrl(tableHtml, baseUrl) {
  const trList = [...tableHtml.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map(m => m[1]);
  const rows = trList.map(tr => {
    const cells = [...tr.matchAll(/<t[dh][^>]*>([\s\S]*?)<\/t[dh]>/gi)].map(m => cleanCell(m[1]));
    return { tr, cells };
  }).filter(x => x.cells.length);

  const headerRowObj = rows.find(r => r.cells.includes('場次') && r.cells.includes('名次') && r.cells.includes('日期'));
  if (!headerRowObj) return { header: null, rows: [] };
  const header = headerRowObj.cells;
  const idxHeader = rows.indexOf(headerRowObj);

  const data = rows.slice(idxHeader + 1)
    .filter(r => r.cells.some(x => /^\d{2}\/\d{2}\/\d{2}$/.test(x)))
    .map(r => {
      const hrefRel = r.tr.match(/href="([^"]*localresults\?[^\"]+)"/i)?.[1] ?? null;
      const raceUrl = hrefRel ? new URL(decodeEntities(hrefRel), baseUrl).toString() : null;
      return { cells: r.cells, raceUrl };
    });

  return { header, rows: data };
}

function idxMap(header) {
  const eq = (a, b) => (a ?? '').replace(/\s+/g, '') === (b ?? '').replace(/\s+/g, '');
  const find = (name) => header.findIndex(x => eq(x, name));
  return {
    raceNo: find('場次'),
    pos: find('名次'),
    date: find('日期'),
    course: header.findIndex(x => x.includes('馬場') || x.includes('跑道') || x.includes('賽道')),
    dist: find('途程'),
    going: find('場地狀況'),
    class: find('賽事班次'),
    draw: find('檔位'),
    rating: find('評分'),
    trainer: find('練馬師'),
    jockey: find('騎師'),
    margin: header.findIndex(x => eq(x, '頭馬距離')),
    odds: header.findIndex(x => eq(x, '獨贏賠率')),
    weight: find('實際負磅'),
    runpos: find('沿途走位'),
    time: find('完成時間'),
    bodywt: find('排位體重'),
    gear: find('配備'),
  };
}

function get(row, i) {
  return (i >= 0 && i < row.length) ? row[i] : null;
}

// ----- Sectionals parse -----
function extractSectionalTable(html) {
  const tables = [...html.matchAll(/<table[\s\S]*?<\/table>/gi)].map(m => m[0]);
  return tables.find(x => x.includes('過終點') && x.includes('分段時間')) || null;
}
function parseSegment(segText) {
  const parts = norm(segText).split(' ').filter(Boolean);
  if (parts.length < 2) return null;
  const pos = /^\d+$/.test(parts[0]) ? Number(parts[0]) : null;
  const idxFirstTime = parts.findIndex(p => /^\d+\.\d{2}$/.test(p) || /^\d{1,2}:\d{2}\.\d{2}$/.test(p));
  const times = parts.filter(p => /^\d+\.\d{2}$/.test(p) || /^\d{1,2}:\d{2}\.\d{2}$/.test(p));
  const margin = idxFirstTime > 1 ? parts.slice(1, idxFirstTime).join(' ') : null;
  const segmentTime = times.length ? times[0] : null;
  const splits = times.length > 1 ? times.slice(1) : [];
  return { pos, margin, segmentTime, splits };
}
function parseSectionalsRows(tableHtml) {
  const trList = [...tableHtml.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map(m => m[1]);
  const rows = [];
  for (const tr of trList) {
    if (!tr.includes('<td')) continue;
    const cells = [...tr.matchAll(/<td[^>]*>([\s\S]*?)<\/td>/gi)].map(m => cleanCell(m[1]));
    if (cells.length < 6) continue;
    const finishPos = cells[0];
    if (!/^\d+$/.test(finishPos)) continue;
    const finalTime = cells.find(c => /^\d{1,2}:\d{2}\.\d{2}$/.test(c)) || null;
    rows.push({
      finishPos: Number(finishPos),
      horseNo: cells[1],
      horseName: cells[2],
      seg1: parseSegment(cells[3]),
      seg2: parseSegment(cells[4]),
      seg3: parseSegment(cells[5]),
      finalTime
    });
  }
  return rows;
}

// ----- Standard time lookup (from screenshot JSON) -----
const std = JSON.parse(await fs.readFile('hkjc_standard_times_from_screenshot_2026-02-11.json', 'utf8'));
function stdLookup({ venue, surface, distance, clsNum }) {
  const distKey = String(distance);
  if (venue === 'HV' && surface === 'turf') {
    const row = std.HV_turf?.[distKey];
    if (!row) return null;
    const v = row[`G${clsNum}`];
    return v && v !== '-' ? parseFinishTimeToSec(v) : null;
  }
  if (venue === 'ST' && surface === 'turf') {
    const row = std.ST_turf?.[distKey];
    if (!row) return null;
    const v = row[`G${clsNum}`];
    return v && v !== '-' ? parseFinishTimeToSec(v) : null;
  }
  if (venue === 'ST' && surface === 'awt') {
    const row = std.ST_all_weather?.[distKey];
    if (!row) return null;
    const v = row[`G${clsNum}`];
    return v && v !== '-' ? parseFinishTimeToSec(v) : null;
  }
  return null;
}

// ----- Feature hashing -----
function fnv1a(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
function addHashed(vec, dim, key, value = 1) {
  const idx = fnv1a(key) % dim;
  vec[idx] += value;
}

// Logistic regression (batch GD)
function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function trainLogReg(X, y, { lr = 0.1, epochs = 2500, l2 = 1e-2 } = {}) {
  const n = X.length, d = X[0].length;
  const w = new Array(d).fill(0);
  for (let ep = 0; ep < epochs; ep++) {
    const grad = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      const p = sigmoid(dot(w, X[i]));
      const err = p - y[i];
      for (let j = 0; j < d; j++) grad[j] += err * X[i][j];
    }
    for (let j = 0; j < d; j++) {
      grad[j] = grad[j] / n + l2 * w[j];
      w[j] -= lr * grad[j];
    }
  }
  return w;
}
function predict(w, x) { return sigmoid(dot(w, x)); }

// ----- Pilot data: today's runners (odds snapshot we scraped) -----
const race = { racedate: '2026-02-11', venue: 'HV', raceNo: 4, scheduledTime: '20:10', distanceMeters: 1200 };
const runners = [
  { no: 1, horse: '巴閉王', draw: 12, wt: 135, win: 8.9, place: 2.7, horseUrl: 'https://racing.hkjc.com/zh-hk/local/information/horse?horseid=HK_2024_K479' },
  { no: 2, horse: '紅愛舍', draw: 8, wt: 133, win: 4.1, place: 1.8, horseUrl: 'https://racing.hkjc.com/zh-hk/local/information/horse?horseid=HK_2022_H110' },
  { no: 3, horse: '運來伍寶', draw: 5, wt: 129, win: 9.6, place: 2.6, horseUrl: 'https://racing.hkjc.com/zh-hk/local/information/horse?horseid=HK_2023_J153' },
  { no: 4, horse: '風火猴王', draw: 9, wt: 129, win: 20, place: 6.8, horseUrl: 'https://racing.hkjc.com/zh-hk/local/information/horse?horseid=HK_2023_J511' },
  { no: 5, horse: '悠然常勝', draw: 2, wt: 127, win: 32, place: 7.2, horseUrl: 'https://racing.hkjc.com/zh-hk/local/information/horse?horseid=HK_2025_L136' },
  { no: 6, horse: '比特星', draw: 6, wt: 126, win: 3.3, place: 1.4, horseUrl: 'https://racing.hkjc.com/zh-hk/local/information/horse?horseid=HK_2023_J362' },
  { no: 7, horse: '多多好馬', draw: 1, wt: 124, win: 6.2, place: 2.3, horseUrl: 'https://racing.hkjc.com/zh-hk/local/information/horse?horseid=HK_2024_K403' },
  { no: 8, horse: '靖哥哥', draw: 11, wt: 121, win: 55, place: 10, horseUrl: 'https://racing.hkjc.com/zh-hk/local/information/horse?horseid=HK_2024_K503' },
  { no: 9, horse: '文明福星', draw: 4, wt: 120, win: 17, place: 3.7, horseUrl: 'https://racing.hkjc.com/zh-hk/local/information/horse?horseid=HK_2023_J315' },
  { no: 10, horse: '幽默大師', draw: 10, wt: 119, win: 57, place: 12, horseUrl: 'https://racing.hkjc.com/zh-hk/local/information/horse?horseid=HK_2024_K281' },
  { no: 11, horse: '眼健康', draw: 7, wt: 119, win: 21, place: 5.3, horseUrl: 'https://racing.hkjc.com/zh-hk/local/information/horse?horseid=HK_2024_K455' },
  { no: 12, horse: '快樂高球', draw: 3, wt: 118, win: 13, place: 3.7, horseUrl: 'https://racing.hkjc.com/zh-hk/local/information/horse?horseid=HK_2023_J222' }
];

const SINCE_ISO = '2024-09-01';
const LAST_N = 5;

// Preload horse histories + sectionals for last N
const sectionalCache = new Map();
async function getSectionalRaceRows(sectionalUrl) {
  if (sectionalCache.has(sectionalUrl)) return sectionalCache.get(sectionalUrl);
  const html = await fetchHtml(sectionalUrl);
  const table = extractSectionalTable(html);
  const rows = table ? parseSectionalsRows(table) : [];
  sectionalCache.set(sectionalUrl, rows);
  return rows;
}

const NO_ODDS = process.env.NO_ODDS === '1' || process.env.NO_ODDS === 'true';

function makeFeatureVector({
  winOdds,
  rating,
  draw,
  weight,
  marginLen,
  timeDelta,
  seg2Time,
  seg3Time,
  pos1,
  pos3,
  kick,
  cat
}, { dimHash = 256 } = {}) {
  const oddsForModel = NO_ODDS ? 1 : (winOdds ?? 50);

  const numeric = [
    1,
    Math.log(oddsForModel),
    rating ?? 0,
    draw ?? 0,
    weight ?? 0,
    marginLen ?? 0,
    timeDelta ?? 0,
    seg2Time ?? 0,
    seg3Time ?? 0,
    pos1 ?? 0,
    pos3 ?? 0,
    kick ?? 0,
  ];
  const hashed = new Array(dimHash).fill(0);
  for (const [k, v] of Object.entries(cat || {})) {
    if (v == null || v === '') continue;
    addHashed(hashed, dimHash, `${k}=${String(v)}`, 1);
  }
  return numeric.concat(hashed);
}

function summarizeRecent(recentRuns) {
  const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
  const pick = (k) => recentRuns.map(r => r[k]).filter(v => v != null);

  // Kick = seg3Time - seg2Time (rough proxy; depends on distance)
  const kicks = recentRuns
    .map(r => (r.seg3Time != null && r.seg2Time != null) ? (r.seg3Time - r.seg2Time) : null)
    .filter(v => v != null);

  return {
    n: recentRuns.length,
    top3Rate: recentRuns.length ? recentRuns.filter(r => r.y === 1).length / recentRuns.length : 0,
    avgMarginLen: avg(pick('marginLen')),
    avgTimeDelta: avg(pick('timeDelta')),
    avgSeg2Time: avg(pick('seg2Time')),
    avgSeg3Time: avg(pick('seg3Time')),
    avgPos1: avg(pick('pos1')),
    avgPos3: avg(pick('pos3')),
    avgKick: avg(kicks)
  };
}

const trainX = [];
const trainY = [];
const runnerRecent = [];

for (const runner of runners) {
  const html = await fetchHtml(runner.horseUrl);
  const ratingNow = parseCurrentRating(html);

  const tableHtml = extractHistoryTableHtml(html);
  if (!tableHtml) throw new Error(`No history table for ${runner.horse}`);
  const parsed = parseHistoryRowsWithRaceUrl(tableHtml, runner.horseUrl);
  const im = idxMap(parsed.header);

  const recent = [];
  for (const row of parsed.rows) {
    const dateStr = get(row.cells, im.date);
    const dateIso = parseDateDDMMYY(dateStr)?.toISOString().slice(0, 10);
    if (!dateIso || dateIso < SINCE_ISO) continue;

    const dist = toNum(get(row.cells, im.dist));
    const clsNum = parseClassNum(get(row.cells, im.class));
    const { venue, surface } = parseVenueSurface(get(row.cells, im.course));

    const finishTimeSec = parseFinishTimeToSec(get(row.cells, im.time));
    const stdTimeSec = (venue && surface && dist && clsNum) ? stdLookup({ venue, surface, distance: dist, clsNum }) : null;
    const timeDelta = (finishTimeSec != null && stdTimeSec != null) ? (finishTimeSec - stdTimeSec) : null;

    // Sectionals (optional; only if raceUrl exists and sectional page exists)
    let seg2Time = null, seg3Time = null, pos1 = null, pos3 = null;
    if (row.raceUrl) {
      const sectionalUrl = toSectionalUrlFromRaceUrl(row.raceUrl);
      const rowsSec = await getSectionalRaceRows(sectionalUrl);
      const secRow = rowsSec.find(r => String(r.horseName || '').includes(runner.horse));
      if (secRow) {
        seg2Time = toNum(secRow.seg2?.segmentTime);
        seg3Time = toNum(secRow.seg3?.segmentTime);
        pos1 = secRow.seg1?.pos ?? null;
        pos3 = secRow.seg3?.pos ?? null;
      }
    }

    const run = {
      dateIso,
      y: isTop3(get(row.cells, im.pos)) ? 1 : 0,
      winOdds: toNum(get(row.cells, im.odds)),
      rating: toNum(get(row.cells, im.rating)),
      draw: toNum(get(row.cells, im.draw)),
      weight: toNum(get(row.cells, im.weight)),
      marginLen: parseMarginLen(get(row.cells, im.margin)),
      timeDelta,
      seg2Time,
      seg3Time,
      pos1,
      pos3,
      cat: {
        course: get(row.cells, im.course),
        going: get(row.cells, im.going),
        class: get(row.cells, im.class),
        dist: get(row.cells, im.dist),
        jockey: get(row.cells, im.jockey),
        trainer: get(row.cells, im.trainer),
        gear: get(row.cells, im.gear)
      }
    };

    // For training we use all runs since SINCE_ISO (not only last 5) to have more samples.
    if (run.winOdds != null && run.rating != null && run.draw != null && run.weight != null) {
      trainX.push(makeFeatureVector(run));
      trainY.push(run.y);
    }

    // Keep last 5 runs for per-runner prediction summary
    if (recent.length < LAST_N) recent.push(run);
  }

  runnerRecent.push({
    ...runner,
    ratingNow,
    recent,
    recentSummary: summarizeRecent(recent)
  });
}

// Standardize numeric part only (first 12)
const D_NUM = 12;
const d = trainX[0].length;
const means = new Array(D_NUM).fill(0);
const stds = new Array(D_NUM).fill(1);
for (let j = 1; j < D_NUM; j++) {
  let s = 0;
  for (const row of trainX) s += row[j];
  means[j] = s / trainX.length;
  let v = 0;
  for (const row of trainX) v += (row[j] - means[j]) ** 2;
  stds[j] = Math.sqrt(v / trainX.length) || 1;
  for (const row of trainX) row[j] = (row[j] - means[j]) / stds[j];
}

const w = trainLogReg(trainX, trainY);

// Predict: use today's odds + current rating/draw/weight, and recent aggregates for the rest
const preds = [];
for (const rr of runnerRecent) {
  const s = rr.recentSummary;
  const feat = makeFeatureVector({
    winOdds: rr.win,
    rating: rr.ratingNow,
    draw: rr.draw,
    weight: rr.wt,
    marginLen: s.avgMarginLen,
    timeDelta: s.avgTimeDelta,
    seg2Time: s.avgSeg2Time,
    seg3Time: s.avgSeg3Time,
    pos1: s.avgPos1,
    pos3: s.avgPos3,
    kick: s.avgKick,
    cat: {
      // include some stable fields from today's race context
      venue: race.venue,
      dist: race.distanceMeters,
    }
  });
  for (let j = 1; j < D_NUM; j++) feat[j] = (feat[j] - means[j]) / stds[j];
  const p = predict(w, feat);
  preds.push({
    no: rr.no,
    horse: rr.horse,
    win: rr.win,
    place: rr.place,
    draw: rr.draw,
    weight: rr.wt,
    rating: rr.ratingNow,
    pTop3: Number(p.toFixed(4)),
    recentSummary: rr.recentSummary,
    recentRuns: rr.recent.map(r => ({
      dateIso: r.dateIso,
      yTop3: r.y,
      winOdds: r.winOdds,
      dist: r.cat.dist,
      class: r.cat.class,
      course: r.cat.course,
      going: r.cat.going,
      marginLen: r.marginLen,
      timeDelta: r.timeDelta,
      seg2Time: r.seg2Time,
      seg3Time: r.seg3Time,
      pos1: r.pos1,
      pos3: r.pos3
    }))
  });
}

preds.sort((a, b) => b.pTop3 - a.pTop3);

const out = {
  race,
  model: {
    kind: 'pilot_logreg_hashed_extended',
    variant: NO_ODDS ? 'B_without_odds' : 'A_with_odds',
    since: SINCE_ISO,
    lastNRunsForPrediction: LAST_N,
    trainRows: trainX.length,
    numericFeatures: ['log(winOdds)','rating','draw','weight','marginLen','timeDelta','seg2Time','seg3Time','pos1','pos3','kick'],
    hashedDim: 256,
    categoricalHashed: ['course','going','class','dist','jockey','trainer','gear'],
    noOdds: NO_ODDS
  },
  rankings: preds
};

const outPath = NO_ODDS ? 'race4_pilot_prediction_extended_v2.B_no_odds.json' : 'race4_pilot_prediction_extended_v2.A_with_odds.json';
await fs.writeFile(outPath, JSON.stringify(out, null, 2));

console.log(JSON.stringify({
  trainRows: trainX.length,
  top5: preds.slice(0, 5).map(p => ({ no: p.no, horse: p.horse, win: p.win, place: p.place, rating: p.rating, pTop3: p.pTop3 }))
}, null, 2));
