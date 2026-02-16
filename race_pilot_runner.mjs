#!/usr/bin/env node
// Generic wrapper to run the pilot model (A/B) for a given race input JSON.
// Usage:
//   node race_pilot_runner.mjs --in race5_pilot_input.json --outPrefix race5
// Produces:
//   <outPrefix>.A_with_odds.json
//   <outPrefix>.B_no_odds.json

import fs from 'node:fs/promises';

const UA = 'openclaw-hkjc-pilot-runner/0.1';

function arg(name) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : null;
}

const inPath = arg('--in');
const outPrefix = arg('--outPrefix');
if (!inPath || !outPrefix) {
  console.error('Usage: node race_pilot_runner.mjs --in <race_input.json> --outPrefix <prefix>');
  process.exit(2);
}

function norm(s) { return (s ?? '').replace(/\s+/g, ' ').replace(/\u00a0/g, ' ').trim(); }
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
    time: find('完成時間'),
    bodywt: find('排位體重'),
    gear: find('配備')
  };
}

function get(row, i) { return (i >= 0 && i < row.length) ? row[i] : null; }

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

const std = JSON.parse(await fs.readFile('hkjc_standard_times_from_screenshot_2026-02-11.json', 'utf8'));
function stdLookup({ venue, surface, distance, clsNum }) {
  const distKey = String(distance);
  if (venue === 'HV' && surface === 'turf') {
    const row = std.HV_turf?.[distKey];
    const v = row?.[`G${clsNum}`];
    return v && v !== '-' ? parseFinishTimeToSec(v) : null;
  }
  if (venue === 'ST' && surface === 'turf') {
    const row = std.ST_turf?.[distKey];
    const v = row?.[`G${clsNum}`];
    return v && v !== '-' ? parseFinishTimeToSec(v) : null;
  }
  if (venue === 'ST' && surface === 'awt') {
    const row = std.ST_all_weather?.[distKey];
    const v = row?.[`G${clsNum}`];
    return v && v !== '-' ? parseFinishTimeToSec(v) : null;
  }
  return null;
}

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

function summarizeRecent(recentRuns) {
  const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
  const pick = (k) => recentRuns.map(r => r[k]).filter(v => v != null);
  const kicks = recentRuns.map(r => (r.seg3Time != null && r.seg2Time != null) ? (r.seg3Time - r.seg2Time) : null).filter(v => v != null);
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

function makeFeatureVector({ winOdds, rating, draw, weight, marginLen, timeDelta, seg2Time, seg3Time, pos1, pos3, kick, cat }, { dimHash = 256, noOdds = false } = {}) {
  const oddsForModel = noOdds ? 1 : (winOdds ?? 50);
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

async function runVariant({ variant, noOdds }) {
  const input = JSON.parse(await fs.readFile(inPath, 'utf8'));
  const race = input.race;
  const runners = input.runners;

  const SINCE_ISO = '2024-09-01';
  const LAST_N = 5;

  const sectionalCache = new Map();
  async function getSectionalRaceRows(sectionalUrl) {
    if (sectionalCache.has(sectionalUrl)) return sectionalCache.get(sectionalUrl);
    const html = await fetchHtml(sectionalUrl);
    const table = extractSectionalTable(html);
    const rows = table ? parseSectionalsRows(table) : [];
    sectionalCache.set(sectionalUrl, rows);
    return rows;
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

      if (run.winOdds != null && run.rating != null && run.draw != null && run.weight != null) {
        trainX.push(makeFeatureVector(run, { noOdds }));
        trainY.push(run.y);
      }

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
      cat: { venue: race.venue, dist: race.distanceMeters }
    }, { noOdds });

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
      recentSummary: rr.recentSummary
    });
  }

  preds.sort((a, b) => b.pTop3 - a.pTop3);

  const out = {
    race,
    model: {
      kind: 'pilot_logreg_hashed_extended',
      variant,
      since: '2024-09-01',
      lastNRunsForPrediction: 5,
      trainRows: trainX.length,
      noOdds,
      hashedDim: 256
    },
    rankings: preds
  };

  await fs.writeFile(`${outPrefix}.${variant}.json`, JSON.stringify(out, null, 2));
  return out;
}

const A = await runVariant({ variant: 'A_with_odds', noOdds: false });
const B = await runVariant({ variant: 'B_no_odds', noOdds: true });

console.log(JSON.stringify({
  outFiles: [`${outPrefix}.A_with_odds.json`, `${outPrefix}.B_no_odds.json`],
  topA: A.rankings.slice(0, 5).map(r => ({ no: r.no, horse: r.horse, p: r.pTop3, win: r.win })),
  topB: B.rankings.slice(0, 5).map(r => ({ no: r.no, horse: r.horse, p: r.pTop3, win: r.win }))
}, null, 2));
