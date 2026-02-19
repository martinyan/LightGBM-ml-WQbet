#!/usr/bin/env node
/* Incrementally ingest a single HKJC meeting into existing hkjc.sqlite.

   - Upserts meetings/races/horses/runners/results
   - Fetches displaysectionaltime and writes ALL split columns into sectional_splits

   Usage:
     node hkjc_ingest_meeting_into_sqlite.mjs --db hkjc.sqlite --racedate 2026/02/19 --venue ST --races 1-11

   Notes:
   - This does NOT rebuild the DB.
   - It relies on the existing schema (v1 + v2 sectional_splits).
*/

import fs from 'node:fs/promises';
import Database from 'better-sqlite3';

const UA='openclaw-hkjc-ingest/1.0';

function arg(name,dflt=null){const i=process.argv.indexOf(name);return i>=0?process.argv[i+1]:dflt;}
const dbPath=arg('--db','hkjc.sqlite');
const racedate=arg('--racedate'); // YYYY/MM/DD
const venue=arg('--venue'); // ST|HV
const racesArg=arg('--races'); // 1-11 or 1,2
const throttleMs=Number(arg('--throttleMs','250'));

if(!racedate || !venue || !racesArg){
  console.error('Usage: node hkjc_ingest_meeting_into_sqlite.mjs --db hkjc.sqlite --racedate YYYY/MM/DD --venue ST|HV --races 1-11');
  process.exit(2);
}

function sleep(ms){return new Promise(r=>setTimeout(r,ms));}
function parseRaceRange(s){
  if(s.includes('-')){const[a,b]=s.split('-').map(x=>Number(x.trim()));return Array.from({length:b-a+1},(_,i)=>a+i);} 
  return s.split(',').map(x=>Number(x.trim())).filter(Boolean);
}
function norm(s){return (s??'').replace(/\s+/g,' ').replace(/\u00a0/g,' ').trim();}
function decodeEntities(s){
  return (s??'')
    .replace(/&nbsp;/g,' ')
    .replace(/&amp;/g,'&')
    .replace(/&quot;/g,'"')
    .replace(/&#39;/g,"'")
    .replace(/&#(\d+);/g,(_,n)=>String.fromCharCode(Number(n)));
}
function cleanCell(html){
  return norm(decodeEntities((html??'').replace(/<br\s*\/?\s*>/gi,' ').replace(/<[^>]+>/g,' ')));
}
async function fetchHtml(url){
  const res = await fetch(url,{headers:{'user-agent':UA}});
  if(!res.ok) return null;
  return await res.text();
}

function parseFinishTimeToSec(s){
  const t=String(s||'').trim();
  let m=t.match(/^(\d{1,2}):(\d{2})\.(\d{2})$/);
  if(m) return Number(m[1])*60+Number(m[2])+Number(m[3])/100;
  m=t.match(/^(\d+)\.(\d{2})\.(\d{2})$/);
  if(m) return Number(m[1])*60+Number(m[2])+Number(m[3])/100;
  return null;
}

function parseMeta(html){
  const raw=html;
  const distMatch = raw.match(/(\d{3,4})\s*米/);
  const surface = raw.includes('全天候') ? 'awt' : (raw.includes('草地') ? 'turf' : null);
  const goingMatch = raw.match(/(好地|好\/+快|好\/快|濕快|濕慢|黏地|黏|軟地|重地)/);
  const railMatch = raw.match(/欄位\s*[:：]?\s*([^\s<]{1,10})/);
  const courseMatch = raw.match(/跑道\s*[:：]?\s*([^\s<]{1,20})/);
  const classDigit = raw.match(/第(\d)班/)?.[1] ?? null;
  const classZh = raw.match(/第([一二三四五])班/)?.[1] ?? null;
  const zhMap = { '一': 1, '二': 2, '三': 3, '四': 4, '五': 5 };
  const classNum = classDigit ? Number(classDigit) : (classZh ? zhMap[classZh] : null);
  const schedMatch = raw.match(/\b(\d{1,2}:\d{2})\b/);
  return {
    distance_m: distMatch ? Number(distMatch[1]) : null,
    surface,
    going: goingMatch ? goingMatch[1].replace('好/+快','好/快') : null,
    rail: railMatch ? railMatch[1] : null,
    course: courseMatch ? courseMatch[1] : null,
    class_num: classNum,
    scheduled_time: schedMatch ? schedMatch[1] : null
  };
}

function parseRunnersFromLocalresults(html){
  const tables=[...html.matchAll(/<table[\s\S]*?<\/table>/gi)].map(m=>m[0]);
  const t=tables.find(x=>x.includes('名次')&&x.includes('馬號')&&x.includes('馬名'));
  if(!t) return [];
  const trList=[...t.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map(m=>m[1]);
  const rows=trList.map(tr=>[...tr.matchAll(/<t[dh][^>]*>([\s\S]*?)<\/t[dh]>/gi)].map(m=>cleanCell(m[1])));
  const header=rows.find(r=>r.includes('名次')&&r.includes('馬號')&&r.includes('馬名'));
  if(!header) return [];

  const idxPos=header.indexOf('名次');
  const idxNo=header.indexOf('馬號');
  const idxName=header.indexOf('馬名');
  const idxDraw=header.indexOf('檔位');
  const idxJockey=header.indexOf('騎師');
  const idxTrainer=header.indexOf('練馬師');
  const idxWt=header.findIndex(x=>x.replace(/\s+/g,'')==='實際負磅');
  const idxTime=header.findIndex(x=>x.replace(/\s+/g,'')==='完成時間')
    || header.findIndex(x=>x.includes('完成')&&x.includes('時間'));
  const idxMargin=header.findIndex(x=>x.replace(/\s+/g,'')==='頭馬距離');
  const idxWinOdds=header.findIndex(x=>x.replace(/\s+/g,'')==='獨贏賠率');

  const data=rows.slice(rows.indexOf(header)+1).filter(r=>/^(\d+)$/.test(r[idxPos]||''));
  const rowHtmls=trList.slice(rows.indexOf(header)+1);

  const out=[];
  for(let i=0;i<data.length;i++){
    const r=data[i];
    const trHtml=rowHtmls[i]||'';
    const horseCode = (trHtml.match(/HorseNo=([A-Z]\d{3})/)||[])[1] || null;
    out.push({
      finish_pos: Number(r[idxPos]),
      horse_no: idxNo>=0 ? Number(r[idxNo]) : null,
      horse_name_zh: r[idxName] || null,
      horse_code: horseCode,
      draw: idxDraw>=0 ? Number(r[idxDraw]) : null,
      weight: idxWt>=0 ? Number(String(r[idxWt]).match(/\d+/)?.[0]||'') : null,
      jockey: idxJockey>=0 ? r[idxJockey] : null,
      trainer: idxTrainer>=0 ? r[idxTrainer] : null,
      finish_time_sec: idxTime>=0 ? parseFinishTimeToSec(r[idxTime]) : null,
      margin_text: idxMargin>=0 ? (r[idxMargin]||null) : null,
      win_odds: idxWinOdds>=0 ? Number(String(r[idxWinOdds]).replace(/[^0-9.]/g,''))||null : null
    });
  }
  return out;
}

function marginToLen(txt){
  const t=String(txt||'').trim();
  if(!t) return null;
  if(t==='-'||t==='—') return 0;
  const map={ '鼻位':0.05, '短馬頭位':0.1, '馬頭位':0.2, '頭位':0.3, '頸位':0.4 };
  for(const [k,v] of Object.entries(map)) if(t.includes(k)) return v;
  const m=t.match(/(\d+)-(\d+)\/(\d+)/);
  if(m){
    const a=Number(m[1]), n=Number(m[2]), d=Number(m[3]);
    return a + (d? n/d : 0);
  }
  const f=t.match(/(\d+)\/(\d+)/);
  if(f){
    const n=Number(f[1]), d=Number(f[2]);
    return d? n/d : null;
  }
  const n=t.match(/\d+(?:\.\d+)?/);
  if(n) return Number(n[0]);
  return null;
}

function extractSectionalTable(html){
  const tables=[...html.matchAll(/<table[\s\S]*?<\/table>/gi)].map(m=>m[0]);
  return tables.find(x=>x.includes('分段時間') && x.includes('過終點')) || null;
}

function parseSplitCell(txt){
  // Cells often look like:
  //   "1 23.04 11.69  11.35"
  // where the FIRST time (23.04) is the split total, and the later times are sub-splits.
  const parts=norm(txt).split(' ').filter(Boolean);
  if(parts.length<2) return {pos:null, split_time:null};
  const pos = /^\d+$/.test(parts[0]) ? Number(parts[0]) : null;
  const times = parts.filter(p=>/^\d+\.\d{2}$/.test(p)).map(Number);
  const split_time = times.length ? times[0] : null;
  return {pos, split_time};
}

function parseSectionalsDynamic(tableHtml){
  const trList=[...tableHtml.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map(m=>m[1]);
  const headerTr = trList.find(tr=>tr.includes('<th') && tr.includes('分段時間'));
  let header=[];
  if(headerTr){
    header=[...headerTr.matchAll(/<t[hd][^>]*>([\s\S]*?)<\/t[hd]>/gi)].map(m=>cleanCell(m[1]));
  }

  const out=[];
  for(const tr of trList){
    if(!tr.includes('<td')) continue;
    const cells=[...tr.matchAll(/<td[^>]*>([\s\S]*?)<\/td>/gi)].map(m=>cleanCell(m[1]));
    if(cells.length<5) continue;
    if(!/^\d+$/.test(cells[0]||'')) continue;

    const finishPos = Number(cells[0]);
    const horseNo = cells[1];
    const splitStart = 3;
    const splitCells = cells.slice(splitStart);

    let splitLabels = [];
    if(header.length>=splitCells.length){
      splitLabels = header.slice(header.length - splitCells.length);
    } else {
      splitLabels = splitCells.map((_,i)=>`split${i+1}`);
    }

    const splits = splitCells.map((c,i)=>({ split_idx:i+1, split_label: splitLabels[i] ?? null, ...parseSplitCell(c) }));
    out.push({ finishPos, horseNo, splits });
  }
  return out;
}

function sectionalUrl({racedate, raceNo}){
  const [Y,M,D]=racedate.split('/');
  return `https://racing.hkjc.com/zh-hk/local/information/displaysectionaltime?racedate=${D}/${M}/${Y}&RaceNo=${raceNo}`;
}

const db = new Database(dbPath);
// Ensure v1 + v2 schema exist
const schemaV1 = await fs.readFile(new URL('./hkjc_db_schema.sql', import.meta.url), 'utf8');
db.exec(schemaV1);
const schemaV2 = await fs.readFile(new URL('./hkjc_db_schema_v2_sectionals.sql', import.meta.url), 'utf8');
db.exec(schemaV2);

const insMeeting = db.prepare(`INSERT OR IGNORE INTO meetings (racedate, venue, going, rail, surface_hint) VALUES (?,?,?,?,?)`);
const getMeeting = db.prepare(`SELECT meeting_id FROM meetings WHERE racedate=? AND venue=?`);
const updMeeting = db.prepare(`UPDATE meetings SET going=COALESCE(going,?), rail=COALESCE(rail,?), surface_hint=COALESCE(surface_hint,?) WHERE meeting_id=?`);

const insRace = db.prepare(`INSERT OR IGNORE INTO races (meeting_id, race_no, distance_m, class_num, surface, course, scheduled_time, race_name) VALUES (?,?,?,?,?,?,?,?)`);
const getRace = db.prepare(`SELECT race_id FROM races WHERE meeting_id=? AND race_no=?`);

const insHorse = db.prepare(`INSERT OR IGNORE INTO horses (horse_code, horse_name_zh) VALUES (?,?)`);
const insRunner = db.prepare(`INSERT OR IGNORE INTO runners (race_id, horse_code, horse_no, horse_name_zh, draw, weight, jockey, trainer, win_odds) VALUES (?,?,?,?,?,?,?,?,?)`);
const getRunner = db.prepare(`SELECT runner_id FROM runners WHERE race_id=? AND horse_no=?`);
const updRunner = db.prepare(`UPDATE runners SET horse_code=COALESCE(horse_code,?), horse_name_zh=COALESCE(horse_name_zh,?), draw=?, weight=?, jockey=?, trainer=?, win_odds=? WHERE runner_id=?`);

const insResult = db.prepare(`INSERT OR REPLACE INTO results (runner_id, finish_pos, finish_time_sec, margin_text, margin_len, time_delta_sec) VALUES (?,?,?,?,?,?)`);

const delSplitsByRace = db.prepare(`DELETE FROM sectional_splits WHERE runner_id IN (SELECT runner_id FROM runners WHERE race_id=?)`);
const insSplit = db.prepare(`INSERT OR REPLACE INTO sectional_splits (runner_id, split_idx, split_label, pos, split_time) VALUES (?,?,?,?,?)`);

const raceNos = parseRaceRange(racesArg);

let ingestedRaces=0, ingestedRunners=0, ingestedSplits=0;

for (const raceNo of raceNos) {
  const resultsUrl = `https://racing.hkjc.com/zh-hk/local/information/localresults?racedate=${racedate}&Racecourse=${venue}&RaceNo=${raceNo}`;
  const html = await fetchHtml(resultsUrl);
  await sleep(throttleMs);
  if (!html || !html.includes('名次') || !html.includes('馬號') || !html.includes('馬名')) {
    console.log(JSON.stringify({ raceNo, skipped: true, reason: 'no localresults table' }));
    continue;
  }

  const meta = parseMeta(html);
  insMeeting.run(racedate, venue, meta.going, meta.rail, meta.surface);
  const meetingId = getMeeting.get(racedate, venue)?.meeting_id;
  if (!meetingId) throw new Error('meeting_id not found after insert');
  updMeeting.run(meta.going, meta.rail, meta.surface, meetingId);

  insRace.run(meetingId, raceNo, meta.distance_m, meta.class_num, meta.surface, meta.course, meta.scheduled_time, null);
  const raceId = getRace.get(meetingId, raceNo)?.race_id;
  if (!raceId) throw new Error('race_id not found after insert');

  const runners = parseRunnersFromLocalresults(html);
  if (!runners.length) {
    console.log(JSON.stringify({ raceNo, raceId, skipped: true, reason: 'no runners parsed' }));
    continue;
  }

  const winner = runners.find(r => r.finish_pos === 1);
  const winnerTime = winner?.finish_time_sec ?? null;

  const tx = db.transaction(() => {
    for (const r of runners) {
      if (r.horse_code) insHorse.run(r.horse_code, r.horse_name_zh);
      insRunner.run(raceId, r.horse_code, r.horse_no, r.horse_name_zh, r.draw, r.weight, r.jockey, r.trainer, r.win_odds);
      const runnerId = getRunner.get(raceId, r.horse_no)?.runner_id;
      if (!runnerId) continue;
      // update in case runner existed already
      updRunner.run(r.horse_code, r.horse_name_zh, r.draw, r.weight, r.jockey, r.trainer, r.win_odds, runnerId);

      const marginLen = marginToLen(r.margin_text);
      const timeDelta = (winnerTime != null && r.finish_time_sec != null) ? (r.finish_time_sec - winnerTime) : null;
      insResult.run(runnerId, r.finish_pos, r.finish_time_sec, r.margin_text, marginLen, timeDelta);
      ingestedRunners++;
    }
  });
  tx();

  // Sectionals: fetch and insert all splits for this race
  const secUrl = sectionalUrl({ racedate, raceNo });
  const secHtml = await fetchHtml(secUrl);
  await sleep(throttleMs);
  if (secHtml) {
    const secTable = extractSectionalTable(secHtml);
    if (secTable) {
      const parsed = parseSectionalsDynamic(secTable);
      const runnerMap = db.prepare(`SELECT runner_id, horse_no FROM runners WHERE race_id=?`).all(raceId);
      const horseNoToRunnerId = new Map(runnerMap.map(x => [String(x.horse_no), x.runner_id]));

      const tx2 = db.transaction(() => {
        delSplitsByRace.run(raceId);
        for (const row of parsed) {
          const runnerId = horseNoToRunnerId.get(String(row.horseNo));
          if (!runnerId) continue;
          for (const s of row.splits) {
            if (s.split_time == null && s.pos == null) continue;
            insSplit.run(runnerId, s.split_idx, s.split_label, s.pos, s.split_time);
            ingestedSplits++;
          }
        }
      });
      tx2();
    }
  }

  ingestedRaces++;
  console.log(JSON.stringify({ raceNo, raceId, runners: runners.length, ingestedRaces, ingestedRunners, ingestedSplits }));
}

console.log(JSON.stringify({ done:true, db:dbPath, racedate, venue, races: raceNos, ingestedRaces, ingestedRunners, ingestedSplits }, null, 2));
