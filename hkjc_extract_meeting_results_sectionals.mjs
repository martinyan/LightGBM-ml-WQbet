#!/usr/bin/env node
// Extract HKJC meeting results (top3 + key fields) and sectionals for all horses.
// Uses racing.hkjc.com localresults + displaysectionaltime pages.
//
// Usage:
//   node hkjc_extract_meeting_results_sectionals.mjs --racedate 2026/02/11 --racecourse HV --races 1-9 --out hv_2026-02-11_results_sectionals.json

import fs from 'node:fs/promises';

const UA = 'openclaw-hkjc-results/1.0';

function arg(name, dflt=null) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i+1] : dflt;
}

const racedate = arg('--racedate');
const racecourse = arg('--racecourse');
const racesArg = arg('--races');
const outPath = arg('--out');

if (!racedate || !racecourse || !racesArg || !outPath) {
  console.error('Missing args');
  process.exit(2);
}

function parseRaceRange(s) {
  // "1-9" or "1,2,3"
  if (s.includes('-')) {
    const [a,b] = s.split('-').map(x=>Number(x.trim()));
    return Array.from({length: b-a+1}, (_,i)=>a+i);
  }
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
  const res = await fetch(url, {headers:{'user-agent':UA}});
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
  return await res.text();
}

function extractTableByHeaders(html, mustInclude=[]) {
  const tables = [...html.matchAll(/<table[\s\S]*?<\/table>/gi)].map(m=>m[0]);
  return tables.find(t => mustInclude.every(x => t.includes(x))) || null;
}

function parseResultsTable(tableHtml){
  // Find header row containing 名次 and 馬號 and 馬名
  const trList=[...tableHtml.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map(m=>m[1]);
  const rows = trList.map(tr=>[...tr.matchAll(/<t[dh][^>]*>([\s\S]*?)<\/t[dh]>/gi)].map(m=>cleanCell(m[1])));
  const headerRow = rows.find(r => r.includes('名次') && r.includes('馬號') && r.includes('馬名'));
  if (!headerRow) return {header:null, runners:[]};
  const header = headerRow;
  const idxHeader = rows.indexOf(headerRow);

  const idx = {
    pos: header.findIndex(x=>x==='名次'),
    horseNo: header.findIndex(x=>x==='馬號'),
    horseName: header.findIndex(x=>x==='馬名'),
    draw: header.findIndex(x=>x==='檔位'),
    weight: header.findIndex(x=>x==='負磅' || x==='實際負磅'),
    jockey: header.findIndex(x=>x==='騎師'),
    trainer: header.findIndex(x=>x==='練馬師'),
    margin: header.findIndex(x=>x.replace(/\s+/g,'')==='頭馬距離'),
    winOdds: header.findIndex(x=>x==='獨贏賠率'),
    finishTime: header.findIndex(x=>x==='完成時間' || x==='時間')
  };

  const dataRows = rows.slice(idxHeader+1).filter(r => /^\d+$/.test(r[idx.pos]||''));
  const runners = dataRows.map(r => ({
    pos: r[idx.pos] ?? null,
    horseNo: r[idx.horseNo] ?? null,
    horseName: r[idx.horseName] ?? null,
    draw: r[idx.draw] ?? null,
    weight: r[idx.weight] ?? null,
    jockey: r[idx.jockey] ?? null,
    trainer: r[idx.trainer] ?? null,
    margin: r[idx.margin] ?? null,
    winOdds: r[idx.winOdds] ?? null,
    finishTime: r[idx.finishTime] ?? null,
    cells: r
  }));
  return {header, runners};
}

function sectionalUrl({racedate, raceNo}) {
  // racedate: YYYY/MM/DD -> DD/MM/YYYY
  const [yyyy,mm,dd] = racedate.split('/');
  return `https://racing.hkjc.com/zh-hk/local/information/displaysectionaltime?racedate=${dd}/${mm}/${yyyy}&RaceNo=${raceNo}`;
}

function parseSectionals(tableHtml){
  const trList=[...tableHtml.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map(m=>m[1]);
  const rows=[];
  for(const tr of trList){
    if(!tr.includes('<td')) continue;
    const cells=[...tr.matchAll(/<td[^>]*>([\s\S]*?)<\/td>/gi)].map(m=>cleanCell(m[1]));
    if(cells.length<6) continue;
    const finishPos=cells[0];
    if(!/^\d+$/.test(finishPos)) continue;
    const finalTime = cells.find(c => /^\d{1,2}:\d{2}\.\d{2}$/.test(c)) || null;
    rows.push({
      finishPos: Number(finishPos),
      horseNo: cells[1] || null,
      horseName: cells[2] || null,
      seg1: cells[3] || null,
      seg2: cells[4] || null,
      seg3: cells[5] || null,
      finalTime
    });
  }
  return rows;
}

const raceNos = parseRaceRange(racesArg);

const meeting = {
  racedate,
  racecourse,
  generatedAt: new Date().toISOString(),
  races: []
};

for (const raceNo of raceNos) {
  const resultsUrl = `https://racing.hkjc.com/zh-hk/local/information/localresults?racedate=${racedate}&Racecourse=${racecourse}&RaceNo=${raceNo}`;
  const html = await fetchHtml(resultsUrl);

  // results runners table usually includes 馬號/馬名/名次
  const t = extractTableByHeaders(html, ['馬號', '馬名', '名次']);
  if (!t) {
    meeting.races.push({ raceNo, resultsUrl, error: 'results table not found' });
    continue;
  }
  const parsed = parseResultsTable(t);

  const secUrl = sectionalUrl({racedate, raceNo});
  let sectionals = null;
  try {
    const secHtml = await fetchHtml(secUrl);
    const secTable = extractTableByHeaders(secHtml, ['過終點', '分段時間']);
    sectionals = secTable ? parseSectionals(secTable) : null;
  } catch {
    sectionals = null;
  }

  const top3 = parsed.runners
    .filter(r => r.pos === '1' || r.pos === '2' || r.pos === '3' || r.pos === '01' || r.pos === '02' || r.pos === '03')
    .map(r => ({ pos: r.pos, horseNo: r.horseNo, horseName: r.horseName }));

  meeting.races.push({
    raceNo,
    resultsUrl,
    sectionalsUrl: secUrl,
    top3,
    runnerCount: parsed.runners.length,
    runners: parsed.runners,
    sectionals
  });
}

await fs.writeFile(outPath, JSON.stringify(meeting, null, 2));
console.log(JSON.stringify({ out: outPath, races: meeting.races.length }, null, 2));
