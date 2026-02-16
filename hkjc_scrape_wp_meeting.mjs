#!/usr/bin/env node
// Cheap scraper: fetch bet.hkjc.com "wp" race pages via HTTP and parse runner/odds table.
// No browser / no LLM.
//
// Usage:
//   node hkjc_scrape_wp_meeting.mjs --racedate 2026-02-14 --venue ST --races 1-10 --outDir .
//
// Output files:
//   <venue>_<racedate>_race<no>_bet_scrape.json

import fs from 'node:fs/promises';
import path from 'node:path';

const UA='openclaw-hkjc-wp-scrape/1.0';
function arg(name,dflt=null){const i=process.argv.indexOf(name);return i>=0?process.argv[i+1]:dflt;}
const racedate=arg('--racedate'); // YYYY-MM-DD
const venue=arg('--venue'); // ST|HV
const racesArg=arg('--races');
const outDir=arg('--outDir','.')
;
if(!racedate||!venue||!racesArg){
  console.error('Missing args');
  process.exit(2);
}

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
  const res = await fetch(url, { headers: { 'user-agent': UA }});
  if(!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
  return await res.text();
}

function parseWp(html, url){
  // Find the runner table by looking for 馬號 + 馬名 + 獨贏/位置.
  const tables=[...html.matchAll(/<table[\s\S]*?<\/table>/gi)].map(m=>m[0]);
  const t = tables.find(x => x.includes('馬號') && x.includes('馬名') && (x.includes('獨贏') || x.includes('位置')));
  if(!t) return { error: 'runner table not found' };

  const trList=[...t.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map(m=>m[1]);
  const rows=[];
  for(const tr of trList){
    const tds=[...tr.matchAll(/<td[^>]*>([\s\S]*?)<\/td>/gi)].map(m=>m[1]);
    if(!tds.length) continue;
    const no = cleanCell(tds[0]);
    if(!/^\d+$/.test(no)) continue;

    // silk cell usually td[1] contains background-image .../Horse/<letter>/<CODE>/...
    const silkHtml = tds[1] ?? '';
    const style = (silkHtml.match(/style\s*=\s*"([^"]+)"/i)||[])[1] || '';
    const m = style.match(/\/Horse\/[A-Z]\/([A-Z]\d{3})\//);
    const code = m ? m[1] : null;

    const horse = cleanCell(tds[2]);
    const draw = Number(cleanCell(tds[3])) || null;
    const wt = Number(cleanCell(tds[4])) || null;
    const jockey = cleanCell(tds[5]);
    const trainer = cleanCell(tds[6]);
    const win = Number(cleanCell(tds[7])) || null;
    const place = Number(cleanCell(tds[8])) || null;

    rows.push({ no: Number(no), horse, code, draw, wt, jockey, trainer, win, place });
  }

  const text = cleanCell(html);
  const scheduledTime = (text.match(/\b(\d{1,2}:\d{2})\b/)||[])[1] || null;
  const distanceMeters = Number((text.match(/(\d{3,4})\s*米/)||[])[1])||null;
  return { url, scrapedAt: new Date().toISOString(), scheduledTime, distanceMeters, rows };
}

const raceNos = parseRaceRange(racesArg);
const outFiles=[];

for(const raceNo of raceNos){
  const url = `https://bet.hkjc.com/ch/racing/wp/${racedate}/${venue}/${raceNo}`;
  const html = await fetchHtml(url);
  const parsed = parseWp(html, url);
  const out = {
    url,
    scrapedAt: parsed.scrapedAt,
    raceNo,
    scheduledTime: parsed.scheduledTime,
    distanceMeters: parsed.distanceMeters,
    rows: parsed.rows
  };
  const outPath = path.join(outDir, `${venue.toLowerCase()}_${racedate}_race${raceNo}_bet_scrape.json`);
  await fs.writeFile(outPath, JSON.stringify(out,null,2));
  outFiles.push(outPath);
  console.log(JSON.stringify({ raceNo, out: outPath, runners: out.rows?.length ?? 0 }, null, 2));
}

console.log(JSON.stringify({done:true, outFiles}, null, 2));
