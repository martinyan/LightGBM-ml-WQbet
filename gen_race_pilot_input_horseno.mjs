#!/usr/bin/env node
// Convert a bet.hkjc.com wp race scrape (st_..._bet_scrape.json) into pilot input JSON.
// Uses HorseNo=<code> pages on racing.hkjc.com for history.
//
// Usage:
//   node gen_race_pilot_input_horseno.mjs --in st_2026-02-14_race1_bet_scrape.json --out st_2026-02-14_race1_pilot_input.json

import fs from 'node:fs/promises';

function arg(name,dflt=null){const i=process.argv.indexOf(name);return i>=0?process.argv[i+1]:dflt;}
const inPath=arg('--in');
const outPath=arg('--out');
if(!inPath||!outPath){console.error('Missing --in/--out');process.exit(2);} 

const src=JSON.parse(await fs.readFile(inPath,'utf8'));
const rows=src.rows||[];

const runners = rows.map(r=>({
  no: r.no,
  horse: r.horse,
  code: r.code,
  draw: r.draw ?? null,
  wt: r.wt ?? null,
  jockey: r.jockey,
  trainer: r.trainer,
  win: r.win ?? null,
  place: r.place ?? null,
  horseUrl: r.code ? `https://racing.hkjc.com/zh-hk/local/information/horse?HorseNo=${encodeURIComponent(r.code)}` : null
}));

const race = {
  racedate: '2026/02/14',
  venue: 'ST',
  raceNo: src.raceNo ?? null,
  scheduledTime: src.scheduledTime ?? null,
  distanceMeters: src.distanceMeters ?? null
};

const out={
  race,
  runners,
  source: src.url,
  scrapedAt: src.scrapedAt
};

await fs.writeFile(outPath, JSON.stringify(out,null,2));
console.log(JSON.stringify({out: outPath, runners: runners.length},null,2));
