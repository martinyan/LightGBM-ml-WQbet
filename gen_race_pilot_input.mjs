#!/usr/bin/env node
// Convert a bet scrape JSON (from bet.hkjc.com) into a pilot input JSON with horse profile URLs.
//
// Usage:
//   node gen_race_pilot_input.mjs --bet race6_bet_scrape.json --links hkjc_links_2026-02-11.json --raceNo 6 --venue HV --racedate 2026-02-11 --out race6_pilot_input.json

import fs from 'node:fs/promises';

function arg(name) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : null;
}

const betPath = arg('--bet');
const linksPath = arg('--links');
const outPath = arg('--out');
const raceNo = Number(arg('--raceNo'));
const venue = arg('--venue');
const racedate = arg('--racedate');

if (!betPath || !linksPath || !outPath || !raceNo || !venue || !racedate) {
  console.error('Missing args.');
  process.exit(2);
}

const bet = JSON.parse(await fs.readFile(betPath, 'utf8'));
const links = JSON.parse(await fs.readFile(linksPath, 'utf8')).links;

const codeToUrl = new Map();
for (const l of links) {
  const horseid = new URL(l.href).searchParams.get('horseid');
  const code = horseid.split('_').slice(-1)[0];
  codeToUrl.set(code, l.href);
}

const missing = [];
const runners = (bet.rows || []).map(r => {
  const horseUrl = codeToUrl.get(r.code);
  if (!horseUrl) missing.push(r);
  return {
    no: r.no,
    horse: r.horse,
    code: r.code,
    draw: r.draw,
    wt: r.wt,
    jockey: r.jockey,
    trainer: r.trainer,
    win: r.win,
    place: r.place,
    horseUrl
  };
});

const race = {
  racedate,
  venue,
  raceNo,
  scheduledTime: bet.scheduledTime,
  distanceMeters: bet.distanceMeters
};

const out = {
  race,
  runners,
  source: bet.url,
  scrapedAt: bet.scrapedAt,
  missingCount: missing.length,
  missing
};

await fs.writeFile(outPath, JSON.stringify(out, null, 2));
console.log(JSON.stringify({ out: outPath, runners: runners.length, missing: missing.length }, null, 2));
