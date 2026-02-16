#!/usr/bin/env node
// Backtest using HKJC localresults pages for dividends (win/place) and our saved predictions.
//
// Usage:
//   node hkjc_backtest_dividends_and_strategies.mjs --racedate 2026/02/11 --racecourse HV --races 1-9 --predGlob 'race*_pilot_prediction*.json' --out hv_2026-02-11_backtest.json

import fs from 'node:fs/promises';
import path from 'node:path';

const UA = 'openclaw-hkjc-backtest/1.0';

function arg(name, dflt=null){ const i=process.argv.indexOf(name); return i>=0?process.argv[i+1]:dflt; }
const racedate = arg('--racedate');
const racecourse = arg('--racecourse');
const racesArg = arg('--races');
const outPath = arg('--out','backtest.json');

if(!racedate||!racecourse||!racesArg){
  console.error('Missing args');
  process.exit(2);
}

function parseRaceRange(s){
  if(s.includes('-')){ const [a,b]=s.split('-').map(x=>Number(x.trim())); return Array.from({length:b-a+1},(_,i)=>a+i); }
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
  if(!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
  return await res.text();
}

function parseResultsBasic(html){
  // extract top3 horse numbers and names from results runner table
  const tables=[...html.matchAll(/<table[\s\S]*?<\/table>/gi)].map(m=>m[0]);
  const t = tables.find(x=>x.includes('名次') && x.includes('馬號') && x.includes('馬名'));
  if(!t) return {top3:[], runners:[]};
  const trList=[...t.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map(m=>m[1]);
  const rows=trList.map(tr=>[...tr.matchAll(/<t[dh][^>]*>([\s\S]*?)<\/t[dh]>/gi)].map(m=>cleanCell(m[1])));
  const header=rows.find(r=>r.includes('名次')&&r.includes('馬號')&&r.includes('馬名'));
  if(!header) return {top3:[], runners:[]};
  const idxPos=header.indexOf('名次');
  const idxNo=header.indexOf('馬號');
  const idxName=header.indexOf('馬名');
  const data=rows.slice(rows.indexOf(header)+1).filter(r=>/^\d+$/.test(r[idxPos]||''));
  const runners=data.map(r=>({pos:r[idxPos], horseNo:r[idxNo], horseName:r[idxName]}));
  const top3=runners.filter(r=>['1','2','3','01','02','03'].includes(r.pos)).map(r=>({pos:r.pos,horseNo:r.horseNo,horseName:r.horseName}));
  return {top3, runners};
}

function parseDividends(html){
  // HKJC localresults has a "派彩" table, but PLACE payouts are shown as:
  //   ["位置", "<horseNo>", "<div>"]
  // followed by 1-2 rows of ["<horseNo>", "<div>"] (no pool label).
  const tables=[...html.matchAll(/<table[\s\S]*?<\/table>/gi)].map(m=>m[0]);
  const divTable = tables.find(t => t.includes('派彩') && t.includes('彩池') && t.includes('勝出組合'));
  if(!divTable) return { win: {pool:'WIN', payouts:[]}, place:{pool:'PLACE', payouts:[]} };

  const trList=[...divTable.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map(m=>m[1]);
  const rows = trList
    .map(tr=>[...tr.matchAll(/<t[dh][^>]*>([\s\S]*?)<\/t[dh]>/gi)].map(m=>cleanCell(m[1])))
    .filter(r=>r.length>=2);

  const win={pool:'WIN', payouts:[]};
  const place={pool:'PLACE', payouts:[]};

  let mode = null; // 'PLACE_CONT'

  for(const r of rows){
    // Header
    if(r[0]==='彩池') continue;

    // 3-col rows with pool label
    if(r.length>=3){
      const pool = r[0];
      const combo = r[1];
      const div = Number(String(r[2]||'').replace(/[^0-9.]/g,''));
      mode = null;
      if(!div) continue;

      if(pool==='獨贏'){
        const horseNo = String(combo).match(/\d+/)?.[0] ?? null;
        if(horseNo) win.payouts.push({horseNo, dividend: div});
      } else if(pool==='位置'){
        const horseNo = String(combo).match(/\d+/)?.[0] ?? null;
        if(horseNo) place.payouts.push({horseNo, dividend: div});
        mode = 'PLACE_CONT';
      }
      continue;
    }

    // 2-col continuation rows
    if(r.length===2 && mode==='PLACE_CONT'){
      const horseNo = String(r[0]).match(/\d+/)?.[0] ?? null;
      const div = Number(String(r[1]||'').replace(/[^0-9.]/g,''));
      if(horseNo && div) place.payouts.push({horseNo, dividend: div});
      continue;
    }

    mode = null;
  }

  return {win, place};
}

function loadPredictionForRace(raceNo){
  // prefer A_with_odds if exists
  const candidates = [
    `race${raceNo}_pilot_prediction.A_with_odds.json`,
    `race${raceNo}_pilot_prediction.B_no_odds.json`,
    `race${raceNo}_pilot_prediction_extended_v2.A_with_odds.json`,
    `race${raceNo}_pilot_prediction_extended_v2.B_no_odds.json`,
    // older naming
    `race${raceNo}_pilot_prediction.json`,
    `race${raceNo}_pilot_prediction_extended.json`
  ];
  return candidates;
}

function pickTop2(pred){
  const rankings = pred?.rankings;
  if(!Array.isArray(rankings)||!rankings.length) return [];
  // Use 'no' field as horse number
  return rankings.slice(0,2).map(r=>String(r.no));
}

function pickTop1(pred){
  const rankings = pred?.rankings;
  if(!Array.isArray(rankings)||!rankings.length) return null;
  return String(rankings[0].no);
}

function payoutFor(pool, horseNo){
  if(!pool?.payouts) return null;
  return pool.payouts.find(p=>String(p.horseNo)===String(horseNo))?.dividend ?? null;
}

const raceNos=parseRaceRange(racesArg);
const races=[];

for(const raceNo of raceNos){
  const resultsUrl = `https://racing.hkjc.com/zh-hk/local/information/localresults?racedate=${racedate}&Racecourse=${racecourse}&RaceNo=${raceNo}`;
  const html = await fetchHtml(resultsUrl);
  const {top3} = parseResultsBasic(html);
  const dividends = parseDividends(html);

  // Load prediction if exists
  let pred=null, predPath=null;
  for(const p of loadPredictionForRace(raceNo)){
    try {
      pred = JSON.parse(await fs.readFile(p,'utf8'));
      predPath = p;
      break;
    } catch {}
  }

  const top1 = pred ? pickTop1(pred) : null;
  const top2 = pred ? pickTop2(pred) : [];

  // Strategy S1: $10 WIN on top1
  const stakeWin = top1 ? 10 : 0;
  const winDiv = top1 ? payoutFor(dividends.win, top1) : null;
  const actualWinnerNo = top3.find(x=>x.pos==='1'||x.pos==='01')?.horseNo ?? null;
  const returnWin = (top1 && actualWinnerNo && String(top1)===String(actualWinnerNo) && winDiv!=null) ? winDiv : 0;

  // Strategy S2: $10 PLACE on top2 horses (total $20)
  const stakePlace = top2.length ? 10*top2.length : 0;
  const top3Nos = new Set(top3.map(x=>String(x.horseNo)));
  let returnPlace=0;
  for(const h of top2){
    if(top3Nos.has(String(h))){
      const div = payoutFor(dividends.place, h);
      if(div!=null) returnPlace += div;
    }
  }

  races.push({
    raceNo,
    resultsUrl,
    top3,
    dividends,
    predictionFile: predPath,
    picks: { top1, top2 },
    strategies: {
      win_top1: { stake: stakeWin, return: returnWin, profit: returnWin-stakeWin },
      place_top2: { stake: stakePlace, return: returnPlace, profit: returnPlace-stakePlace }
    }
  });
}

const totals = races.reduce((acc,r)=>{
  for(const [k,v] of Object.entries(r.strategies)){
    acc[k] ??= {stake:0, return:0, profit:0, racesBet:0};
    acc[k].stake += v.stake;
    acc[k].return += v.return;
    acc[k].profit += v.profit;
    if(v.stake>0) acc[k].racesBet += 1;
  }
  return acc;
}, {});

for(const k of Object.keys(totals)){
  const t = totals[k];
  t.roi = t.stake>0 ? (t.profit/t.stake) : null;
}

const out = { meeting: {racedate, racecourse, races: raceNos}, generatedAt: new Date().toISOString(), totals, races };
await fs.writeFile(outPath, JSON.stringify(out,null,2));
console.log(JSON.stringify({out: outPath, totals}, null, 2));
