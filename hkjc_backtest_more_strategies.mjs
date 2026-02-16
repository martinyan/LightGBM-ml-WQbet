#!/usr/bin/env node
// Extended backtest strategies using localresults dividends and saved prediction files.
// Strategies:
// - WIN top1 always
// - PLACE top2 always
// - WIN top1 only if pTop3 >= threshold
// - PLACE top2 only if pTop3 >= threshold (per horse)
// - Optional odds filter (max win odds)
//
// Usage:
// node hkjc_backtest_more_strategies.mjs --racedate 2026/02/11 --racecourse HV --races 1-9 --out hv_2026-02-11_backtest_more.json

import fs from 'node:fs/promises';

const UA='openclaw-hkjc-backtest/1.1';

function arg(name,dflt=null){const i=process.argv.indexOf(name);return i>=0?process.argv[i+1]:dflt;}
const racedate=arg('--racedate');
const racecourse=arg('--racecourse');
const racesArg=arg('--races');
const outPath=arg('--out','backtest_more.json');
if(!racedate||!racecourse||!racesArg){console.error('Missing args');process.exit(2);} 

function parseRaceRange(s){if(s.includes('-')){const[a,b]=s.split('-').map(x=>Number(x.trim()));return Array.from({length:b-a+1},(_,i)=>a+i);}return s.split(',').map(x=>Number(x.trim())).filter(Boolean);} 
function norm(s){return (s??'').replace(/\s+/g,' ').replace(/\u00a0/g,' ').trim();}
function decodeEntities(s){return (s??'').replace(/&nbsp;/g,' ').replace(/&amp;/g,'&').replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&#(\d+);/g,(_,n)=>String.fromCharCode(Number(n)));}
function cleanCell(html){return norm(decodeEntities((html??'').replace(/<br\s*\/?\s*>/gi,' ').replace(/<[^>]+>/g,' ')));}
async function fetchHtml(url){const res=await fetch(url,{headers:{'user-agent':UA}}); if(!res.ok) throw new Error(`HTTP ${res.status}`); return await res.text();}

function parseResultsBasic(html){
  const tables=[...html.matchAll(/<table[\s\S]*?<\/table>/gi)].map(m=>m[0]);
  const t=tables.find(x=>x.includes('名次')&&x.includes('馬號')&&x.includes('馬名'));
  if(!t) return {top3:[]};
  const trList=[...t.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map(m=>m[1]);
  const rows=trList.map(tr=>[...tr.matchAll(/<t[dh][^>]*>([\s\S]*?)<\/t[dh]>/gi)].map(m=>cleanCell(m[1])));
  const header=rows.find(r=>r.includes('名次')&&r.includes('馬號')&&r.includes('馬名'));
  const idxPos=header.indexOf('名次');
  const idxNo=header.indexOf('馬號');
  const idxName=header.indexOf('馬名');
  const data=rows.slice(rows.indexOf(header)+1).filter(r=>/^\d+$/.test(r[idxPos]||''));
  const runners=data.map(r=>({pos:r[idxPos], horseNo:r[idxNo], horseName:r[idxName]}));
  const top3=runners.filter(r=>['1','2','3','01','02','03'].includes(r.pos)).map(r=>({pos:r.pos,horseNo:r.horseNo,horseName:r.horseName}));
  return {top3};
}

function parseDividends(html){
  const tables=[...html.matchAll(/<table[\s\S]*?<\/table>/gi)].map(m=>m[0]);
  const divTable=tables.find(t=>t.includes('派彩')&&t.includes('彩池')&&t.includes('勝出組合'));
  const win={pool:'WIN', payouts:[]};
  const place={pool:'PLACE', payouts:[]};
  if(!divTable) return {win, place};
  const trList=[...divTable.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map(m=>m[1]);
  const rows=trList.map(tr=>[...tr.matchAll(/<t[dh][^>]*>([\s\S]*?)<\/t[dh]>/gi)].map(m=>cleanCell(m[1]))).filter(r=>r.length>=2);
  let mode=null;
  for(const r of rows){
    if(r[0]==='彩池') continue;
    if(r.length>=3){
      const pool=r[0]; const combo=r[1]; const div=Number(String(r[2]||'').replace(/[^0-9.]/g,''));
      mode=null;
      if(!div) continue;
      if(pool==='獨贏'){
        const horseNo=String(combo).match(/\d+/)?.[0]; if(horseNo) win.payouts.push({horseNo, dividend:div});
      } else if(pool==='位置'){
        const horseNo=String(combo).match(/\d+/)?.[0]; if(horseNo) place.payouts.push({horseNo, dividend:div});
        mode='PLACE_CONT';
      }
      continue;
    }
    if(mode==='PLACE_CONT' && r.length===2){
      const horseNo=String(r[0]).match(/\d+/)?.[0]; const div=Number(String(r[1]||'').replace(/[^0-9.]/g,''));
      if(horseNo && div) place.payouts.push({horseNo, dividend:div});
      continue;
    }
    mode=null;
  }
  return {win, place};
}

function payoutFor(pool, horseNo){
  return pool?.payouts?.find(p=>String(p.horseNo)===String(horseNo))?.dividend ?? null;
}

async function loadPred(raceNo){
  const files=[
    `race${raceNo}_pilot_prediction.A_with_odds.json`,
    `race${raceNo}_pilot_prediction.B_no_odds.json`,
    `race${raceNo}_pilot_prediction_extended_v2.A_with_odds.json`,
    `race${raceNo}_pilot_prediction_extended_v2.B_no_odds.json`,
    `race${raceNo}_pilot_prediction.json`,
    `race${raceNo}_pilot_prediction_extended.json`
  ];
  for(const f of files){
    try {
      const txt = await fs.readFile(f,'utf8');
      return {file:f, json: JSON.parse(txt)};
    } catch {}
  }
  return null;
}

function getRankings(pred){
  const r=pred?.rankings; if(!Array.isArray(r)) return [];
  return r.map(x=>({horseNo:String(x.no), pTop3: x.pTop3 ?? null, win: x.win ?? null, place: x.place ?? null}));
}

const thresholds=[0,0.4,0.5,0.6,0.7,0.8];
const maxWinOdds=[null, 10, 20];

const raceNos=parseRaceRange(racesArg);

const strategies=[];
for(const thr of thresholds){
  for(const maxOdds of maxWinOdds){
    strategies.push({name:`WIN_top1_p>=${thr}${maxOdds?`_odds<=${maxOdds}`:''}`, type:'WIN_TOP1', thr, maxOdds});
    strategies.push({name:`PLACE_top2_each_p>=${thr}${maxOdds?`_odds<=${maxOdds}`:''}`, type:'PLACE_TOP2', thr, maxOdds});
  }
}

const results=[];

for(const raceNo of raceNos){
  const resultsUrl=`https://racing.hkjc.com/zh-hk/local/information/localresults?racedate=${racedate}&Racecourse=${racecourse}&RaceNo=${raceNo}`;
  const html=await fetchHtml(resultsUrl);
  const {top3}=parseResultsBasic(html);
  const dividends=parseDividends(html);
  const top3Nos=new Set(top3.map(x=>String(x.horseNo)));
  const winnerNo=top3.find(x=>x.pos==='1'||x.pos==='01')?.horseNo ?? null;

  const predLoaded = await loadPred(raceNo);

  if(!predLoaded){
    results.push({raceNo, resultsUrl, predictionFile:null, skipped:true});
    continue;
  }

  const rankings=getRankings(predLoaded.json);
  const top1=rankings[0];
  const top2=rankings.slice(0,2);

  const stratOut={};
  for(const s of strategies){
    let stake=0, ret=0;
    if(s.type==='WIN_TOP1'){
      const okP = (top1.pTop3??0) >= s.thr;
      const okOdds = s.maxOdds==null || (top1.win!=null && top1.win <= s.maxOdds);
      if(okP && okOdds){
        stake=10;
        if(winnerNo && String(top1.horseNo)===String(winnerNo)){
          const div=payoutFor(dividends.win, top1.horseNo);
          if(div!=null) ret=div;
        }
      }
    } else if(s.type==='PLACE_TOP2'){
      for(const h of top2){
        const okP = (h.pTop3??0) >= s.thr;
        const okOdds = s.maxOdds==null || (h.win!=null && h.win <= s.maxOdds);
        if(okP && okOdds){
          stake += 10;
          if(top3Nos.has(String(h.horseNo))){
            const div=payoutFor(dividends.place, h.horseNo);
            if(div!=null) ret += div;
          }
        }
      }
    }
    stratOut[s.name]={stake, return:ret, profit:ret-stake};
  }

  results.push({raceNo, resultsUrl, predictionFile: predLoaded.file, top3, dividends, stratOut});
}

const totals={};
for(const s of strategies){ totals[s.name]={stake:0, return:0, profit:0, racesBet:0, roi:null}; }
for(const r of results){
  if(r.skipped) continue;
  for(const [name,val] of Object.entries(r.stratOut)){
    totals[name].stake += val.stake;
    totals[name].return += val.return;
    totals[name].profit += val.profit;
    if(val.stake>0) totals[name].racesBet += 1;
  }
}
for(const [name,t] of Object.entries(totals)) t.roi = t.stake>0 ? t.profit/t.stake : null;

const best = Object.entries(totals)
  .filter(([_,t])=>t.stake>0)
  .sort((a,b)=>b[1].roi-a[1].roi)
  .slice(0,10)
  .map(([name,t])=>({name,...t}));

const out={meeting:{racedate,racecourse,races:raceNos}, generatedAt:new Date().toISOString(), totals, best, results};
await fs.writeFile(outPath, JSON.stringify(out,null,2));
console.log(JSON.stringify({out: outPath, best},null,2));
