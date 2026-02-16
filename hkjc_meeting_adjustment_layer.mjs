#!/usr/bin/env node
// Build a meeting-specific adjustment layer (track variant + draw/pace bias proxies)
// from HKJC localresults + displaysectionaltime.
//
// Usage:
//   node hkjc_meeting_adjustment_layer.mjs --racedate 2026/02/11 --racecourse HV --races 1-9 --out hv_2026-02-11_meeting_adjustment.json

import fs from 'node:fs/promises';

const UA='openclaw-hkjc-meeting-adjust/1.0';

function arg(name,dflt=null){const i=process.argv.indexOf(name);return i>=0?process.argv[i+1]:dflt;}
const racedate=arg('--racedate');
const racecourse=arg('--racecourse');
const racesArg=arg('--races');
const outPath=arg('--out','meeting_adjustment.json');
if(!racedate||!racecourse||!racesArg){console.error('Missing args');process.exit(2);} 

function parseRaceRange(s){if(s.includes('-')){const[a,b]=s.split('-').map(x=>Number(x.trim()));return Array.from({length:b-a+1},(_,i)=>a+i);}return s.split(',').map(x=>Number(x.trim())).filter(Boolean);} 
function norm(s){return (s??'').replace(/\s+/g,' ').replace(/\u00a0/g,' ').trim();}
function decodeEntities(s){return (s??'').replace(/&nbsp;/g,' ').replace(/&amp;/g,'&').replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&#(\d+);/g,(_,n)=>String.fromCharCode(Number(n)));}
function cleanCell(html){return norm(decodeEntities((html??'').replace(/<br\s*\/?\s*>/gi,' ').replace(/<[^>]+>/g,' ')));}
async function fetchHtml(url){const res=await fetch(url,{headers:{'user-agent':UA}}); if(!res.ok) throw new Error(`HTTP ${res.status}`); return await res.text();}

function parseFinishTimeToSec(s){
  const t=String(s||'').trim();
  let m=t.match(/^(\d{1,2}):(\d{2})\.(\d{2})$/);
  if(m) return Number(m[1])*60+Number(m[2])+Number(m[3])/100;
  m=t.match(/^(\d+)\.(\d{2})\.(\d{2})$/);
  if(m) return Number(m[1])*60+Number(m[2])+Number(m[3])/100;
  return null;
}

function parseMeta(html){
  // Prefer regex on raw HTML for class/distance/going because text extraction can lose tokens.
  const raw = html;
  const distMatch = raw.match(/(\d{3,4})\s*米/);
  const surface = raw.includes('全天候') ? 'awt' : (raw.includes('草地') ? 'turf' : null);
  const goingMatch = raw.match(/(好地|好\/快|濕快|濕慢|黏地|黏|軟地|重地)/);

  // Class can appear as Chinese numerals like 第五班
  const classDigit = raw.match(/第(\d)班/)?.[1] ?? null;
  const classZh = raw.match(/第([一二三四五])班/)?.[1] ?? null;
  const zhMap = { '一': 1, '二': 2, '三': 3, '四': 4, '五': 5 };
  const classNum = classDigit ? Number(classDigit) : (classZh ? zhMap[classZh] : null);

  return {
    classNum,
    distance: distMatch ? Number(distMatch[1]) : null,
    surface,
    going: goingMatch ? goingMatch[1] : null
  };
}

function parseRunners(html){
  const tables=[...html.matchAll(/<table[\s\S]*?<\/table>/gi)].map(m=>m[0]);
  const t=tables.find(x=>x.includes('名次')&&x.includes('馬號')&&x.includes('馬名'));
  if(!t) return [];
  const trList=[...t.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map(m=>m[1]);
  const rows=trList.map(tr=>[...tr.matchAll(/<t[dh][^>]*>([\s\S]*?)<\/t[dh]>/gi)].map(m=>cleanCell(m[1])));
  const header=rows.find(r=>r.includes('名次')&&r.includes('馬號')&&r.includes('馬名'));
  if(!header) return [];
  const idxPos=header.indexOf('名次');
  const idxNo=header.indexOf('馬號');
  const idxDraw=header.indexOf('檔位');
  const idxTime=header.findIndex(x=>x.replace(/\s+/g,'')==='完成時間')
    || header.findIndex(x=>x.includes('完成') && x.includes('時間'))
    || header.findIndex(x=>x.replace(/\s+/g,'')==='時間');
  const data=rows.slice(rows.indexOf(header)+1).filter(r=>/^\d+$/.test(r[idxPos]||''));
  return data.map(r=>({
    pos: Number(r[idxPos]),
    horseNo: r[idxNo],
    draw: idxDraw>=0 ? Number(r[idxDraw]) : null,
    finishTime: idxTime>=0 ? r[idxTime] : null
  }));
}

function extractSectionalTable(html){
  const tables=[...html.matchAll(/<table[\s\S]*?<\/table>/gi)].map(m=>m[0]);
  return tables.find(x=>x.includes('過終點') && x.includes('分段時間')) || null;
}
function parseSegment(segText){
  const parts=norm(segText).split(' ').filter(Boolean);
  if(parts.length<2) return null;
  const pos=/^\d+$/.test(parts[0])?Number(parts[0]):null;
  const idxFirstTime=parts.findIndex(p=>/^\d+\.\d{2}$/.test(p));
  const times=parts.filter(p=>/^\d+\.\d{2}$/.test(p));
  const margin=idxFirstTime>1?parts.slice(1,idxFirstTime).join(' '):null;
  const segTime=times.length?Number(times[0]):null;
  return {pos, margin, segTime};
}
function parseSectionals(tableHtml){
  const trList=[...tableHtml.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map(m=>m[1]);
  const out=[];
  for(const tr of trList){
    if(!tr.includes('<td')) continue;
    const cells=[...tr.matchAll(/<td[^>]*>([\s\S]*?)<\/td>/gi)].map(m=>cleanCell(m[1]));
    if(cells.length<6) continue;
    if(!/^\d+$/.test(cells[0]||'')) continue;
    out.push({
      finishPos: Number(cells[0]),
      horseNo: cells[1],
      seg1: parseSegment(cells[3]),
      seg2: parseSegment(cells[4]),
      seg3: parseSegment(cells[5])
    });
  }
  return out;
}

// Standard times from screenshot
const std = JSON.parse(await fs.readFile('hkjc_standard_times_from_screenshot_2026-02-11.json','utf8'));
function stdTimeSec({venue,surface,distance,classNum}){
  const key=String(distance);
  if(venue==='HV' && surface==='turf'){
    const v=std.HV_turf?.[key]?.[`G${classNum}`];
    return v && v!=='-' ? parseFinishTimeToSec(v) : null;
  }
  if(venue==='ST' && surface==='turf'){
    const v=std.ST_turf?.[key]?.[`G${classNum}`];
    return v && v!=='-' ? parseFinishTimeToSec(v) : null;
  }
  if(venue==='ST' && surface==='awt'){
    const v=std.ST_all_weather?.[key]?.[`G${classNum}`];
    return v && v!=='-' ? parseFinishTimeToSec(v) : null;
  }
  return null;
}

const raceNos=parseRaceRange(racesArg);
const races=[];

for(const raceNo of raceNos){
  const resultsUrl=`https://racing.hkjc.com/zh-hk/local/information/localresults?racedate=${racedate}&Racecourse=${racecourse}&RaceNo=${raceNo}`;
  const html=await fetchHtml(resultsUrl);
  const meta=parseMeta(html);
  const runners=parseRunners(html);
  const winner=runners.find(r=>r.pos===1);
  const winnerTimeSec=parseFinishTimeToSec(winner?.finishTime);
  const stdSec=(meta.classNum&&meta.distance&&meta.surface)?stdTimeSec({venue:racecourse,surface:meta.surface,distance:meta.distance,classNum:meta.classNum}):null;
  const variant=(winnerTimeSec!=null && stdSec!=null)?(winnerTimeSec-stdSec):null;

  // sectionals
  const [yyyy,mm,dd]=racedate.split('/');
  const sectionalsUrl=`https://racing.hkjc.com/zh-hk/local/information/displaysectionaltime?racedate=${dd}/${mm}/${yyyy}&RaceNo=${raceNo}`;
  let secWinnerPos1=null;
  try{
    const secHtml=await fetchHtml(sectionalsUrl);
    const secTable=extractSectionalTable(secHtml);
    if(secTable){
      const sec=parseSectionals(secTable);
      const w=sec.find(r=>String(r.horseNo)===String(winner?.horseNo));
      secWinnerPos1=w?.seg1?.pos ?? null;
    }
  }catch{}

  races.push({raceNo, resultsUrl, sectionalsUrl, meta, winnerHorseNo:winner?.horseNo??null, winnerTimeSec, stdSec, variant, winnerPos1: secWinnerPos1, runners});
}

// Simple bias estimates
const allRunners = races.flatMap(r=>r.runners.map(x=>({raceNo:r.raceNo, pos:x.pos, draw:x.draw})));
const valid = allRunners.filter(r=>r.draw!=null && r.pos!=null);
const drawMeanPos = {};
for(const r of valid){
  drawMeanPos[r.draw] ??= {sum:0,n:0};
  drawMeanPos[r.draw].sum += r.pos;
  drawMeanPos[r.draw].n += 1;
}
const drawBias = Object.entries(drawMeanPos).map(([draw,v])=>({draw:Number(draw), meanPos:v.sum/v.n, n:v.n})).sort((a,b)=>a.draw-b.draw);

const variants = races.map(r=>r.variant).filter(v=>v!=null);
const avgVariant = variants.length ? variants.reduce((a,b)=>a+b,0)/variants.length : null;

const winnerPos1s = races.map(r=>r.winnerPos1).filter(v=>v!=null);
const avgWinnerPos1 = winnerPos1s.length ? winnerPos1s.reduce((a,b)=>a+b,0)/winnerPos1s.length : null;

const out={
  meeting:{racedate,racecourse,races: raceNos},
  generatedAt: new Date().toISOString(),
  trackVariantSeconds: { avg: avgVariant, byRace: races.map(r=>({raceNo:r.raceNo, variant:r.variant, classNum:r.meta.classNum, distance:r.meta.distance, surface:r.meta.surface, winnerTimeSec:r.winnerTimeSec, stdSec:r.stdSec})) },
  drawBias: { byDraw: drawBias, interpretation: 'Lower meanPos is better. Small sample sizes per draw.' },
  paceBias: { avgWinnerPos1, interpretation: 'Average winner position at first call (from sectionals). Lower implies leaders/handy winners.' },
  raw: races
};

await fs.writeFile(outPath, JSON.stringify(out,null,2));
console.log(JSON.stringify({out: outPath, avgVariant, avgWinnerPos1},null,2));
