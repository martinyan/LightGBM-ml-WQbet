#!/usr/bin/env node
/* Backfill runners.horse_code by re-fetching localresults pages and extracting HorseNo=CODE.

   Usage:
     node hkjc_backfill_horse_codes_from_localresults.mjs --db hkjc.sqlite --startRaceId 1 --limit 0 --throttleMs 200 --progressEvery 200

   This is a targeted re-scrape of localresults only. It updates:
   - runners.horse_code
   - horses(horse_code, horse_name_zh)

   NOTE: localresults runner table includes <a href="...horse?HorseNo=K165">; we regex HorseNo.
*/

import fs from 'node:fs/promises';
import Database from 'better-sqlite3';

const UA='openclaw-hkjc-horsecode-backfill/1.0';

function arg(name,dflt=null){const i=process.argv.indexOf(name);return i>=0?process.argv[i+1]:dflt;}
const dbPath=arg('--db','hkjc.sqlite');
const startRaceId=Number(arg('--startRaceId','1'));
const limit=Number(arg('--limit','0'));
const throttleMs=Number(arg('--throttleMs','200'));
const progressEvery=Number(arg('--progressEvery','200'));
const outLog=arg('--log','hkjc_horsecode_backfill.log');

function sleep(ms){return new Promise(r=>setTimeout(r,ms));}
async function fetchHtml(url){
  const res = await fetch(url,{headers:{'user-agent':UA}});
  if(!res.ok) return null;
  return await res.text();
}

function extractRunnerLinks(html){
  // Extract all HorseNo codes in order of appearance.
  // We only care about 1 race page at a time; there should be exactly one per runner.
  const codes=[...html.matchAll(/HorseNo=([A-Z]\d{3})/g)].map(m=>m[1]);
  return codes;
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

function parseRunnerTable(html){
  const tables=[...html.matchAll(/<table[\s\S]*?<\/table>/gi)].map(m=>m[0]);
  const t=tables.find(x=>x.includes('名次')&&x.includes('馬號')&&x.includes('馬名'));
  if(!t) return [];
  const trList=[...t.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map(m=>m[1]);
  const rows=trList.map(tr=>[...tr.matchAll(/<t[dh][^>]*>([\s\S]*?)<\/t[dh]>/gi)].map(m=>cleanCell(m[1])));
  const header=rows.find(r=>r.includes('名次')&&r.includes('馬號')&&r.includes('馬名'));
  if(!header) return [];
  const idxNo=header.indexOf('馬號');
  const idxName=header.indexOf('馬名');
  const data=rows.slice(rows.indexOf(header)+1).filter(r=>/^\d+$/.test(r[0]||'') || /^\d+$/.test(r[idxNo]||''));

  // Use tr HTML to get horse codes per row.
  const dataTrHtmls = trList.slice(rows.indexOf(header)+1);
  const out=[];
  for(let i=0;i<data.length;i++){
    const r=data[i];
    const trHtml=dataTrHtmls[i]||'';
    const horseNo = Number(r[idxNo]) || null;
    const horseName = r[idxName] || null;
    // localresults pages link horses like: horse?horseid=HK_2023_J332 (take last token as code)
    const horseCode = (trHtml.match(/horse\?horseid=HK_\d{4}_([A-Z]\d{3})/i)||[])[1] || null;
    if(horseNo!=null) out.push({horse_no: horseNo, horse_name_zh: horseName, horse_code: horseCode});
  }
  return out;
}

async function main(){
  const db = new Database(dbPath);
  const racesQ = db.prepare(`
    SELECT r.race_id, r.race_no, m.racedate, m.venue
    FROM races r
    JOIN meetings m ON m.meeting_id=r.meeting_id
    WHERE r.race_id >= ?
    ORDER BY r.race_id ASC
    ${limit>0 ? 'LIMIT '+limit : ''}
  `);
  // Avoid FK failures: insert horse row before updating runner.horse_code
  const insHorse = db.prepare(`INSERT OR IGNORE INTO horses (horse_code, horse_name_zh) VALUES (?,?)`);
  const updRunner = db.prepare(`UPDATE runners SET horse_code=? WHERE race_id=? AND horse_no=?`);

  const races = racesQ.all(startRaceId);
  let doneRaces=0, updated=0, missing=0, skipped=0;

  for(const race of races){
    const url = `https://racing.hkjc.com/zh-hk/local/information/localresults?racedate=${race.racedate}&Racecourse=${race.venue}&RaceNo=${race.race_no}`;
    const html = await fetchHtml(url);
    await sleep(throttleMs);
    if(!html){ skipped++; continue; }

    const rows = parseRunnerTable(html);
    if(!rows.length){ skipped++; continue; }

    const tx = db.transaction(()=>{
      for(const rr of rows){
        if(rr.horse_code){
          insHorse.run(rr.horse_code, rr.horse_name_zh);
          updRunner.run(rr.horse_code, race.race_id, rr.horse_no);
          updated++;
        } else {
          missing++;
        }
      }
    });
    tx();

    doneRaces++;
    if(doneRaces % progressEvery === 0){
      const msg = { progressRaceId: race.race_id, racedate: race.racedate, venue: race.venue, raceNo: race.race_no, doneRaces, updated, missing, skipped };
      await fs.appendFile(outLog, JSON.stringify(msg)+"\n");
      console.log(JSON.stringify(msg));
    }
  }

  const final = { done:true, db: dbPath, startRaceId, limit, doneRaces, updated, missing, skipped };
  await fs.appendFile(outLog, JSON.stringify(final)+"\n");
  console.log(JSON.stringify(final));
}

main().catch(e=>{console.error(e);process.exit(1);});
