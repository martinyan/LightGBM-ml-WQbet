#!/usr/bin/env node
/* Backfill dynamic sectional splits into SQLite.

   Reads races in hkjc.sqlite, fetches displaysectionaltime for each race,
   parses ALL split columns, and writes to sectional_splits(runner_id, split_idx,...).

   Usage:
     node hkjc_backfill_sectional_splits.mjs --db hkjc.sqlite --startRaceId 1 --limit 0 --throttleMs 200 --progressEvery 200

   Notes:
   - This is a targeted rescrape of sectionals only (not localresults).
   - If a race has no sectionals table, it is skipped.
*/

import fs from 'node:fs/promises';
import Database from 'better-sqlite3';

const UA='openclaw-hkjc-sectionals-v2/1.0';

function arg(name,dflt=null){const i=process.argv.indexOf(name);return i>=0?process.argv[i+1]:dflt;}
const dbPath=arg('--db','hkjc.sqlite');
const startRaceId=Number(arg('--startRaceId','1'));
const limit=Number(arg('--limit','0')); // 0 = no limit
const throttleMs=Number(arg('--throttleMs','200'));
const progressEvery=Number(arg('--progressEvery','200'));
const outLog=arg('--log','hkjc_sectionals_backfill.log');

function sleep(ms){return new Promise(r=>setTimeout(r,ms));}
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

function extractSectionalTable(html){
  const tables=[...html.matchAll(/<table[\s\S]*?<\/table>/gi)].map(m=>m[0]);
  // table containing 分段時間 and 過終點 is the one we used before
  return tables.find(x=>x.includes('分段時間') && x.includes('過終點')) || null;
}

function parseSplitCell(txt){
  // Cells look like: "8 1-1/2 22.34" or similar; we want pos and the split time.
  const parts=norm(txt).split(' ').filter(Boolean);
  if(parts.length<2) return {pos:null, split_time:null};
  const pos = /^\d+$/.test(parts[0]) ? Number(parts[0]) : null;
  // pick last numeric time like 22.34
  const timeMatch = parts.slice().reverse().find(p=>/^\d+\.\d{2}$/.test(p));
  const split_time = timeMatch ? Number(timeMatch) : null;
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
    // First cell should be finish position
    if(!/^\d+$/.test(cells[0]||'')) continue;

    const finishPos = Number(cells[0]);
    const horseNo = cells[1];
    // Heuristic: split columns begin after first 3 columns: pos, horseNo, horseName
    const splitStart = 3;
    const splitCells = cells.slice(splitStart);

    // labels from header if present, align to splitCells length from the tail.
    // header often includes non-split columns too; try to map last K header labels.
    let splitLabels = [];
    if(header.length>=splitCells.length){
      splitLabels = header.slice(header.length - splitCells.length);
    } else {
      splitLabels = splitCells.map((_,i)=>`split${i+1}`);
    }

    const splits = splitCells.map((c,i)=>({ split_idx: i+1, split_label: splitLabels[i] ?? null, ...parseSplitCell(c) }));
    out.push({ finishPos, horseNo, splits });
  }
  return out;
}

async function main(){
  const db = new Database(dbPath);
  const schema = await fs.readFile(new URL('./hkjc_db_schema_v2_sectionals.sql', import.meta.url), 'utf8');
  db.exec(schema);

  const racesQ = db.prepare(`
    SELECT r.race_id, m.racedate, m.venue, r.race_no
    FROM races r
    JOIN meetings m ON m.meeting_id = r.meeting_id
    WHERE r.race_id >= ?
    ORDER BY r.race_id ASC
    ${limit>0 ? 'LIMIT '+limit : ''}
  `);

  const runnerMapQ = db.prepare(`SELECT runner_id, horse_no FROM runners WHERE race_id=?`);
  const delExisting = db.prepare(`DELETE FROM sectional_splits WHERE runner_id IN (SELECT runner_id FROM runners WHERE race_id=?)`);
  const insSplit = db.prepare(`INSERT OR REPLACE INTO sectional_splits (runner_id, split_idx, split_label, pos, split_time) VALUES (?,?,?,?,?)`);

  const races = racesQ.all(startRaceId);

  let doneRaces=0, doneSplits=0, skipped=0;

  for(const race of races){
    const [Y,M,D]=race.racedate.split('/');
    const secUrl = `https://racing.hkjc.com/zh-hk/local/information/displaysectionaltime?racedate=${D}/${M}/${Y}&RaceNo=${race.race_no}`;

    const html = await fetchHtml(secUrl);
    await sleep(throttleMs);
    if(!html){ skipped++; continue; }
    const table = extractSectionalTable(html);
    if(!table){ skipped++; continue; }

    const parsed = parseSectionalsDynamic(table);
    if(!parsed.length){ skipped++; continue; }

    const runners = runnerMapQ.all(race.race_id);
    const horseNoToRunnerId = new Map(runners.map(r=>[String(r.horse_no), r.runner_id]));

    const tx = db.transaction(()=>{
      delExisting.run(race.race_id);
      for(const row of parsed){
        const runnerId = horseNoToRunnerId.get(String(row.horseNo));
        if(!runnerId) continue;
        for(const s of row.splits){
          if(s.split_time==null && s.pos==null) continue;
          insSplit.run(runnerId, s.split_idx, s.split_label, s.pos, s.split_time);
          doneSplits++;
        }
      }
    });
    tx();

    doneRaces++;
    if(doneRaces % progressEvery === 0){
      const msg = { progressRaceId: race.race_id, racedate: race.racedate, raceNo: race.race_no, doneRaces, doneSplits, skipped };
      await fs.appendFile(outLog, JSON.stringify(msg)+"\n");
      console.log(JSON.stringify(msg));
    }
  }

  const final = { done:true, db: dbPath, startRaceId, limit, doneRaces, doneSplits, skipped };
  await fs.appendFile(outLog, JSON.stringify(final)+"\n");
  console.log(JSON.stringify(final));
}

main().catch(e=>{console.error(e);process.exit(1);});
