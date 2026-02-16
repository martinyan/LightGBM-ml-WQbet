#!/usr/bin/env node
/* Build a local SQLite DB of HKJC races from racing.hkjc.com localresults + displaysectionaltime.

   Strategy: brute-force scan dates in [--start, --end] for both venues (ST/HV).
   For each date+venue: probe RaceNo=1; if a runner table exists, ingest races until missing.

   Usage:
     node hkjc_build_sqlite_from_localresults.mjs --db hkjc.sqlite --start 2024/09/01 --end 2026/02/14 --maxRaces 12

   Notes:
   - This is v1 ingestion to support the pilot model features.
   - It stores structured data; raw HTML is not stored.
*/

import fs from 'node:fs/promises';
import Database from 'better-sqlite3';

const UA='openclaw-hkjc-db/1.0';

function arg(name,dflt=null){const i=process.argv.indexOf(name);return i>=0?process.argv[i+1]:dflt;}
const dbPath=arg('--db','hkjc.sqlite');
const start=arg('--start');
const end=arg('--end');
const maxRaces=Number(arg('--maxRaces','12'));
const throttleMs=Number(arg('--throttleMs','250'));

if(!start||!end){
  console.error('Missing --start/--end (YYYY/MM/DD)');
  process.exit(2);
}

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

function ymdToDate(ymd){
  const [Y,M,D]=ymd.split('/').map(Number);
  return new Date(Date.UTC(Y,M-1,D));
}
function dateToYmd(d){
  const Y=d.getUTCFullYear();
  const M=String(d.getUTCMonth()+1).padStart(2,'0');
  const D=String(d.getUTCDate()).padStart(2,'0');
  return `${Y}/${M}/${D}`;
}
function* dateRange(startYmd,endYmd){
  let d=ymdToDate(startYmd);
  const e=ymdToDate(endYmd);
  while(d<=e){
    yield dateToYmd(d);
    d=new Date(d.getTime()+24*3600*1000);
  }
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
  const goingMatch = raw.match(/(好地|好\/快|濕快|濕慢|黏地|黏|軟地|重地)/);
  const railMatch = raw.match(/欄位\s*[:：]?\s*([^\s<]{1,10})/);
  const courseMatch = raw.match(/跑道\s*[:：]?\s*([^\s<]{1,20})/);
  const classDigit = raw.match(/第(\d)班/)?.[1] ?? null;
  const classZh = raw.match(/第([一二三四五])班/)?.[1] ?? null;
  const zhMap = { '一': 1, '二': 2, '三': 3, '四': 4, '五': 5 };
  const classNum = classDigit ? Number(classDigit) : (classZh ? zhMap[classZh] : null);
  return {
    distance_m: distMatch ? Number(distMatch[1]) : null,
    surface,
    going: goingMatch ? goingMatch[1] : null,
    rail: railMatch ? railMatch[1] : null,
    course: courseMatch ? courseMatch[1] : null,
    class_num: classNum
  };
}

function parseRunnersFromLocalresults(html){
  const tables=[...html.matchAll(/<table[\s\S]*?<\/table>/gi)].map(m=>m[0]);
  const t=tables.find(x=>x.includes('名次')&&x.includes('馬號')&&x.includes('馬名'));
  if(!t) return {header:null, rows:[]};
  const trList=[...t.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map(m=>m[1]);
  const rows=trList.map(tr=>[...tr.matchAll(/<t[dh][^>]*>([\s\S]*?)<\/t[dh]>/gi)].map(m=>cleanCell(m[1])));
  const header=rows.find(r=>r.includes('名次')&&r.includes('馬號')&&r.includes('馬名'));
  if(!header) return {header:null, rows:[]};

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

  // Try to pull horse code from links in the row html: HorseNo=K165 etc.
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
  return {header, rows: out};
}

// margin text to lengths (approx) for key values; extend later.
function marginToLen(txt){
  const t=String(txt||'').trim();
  if(!t) return null;
  // common Chinese units
  const map={
    '鼻位':0.05,
    '短馬頭位':0.1,
    '馬頭位':0.2,
    '頭位':0.3,
    '頸位':0.4
  };
  for(const [k,v] of Object.entries(map)) if(t.includes(k)) return v;
  const m=t.match(/(\d+)\/(\d+)/);
  if(m){
    const a=Number(m[1]), b=Number(m[2]);
    return b? a/b : null;
  }
  const n=t.match(/\d+(?:\.\d+)?/);
  if(n) return Number(n[0]);
  return null;
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
  const segTime=times.length?Number(times[0]):null;
  return {pos, segTime};
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

async function main(){
  const schema = await fs.readFile(new URL('./hkjc_db_schema.sql', import.meta.url), 'utf8');
  const db = new Database(dbPath);
  db.exec(schema);

  const insMeeting = db.prepare(`INSERT OR IGNORE INTO meetings (racedate, venue, going, rail, surface_hint) VALUES (?,?,?,?,?)`);
  const getMeeting = db.prepare(`SELECT meeting_id FROM meetings WHERE racedate=? AND venue=?`);
  const updMeeting = db.prepare(`UPDATE meetings SET going=COALESCE(going,?), rail=COALESCE(rail,?), surface_hint=COALESCE(surface_hint,?) WHERE meeting_id=?`);

  const insRace = db.prepare(`INSERT OR IGNORE INTO races (meeting_id, race_no, distance_m, class_num, surface, course, scheduled_time, race_name) VALUES (?,?,?,?,?,?,?,?)`);
  const getRace = db.prepare(`SELECT race_id FROM races WHERE meeting_id=? AND race_no=?`);

  const insHorse = db.prepare(`INSERT OR IGNORE INTO horses (horse_code, horse_name_zh) VALUES (?,?)`);
  const insRunner = db.prepare(`INSERT OR IGNORE INTO runners (race_id, horse_code, horse_no, horse_name_zh, draw, weight, jockey, trainer, win_odds) VALUES (?,?,?,?,?,?,?,?,?)`);
  const getRunner = db.prepare(`SELECT runner_id FROM runners WHERE race_id=? AND horse_no=?`);

  const insResult = db.prepare(`INSERT OR REPLACE INTO results (runner_id, finish_pos, finish_time_sec, margin_text, margin_len, time_delta_sec) VALUES (?,?,?,?,?,?)`);
  const insSect = db.prepare(`INSERT OR REPLACE INTO sectionals (runner_id, pos1, pos3, seg2_time, seg3_time, kick_time) VALUES (?,?,?,?,?,?)`);

  let meetingsFound=0, racesFound=0, runnersFound=0;

  for(const racedate of dateRange(start,end)){
    for(const venue of ['ST','HV']){
      const url1 = `https://racing.hkjc.com/zh-hk/local/information/localresults?racedate=${racedate}&Racecourse=${venue}&RaceNo=1`;
      const html1 = await fetchHtml(url1);
      await sleep(throttleMs);
      if(!html1) continue;
      if(!html1.includes('名次') || !html1.includes('馬號') || !html1.includes('馬名')) continue;

      const meta1=parseMeta(html1);
      insMeeting.run(racedate, venue, meta1.going, meta1.rail, meta1.surface);
      const meetingId = getMeeting.get(racedate, venue)?.meeting_id;
      if(!meetingId) continue;
      updMeeting.run(meta1.going, meta1.rail, meta1.surface, meetingId);
      meetingsFound++;

      for(let raceNo=1; raceNo<=maxRaces; raceNo++){
        const url = `https://racing.hkjc.com/zh-hk/local/information/localresults?racedate=${racedate}&Racecourse=${venue}&RaceNo=${raceNo}`;
        const html = raceNo===1 ? html1 : await fetchHtml(url);
        await sleep(throttleMs);
        if(!html) break;
        if(!html.includes('名次') || !html.includes('馬號') || !html.includes('馬名')) break;

        const meta=parseMeta(html);
        insRace.run(meetingId, raceNo, meta.distance_m, meta.class_num, meta.surface, meta.course, null, null);
        const raceId = getRace.get(meetingId, raceNo)?.race_id;
        if(!raceId) continue;
        racesFound++;

        const parsed=parseRunnersFromLocalresults(html);
        if(!parsed.rows.length) continue;

        // Winner time baseline for time_delta
        const winner=parsed.rows.find(x=>x.finish_pos===1);
        const winnerTime=winner?.finish_time_sec ?? null;

        // sectionals page
        const [Y,M,D]=racedate.split('/');
        const secUrl = `https://racing.hkjc.com/zh-hk/local/information/displaysectionaltime?racedate=${D}/${M}/${Y}&RaceNo=${raceNo}`;
        const secHtml = await fetchHtml(secUrl);
        await sleep(throttleMs);
        const secTable = secHtml ? extractSectionalTable(secHtml) : null;
        const secRows = secTable ? parseSectionals(secTable) : [];
        const secByHorseNo = new Map(secRows.map(r=>[String(r.horseNo), r]));

        const tx = db.transaction(()=>{
          for(const rr of parsed.rows){
            if(rr.horse_code) insHorse.run(rr.horse_code, rr.horse_name_zh);
            insRunner.run(raceId, rr.horse_code, rr.horse_no, rr.horse_name_zh, rr.draw, rr.weight, rr.jockey, rr.trainer, rr.win_odds);
            const runnerId = getRunner.get(raceId, rr.horse_no)?.runner_id;
            if(!runnerId) continue;
            runnersFound++;

            const marginLen = marginToLen(rr.margin_text);
            const timeDelta = (winnerTime!=null && rr.finish_time_sec!=null) ? (rr.finish_time_sec - winnerTime) : null;
            insResult.run(runnerId, rr.finish_pos, rr.finish_time_sec, rr.margin_text, marginLen, timeDelta);

            const s = secByHorseNo.get(String(rr.horse_no));
            if(s){
              const pos1 = s.seg1?.pos ?? null;
              const pos3 = s.seg3?.pos ?? null;
              const seg2 = s.seg2?.segTime ?? null;
              const seg3 = s.seg3?.segTime ?? null;
              const kick = (seg2!=null && seg3!=null) ? (seg3 - seg2) : null;
              insSect.run(runnerId, pos1, pos3, seg2, seg3, kick);
            }
          }
        });
        tx();
      }
    }

    // progress line
    if(racedate.endsWith('/01')){
      console.log(JSON.stringify({progress:racedate, meetingsFound, racesFound, runnersFound},null,2));
    }
  }

  console.log(JSON.stringify({done:true, db: dbPath, meetingsFound, racesFound, runnersFound},null,2));
}

main().catch(e=>{console.error(e);process.exit(1);});
