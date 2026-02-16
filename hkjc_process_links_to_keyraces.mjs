#!/usr/bin/env node
import fs from 'node:fs/promises';

// Usage:
//   node hkjc_process_links_to_keyraces.mjs --links <links.json> --out <out.json>
//
// links.json schema expected (from browser extract): { source, scrapedAt, links:[{href,...}] }
// Output schema: {source, generatedAt, horseCount, totalKeyRaces, horses:[{horseid,name,trainer,rating,keyRaces...}]}

const UA = 'openclaw-hkjc-scan/1.0';

function arg(name) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : null;
}

const linksPath = arg('--links');
const outPath = arg('--out');
if (!linksPath || !outPath) {
  console.error('Missing args. Usage: node hkjc_process_links_to_keyraces.mjs --links <links.json> --out <out.json>');
  process.exit(2);
}

function uniq(arr) { return [...new Set(arr)]; }
function norm(s) { return (s ?? '').replace(/\s+/g, ' ').replace(/\u00a0/g, ' ').trim(); }
function decodeEntities(s) {
  return (s ?? '')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)));
}

function parseMarginToLen(raw) {
  const t = norm(decodeEntities(raw).replace(/<br\s*\/?\s*>/gi, ' '));
  if (!t) return null;
  if (t.includes('鼻位')) return 0.05;
  if (t.includes('短馬頭位')) return 0.10;
  if (t === '頭位' || t.includes('馬頭位')) return 0.20;
  if (t.includes('頸位')) return 0.30;
  if (/^\d+(?:-\d+\/\d+|\/\d+)?$/.test(t)) {
    if (t.includes('-')) {
      const [a, frac] = t.split('-');
      const [n, d] = frac.split('/');
      return Number(a) + Number(n) / Number(d);
    }
    if (t.includes('/')) {
      const [n, d] = t.split('/');
      return Number(n) / Number(d);
    }
    return Number(t);
  }
  return null;
}

function stripTags(html) {
  return norm(decodeEntities((html ?? '').replace(/<[^>]+>/g, ' ')));
}

function extractHorseMeta(html) {
  const title = html.match(/<title>([\s\S]*?)<\/title>/i)?.[1] ?? '';
  const name = norm(decodeEntities(title.split('-')[0] ?? ''));

  const trainer = norm(stripTags(html.match(/練馬師[\s\S]{0,200}?<td[^>]*>\s*:?\s*<\/td>[\s\S]{0,200}?<td[^>]*>([\s\S]*?)<\/td>/i)?.[1] ?? '')) ||
    norm(stripTags(html.match(/>\s*練馬師\s*<[\s\S]{0,200}?>\s*:?\s*<[\s\S]{0,200}?>([\s\S]*?)<\/a>/i)?.[1] ?? ''));

  const ratingStr = html.match(/現時評分[\s\S]{0,400}?>(\d+)<\/td>/i)?.[1]
    ?? html.match(/現時評分[\s\S]{0,400}?(\d{1,3})/i)?.[1];
  const rating = ratingStr ? Number(ratingStr) : null;

  return { name, trainer, rating };
}

function extractRaceTable(html) {
  // Locate the race-history table by the header cell "頭馬<br />距離".
  const idx = html.indexOf('頭馬<br />距離');
  if (idx < 0) return null;
  const start = html.lastIndexOf('<table', idx);
  if (start < 0) return null;
  const end = html.indexOf('</table>', idx);
  if (end < 0) return null;
  return html.slice(start, end + 8);
}

function parseRacesFromTable(tableHtml, baseUrl) {
  const races = [];

  const cleanCell = (s) => norm(
    decodeEntities(
      (s ?? '')
        .replace(/<br\s*\/?\s*>/gi, ' ')
        .replace(/<[^>]+>/g, ' ')
    )
  );

  const trList = [...tableHtml.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)].map(m => m[1]);
  const getTds = (trHtml) => [...trHtml.matchAll(/<td[^>]*>([\s\S]*?)<\/td>/gi)].map(m => cleanCell(m[1]));

  const headerTr = trList.find(tr => tr.includes('名次') && tr.includes('頭馬'));
  if (!headerTr) return races;
  const header = getTds(headerTr);

  const idxRaceNo = header.findIndex(x => x === '場次');
  const idxPos = header.findIndex(x => x === '名次');
  const idxDate = header.findIndex(x => x === '日期');
  const idxMargin = header.findIndex(x => x.replace(/\s+/g, '') === '頭馬距離');

  if (idxPos < 0 || idxMargin < 0) return races;

  for (const trHtml of trList) {
    if (!/localresults\?racedate=/i.test(trHtml)) continue;
    const tds = getTds(trHtml);
    if (!tds.length) continue;

    const raceNo = tds[idxRaceNo] ?? tds[0] ?? null;
    const pos = tds[idxPos] ?? null;
    const date = tds[idxDate] ?? null;
    const marginText = tds[idxMargin] ?? null;
    const marginLen = parseMarginToLen(marginText);

    const hrefRel = trHtml.match(/href="([^"]*localresults\?[^\"]+)"/i)?.[1] ?? null;
    const href = hrefRel ? new URL(decodeEntities(hrefRel), baseUrl).toString() : null;

    races.push({ raceNo, pos, date, marginText, marginLen, raceUrl: href });
  }

  return races;
}

async function fetchHtml(url) {
  const res = await fetch(url, { headers: { 'user-agent': UA } });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
  return await res.text();
}

async function main() {
  const linksJson = JSON.parse(await fs.readFile(linksPath, 'utf8'));
  const horseLinks = uniq((linksJson.links ?? []).map(x => x.href).filter(h => h.includes('/local/information/horse?horseid=')));

  const results = [];
  const concurrency = 8;
  let next = 0;

  async function worker() {
    while (true) {
      const i = next++;
      if (i >= horseLinks.length) return;
      const href = horseLinks[i];
      try {
        const html = await fetchHtml(href);
        const horseid = new URL(href).searchParams.get('horseid');
        const meta = extractHorseMeta(html);
        const table = extractRaceTable(html);
        const races = table ? parseRacesFromTable(table, href) : [];

        // Key race rule: horse finished 01 or 02, and head margin <= 0.5
        const keyRaces = races
          .filter(r => r.pos === '01' || r.pos === '02')
          .filter(r => r.marginLen !== null && r.marginLen <= 0.5);

        results[i] = {
          horseid,
          href,
          name: meta.name,
          trainer: meta.trainer,
          rating: meta.rating,
          keyRaces,
          keyRaceCount: keyRaces.length
        };
      } catch (e) {
        results[i] = { href, error: String(e) };
      }
    }
  }

  await Promise.all(Array.from({ length: concurrency }, worker));

  const ok = results.filter(r => r && !r.error);
  const out = {
    source: linksJson.source ?? null,
    linksScrapedAt: linksJson.scrapedAt ?? null,
    generatedAt: new Date().toISOString(),
    rule: 'Key race = horse pos 01/02 AND 頭馬距離 <= 0.5 (鼻位/短馬頭位/頭位/頸位/1/4/1/2 etc.)',
    horseCount: horseLinks.length,
    okCount: ok.length,
    errorCount: results.filter(r => r && r.error).length,
    keyRaceHorseCount: ok.filter(r => r.keyRaceCount > 0).length,
    totalKeyRaces: ok.reduce((a, r) => a + (r.keyRaceCount || 0), 0),
    horses: ok
  };

  await fs.writeFile(outPath, JSON.stringify(out, null, 2), 'utf8');
  console.log(JSON.stringify({
    horseCount: out.horseCount,
    okCount: out.okCount,
    errorCount: out.errorCount,
    keyRaceHorseCount: out.keyRaceHorseCount,
    totalKeyRaces: out.totalKeyRaces,
    out: outPath
  }, null, 2));
}

await main();
