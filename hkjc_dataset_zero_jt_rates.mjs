#!/usr/bin/env node
// Create a copy of a JSONL dataset with jockey/trainer rolling-365d stats zeroed out.
//
// Usage:
//   node hkjc_dataset_zero_jt_rates.mjs --in hkjc_dataset_v2_code_prev1.jsonl --out hkjc_dataset_v2_no_jt_rates.jsonl

import fs from 'node:fs/promises';

function arg(name,dflt=null){const i=process.argv.indexOf(name);return i>=0?process.argv[i+1]:dflt;}
const inPath=arg('--in');
const outPath=arg('--out');
if(!inPath||!outPath){console.error('Missing --in/--out');process.exit(2);} 

const txt=await fs.readFile(inPath,'utf8');
const lines=txt.split(/\n+/).filter(Boolean);

const keysToZero=[
  'jockey_365d_starts','jockey_365d_win_rate','jockey_365d_place_rate',
  'trainer_365d_starts','trainer_365d_win_rate','trainer_365d_place_rate'
];

let wrote=0;
const out=[];
for(const line of lines){
  const r=JSON.parse(line);
  for(const k of keysToZero){ if(k in r) r[k]=0; }
  out.push(JSON.stringify(r));
  wrote++;
}
await fs.writeFile(outPath, out.join('\n')+'\n');
console.log(JSON.stringify({out: outPath, rows: wrote},null,2));
