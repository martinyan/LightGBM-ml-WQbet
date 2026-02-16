#!/usr/bin/env node
// Train a simple logistic regression model from JSONL dataset produced by hkjc_ml_build_dataset_sqlite.mjs
// and evaluate on holdout dates.
//
// Usage:
//   node hkjc_train_logreg_from_jsonl.mjs --data hkjc_dataset_v2_code_prev1.jsonl \
//     --holdoutDates 2026/02/11,2026/02/14 --outModel hkjc_model_v2.json

import fs from 'node:fs/promises';

function arg(name,dflt=null){const i=process.argv.indexOf(name);return i>=0?process.argv[i+1]:dflt;}
const dataPath=arg('--data');
const holdoutDates=(arg('--holdoutDates','')||'').split(',').map(s=>s.trim()).filter(Boolean);
const outModel=arg('--outModel','hkjc_model.json');

if(!dataPath){console.error('Missing --data');process.exit(2);} 

function sigmoid(z){return 1/(1+Math.exp(-z));}
function dot(w,x){let s=0; for(let i=0;i<w.length;i++) s+=w[i]*x[i]; return s;}

function trainLogReg(X,y,{lr=0.1,epochs=1500,l2=1e-2}={}){
  const n=X.length,d=X[0].length;
  const w=new Array(d).fill(0);
  for(let ep=0; ep<epochs; ep++){
    const grad=new Array(d).fill(0);
    for(let i=0;i<n;i++){
      const p=sigmoid(dot(w,X[i]));
      const err=p-y[i];
      for(let j=0;j<d;j++) grad[j]+=err*X[i][j];
    }
    for(let j=0;j<d;j++){
      grad[j]=grad[j]/n + l2*w[j];
      w[j]-=lr*grad[j];
    }
  }
  return w;
}

function metrics(yTrue,p){
  let ll=0, correct=0;
  for(let i=0;i<yTrue.length;i++){
    const ytRaw = yTrue[i];
    const yt = (ytRaw === 1 || ytRaw === '1' || ytRaw === true) ? 1 : 0;
    const pr=Math.min(0.999999, Math.max(1e-6, p[i]));
    ll += -(yt*Math.log(pr)+(1-yt)*Math.log(1-pr));
    const pred = pr>=0.5?1:0;
    if(pred===yt) correct++;
  }
  return {logloss: yTrue.length ? ll/yTrue.length : null, acc: yTrue.length ? correct/yTrue.length : null};
}

function top2HitRate(rows){
  // rows: [{racedate, venue, race_no, horse_no, y, p}]
  const byRace=new Map();
  for(const r of rows){
    const k=`${r.racedate}|${r.venue}|${r.race_no}`;
    if(!byRace.has(k)) byRace.set(k,[]);
    byRace.get(k).push(r);
  }
  let races=0, hits=0;
  for(const arr of byRace.values()){
    races++;
    arr.sort((a,b)=>b.p-a.p);
    const top2=arr.slice(0,2);
    if(top2.some(x=>x.y===1)) hits++;
  }
  return {races, top2_hit_rate: races ? hits/races : null};
}

const txt=await fs.readFile(dataPath,'utf8');
const lines=txt.split(/\n+/).filter(Boolean);

// Dataset rows are flat key/value (not nested under .features)
const sample=JSON.parse(lines[0]);
const metaKeys = new Set([
  'racedate','venue','race_no','runner_id','horse_code','horse_name_zh','horse_no',
  'y_finish_pos','y_win','y_place',
  // non-numeric categoricals we should never include as numeric features
  'cur_jockey','cur_trainer',
  'cur_surface'
]);
// Prefer y_place (top3 label)
const labelKey = 'y_place';

const featKeys = Object.keys(sample)
  .filter(k => !metaKeys.has(k) && k !== labelKey)
  .sort();

function vectorize(row){
  const x=[1];
  for(const k of featKeys) x.push(Number(row[k]??0));
  return x;
}

const trainRows=[];
const holdoutRowsByDate=new Map();
for(const line of lines){
  const r=JSON.parse(line);
  if(holdoutDates.includes(r.racedate)){
    if(!holdoutRowsByDate.has(r.racedate)) holdoutRowsByDate.set(r.racedate,[]);
    holdoutRowsByDate.get(r.racedate).push(r);
  } else {
    trainRows.push(r);
  }
}

// Standardize features using training set (excluding intercept)
const X=trainRows.map(vectorize);
const y=trainRows.map(r=>Number(r[labelKey]??0));
const d=X[0].length;
const means=new Array(d).fill(0);
const stds=new Array(d).fill(1);
for(let j=1;j<d;j++){
  let s=0;
  for(const x of X) s+=x[j];
  means[j]=s/X.length;
  let v=0;
  for(const x of X) v+=(x[j]-means[j])**2;
  stds[j]=Math.sqrt(v/X.length)||1;
  for(const x of X) x[j]=(x[j]-means[j])/stds[j];
}

const w=trainLogReg(X,y,{lr:0.1,epochs:2000,l2:1e-2});

const report={
  trainedAt: new Date().toISOString(),
  dataPath,
  featureCount: featKeys.length,
  trainRows: trainRows.length,
  holdoutDates,
  holdout: {}
};

for(const [date,rows] of holdoutRowsByDate.entries()){
  const Xh=rows.map(vectorize);
  for(const x of Xh) for(let j=1;j<d;j++) x[j]=(x[j]-means[j])/stds[j];
  const ph=Xh.map(x=>sigmoid(dot(w,x)));
  const yh=rows.map(r=>Number(r[labelKey]??0));
  const m=metrics(yh,ph);
  const raceRows=rows.map((r,i)=>({racedate:r.racedate, venue:r.venue, race_no:r.race_no, horse_no:r.horse_no, y:Number(r[labelKey]??0), p:ph[i]}));
  report.holdout[date]={...m, ...top2HitRate(raceRows), rows: rows.length};
}

await fs.writeFile(outModel, JSON.stringify({w, means, stds, featKeys, report},null,2));
console.log(JSON.stringify(report,null,2));
