// Independent financial arithmetic from frozen raw local inputs and native books.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';

const study=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(study,'../..');
const read=file=>JSON.parse(fs.readFileSync(file,'utf8'));
const sources={HIST:'experiments/v2-annual-repool/data',YTD2026:'experiments/annual-2026/data/science'};
const signals=read(path.join(study,'SIGNALS.json'));
const summary=read(path.join(study,'RESULTS.json'));
const fees=read(path.join(root,'standard/STRATEGY.json')).fees;
const profiles={};
for(const [window,directory] of Object.entries(sources)){
  const folder=path.join(root,directory),manifest=read(path.join(folder,'MANIFEST.json'));
  const data={};
  for(const [name,hash] of Object.entries(manifest.hashes)){
    const bytes=fs.readFileSync(path.join(folder,name));
    assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),hash);
    if(!name.endsWith('.csv.gz'))continue;
    const [header,...lines]=zlib.gunzipSync(bytes).toString().trim().split(/\r?\n/);
    const columns=header.split(',');
    data[name.split('.')[0]]=lines.map(line=>Object.fromEntries(line.split(',').map((v,i)=>
      [columns[i],columns[i]==='date'?v:Number(v)])));
  }
  if(window==='YTD2026')for(const bar of read(path.join(root,'experiments/conditional-sleeve-2026/NOV11_BARS.json'))){
    const last=data[bar.symbol].at(-1);
    data[bar.symbol].push({...bar,date:bar.day,tr:last.tr*bar.close/last.close});
  }
  profiles[window]={data,cash:read(path.join(folder,'CASH.json')),splits:read(path.join(folder,'SPLITS.json')),
    byDate:Object.fromEntries(Object.entries(data).map(([s,rows])=>[s,new Map(rows.map(r=>[r.date,r]))]))};
}
const weights={'45_45':[.45,.45],'50_40':[.5,.4],'60_30':[.6,.3]};
const eq=(a,b,label,tolerance=1e-5)=>assert.ok(Math.abs(a-b)<=tolerance,`${label}: ${a} != ${b}`);
const fee=o=>{
  const q=Math.abs(o.size);
  return Math.max(fees.commission_min,fees.commission_per_share*q)
    +Math.max(fees.platform_min,fees.platform_per_share*q)
    +(fees.settlement_per_share+fees.nms_per_share)*q
    +(o.size<0?Math.max(fees.sec_min,fees.sec_rate*q*o.price):0);
};
let scoringChecks=0,targetChecks=0,financialChecks=0;
for(const window of Object.keys(sources))for(let i=0;i<signals.candidates[window].length;i++){
  const values=signals.candidates[window][i],rows=[];
  const schedule=read(path.join(root,window==='HIST'?'experiments/v2-annual-repool/ANNUAL.json':'experiments/annual-2026/SCHEDULE.json'));
  const executionYear=Number(values.day.slice(0,4))+(values.day.slice(5,7)==='12'?1:0);
  const pool=schedule.entries.find(p=>p.year===executionYear);
  assert.deepEqual(values.rows.map(r=>r.symbol).sort(),[...pool.risk_names].sort());
  for(const r of values.rows){
    const f=profiles[window].data[r.symbol].filter(d=>d.date<=values.day);
    assert.equal(f.at(-1).date,values.day);assert.ok(f.length>=253);
    const score=f.at(-1).tr/f.at(-253).tr-1,last20=f.slice(-20);
    eq(score,r.score,'raw momentum',1e-12);
    const qualified=f.at(-1).close>=5&&last20.reduce((v,d)=>v+d.turnover,0)/20>=1e7&&last20.every(d=>d.volume>0);
    assert.equal(qualified,r.qualified);
    const group=Object.keys(pool.groups.groups).find(g=>pool.groups.groups[g].includes(r.symbol));
    assert.equal(group,r.group);rows.push({...r,score,qualified,group});scoringChecks+=3;
  }
  rows.sort((a,b)=>b.score-a.score||(a.symbol<b.symbol?-1:a.symbol>b.symbol?1:0));
  const groups=new Set(),picks=[];
  for(const r of rows)if(r.qualified&&!groups.has(r.group)){groups.add(r.group);picks.push(r.symbol);if(picks.length===2)break;}
  const archived=signals.picks[window][i];assert.equal(picks[0],archived.first);assert.equal(picks[1],archived.second);assert.equal(archived.day,values.day);targetChecks++;
}
function expectedTargets(window,weight){
  return signals.picks[window].map(p=>{
    const [a,b]=weights[weight],sleeve=[p.first,p.second].some(s=>s==='QQQ'||s==='SPY')?'SHY':'QQQ';
    return [p.day,{[p.first]:a,[p.second]:b,[sleeve]:.1}];
  });
}
function financial(a,window,weight,cost){
  assert.deepEqual(a.targets.map(r=>[r.day,r.q]),expectedTargets(window,weight));
  assert.equal(a.contract.cost_bps,cost);assert.equal(a.contract.cohort,window==='HIST'?'historical2022_2025':'reset2026');
  const {data,cash:cashEvents,splits,byDate}=profiles[window];
  const b=a.account,names=[...new Set(b.holdings.map(h=>h.symbol))],qty=Object.fromEntries(names.map(s=>[s,0]));
  const start=window==='HIST'?'2022-01-03':'2026-01-02',end=window==='HIST'?'2025-12-31':'2026-11-11';
  const days=data.SPY.filter(r=>r.date>=start&&r.date<=end).map(r=>r.date);
  assert.deepEqual(b.daily.map(d=>d.day),days);
  const group=(rows,key)=>{const m=new Map();for(const r of rows){const k=r[key];if(!m.has(k))m.set(k,[]);m.get(k).push(r);}return m;};
  const orders=group(b.orders,'day'),holdings=group(b.holdings,'day'),distributions=group(b.distributions,'ex_date');
  let cash=50000,previous=50000,maxError=0;
  for(let i=0;i<days.length;i++){
    const day=days[i],row=b.daily[i];
    for(const s of names)for(const e of splits[s])if(e.day===day){qty[s]*=e.shares_after_per_before;eq(qty[s],Math.round(qty[s]),'whole split shares');}
    const expected=names.flatMap(s=>cashEvents[s].filter(e=>e.ex_date===day&&qty[s]>0)),actual=distributions.get(day)||[];
    assert.deepEqual(actual.map(d=>d.symbol).sort(),expected.map(d=>d.symbol).sort());
    for(const e of expected){const d=actual.find(d=>d.symbol===e.symbol);eq(d.qty,qty[e.symbol],'dividend units');eq(d.per_share,e.amount,'source distribution');eq(d.net_amount,d.qty*e.amount,'dividend cash');assert.equal(d.pay_date,e.pay_date);const release=days.find(x=>x>e.pay_date);if(release)assert.equal(d.release_date,release);else assert.ok(d.release_date>days.at(-1));cash+=d.net_amount;financialChecks+=5;}
    for(const o of orders.get(day)||[]){eq(o.price,byDate[o.symbol].get(day).open*(1+(o.size>0?1:-1)*cost/10000),'raw open fill',1e-8);eq(o.fee,fee(o),'commission',1e-7);qty[o.symbol]+=o.size;assert.ok(qty[o.symbol]>=0);cash-=o.size*o.price+o.fee;financialChecks+=3;}
    const hs=holdings.get(day);assert.equal(hs.length,names.length);let market=0;
    for(const h of hs){eq(h.size,qty[h.symbol],'holding size');eq(h.close,byDate[h.symbol].get(day).close,'raw holding close',1e-8);market+=h.size*h.close;financialChecks+=2;}
    const locked=b.distributions.filter(d=>d.ex_date<=day&&day<d.release_date).reduce((v,d)=>v+d.net_amount,0);
    maxError=Math.max(maxError,Math.abs(cash-row.cash));eq(row.cash,cash,'cash');eq(row.equity,cash+market,'NAV');eq(row.locked,locked,'locked');eq(row.available,cash-locked,'available');assert.ok(row.available>=-1e-5);eq(a.native_returns[i].value,row.equity/previous-1,'native TimeReturn',1e-12);previous=row.equity;financialChecks+=5;
  }
  for(const r of summary.filter(r=>r.window===window&&r.weight===weight&&r.cost_bps===cost)){
    const indices=b.daily.flatMap((d,i)=>d.day>=r.stats.first&&d.day<=r.stats.last?[i]:[]);
    let previous=indices[0]?b.daily[indices[0]-1].equity:50000,peak=previous,dd=0;
    const opening=previous,returns=[];
    for(const i of indices){const v=b.daily[i].equity;returns.push(v/previous-1);peak=Math.max(peak,v);dd=Math.max(dd,1-v/peak);previous=v;}
    const avg=returns.reduce((a,b)=>a+b,0)/returns.length;
    const sd=Math.sqrt(returns.reduce((v,r)=>v+(r-avg)**2,0)/(returns.length-1));
    eq(r.stats.total_return,previous/opening-1,'period return',1e-12);eq(r.stats.max_drawdown,dd,'period DD',1e-12);eq(r.stats.sharpe_rf0,avg/sd*Math.sqrt(252),'period Sharpe',1e-10);
    if(r.stats.cagr!==null)eq(r.stats.cagr,(previous/opening)**(252/returns.length)-1,'period CAGR',1e-12);financialChecks+=4;
  }
  return {window,weight,cost,days:days.length,orders:b.orders.length,dividends:b.distributions.length,maxCashError:maxError};
}
const accounts=[];
for(const window of Object.keys(sources))for(const weight of Object.keys(weights))for(const cost of [5,20]){
  const a=JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(study,`reference/${window}/${weight}/${cost}bp/ACCOUNT.json.gz`))));
  accounts.push(financial(a,window,weight,cost));
}
const original=JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(study,'reference/HIST/45_45/5bp/ACCOUNT.json.gz'))));
const badCash=structuredClone(original);badCash.account.daily[0].cash+=100;assert.throws(()=>financial(badCash,'HIST','45_45',5),/cash/);
const badPosition=structuredClone(original);badPosition.account.holdings.find(h=>h.size>0).size+=1;assert.throws(()=>financial(badPosition,'HIST','45_45',5),/holding/);
const badPrice=structuredClone(original);badPrice.account.orders[0].price*=1.01;assert.throws(()=>financial(badPrice,'HIST','45_45',5),/fill/);
const output={language:'JavaScript',accounts,scoringChecks,targetChecks,financialChecks,corruptionChecks:3,
  boundary:'Separate arithmetic from raw local inputs; no independent-author review or replacement broker.'};
fs.writeFileSync(path.join(study,'INDEPENDENT.json'),JSON.stringify(output,null,2)+'\n');
console.log(JSON.stringify({accounts:accounts.length,financialChecks,scoringChecks,targetChecks,corruptionChecks:3,maxCashError:Math.max(...accounts.map(a=>a.maxCashError))}));
