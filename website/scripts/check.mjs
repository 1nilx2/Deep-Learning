import assert from 'node:assert/strict';
import {readFile, access} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const site=fileURLToPath(new URL('../',import.meta.url));
const root=path.dirname(site.replace(/\/$/,''));
const content=JSON.parse(await readFile(path.join(site,'content.json'),'utf8'));
for (const work of content.works) {
  await access(path.join(root,work.path));
  if(work.heroImage) await access(path.join(site,'dist/assets',work.heroImage));
  await access(path.join(site,'dist/assets',work.image));
  if(work.page) await access(path.join(site,'dist',work.page,'index.html'));
}
for(const entry of content.entries) await access(path.join(root,entry[4]));
const source=await readFile(path.join(root,'paper/spd/README.md'),'utf8');
assert.equal(await readFile(path.join(site,'dist/studies/spd/SPD_notes.md'),'utf8'),source,'Downloaded notes must match repository notes');
const study=await readFile(path.join(site,'dist/studies/spd/index.html'),'utf8');
const embedded=study.match(/var S1=("(?:[^"\\]|\\.)*");document\.getElementById\("download"\)/);
assert.ok(embedded,'Study must have a synchronized notes download');
assert.equal(JSON.parse(embedded[1]),source);
assert.ok(study.includes('href="../../"'),'Study must link back to homepage');
const shell=await readFile(path.join(site,'dist/index.html'),'utf8');
for(const [,resource] of shell.matchAll(/(?:href|src)="([^"#]+)"/g)) {
  if(!/^https?:/.test(resource)) await access(path.join(site,'dist',resource));
}
console.log('PASS: content links, generated assets, SPD route, back navigation, and both downloaded notes versions match their repository sources.');

// Verify the new studies use their canonical chapter content and current notes.
for(const folder of ['genrec','netflix-artwork-personalization','gr2']) {
  const config=JSON.parse(await readFile(path.join(root,'paper',folder,'study.json'),'utf8'));
  assert.equal(config.chapters.length,config.id==='gr2'?7:8);
  const html=await readFile(path.join(site,'dist/studies',config.id,'index.html'),'utf8');
  const embedded=JSON.parse(html.match(/<script id="study-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
  assert.deepEqual(embedded,config,'Published chapter content must match its paper source');
  assert.equal(await readFile(path.join(site,'dist/studies',config.id,config.notesFile),'utf8'),await readFile(path.join(root,config.notesPath),'utf8'));
  for(let i=0;i<config.chapters.length;i++) await access(path.join(site,'dist/assets/plates',`${config.id}-${i}.svg`));
  await access(path.join(site,'dist/studies',config.next,'index.html'));
}
const {softmax,weightedCE,dpoLoss,bigramMatch}=await import('../src/study/math.mjs');
const probabilities=softmax([1001,1002,1000]);
assert.ok(probabilities.every(Number.isFinite));
assert.ok(Math.abs(probabilities.reduce((a,b)=>a+b,0)-1)<1e-12);
assert.ok(probabilities[1]>probabilities[0]&&probabilities[0]>probabilities[2]);
assert.equal(weightedCE([.4,1.3,-.2],1,0),0);
assert.ok(weightedCE([0,2,0],1,1)<weightedCE([0,1,0],1,1));
assert.ok(Number.isFinite(weightedCE([1000,-1000,0],1,1)));
assert.ok(Math.abs(dpoLoss(0,0,1)-Math.log(2))<1e-12);
assert.ok(dpoLoss(2,0,1)<dpoLoss(1,0,1));
assert.ok(Number.isFinite(dpoLoss(-1000,1000,2)));
assert.equal(bigramMatch('A warm AMBER room!',['A cold blue moon.','A warm amber room.']).index,1);
assert.equal(bigramMatch('Unrelated words',['A cold blue moon.','A warm amber room.']).index,null);
assert.equal(bigramMatch('cold blue',['A cold blue moon.','A cold blue sky.']).index,null);
console.log('PASS: all paper study chapters, current downloads, companion links, stable softmax/CE/DPO, and caption-matching edge cases.');

for(const id of ['embedding','laya','turbo']){
  const config=JSON.parse(await readFile(path.join(site,'src/study/data',id+'.json'),'utf8'));
  const html=await readFile(path.join(site,'dist/studies',id,'index.html'),'utf8');
  const data=JSON.parse(html.match(/<script id="study-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
  assert.deepEqual(data,config);
  assert.equal(await readFile(path.join(site,'dist/studies',id,config.notesFile),'utf8'),await readFile(path.join(root,config.notesPath),'utf8'));
  for(let i=0;i<config.chapters.length;i++)await access(path.join(site,'dist/assets/plates',`${id}-${i}.svg`));
  await access(path.join(site,'dist/studies',config.next,'index.html'));
}
assert.ok(study.includes('../../assets/plates/spd-0.svg'),'SPD must use the new chapter plates');
assert.ok(!study.includes('var UM=[{src:"data:image'),'Old SPD carousel art must be replaced in the deployed study');
console.log('PASS: all seven Study/Lab routes, 46 chapter plates, synchronized downloads, and the SPD carousel artwork hook.');

// Verify the conditional reward boundary exactly as published, including the initial-#1 exception.
const {rankReward,groupAdvantages}=await import('../src/study/gr2-math.mjs');
assert.equal(rankReward(6,6).total,0,'Copied non-top ranking must not earn the format bonus');
assert.equal(rankReward(6,6,10,true,.1,false).total,.1);
assert.equal(rankReward(6,2).total,.5);
assert.equal(rankReward(6,8).total,-.2);
assert.equal(rankReward(1,1).total,.1);
assert.equal(rankReward(1,2).total,0,'Eq.15 permits bonus when initially #1 even if demoted');
assert.equal(rankReward(6,2,10,false).total,.4);
assert.equal(rankReward(6,6,10,true,0).total,0);
assert.throws(()=>rankReward(11,1),RangeError);
for(let before=1;before<=10;before++)for(let after=1;after<=10;after++){
 const r=rankReward(before,after);
 assert.ok(Math.abs(r.rank-(before-after)/10)<1e-12);
 assert.equal(r.bonus,(after<before||before===1)?.1:0);
}
const mixed=groupAdvantages([.5,0,-.2,.3]);
assert.ok(Math.abs(mixed.advantages.reduce((a,b)=>a+b,0))<1e-12);
assert.ok(mixed.advantages[0]>mixed.advantages[3]);
assert.deepEqual(groupAdvantages([.1,.1,.1,.1]).advantages,[0,0,0,0]);
assert.deepEqual(await readFile(path.join(site,'dist/assets/gr2-paper.pdf')),await readFile(path.join(root,'paper/gr2/source/2602.07774v6.pdf')));
console.log('PASS: GR2 conditional rewards, rank-one exception, format gating, valid-rank bounds and equal-reward groups.');
