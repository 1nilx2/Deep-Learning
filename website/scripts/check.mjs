import assert from 'node:assert/strict';
import {readFile, access} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const site=fileURLToPath(new URL('../',import.meta.url));
const root=path.dirname(site.replace(/\/$/,''));
const content=JSON.parse(await readFile(path.join(site,'content.json'),'utf8'));
for (const work of content.works) {
  await access(path.join(root,work.path));
  await access(path.join(site,'dist/assets',work.image));
  await access(path.join(site,'dist/assets',`${work.card}-card.svg`));
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
for(const folder of ['genrec','netflix-artwork-personalization']) {
  const config=JSON.parse(await readFile(path.join(root,'paper',folder,'study.json'),'utf8'));
  assert.equal(config.chapters.length,8);
  const html=await readFile(path.join(site,'dist/studies',config.id,'index.html'),'utf8');
  const embedded=JSON.parse(html.match(/<script id="study-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
  assert.deepEqual(embedded,config,'Published chapter content must match its paper source');
  assert.equal(await readFile(path.join(site,'dist/studies',config.id,config.notesFile),'utf8'),await readFile(path.join(root,config.notesPath),'utf8'));
  for(let i=0;i<8;i++) await access(path.join(site,'dist/assets/studies',`${config.id}-${i}.svg`));
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
console.log('PASS: both eight-chapter studies, current downloads, companion links, stable softmax/CE/DPO, and caption-matching edge cases.');
