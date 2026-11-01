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
