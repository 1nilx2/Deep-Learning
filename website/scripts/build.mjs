import {build} from 'esbuild';
import {writeResearchPlates} from './research-plates.mjs';
import {cp, mkdir, readFile, writeFile, rm} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const site = fileURLToPath(new URL('../', import.meta.url));
const root = path.dirname(site.replace(/\/$/, ''));
const fromRepo = p => path.join(root, p);
const dist = path.join(site, 'dist');
const content = JSON.parse(await readFile(path.join(site, 'content.json'), 'utf8'));
// Verify referenced source files before replacing the generated output.
for (const entry of content.entries) await readFile(fromRepo(entry[4]));
for (const work of content.works) await readFile(fromRepo(work.path));
await rm(dist, {recursive:true, force:true});
await mkdir(path.join(dist, 'assets'), {recursive:true});
await cp(path.join(site, 'src/index.html'), path.join(dist, 'index.html'));
await cp(path.join(site, 'src/thumbnails/review.html'), path.join(dist, 'thumbnail-review.html'));
for (const [name, source] of Object.entries(content.assets)) {
  const destination = path.join(dist, 'assets', name);
  await mkdir(path.dirname(destination), {recursive:true});
  await cp(fromRepo(source), destination);
}
await build({absWorkingDir:site, entryPoints:['src/app.tsx'], bundle:true, jsx:'automatic', minify:true, outfile:'dist/app.js', legalComments:'eof', define:{'process.env.NODE_ENV':'"production"'}});
const css = await readFile(path.join(site, 'dist/app.css'), 'utf8');
await writeFile(path.join(dist, 'styles.css'), (await readFile(path.join(site,'src/styles.css'),'utf8'))+'\n'+css);
await rm(path.join(dist, 'app.css'));
const plateStudies = await writeResearchPlates(dist);
// Canonical study files live in paper/spd; every build regenerates the website.
const study = path.join(dist, 'studies/spd');
await mkdir(study, {recursive:true});
const notes = await readFile(fromRepo('paper/spd/README.md'), 'utf8');
let html = await readFile(fromRepo('paper/spd/index.html'), 'utf8');
const downloadPattern = /var S1=`[\s\S]*?`;document\.getElementById\("download"\)/;
if (!downloadPattern.test(html)) throw new Error('SPD download hook changed; update the build before publishing.');
html = html.replace(downloadPattern, () => `var S1=${JSON.stringify(notes).replaceAll('<','\\u003c')};document.getElementById("download")`);
html = html.replace('<div class="headlinks">', '<div class="headlinks"><a href="../../">← Fieldnotes</a><a href="https://github.com/1nilx2/Deep-Learning/blob/main/paper/spd/README.md">GitHub notes ↗</a>');
const spdItems=plateStudies.spd.chapters.map((chapter,i)=>({src:`../../assets/plates/spd-${i}.svg`,alt:chapter.title.replaceAll('\n',' '),title:chapter.title.replaceAll('\n',' '),subtitle:chapter.tag}));
const spdImages=/var UM=\[[\s\S]*?\];var Ri=/;
if(!spdImages.test(html)) throw new Error('SPD carousel data hook changed.');
html=html.replace(spdImages,()=>`var UM=${JSON.stringify(spdItems)};var Ri=`);
html=html.replaceAll('#c9b8ff','#695779').replaceAll('#f3bc97','#8d6246').replaceAll('#badea3','#4f7059');
const paperPanelCSS=`.lab{--accent:#695779;--muted:#756d7c;background:#e6e0eb!important;color:#342b3d;border-color:#ffffff20}.lab p,.lab td,.lab th,.lab .path-equation{color:#554a60}.lab strong,.lab .result,.lab .formula{color:var(--accent)}.lab .matrix td,.lab .objective-matrix td{background:#d5ccdE55;border-color:#8b7c9920}.lab .matrix td.chosen{background:#a99bb955;border-color:#695779}.lab .matrix input{color:#342b3d}.lab .tabs button[aria-selected=true]{background:#69577914;color:#695779}.lab .lablabel,.lab .objective-slider{color:#756d7c}.lab .latency th,.lab .latency td{color:#554a60;border-color:#69577928}.lab .path-equation b,.lab .matrix-guide strong{color:#695779}`;
html=html.replace('</head>',`<style>${paperPanelCSS}</style></head>`);
html = html.replace('</head>', '<style>.headlinks{flex-wrap:wrap;justify-content:flex-end;gap:12px 18px}header{gap:16px}.headlinks a{white-space:nowrap}@media(max-width:720px){header{display:block}.headlinks{justify-content:flex-start;margin-top:16px}}</style></head>');
await writeFile(path.join(study,'index.html'), html);
await writeFile(path.join(study,'SPD_notes.md'), notes);
// Shared interactive reading pages: their chapter content lives beside the papers.
await build({absWorkingDir:site, entryPoints:['src/study/app.jsx'], bundle:true, jsx:'automatic', minify:true, outfile:'dist/study.js', legalComments:'eof', define:{'process.env.NODE_ENV':'"production"'}});
await mkdir(path.join(dist,'assets/studies'), {recursive:true});
await cp(path.join(site,'src/study/assets'), path.join(dist,'assets/studies'), {recursive:true});
for (const config of Object.values(plateStudies).filter(study=>study.id!=='spd')) {
  const directory=path.join(dist,'studies',config.id);
  await mkdir(directory,{recursive:true});
  const data=JSON.stringify(config).replaceAll('<',String.fromCharCode(92)+'u003c');
  const shell=`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#100f14"><meta name="description" content="An interactive Study or Lab dossier from Deep Learning Fieldnotes."><title>${config.name} — Fieldnotes</title><link rel="stylesheet" href="../../study.css"><script id="study-data" type="application/json">${data}</script><script src="../../study.js" defer></script></head><body><div id="study-root"></div><noscript><main style="padding:8%;color:#ddd;background:#100f14;font-family:system-ui"><h1>${config.name}</h1><p>Enable JavaScript for the interactive study, or read the complete notes.</p><a href="https://github.com/1nilx2/Deep-Learning/blob/main/${config.notesPath}">Read the GitHub notes</a> · <a href="../../">Fieldnotes</a></main></noscript></body></html>`;
  await writeFile(path.join(directory,'index.html'),shell);
  await cp(fromRepo(config.notesPath),path.join(directory,config.notesFile));
}

await writeFile(path.join(dist,'.nojekyll'), '');
await cp(path.join(site,'THIRD_PARTY_NOTICES.md'), path.join(dist,'THIRD_PARTY_NOTICES.md'));
console.log(`Built ${content.works.length} featured works, ${content.entries.length} archive entries, and six coordinated Study/Lab pages.`);
