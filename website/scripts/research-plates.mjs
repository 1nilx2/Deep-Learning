import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const site=fileURLToPath(new URL('../',import.meta.url));
const root=path.dirname(site.replace(/\/$/,''));
const palettes={gr2:['#dfe6db','#536b50'],spd:['#e6e0eb','#695779'],genrec:['#e5e1d6','#75674e'],artwork:['#e7dfe4','#795a70'],embedding:['#dde7e0','#42685f'],laya:['#e7e0d6','#785e49'],turbo:['#dfe5eb','#4d637b']};
const escape=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const text=(x,y,s,size=28,ink='#39323e',extra='')=>`<text x="${x}" y="${y}" fill="${ink}" font-family="Inter, Helvetica Neue, Arial, sans-serif" font-size="${size}" ${extra}>${escape(s)}</text>`;
const line=(x,y,xx,yy,ink)=>`<path d="M${x} ${y}H${xx}" stroke="${ink}" stroke-opacity=".18" fill="none"/>`;
const wrap=(s,length=33)=>{const lines=[];let row='';for(const word of s.replaceAll('\n',' ').split(' ')){if(row.length+word.length+1>length&&row){lines.push(row);row=word;}else row+=(row?' ':'')+word;}if(row)lines.push(row);return lines;};
const rect=(x,y,w,h,ink,opacity='.1')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="${ink}" fill-opacity="${opacity}"/>`;
function rows(items,ink){return items.map(([label,detail],i)=>{const y=414+i*113;return text(92,y,String(i+1).padStart(2,'0'),24,ink)+text(160,y,label,35,ink)+text(160,y+46,detail,24,'#6e6670')+line(160,y+72,1100,y+72,ink)}).join('');}
function columns(left,right,ink){let svg='';for(const [j,items] of [left,right].entries()){const x=92+j*566;svg+=text(x,418,items[0],25,ink,'letter-spacing="2"');for(let i=1;i<items.length;i++)svg+=rect(x,460+(i-1)*105,505,84,ink,'.08')+text(x+22,513+(i-1)*105,items[i],27,ink);}return svg;}
function metrics(items,ink){return items.map(([value,label,base],i)=>{const x=92+(i%2)*561,y=470+Math.floor(i/2)*263;return text(x,y,value,75,ink,'letter-spacing="-3"')+text(x,y+55,label,26,ink)+text(x,y+100,base,21,'#6e6670')+line(x,y+140,x+480,y+140,ink)}).join('');}
function bars(items,max,unit,ink){return items.map(([label,value],i)=>{const y=430+i*137;return text(92,y,label,25,ink)+rect(92,y+24,1008,45,ink,'.1')+rect(92,y+24,1008*value/max,45,ink,'.55')+text(1100,y+109,value+' '+unit,31,ink,'text-anchor="end"')}).join('');}
function matrix(values,ink,label='M',x=351,y=417,size=148){let svg=text(x+size*1.5,y-45,label,34,ink,'text-anchor="middle"');for(let i=0;i<3;i++){svg+=text(x-28,y+i*size+size*.6,'ABC'[i],24,ink,'text-anchor="end"')+text(x+i*size+(size-10)/2,y-13,i+1,21,ink,'text-anchor="middle"');}values.forEach((row,i)=>row.forEach((v,j)=>{const chosen=[[0,1],[1,0],[2,2]].some(([r,c])=>r===i&&c===j);svg+=rect(x+j*size,y+i*size,size-10,size-10,ink,chosen?'.35':'.1')+text(x+j*size+(size-10)/2,y+i*size+size*.6,typeof v==='number'&&!Number.isInteger(v)?v.toFixed(2):v,size*.29,ink,'text-anchor="middle"')}));return svg;}
const M=[[9,8,1],[10,2,0],[1,3,9]],P=[[0,1,0],[1,0,0],[0,0,1]];
function sinkhorn(){let s=M.map(r=>r.map(v=>Math.exp((v-10)/2)));for(let k=0;k<120;k++){s=s.map(r=>{const sum=r.reduce((a,b)=>a+b);return r.map(v=>v/sum)});const sums=[0,1,2].map(j=>s.reduce((sum,r)=>sum+r[j],0));s=s.map(r=>r.map((v,j)=>v/sums[j]));}return s;}
async function posters(ink,caption=false){let s='';const descriptions=['A lone figure at sea.','Two characters, one room.','A runner on the ridge.'];for(let i=0;i<3;i++){let svg=await readFile(path.join(site,`src/study/assets/poster-${i}.svg`),'utf8');svg=svg.replace(/<svg\b[^>]*>/,`<svg x="${87+i*352}" y="405" width="320" height="426" viewBox="0 0 600 800">`).replace(/id="([^"]+)"/g,(_,id)=>`id="p${i}-${id}"`).replace(/url\(#([^)]+)\)/g,(_,id)=>`url(#p${i}-${id})`);s+=svg+text(247+i*352,884,caption?descriptions[i]:['SUSPENSE','RELATIONSHIPS','ADVENTURE'][i],caption?21:22,ink,'text-anchor="middle"');}return s;}
// The SID plate follows the reading map's problem → alternatives → solution argument.
function sidArgument(plate,ink){
 let svg='';
 plate.problems.forEach(([question,consequence],i)=>{
  const y=366+i*100;
  svg+=text(78,y,String(i+1).padStart(2,'0'),22,ink)
   +text(137,y,question,32,ink,'font-weight="500"')
   +text(137,y+41,consequence,25,'#626c60')
   +line(137,y+62,1120,y+62,ink);
 });
 svg+=rect(70,662,1060,237,ink,'.09')+text(99,710,plate.solution,35,ink,'font-weight="500"');
 plate.examples.forEach(([label,code],i)=>{
  svg+=text(105,762+i*43,label,27,ink)+text(300,762+i*43,code,31,ink);
 });
 svg+=text(667,765,'Share early codes',25,ink)+text(667,805,'Distinguish by the full tuple',25,ink)
  +text(103,869,plate.analogy,25,ink)
  +text(78,947,'↓',33,ink)+text(137,945,plate.alignment[0],28,ink)
  +text(137,986,plate.alignment[1],28,ink);
 return svg;
}
function traceDistillation(plate,ink){
 const arrow=(d)=>`<path d="${d}" fill="none" stroke="${ink}" stroke-width="2" stroke-opacity=".55" marker-end="url(#trace-arrow)"/>`;
 let svg=`<defs><marker id="trace-arrow" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="${ink}" stroke-width="1.2"/></marker></defs>`;
 svg+=text(72,353,plate.mainLabel,23,ink)
  +rect(70,379,460,88,ink,'.07')+text(94,414,plate.input[0],30,ink)+text(94,449,plate.input[1],24,ink)
  +arrow('M545 423H632')
  +rect(650,379,480,88,ink,'.1')+text(677,414,plate.teacher[0],30,ink)+text(677,449,plate.teacher[1],27,ink)
  +arrow('M890 470V492H600V516')
  +rect(300,523,600,62,ink,'.12')+text(600,564,plate.verification,31,ink,'text-anchor="middle"')
  +arrow('M300 554H278V612')+arrow('M900 554H910V612')
  +rect(70,623,460,85,ink,'.05')+text(94,658,plate.reject[0],29,ink)+text(94,691,plate.reject[1],23,ink)
  +rect(650,623,480,85,ink,'.17')+text(674,658,plate.accept[0],29,ink)+text(674,691,plate.accept[1],24,ink)
  +text(73,747,plate.verificationNote,24,ink)
  +rect(70,781,1060,203,ink,'.06')+text(97,825,plate.alternative[0],28,ink,'font-weight="500"')
  +text(97,868,plate.alternative[1],27,ink)+text(97,918,plate.alternative[2],25,ink)+text(97,955,plate.alternative[3],25,ink);
 return svg;
}
const specs={
gr2:[
['rows',[['Represent the items','Text → RQ-VAE → Semantic IDs'],['Align the language','SID + natural language → mid-training'],['Distill the reasoning','Teacher → accepted traces → weighted SFT'],['Optimize the order','Conditional rewards → DAPO']]],
['gr2-sid'],
['gr2-traces'],
['rows',[['Rank reward','Rrank = (rank before − rank after) / N'],['Format reward','Can reasoning + ranking be extracted?'],['Conditional bonus','Add αRfmt if improved OR initially #1'],['Copying below #1','No improvement → no format bonus']]],
['rows',[['Generate a group','Same prompt → several responses'],['Compute rewards','Rank movement + conditional format'],['Compare within group','Aᵢ = (Rᵢ − mean) / std'],['DAPO update','Asymmetric clipping + dynamic sampling']]],
['metrics',[['0.7460','Beauty · Recall@5','Pre-rank: 0.7227 · Table 5'],['0.5234','Beauty · NDCG@5','Pre-rank: 0.5101 · Table 5'],['0.7077','Sports · Recall@5','Pre-rank: 0.7077 · Table 6'],['0.4791','Sports · NDCG@5','Pre-rank: 0.4796 · Table 6']]],
['columns',['KEEP THE INTUITION','Item representation matters','Select teacher traces','Reward better rankings'],['REFINE THE DETAILS','RQ-VAE, not BPE','Assistant-token weighted NLL','Rank lift, not an AUC reward']],
],
spd:[
['rows',[['Shared backbone','Context + candidate descriptions → H'],['Learn the placements','Attention + projection → M'],['Training path','M → Sinkhorn → S → CE against P'],['Inference path','M → Hungarian → complete ranking']]],
['rows',[['Candidate A','Description → final-token hidden state h₁'],['Candidate B','Description → final-token hidden state h₂'],['Candidate C','Description → final-token hidden state h₃'],['Stack the representations','H has one contextual vector per candidate']]],
['matrix',M,'M · ITEM–POSITION SCORES'],
['matrix','S','S · SINKHORN SOFT ASSIGNMENT'],
['paired'],
['matrix',M,'MAXIMUM-SCORE ASSIGNMENT / B → A → C'],
['metrics',[['1,807 ms','Teacher with reasoning','Internal dataset · NVIDIA A100'],['88 ms','Ranking-only teacher','Same reported internal comparison'],['28 ms','SPD','Single-pass ranking configuration'],['3.1×','Versus ranking-only teacher','64.5× versus reasoning teacher']]],
['columns',['REPRESENTATION','Candidate descriptions','Contextual vectors hᵢ','Item–position scores M'],['RANKING','One-to-one constraints','Hungarian assignment','A complete permutation']],
],
genrec:[
['rows',[['Read the context','History + metadata + request → x'],['Encode once','LLM → context-dependent h'],['Score the catalog','Learned item vectors eᵢ + head φ'],['Return the order','Sort item scores at serving time']]],
['columns',['PHASE 1 / FOUNDATION','User + content understanding','Language + instructions','Less frequent refresh'],['PHASE 2 / RANKER','Recommendation supervision','Backbone + head + items','More frequent refresh']],
['rows',[['Keep strong signals','Long plays + explicit positive feedback'],['Compress repetition','Binge sessions + repeated activity'],['Summarize older history','Retain interests, reduce detail'],['Reported context ablation','About 5,000 → 1,700 tokens']]],
['columns',['CONTEXT SIDE','Verbalized input x','Backbone forward pass','Context activation h'],['CATALOG SIDE','Learned parameters eᵢ','Catalog-aware φ(h, eᵢ)','Logits → probabilities']],
['metrics',[['0.30','Illustrative target probability','Target = item B'],['1.204','−log p(target)','Cross-entropy for one example'],['w × CE','Reward-weighted contribution','Weight mapping not disclosed'],['Joint','Backbone + head + items','Plus auxiliary language objective']]],
['rows',[['Request context','No target labels at serving time'],['One prefill pass','Compute contextual representation h'],['Catalog-aware scoring','One score per candidate'],['Sort the scores','No token-by-token ranking text']]],
['metrics',[['+1.6%','Relative offline MRR','Versus production ranker'],['40×','Fewer Phase-2 examples','Excludes pretraining + Phase 1'],['+0.115%','Homepage engagement','Reported treatment vs production'],['~⅓','Compressed-context cost','Versus original context, not ranker']]],
['rows',[['GenRec','Context h + learned eᵢ → scores'],['SPD','Candidate hᵢ → M → assignment'],['Artwork','History + captions → selected image'],['Read the task boundary','Similar backbones, different outputs']]],
],
artwork:[
['posters'],['captions'],
['rows',[['Viewing history','Preference evidence from past activity'],['Target title','The selected title and its context'],['Candidate captions','Visual cues converted to text'],['Selection instruction','Return the chosen artwork caption']]],
['columns',['STANDARD SFT','Shared context x','Selected caption c*','Next-token supervision'],['REASONING-SFT','Same shared context x','Reason r → caption c*','LoRA adapter updates']],
['columns',['CHOSEN RESPONSE','Ground-truth artwork','log πθ(yw) / πref(yw)','Reference stays frozen'],['REJECTED RESPONSE','Sampled alternative','log πθ(yl) / πref(yl)','Not an observed dislike']],
['rows',[['Model output','Generated caption text'],['Candidate comparison','Match output to supplied captions'],['Illustrative matcher','Shared word bigrams'],['Ambiguous or no overlap','No unique match in the teaching demo']]],
['bars',[['SFT',2.45],['DPO',2.82],['Reasoning-SFT',5.21]],6,'% relative IPS'],
['rows',[['Visible evidence','What the supplied artwork depicts'],['Generated explanation','A learned response, not observed intent'],['Preference supervision','Chosen vs sampled alternative'],['Evaluation boundary','Offline IPS, not an online A/B test']]],
],
embedding:[
['metrics',[['221','Paired dialogues','Six children'],['72.4%','Text → audio Top-1','Exp 1'],['65.6%','Audio → text Top-1','Exp 1'],['3','Directly reviewed pairs','218 alignments assumed']]],
['bars',[['Text → audio · Top-1',72.4],['Audio → text · Top-1',65.6]],100,'%'],
['metrics',[['0.810','Mean paired cosine similarity','Original embedding space'],['0.720','Mean mismatched similarity','Exp 1 comparison'],['0.090','Mean paired-minus-mismatched','Rounded reported difference'],['100%','Modality classification','Held-out children; Exp 1']]],
['rows',[['Group sizes','59 / 30 / 60 / 30 / 12 / 30 pairs'],['Direction matters','Text queries and audio queries differ'],['Small groups','Rates can be unstable'],['Read the full report','Per-child values and uncertainty']]],
['rows',[['Observed','Cross-modal retrieval works here'],['Also observed','A strong modality signature remains'],['Alignment limit','Only three pairs directly reviewed'],['Semantic limit','No independent gold topic labels']]],
],
laya:[
['metrics',[['8','Related payment intents','A BANKING77 subset'],['320','Held-out test requests','40 official test requests per intent'],['400','Allocated training requests','Laya retains 40 for calibration'],['CPU','Batch size four','Single local timing runs']]],
['columns',['LAYA','One request','One eight-option question','Option probabilities'],['CROSSENCODER','One request','Eight request–option pairs','Pair scores → decision']],
['bars',[['Laya · original',77.5],['Laya · fine-tuned',90.6],['CrossEncoder · base',89.7],['CrossEncoder · large',93.4]],100,'% accuracy'],
['bars',[['Laya · original',98.6],['Laya · fine-tuned',98.8],['CrossEncoder · base',76.3],['CrossEncoder · large',208]],250,'ms / request'],
['rows',[['Gradient-training items','Laya 360; CrossEncoder 400'],['Calibration','Held-back items vs validation logits'],['Initialization + objectives','Not a controlled architecture ablation'],['Conclusion','A system comparison with limits']]],
],
turbo:[
['columns',['BASELINE','FP16 keys','FP16 values','Qwen2.5-1.5B · M1 Max'],['EXPERIMENT','Q8_0 keys','TurboQuant 4-bit values','Same local model and device']],
['bars',[['FP16 KV',896],['Q8 K + TurboQuant V',357]],1000,'MiB'],
['metrics',[['1,801 MiB','Model allocation','Same measured local runs'],['1,671 MiB','Compute buffers','Experimental format · 30.3k'],['357 MiB','KV allocation','Experimental format · 30.3k'],['3,829 MiB','Total Metal allocation','Weights + compute + KV']]],
['metrics',[['37.74 s','FP16 prefill','30.3k · median across two prompts'],['73.09 s','Experimental prefill','30.3k · median across two prompts'],['2 / 2','FP16 first code correct','Synthetic retrieval cases'],['1 / 2','Experimental first code correct','Too few cases for a general rate']]],
['metrics',[['26.7 GiB','FP16 KV projection','One million allocated tokens'],['10.6 GiB','Experimental KV projection','Same architecture and format'],['32,768','Native position limit','This model cannot run 1M context'],['Estimate','Storage arithmetic only','Not a one-million-token benchmark']]],
]};
export async function writeResearchPlates(dist){
 const dir=path.join(dist,'assets/plates');await mkdir(dir,{recursive:true});
 const studies={spd:{id:'spd',name:'SPD',chapters:JSON.parse(await readFile(path.join(site,'src/study/data/spd-chapters.json'),'utf8'))}};
 for(const folder of ['genrec','netflix-artwork-personalization','gr2']){const c=JSON.parse(await readFile(path.join(root,'paper',folder,'study.json'),'utf8'));studies[c.id]=c;}
 for(const id of ['embedding','laya','turbo'])studies[id]=JSON.parse(await readFile(path.join(site,'src/study/data',id+'.json'),'utf8'));
 for(const [id,study] of Object.entries(studies)){
  const [paper,ink]=palettes[id];
  for(let i=0;i<study.chapters.length;i++){
   const ch=study.chapters[i],spec=specs[id][i];if(!spec)throw new Error(`Missing plate: ${id}/${i}`);
   let body='';const [type,data,label]=spec;
   if(type==='rows')body=rows(data,ink);
   if(type==='gr2-sid')body=sidArgument(ch.plate,ink);
   if(type==='gr2-traces')body=traceDistillation(ch.plate,ink);
   if(type==='columns')body=columns(data,label,ink);
   if(type==='metrics')body=metrics(data,ink);
   if(type==='bars')body=bars(data,label,spec[3],ink);
   if(type==='matrix')body=matrix(data==='S'?sinkhorn():data,ink)+text(600,969,label,25,ink,'text-anchor="middle" letter-spacing="1"');
   if(type==='paired')body=matrix(sinkhorn(),ink,'S / STUDENT',120,447,137)+matrix(P,ink,'P / TEACHER',679,447,137)+text(600,968,'CE = −log S(A,2) − log S(B,1) − log S(C,3)',28,ink,'text-anchor="middle"');
   if(type==='posters'||type==='captions')body=await posters(ink,type==='captions');
   const title=ch.title.replaceAll('\n',' ');const lines=ch.plate?.titleLines||wrap(title,32);
   let svg=`<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1140" viewBox="0 0 1200 1140" role="img" aria-labelledby="title desc"><title id="title">${escape(study.name+' — '+title)}</title><desc id="desc">${escape(ch.note||'Study mechanism plate; numerical examples are illustrative unless explicitly marked as measured results.')}</desc><rect width="1200" height="1140" fill="${paper}"/>`;
   svg+=text(70,71,'FIELDNOTES / '+study.name.toUpperCase(),17,ink,'letter-spacing="2.5"')+text(1130,71,String(i+1).padStart(2,'0'),21,ink,'text-anchor="end"')+line(70,99,1130,99,ink);
   lines.forEach((s,j)=>svg+=text(67,184+j*70,s,61,ink,'font-weight="500" letter-spacing="-2"'));
   svg+=text(72,lines.length>1?308:249,ch.subtitle||ch.sub||'',23,'#716874')+body+line(70,1020,1130,1020,ink)+text(72,1071,ch.tag,20,ink,'letter-spacing="1.3"')+text(1130,1071,'↗',26,ink,'text-anchor="end"')+'</svg>';
   await writeFile(path.join(dir,`${id}-${i}.svg`),svg);
  }
 }
 return studies;
}
