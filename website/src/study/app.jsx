import React,{useRef,useState,useEffect} from 'react';
import {createRoot} from 'react-dom/client';
import FlexCarousel from '../components/FlexCarousel';
import Lab from './labs';
import './study.css';
const study=JSON.parse(document.getElementById('study-data').textContent);
const github=`https://github.com/1nilx2/Deep-Learning/blob/main/${study.notesPath}`;
const items=study.chapters.map((chapter,i)=>({src:`../../assets/studies/${study.id}-${i}.svg`,alt:chapter.title.replaceAll('\n',' '),title:chapter.title.replaceAll('\n',' '),subtitle:chapter.tag}));
function HeroLine({text}){const i=text.lastIndexOf(study.emphasis);return i>=0?<>{text.slice(0,i)}<em>{study.emphasis}</em>{text.slice(i+study.emphasis.length)}</>:text}
function App(){
 const navigation=useRef(null);const [active,setActive]=useState(0);
 const [supported]=useState(()=>{try{const gl=document.createElement('canvas').getContext('webgl2');if(!gl)return false;gl.getExtension('WEBGL_lose_context')?.loseContext();return true}catch{return false}});
 const goTo=i=>{const next=(i+items.length)%items.length;setActive(next);if(supported&&navigation.current)navigation.current.goTo(next)};
 const chapter=study.chapters[active];
 useEffect(()=>{document.documentElement.style.setProperty('--accent',study.accent);document.title=`${study.name} — Fieldnotes`;},[]);
 return <>
  <a className="skip" href="#chapter">Skip to the chapter</a>
  <header className="shell"><a className="brand" href="../../">PAPER / NOTES <span>{study.name}</span></a><nav aria-label="Study navigation"><a href="../../">← Fieldnotes</a><a href={study.paper} target="_blank" rel="noopener">Read the paper ↗</a><a href={study.notesFile} download>Full notes ↓</a></nav></header>
  <main><section className="hero shell"><div><p className="eyebrow">{study.eyebrow}</p><h1>{study.hero.map((line,i)=><React.Fragment key={line}>{i>0&&<br/>}<HeroLine text={line}/></React.Fragment>)}</h1></div><p className="intro">{study.intro}</p></section>
  <section className="showcase" aria-label="Eight chapters of the study"><div className="gallery-real">{supported?<FlexCarousel navigationRef={navigation} items={items} preset="liquid" intro="rise" cardHeight={.6} gap={18} radius={2} squeeze={.12} focusOnClick captions captureWheel={false} liquid={.15} onChange={setActive}/>:<div className="fallback-gallery">{items.map((item,i)=><button key={item.src} onClick={()=>setActive(i)} aria-label={item.title}><img src={item.src} alt={item.title}/></button>)}</div>}</div><p className="help">Drag · Two-finger scroll · ← → · Click to expand</p><nav className="controls" aria-label="Chapter navigation"><button className="arrow" onClick={()=>goTo(active-1)} aria-label="Previous chapter">←</button><span className="counter"><b>{String(active+1).padStart(2,'0')}</b> / 08</span><div className="dots">{items.map((item,i)=><button key={item.src} className="dot" aria-label={`Go to chapter ${i+1}: ${item.title}`} aria-current={active===i?'step':undefined} onClick={()=>goTo(i)}/>)}</div><button className="arrow" onClick={()=>goTo(active+1)} aria-label="Next chapter">→</button></nav></section>
  <section className="chapter shell" id="chapter"><article className="copy" aria-live="polite" aria-atomic="true"><p className="eyebrow">{chapter.tag}</p><h2>{chapter.title}</h2><p className="subtitle">{chapter.subtitle}</p>{chapter.paragraphs.map(p=><p key={p}>{p}</p>)}<aside className="note">{chapter.note}</aside><a className="source-link" href={github}>Read the complete GitHub notes ↗</a></article><Lab key={active} type={chapter.lab}/></section>
  <section className="reading-map shell"><div className="map-heading"><div><p className="eyebrow">THE WHOLE ARGUMENT</p><h2>Keep the thread.</h2></div><p>Return to any chapter.</p></div><div className="chapter-index">{study.chapters.map((c,i)=><button className={active===i?'current':''} key={c.tag} onClick={()=>{goTo(i);document.getElementById('chapter').scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'})}}><span>{String(i+1).padStart(2,'0')}</span><strong>{c.title.replaceAll('\n',' ')}</strong><b aria-hidden="true">↗</b></button>)}</div><details><summary>Open the full study diagram <span>+</span></summary><a href={`../../assets/${study.figure}`} target="_blank" rel="noopener"><img loading="lazy" src={`../../assets/${study.figure}`} alt={study.figureAlt}/></a><p>The original repository diagram. Open it at full size or find its editable source in the GitHub notes.</p></details></section>
  <a className="next-study shell" href={`../${study.next}/`}><div><p className="eyebrow">CONTINUE THE STUDY</p><h2>{study.nextTitle}</h2></div><span>↗</span></a>
  </main><footer className="shell"><span>{study.name}<br/>{study.version}</span><span>Interactive examples are illustrations, not model runs.<br/>Original React Bits FlexCarousel · React + OGL / WebGL</span><a href={github}>Notes & sources ↗</a></footer>
 </>;
}
class ErrorBoundary extends React.Component{state={error:false};static getDerivedStateFromError(){return {error:true}}render(){return this.state.error?<main className="error-state"><h1>Continue with the notes.</h1><p>This interactive study could not start in this browser.</p><a href={github}>Open the full study notes ↗</a><a href="../../">← Fieldnotes</a></main>:this.props.children}}
createRoot(document.getElementById('study-root')).render(<ErrorBoundary><App/></ErrorBoundary>);
