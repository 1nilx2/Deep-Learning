import React, {useState, useEffect} from 'react';
import {createRoot} from 'react-dom/client';
import {TiltedGridHero} from './components/ui/tilted-grid-hero';
import content from '../content.json';
import './archive.js';
import './home.css';

const source = (path: string) => `https://github.com/1nilx2/Deep-Learning/blob/main/${path}`;
const images = content.works.map(work => ({src:`assets/${work.heroImage}`,alt:work.heroAlt}));
const $ = (id: string) => document.getElementById(id)!;

function App() {
  const [selected, setSelected] = useState(0);
  const [paused, setPaused] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  useEffect(() => {
    const work=content.works[selected];
    document.documentElement.style.setProperty('--accent',work.color);
    $('detail-tag').textContent=work.tag;
    $('detail-title').textContent=work.title;
    $('detail-description').textContent=work.description;
    $('detail-caveat').textContent=work.caveat;
    ($('detail-link') as HTMLAnchorElement).href=work.page||source(work.path);
    $('detail-link').firstChild!.textContent=work.page?'Open the study ':'Read the notes ';
    ($('source-link') as HTMLAnchorElement).href=source(work.path);
    ($('figure-link') as HTMLAnchorElement).href=work.page||source(work.path);
    ($('detail-image') as HTMLImageElement).src=`assets/${work.image}`;
    ($('detail-image') as HTMLImageElement).alt=work.alt;
  },[selected]);
  return <>
    <section id="selected" aria-label="Fieldnotes introduction">
      <TiltedGridHero images={images} speed={5.5} tileHeight={34} aspectRatio={1.05} gap={8} axis={60} curve={75} fade={14} className="tilted-hero" data-paused={paused}>
        <div className="tilted-content">
          <div className="tilted-heading"><p className="eyebrow">A PERSONAL RESEARCH NOTEBOOK</p><h1>Every paper,<br/>a new <em>perspective.</em></h1><p className="tilted-subtitle">Read the mechanisms. Build the intuition.</p></div>
          <div className="tilted-bottom"><p>Papers, notes, and experiments.<br/>A closer look at the ideas behind the models.</p><div className="hero-actions"><a className="primary-action" href="#featured">Explore the studies <span>↘</span></a><button className="motion-toggle" onClick={()=>setPaused(!paused)} aria-pressed={paused}>{paused?'Play motion':'Pause motion'}</button></div></div>
        </div>
      </TiltedGridHero>
    </section>
    <section className="featured-selection shell" id="featured" aria-label="Select a featured study"><div className="selection-heading"><span className="eyebrow">SELECTED WORK / 06 STUDIES</span><span>Follow an idea.</span></div><div className="study-tabs" role="group" aria-label="Featured studies">{content.works.map((work,i)=><button key={work.card} aria-pressed={selected===i} onClick={()=>setSelected(i)}><span>{String(i+1).padStart(2,'0')}</span>{['SPD','GenRec','EmbeddingGemma','Laya','TurboQuant','Artwork'][i]}<b aria-hidden="true">↗</b></button>)}</div></section>
  </>;
}
createRoot($('home-root')).render(<App/>);
