import React, {useEffect, useRef} from 'react';
import {createRoot} from 'react-dom/client';
import FlexCarousel from './components/FlexCarousel';
import content from '../content.json';
import './archive.js';

const {works} = content;
const $ = id => document.getElementById(id);
const source = path => `https://github.com/1nilx2/Deep-Learning/blob/main/${path}`;
const items = works.map(work => ({src:`assets/${work.card}-card.svg`, alt:work.title, title:work.title, subtitle:work.tag}));

function selectWork(index) {
  const work = works[index];
  document.documentElement.style.setProperty('--accent', work.color);
  $('current').textContent = String(index + 1).padStart(2, '0');
  $('detail-tag').textContent = work.tag;
  $('detail-title').textContent = work.title;
  $('detail-description').textContent = work.description;
  $('detail-caveat').textContent = work.caveat;
  $('detail-link').href = work.page || source(work.path);
  $('detail-link').firstChild.textContent = work.page ? 'Open the study ' : 'Read the notes ';
  $('source-link').href = source(work.path);
  $('figure-link').href = work.page || source(work.path);
  $('detail-image').src = `assets/${work.image}`;
  $('detail-image').alt = work.alt;
  $('announcement').textContent = `${index + 1} / ${works.length}: ${work.title}`;
  document.querySelectorAll('.dot').forEach((dot, i) => dot.setAttribute('aria-current', String(i === index)));
}

function Fallback({goTo}) {
  return <><div className="fallback-gallery">{items.map((item, i) => <button key={item.src} onClick={() => goTo(i)} aria-label={item.title}><img src={item.src} alt={item.title}/></button>)}</div><p className="fallback-message">Select a card to read the study.</p></>;
}

function App() {
  const navigation = useRef(null);
  const active = useRef(0);
  // Detect unsupported graphics before mounting the original WebGL component.
  const [supported] = React.useState(() => {
    try {
      const gl = document.createElement('canvas').getContext('webgl2');
      if (!gl) return false;
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      return true;
    } catch { return false; }
  });
  const change = index => {active.current = (index + works.length) % works.length; selectWork(active.current);};
  useEffect(() => {
    const goTo = index => supported ? navigation.current?.goTo(index) : change(index);
    const previous = () => goTo(active.current - 1);
    const next = () => goTo(active.current + 1);
    $('prev').addEventListener('click', previous);
    $('next').addEventListener('click', next);
    $('work-total').textContent = ` / ${String(works.length).padStart(2,'0')}`;
    const dots = works.map((work, i) => {
      const dot = document.createElement('button');
      dot.className = 'dot'; dot.setAttribute('aria-label', `${i+1}: ${work.title}`);
      dot.onclick = () => goTo(i);
      return dot;
    });
    $('dots').replaceChildren(...dots);
    change(0);
    return () => { $('prev').removeEventListener('click', previous); $('next').removeEventListener('click', next); };
  }, [supported]);
  if (!supported) return <Fallback goTo={change}/>;
  return <FlexCarousel navigationRef={navigation} items={items} preset="liquid" intro="rise" cardHeight={0.6} gap={18} radius={2} squeeze={0.12} focusOnClick captions captureWheel={false} liquid={0.15} onChange={change}/>;
}

class ErrorBoundary extends React.Component {
  state = {error:false};
  static getDerivedStateFromError() { return {error:true}; }
  render() {return this.state.error ? <Fallback goTo={selectWork}/> : this.props.children;}
}
createRoot($('carousel-root')).render(<ErrorBoundary><App/></ErrorBoundary>);
