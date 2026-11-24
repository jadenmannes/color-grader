// Full-quality still: one frame at the source's full resolution with the full grade, lossless, full screen.
// Graded, source and a split compare. Fit or 100% (one image pixel per screen pixel) to judge noise and sharpness.
import { html, useState, useEffect, useRef, useLayoutEffect } from '/static/vendor/htm-preact-standalone.js';
import { qs, fmtTime } from './api.js';

export function StillView({ clip, shot, t, before, after, names, version, onClose }) {
  const root = useRef(null);
  const stage = useRef(null);
  const [mode, setMode] = useState('after');          // after | split | before
  const [zoom, setZoom] = useState('fit');            // fit | full
  const [imgs, setImgs] = useState({});               // object URLs of the rendered PNGs
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState(null);
  const [size, setSize] = useState(null);
  const [split, setSplit] = useState(0.5);
  const [fs, setFs] = useState(false);
  const [nat, setNat] = useState(null);               // natural size of the frame in pixels
  const [room, setRoom] = useState(null);             // the stage's size, for Fit
  useEffect(() => {
    const el = stage.current;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(() => setRoom([el.clientWidth, el.clientHeight]));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const anchor = useRef(null);                        // keeps the clicked point under the pointer when zooming in
  const drag = useRef(null);
  const base = { clip: clip.id, t: t.toFixed(3), shot: shot ? shot.id : undefined };
  const stageOf = { after, before };

  // render after, then before, as blobs: switching between them is then instant
  useEffect(() => {
    let alive = true;
    setBusy(true);
    setError(null);
    const load = async (which) => {
      const res = await fetch('/api/still?' + qs({ ...base, stage: stageOf[which], v: version }));
      if (!res.ok) {
        let msg = res.statusText;
        try { msg = (await res.json()).detail || msg; } catch (e) { /* not JSON */ }
        throw new Error(msg);
      }
      const sz = res.headers.get('X-Still-Size');
      const url = URL.createObjectURL(await res.blob());
      if (!alive) { URL.revokeObjectURL(url); return; }
      if (sz) setSize(sz.replace('x', ' × '));
      setImgs((m) => { if (m[which]) URL.revokeObjectURL(m[which]); return { ...m, [which]: url }; });
    };
    (async () => {
      try {
        await load('after');
        if (alive) setBusy(false);
        await load('before');
      } catch (e) { if (alive) { setError(e.message); setBusy(false); } }
    })();
    return () => { alive = false; };
  }, [clip.id, shot && shot.id, t, before, after, version]);
  useEffect(() => () => setImgs((m) => { Object.values(m).forEach((u) => URL.revokeObjectURL(u)); return {}; }), []);

  // full screen only on request (the button or F); Esc leaves full screen first (the browser does that), then closes
  useEffect(() => {
    const el = root.current;
    const onFs = () => setFs(document.fullscreenElement === el);
    document.addEventListener('fullscreenchange', onFs);
    return () => {
      document.removeEventListener('fullscreenchange', onFs);
      if (document.fullscreenElement === el && document.exitFullscreen) document.exitFullscreen().catch(() => {});
    };
  }, []);
  const toggleFs = () => {
    const el = root.current;
    if (document.fullscreenElement === el) document.exitFullscreen().catch(() => {});
    else if (el.requestFullscreen) el.requestFullscreen().catch(() => {});
  };

  const zoomAt = (e) => {
    const img = stage.current && stage.current.querySelector('img');
    if (zoom === 'fit' && img && e) {
      const r = img.getBoundingClientRect();
      const s = stage.current.getBoundingClientRect();
      anchor.current = { fx: (e.clientX - r.left) / r.width, fy: (e.clientY - r.top) / r.height, px: e.clientX - s.left, py: e.clientY - s.top };
    } else anchor.current = null;
    setZoom(zoom === 'fit' ? 'full' : 'fit');
  };
  useLayoutEffect(() => {
    const el = stage.current;
    const img = el && el.querySelector('img');
    if (!el || !img) return;
    if (zoom === 'full') {
      const a = anchor.current || { fx: 0.5, fy: 0.5, px: el.clientWidth / 2, py: el.clientHeight / 2 };
      el.scrollLeft = a.fx * img.offsetWidth - a.px;
      el.scrollTop = a.fy * img.offsetHeight - a.py;
    }
  }, [zoom]);

  useEffect(() => {
    const key = (e) => {
      if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
      const k = e.key.toLowerCase();
      if (k === 'escape') { e.preventDefault(); onClose(); }
      else if (k === 'b' || k === ' ') { e.preventDefault(); setMode((m) => (m === 'before' ? 'after' : 'before')); }
      else if (k === 's') setMode((m) => (m === 'split' ? 'after' : 'split'));
      else if (k === 'z' || k === '1') zoomAt(null);
      else if (k === 'f') toggleFs();
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  });

  // pointer: drag pans at 100%, a click toggles the zoom; in split mode a drag moves the split
  const down = (e) => {
    if (e.button !== 0) return;
    drag.current = { x: e.clientX, y: e.clientY, sl: stage.current.scrollLeft, st: stage.current.scrollTop, moved: false };
    if (mode === 'split') moveSplit(e);
    stage.current.setPointerCapture(e.pointerId);
  };
  const moveSplit = (e) => {
    const img = stage.current.querySelector('img');
    if (!img) return;
    const r = img.getBoundingClientRect();
    setSplit(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)));
  };
  const move = (e) => {
    const d = drag.current;
    if (!d) return;
    if (Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) > 4) d.moved = true;
    if (mode === 'split') { moveSplit(e); return; }
    if (zoom === 'full') { stage.current.scrollLeft = d.sl - (e.clientX - d.x); stage.current.scrollTop = d.st - (e.clientY - d.y); }
  };
  const up = (e) => {
    const d = drag.current;
    drag.current = null;
    if (d && !d.moved && mode !== 'split') zoomAt(e);
  };

  const shown = mode === 'before' ? imgs.before : imgs.after;
  const savedStage = mode === 'before' ? before : after;
  const where = `${shot ? `Shot ${shot.index + 1}` : clip.name}  ·  ${fmtTime(t)}${size ? `  ·  ${size}` : ''}`;
  return html`
    <div class="still" ref=${root}>
      <div class="still-bar">
        <div class="still-title"><b>Full quality</b><span class="dim">${where}</span></div>
        <div class="seg">
          <button class=${mode === 'after' ? 'on' : ''} onClick=${() => setMode('after')}>${names[after]}</button>
          <button class=${mode === 'split' ? 'on' : ''} disabled=${!imgs.before} onClick=${() => setMode('split')}>Split</button>
          <button class=${mode === 'before' ? 'on' : ''} disabled=${!imgs.before} onClick=${() => setMode('before')}>${names[before]}</button>
        </div>
        <div class="seg">
          <button class=${zoom === 'fit' ? 'on' : ''} onClick=${() => { anchor.current = null; setZoom('fit'); }}>Fit</button>
          <button class=${zoom === 'full' ? 'on' : ''} onClick=${() => { anchor.current = null; setZoom('full'); }}>100%</button>
        </div>
        <a class="btn small" href=${'/api/still?' + qs({ ...base, stage: savedStage, download: 1, v: version })} download
          title="Lossless 16-bit PNG at full resolution">Save PNG</a>
        <button class="small" onClick=${toggleFs}>${fs ? 'Exit full screen' : 'Full screen'}</button>
        <button class="small" title="Close (Esc)" onClick=${onClose}>✕</button>
      </div>
      <div class=${`still-stage ${zoom}${mode === 'split' ? ' split' : ''}`} ref=${stage}
        onPointerDown=${down} onPointerMove=${move} onPointerUp=${up} onPointerCancel=${() => { drag.current = null; }}>
        ${shown && html`<div class="still-pic">
          <img src=${shown} draggable=${false} alt="Full-quality frame" onLoad=${(e) => setNat([e.target.naturalWidth, e.target.naturalHeight])}
            style=${!nat ? null : zoom === 'full' ? { width: `${nat[0] / (window.devicePixelRatio || 1)}px` }
              : room ? { width: `${Math.floor(Math.min(room[0] - 16, (room[1] - 16) * nat[0] / nat[1]))}px` } : null} />
          ${mode === 'split' && imgs.before && html`
            <img class="still-before" src=${imgs.before} draggable=${false} style=${{ clipPath: `inset(0 ${(1 - split) * 100}% 0 0)` }} />
            <div class="still-handle" style=${{ left: `${split * 100}%` }}></div>
            <div class="still-tag l">${names[before]}</div><div class="still-tag r">${names[after]}</div>`}
        </div>`}
        ${(busy || error) && html`<div class="still-wait">${error ? `Could not render the frame: ${error}` : 'Rendering the frame at full resolution with the full grade…'}</div>`}
      </div>
      <div class="still-help">B or Space: before and after   ·   S: split   ·   Z or a click: Fit and 100%   ·   F: full screen   ·   Esc: close</div>
    </div>`;
}
