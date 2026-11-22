// The Color tab: every color control of the app, live. Edits the shared look (all shots) or one shot's own trim.
import { html, useState, useEffect, useRef } from '/static/vendor/htm-preact-standalone.js';
import { api, qs } from './api.js';

const clone = (o) => JSON.parse(JSON.stringify(o));
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const roundTo = (v, d) => { const f = 10 ** d; return Math.round(v * f) / f; };
const digitsOf = (step) => (step >= 1 ? 0 : Math.min(4, Math.max(0, Math.ceil(-Math.log10(step) - 1e-9))));
const ZONES = ['dark', 'shadow', 'light', 'highlight', 'specular'];
const ZONE_NEUTRAL = { exposure: 0, saturation: 1, temp: 0, tint: 0 };
const HUE_SIX = [0, 60, 120, 180, 240, 300];
const SQ3 = Math.sqrt(3);

export function near(a, b) {
  if (Array.isArray(a) || Array.isArray(b)) {
    a = a || []; b = b || [];
    return a.length === b.length && a.every((x, i) => near(x, b[i]));
  }
  if ((a && typeof a === 'object') || (b && typeof b === 'object')) {
    a = a || {}; b = b || {};
    return [...new Set([...Object.keys(a), ...Object.keys(b)])].every((k) => near(a[k], b[k]));
  }
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) < 1e-9;
  return a === b;
}

let SPEC = null;
function useSpec(notify) {
  const [spec, setSpec] = useState(SPEC);
  useEffect(() => {
    if (!SPEC) api.get('/api/controls').then((s) => { SPEC = s; setSpec(s); }).catch((e) => notify(e.message));
  }, []);
  return spec;
}

// ------------------------------------------------------------------ curves
// The app draws its curves with PCHIP (scipy's monotone cubic). The same math here, so the editor shows the real curve.
function pchipSlopes(xs, ys) {
  const n = xs.length;
  const h = []; const m = [];
  for (let k = 0; k < n - 1; k++) { h.push(xs[k + 1] - xs[k]); m.push((ys[k + 1] - ys[k]) / h[k]); }
  const d = new Array(n).fill(0);
  if (n === 2) { d[0] = m[0]; d[1] = m[0]; return d; }
  for (let k = 1; k < n - 1; k++) {
    if (m[k - 1] * m[k] <= 0) d[k] = 0;
    else { const w1 = 2 * h[k] + h[k - 1]; const w2 = h[k] + 2 * h[k - 1]; d[k] = (w1 + w2) / (w1 / m[k - 1] + w2 / m[k]); }
  }
  const edge = (h0, h1, m0, m1) => {
    let v = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if (Math.sign(v) !== Math.sign(m0)) v = 0;
    else if (Math.sign(m0) !== Math.sign(m1) && Math.abs(v) > Math.abs(3 * m0)) v = 3 * m0;
    return v;
  };
  d[0] = edge(h[0], h[1], m[0], m[1]);
  d[n - 1] = edge(h[n - 2], h[n - 3], m[n - 2], m[n - 3]);
  return d;
}

function pchip(xs, ys) {
  const n = xs.length;
  if (n === 1) return () => ys[0];
  const d = pchipSlopes(xs, ys);
  return (x) => {
    let lo = 0; let hi = n - 2;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid - 1; }
    const k = lo; const h = xs[k + 1] - xs[k]; const t = (x - xs[k]) / h; const t2 = t * t; const t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * ys[k] + (t3 - 2 * t2 + t) * h * d[k] + (-2 * t3 + 3 * t2) * ys[k + 1] + (t3 - t2) * h * d[k + 1];
  };
}

function sortedUnique(points) {
  const out = [];
  [...points].sort((a, b) => a[0] - b[0]).forEach((p) => { if (!out.length || p[0] - out[out.length - 1][0] > 1e-6) out.push(p); });
  return out;
}

function curveFn(points, kind) {
  const pts = sortedUnique(kind === 'hue' ? points.map(([h, v]) => [((h % 360) + 360) % 360, v]) : points);
  if (!pts.length) return null;
  if (kind === 'hue') {
    if (pts.length === 1) return () => pts[0][1];
    const xs = [...pts.map((p) => p[0] - 360), ...pts.map((p) => p[0]), ...pts.map((p) => p[0] + 360)];
    const f = pchip(xs, [...pts, ...pts, ...pts].map((p) => p[1]));
    return (x) => f(((x % 360) + 360) % 360);
  }
  const xs = pts.map((p) => p[0]); const f = pchip(xs, pts.map((p) => p[1]));
  const lo = xs[0]; const hi = xs[xs.length - 1];
  return (x) => { const xc = clamp(x, lo, hi); return f(xc) + (kind === 'tone' ? x - xc : 0); };
}

const CURVES = [
  { key: 'tone_curve', label: 'Custom (all channels)', kind: 'tone', xMax: 1, yMin: 0, yMax: 1, color: '#dcdcdc' },
  { key: 'curves_rgb.r', label: 'Red curve', kind: 'tone', xMax: 1, yMin: 0, yMax: 1, color: '#e0675a' },
  { key: 'curves_rgb.g', label: 'Green curve', kind: 'tone', xMax: 1, yMin: 0, yMax: 1, color: '#6fbf73' },
  { key: 'curves_rgb.b', label: 'Blue curve', kind: 'tone', xMax: 1, yMin: 0, yMax: 1, color: '#6b9be6' },
  { key: 'hue_vs_hue', label: 'Hue vs Hue', kind: 'hue', xMax: 360, yMin: -60, yMax: 60, neutral: 0, yd: 1, unit: '°' },
  { key: 'hue_vs_sat', label: 'Hue vs Sat', kind: 'hue', xMax: 360, yMin: 0, yMax: 2, neutral: 1, yd: 3 },
  { key: 'hue_vs_lum', label: 'Hue vs Lum', kind: 'hue', xMax: 360, yMin: 0, yMax: 2, neutral: 1, yd: 3 },
  { key: 'lum_vs_sat', label: 'Lum vs Sat', kind: 'unit', xMax: 1, yMin: 0, yMax: 2, neutral: 1, yd: 3 },
  { key: 'sat_vs_sat', label: 'Sat vs Sat', kind: 'unit', xMax: 0.6, yMin: 0, yMax: 2, neutral: 1, yd: 3 },
  { key: 'sat_vs_lum', label: 'Sat vs Lum', kind: 'unit', xMax: 0.6, yMin: 0, yMax: 2, neutral: 1, yd: 3 },
];

function seedPoints(def) {
  if (def.kind === 'tone') return [[0, 0], [1, 1]];
  if (def.kind === 'hue') return HUE_SIX.map((h) => [h, def.neutral]);
  return [[0, def.neutral], [def.xMax, def.neutral]];
}

function curveGet(values, key) {
  if (key.startsWith('curves_rgb.')) return ((values.curves_rgb || {})[key.slice(11)]) || [];
  return values[key] || [];
}

function curveIsNeutral(def, pts) {
  if (!pts || !pts.length) return true;
  return def.kind === 'tone' ? pts.every(([x, y]) => Math.abs(x - y) < 1e-9) : pts.every(([, y]) => Math.abs(y - def.neutral) < 1e-9);
}

function CurveEditor({ def, points, onSet }) {
  const W = 300; const H = def.kind === 'tone' ? 220 : 150;
  const svg = useRef(null);
  const drag = useRef(null);
  const [local, setLocal] = useState(null);
  const [readout, setReadout] = useState('');
  const pts = local || (points && points.length ? points : null);
  const shown = pts || seedPoints(def);
  const f = curveFn(shown, def.kind);
  const X = (x) => (x / def.xMax) * W;
  const Y = (y) => H - ((y - def.yMin) / (def.yMax - def.yMin)) * H;
  const xd = def.kind === 'hue' ? 1 : 3; const yd = def.yd || 3;
  const at = (e) => {
    const r = svg.current.getBoundingClientRect();
    return [clamp(((e.clientX - r.left) / r.width) * def.xMax, 0, def.xMax),
      clamp(def.yMin + (1 - (e.clientY - r.top) / r.height) * (def.yMax - def.yMin), def.yMin, def.yMax)];
  };
  const say = (p) => setReadout(def.kind === 'hue'
    ? `Hue ${p[0].toFixed(0)}°: ${def.key === 'hue_vs_hue' ? `turn ${p[1] > 0 ? '+' : ''}${p[1].toFixed(1)}°` : `× ${p[1].toFixed(3)}`}`
    : `In ${p[0].toFixed(3)} → ${def.kind === 'tone' ? 'out' : '×'} ${p[1].toFixed(3)}`);
  const start = (e, list, i) => {
    e.preventDefault();
    svg.current.setPointerCapture(e.pointerId);
    drag.current = { i, list };
    setLocal(list);
    say(list[i]);
  };
  const down = (e) => {
    const [x] = at(e);
    const list = clone(shown);
    const y = roundTo(clamp(f(x), def.yMin, def.yMax), yd);
    list.push([roundTo(x, xd), y]);
    list.sort((a, b) => a[0] - b[0]);
    start(e, list, list.findIndex((p) => p[0] === roundTo(x, xd) && p[1] === y));
  };
  const move = (e) => {
    if (!drag.current) return;
    const { i } = drag.current; const list = clone(drag.current.list);
    let [x, y] = at(e);
    const fixedEnd = def.kind !== 'hue' && (i === 0 || i === list.length - 1) && (list[i][0] <= 1e-9 || list[i][0] >= def.xMax - 1e-9);
    if (fixedEnd) x = list[i][0];
    else {
      const lo = i > 0 ? list[i - 1][0] + (def.kind === 'hue' ? 1 : 0.005) : 0;
      const hi = i < list.length - 1 ? list[i + 1][0] - (def.kind === 'hue' ? 1 : 0.005) : def.xMax - (def.kind === 'hue' ? 0.1 : 0);
      x = clamp(x, lo, hi);
    }
    list[i] = [roundTo(x, xd), roundTo(y, yd)];
    drag.current.list = list;
    setLocal(list);
    say(list[i]);
    onSet(list, false);
  };
  const up = () => {
    if (!drag.current) return;
    const list = drag.current.list;
    drag.current = null;
    setLocal(null);
    onSet(list, true);
  };
  const remove = (e, i) => {
    e.stopPropagation();
    const list = clone(shown);
    if (def.kind !== 'hue' && (list[i][0] <= 1e-9 || list[i][0] >= def.xMax - 1e-9)) return;
    list.splice(i, 1);
    onSet(list, true);
  };
  const path = [];
  for (let k = 0; k <= 150; k++) {
    const x = (k / 150) * def.xMax;
    path.push(`${X(x).toFixed(1)},${clamp(Y(f(x)), -4, H + 4).toFixed(1)}`);
  }
  const grid = [0.25, 0.5, 0.75];
  return html`<div>
    <svg ref=${svg} class="curve-svg" viewBox=${`0 0 ${W} ${H}`} onPointerDown=${down} onPointerMove=${move} onPointerUp=${up} onPointerCancel=${up}>
      ${def.kind === 'hue' && html`<defs><linearGradient id="hueband" x1="0" x2="1" y1="0" y2="0">
        ${[0, 60, 120, 180, 240, 300, 360].map((h) => html`<stop offset=${h / 360} stop-color=${`hsl(${h} 70% 50%)`} />`)}</linearGradient></defs>
        <rect x="0" y=${H - 7} width=${W} height="7" fill="url(#hueband)" opacity="0.8" />`}
      ${grid.map((g) => html`<line x1=${g * W} x2=${g * W} y1="0" y2=${H} stroke="#262626" /><line y1=${g * H} y2=${g * H} x1="0" x2=${W} stroke="#262626" />`)}
      ${def.kind === 'tone'
        ? html`<line x1="0" y1=${H} x2=${W} y2="0" stroke="#3a3a3a" stroke-dasharray="3 3" />
               <line x1=${X(0.336)} x2=${X(0.336)} y1="0" y2=${H} stroke="#3a3a3a" stroke-dasharray="2 4" />`
        : html`<line x1="0" x2=${W} y1=${Y(def.neutral)} y2=${Y(def.neutral)} stroke="#3a3a3a" stroke-dasharray="3 3" />`}
      <polyline points=${path.join(' ')} fill="none" stroke=${def.color || '#e8a33d'} stroke-width="1.6" />
      ${pts && pts.map((p, i) => html`<circle cx=${X(p[0])} cy=${Y(p[1])} r="5" fill="#121212" stroke=${def.color || '#e8a33d'} stroke-width="1.6"
        style="cursor:grab" onPointerDown=${(e) => { e.stopPropagation(); start(e, clone(pts), i); }} onDblClick=${(e) => remove(e, i)} />`)}
    </svg>
    <div class="row" style="justify-content:space-between;margin-top:3px">
      <span class="dim" style="font-size:11.5px">${readout || (def.kind === 'tone' ? 'Click to add a point, drag to move, double-click to remove. Dashed: middle gray.' : 'Click to add a point, drag it, double-click to remove.')}</span>
      <button class="small ghost" onClick=${() => onSet([], true)}>Reset</button>
    </div>
  </div>`;
}

function Curves({ values, set }) {
  const [sel, setSel] = useState('tone_curve');
  const def = CURVES.find((c) => c.key === sel);
  const onSet = (list, commit) => {
    if (sel.startsWith('curves_rgb.')) set('curves_rgb', { ...(values.curves_rgb || { r: [], g: [], b: [] }), [sel.slice(11)]: list }, commit);
    else set(sel, list, commit);
  };
  return html`<div>
    <select value=${sel} onChange=${(e) => setSel(e.target.value)} style="margin-bottom:6px">
      ${CURVES.map((c) => html`<option value=${c.key}>${curveIsNeutral(c, curveGet(values, c.key)) ? '' : '● '}${c.label}</option>`)}
    </select>
    <${CurveEditor} key=${sel} def=${def} points=${curveGet(values, sel)} onSet=${onSet} />
  </div>`;
}

// ------------------------------------------------------------------ inputs
function Slider({ label, value, min, max, step, neutral, onSet, help, unit }) {
  const d = digitsOf(step);
  const v = typeof value === 'number' ? value : neutral;
  const changed = Math.abs(v - neutral) > 1e-9;
  return html`<div class="ctl" title=${help || ''}>
    <span class=${'ctl-label' + (changed ? ' on' : '')} title="Double-click to reset" onDblClick=${() => onSet(neutral, true)}>${label}</span>
    <input type="range" min=${min} max=${max} step=${step} value=${v}
      onInput=${(e) => onSet(parseFloat(e.target.value), false)} onChange=${(e) => onSet(parseFloat(e.target.value), true)} />
    <input type="number" class="ctl-num" step=${step} value=${roundTo(v, d)}
      onChange=${(e) => { const x = parseFloat(e.target.value); if (!Number.isNaN(x)) onSet(clamp(x, min, max), true); }} />
    <span class="ctl-unit">${unit || ''}</span>
  </div>`;
}

function Wheel({ title, value, neutral, radius, level, step, onSet }) {
  // The ball moves color (the channels' difference), the ring below moves level (their mean), like Resolve's wheels.
  const pad = useRef(null);
  const grab = useRef(null);
  const d = value.map((v, i) => v - neutral[i]);
  const L = (d[0] + d[1] + d[2]) / 3;
  const c1 = d[0] - (d[1] + d[2]) / 2; const c2 = (SQ3 / 2) * (d[1] - d[2]);
  let bx = c1 / radius; let by = c2 / radius;
  const len = Math.hypot(bx, by); if (len > 1) { bx /= len; by /= len; }
  const dg = digitsOf(step);
  const build = (n1, n2, lev) => [lev + (2 / 3) * n1, lev - n1 / 3 + n2 / SQ3, lev - n1 / 3 - n2 / SQ3].map((x, i) => roundTo(neutral[i] + x, dg));
  const down = (e) => {
    e.preventDefault();
    pad.current.setPointerCapture(e.pointerId);
    grab.current = { x: e.clientX, y: e.clientY, c1, c2 };
  };
  const move = (e) => {
    if (!grab.current) return;
    const r = pad.current.getBoundingClientRect();
    const k = (radius / (r.width / 2)) * (e.shiftKey ? 0.2 : 1);
    const g = grab.current;
    let n1 = g.c1 + (e.clientX - g.x) * k; let n2 = g.c2 - (e.clientY - g.y) * k;
    const l = Math.hypot(n1, n2) / radius; if (l > 1) { n1 /= l; n2 /= l; }
    onSet(build(n1, n2, L), false);
  };
  const up = () => { if (grab.current) { grab.current = null; onSet(value, true); } };
  const changed = !near(value, neutral);
  const stops = [0, 300, 240, 180, 120, 60, 0].map((h, i) => `hsl(${h} 55% 42%) ${i * 60}deg`).join(', ');
  return html`<div class="wheel">
    <div class=${'wheel-title' + (changed ? ' on' : '')} title="Double-click the wheel to reset its color" >${title}</div>
    <div class="wheel-pad" ref=${pad} style=${{ background: `radial-gradient(circle, #2a2a2a 0 22%, rgba(42,42,42,.55) 46%, rgba(42,42,42,0) 72%), conic-gradient(from 90deg, ${stops})` }}
      onPointerDown=${down} onPointerMove=${move} onPointerUp=${up} onPointerCancel=${up} onDblClick=${() => onSet(build(0, 0, L), true)}>
      <div class="wheel-dot" style=${{ left: `${50 + bx * 46}%`, top: `${50 - by * 46}%` }}></div>
    </div>
    <input type="range" min=${level[0]} max=${level[1]} step=${step} value=${L}
      onInput=${(e) => onSet(build(c1, c2, parseFloat(e.target.value)), false)} onChange=${(e) => onSet(build(c1, c2, parseFloat(e.target.value)), true)} />
    <div class="rgbnums">${[0, 1, 2].map((i) => html`<input type="number" step=${step} value=${roundTo(value[i], dg)} title=${'RGB'[i]}
      onChange=${(e) => { const x = parseFloat(e.target.value); if (!Number.isNaN(x)) { const nv = [...value]; nv[i] = x; onSet(nv, true); } }} />`)}</div>
    ${changed ? html`<button class="small ghost" style="margin-top:2px" onClick=${() => onSet([...neutral], true)}>Reset</button>` : ''}
  </div>`;
}

function Group({ title, changed, open, children, note }) {
  return html`<details class="cgroup" open=${open}>
    <summary>${title}${changed ? html`<span class="changed-dot" title="Changed"></span>` : ''}</summary>
    ${note ? html`<div class="dim" style="font-size:11.5px;margin:-2px 0 6px">${note}</div>` : ''}
    <div class="cgroup-body">${children}</div>
  </details>`;
}

function Zones({ value, onSet, spec }) {
  const [zone, setZone] = useState('light');
  const hdr = value || {};
  const z = { ...ZONE_NEUTRAL, ...(hdr[zone] || {}) };
  const f = spec.fields;
  const setField = (k, v, commit) => onSet({ ...hdr, [zone]: { ...z, [k]: v } }, commit);
  const zoneChanged = (name) => !near({ ...ZONE_NEUTRAL, ...(hdr[name] || {}) }, ZONE_NEUTRAL);
  return html`<div>
    <div class="seg" style="margin-bottom:6px">${ZONES.map((n) => html`<button class=${zone === n ? 'on' : ''} onClick=${() => setZone(n)}>
      ${n[0].toUpperCase() + n.slice(1)}${zoneChanged(n) ? ' ●' : ''}</button>`)}</div>
    <${Slider} label="Exposure" value=${z.exposure} min=${f.exposure[0]} max=${f.exposure[1]} step=${f.exposure[2]} neutral=${0} unit="st" onSet=${(v, c) => setField('exposure', v, c)} />
    <${Slider} label="Saturation" value=${z.saturation} min=${f.saturation[0]} max=${f.saturation[1]} step=${f.saturation[2]} neutral=${1} unit="×" onSet=${(v, c) => setField('saturation', v, c)} />
    <${Slider} label="Temp" value=${z.temp} min=${f.temp[0]} max=${f.temp[1]} step=${f.temp[2]} neutral=${0} unit="st" onSet=${(v, c) => setField('temp', v, c)} />
    <${Slider} label="Tint" value=${z.tint} min=${f.tint[0]} max=${f.tint[1]} step=${f.tint[2]} neutral=${0} unit="st" onSet=${(v, c) => setField('tint', v, c)} />
  </div>`;
}

function Mixer({ value, onSet }) {
  const m = value || [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const setCell = (r, c, v) => { const n = clone(m); n[r][c] = v; onSet(n, true); };
  return html`<div class="mixer">
    <span></span><span class="dim">R in</span><span class="dim">G in</span><span class="dim">B in</span>
    ${['Red out', 'Green out', 'Blue out'].map((name, r) => html`<span class="dim">${name}</span>${[0, 1, 2].map((c) => html`
      <input type="number" step="0.01" value=${roundTo(m[r][c], 3)} onChange=${(e) => { const x = parseFloat(e.target.value); if (!Number.isNaN(x)) setCell(r, c, clamp(x, -1, 2)); }} />`)}`)}
  </div>`;
}

// ------------------------------------------------------------- secondaries
const SEC_NEUTRAL = { exposure: 0, temp: 0, tint: 0, saturation: 1, hue: 0 };
const PRESETS = {
  Sky: { hue: [214, 50, 25], sat: [0.015, 1, 0.01], lum: [0.3, 1, 0.08] },
  Skin: { hue: [21, 30, 15], sat: [0.02, 0.11, 0.015], lum: [0.12, 0.6, 0.06] },
  Foliage: { hue: [85, 60, 25], sat: [0.02, 1, 0.01] },
  'Autumn leaves': { hue: [20, 40, 15], sat: [0.1, 1, 0.03] },
};
const QUAL = {
  hue: [['Center', 0, 359, 1, '°'], ['Width', 5, 180, 1, '°'], ['Soft', 0, 90, 1, '°']],
  sat: [['Low', 0, 0.3, 0.002, ''], ['High', 0, 1, 0.005, ''], ['Soft', 0, 0.1, 0.002, '']],
  lum: [['Low', 0, 1, 0.005, ''], ['High', 0, 1, 0.005, ''], ['Soft', 0, 0.3, 0.005, '']],
};
const QUAL_DEFAULT = { hue: [0, 40, 20], sat: [0.02, 1, 0.01], lum: [0.1, 0.7, 0.06] };
const QUAL_NAME = { hue: 'Hue', sat: 'Saturation', lum: 'Luminance' };

function Secondary({ sec, index, onChange, onRemove, onPick, keyOn, onKey }) {
  const adj = { ...SEC_NEUTRAL, ...(sec.adjust || {}) };
  const put = (patch, commit = true) => onChange({ ...sec, ...patch }, commit);
  const setAdj = (k, v, commit) => onChange({ ...sec, adjust: { ...adj, [k]: v } }, commit);
  return html`<div class="card sec-card">
    <div class="row" style="margin-bottom:6px">
      <input type="checkbox" checked=${sec.enabled !== false} title="On or off" onChange=${(e) => put({ enabled: e.target.checked })} />
      <input type="text" value=${sec.name || ''} onChange=${(e) => put({ name: e.target.value })} />
      <button class="small ghost danger" title="Remove" onClick=${onRemove}>✕</button>
    </div>
    <div class="row wrap" style="margin-bottom:6px">
      <button class="small" title="Click a color in the viewer to key it" onClick=${onPick}>Pick color</button>
      <button class=${'small' + (keyOn ? ' primary' : '')} title="Show what the key selects in the viewer" onClick=${onKey}>Show key</button>
      <label class="row" style="margin:0;gap:4px"><input type="checkbox" checked=${!!sec.invert} onChange=${(e) => put({ invert: e.target.checked })} />Invert</label>
    </div>
    ${['hue', 'sat', 'lum'].map((q) => html`<div class="qual">
      <label class="row" style="margin:4px 0 2px;gap:4px"><input type="checkbox" checked=${!!sec[q]}
        onChange=${(e) => { const n = { ...sec }; if (e.target.checked) n[q] = sec[q] || QUAL_DEFAULT[q]; else delete n[q]; onChange(n, true); }} />${QUAL_NAME[q]}</label>
      ${sec[q] && QUAL[q].map(([name, lo, hi, st, unit], i) => html`<${Slider} label=${name} value=${sec[q][i]} min=${lo} max=${hi} step=${st} neutral=${QUAL_DEFAULT[q][i]} unit=${unit}
        onSet=${(v, c) => { const arr = [...sec[q]]; arr[i] = v; put({ [q]: arr }, c); }} />`)}
    </div>`)}
    <div class="dim" style="font-size:11.5px;margin:8px 0 2px">Inside the key</div>
    <${Slider} label="Exposure" value=${adj.exposure} min=${-2} max=${2} step=${0.02} neutral=${0} unit="st" onSet=${(v, c) => setAdj('exposure', v, c)} />
    <${Slider} label="Temp" value=${adj.temp} min=${-1} max=${1} step=${0.02} neutral=${0} unit="st" onSet=${(v, c) => setAdj('temp', v, c)} />
    <${Slider} label="Tint" value=${adj.tint} min=${-1} max=${1} step=${0.02} neutral=${0} unit="st" onSet=${(v, c) => setAdj('tint', v, c)} />
    <${Slider} label="Saturation" value=${adj.saturation} min=${0} max=${2} step=${0.02} neutral=${1} unit="×" onSet=${(v, c) => setAdj('saturation', v, c)} />
    <${Slider} label="Hue" value=${adj.hue} min=${-60} max=${60} step=${0.5} neutral=${0} unit="°" onSet=${(v, c) => setAdj('hue', v, c)} />
    <div class="dim" style="font-size:11px">Secondary ${index + 1}. Later secondaries see the result of earlier ones.</div>
  </div>`;
}

function Secondaries({ list, set, ctx, clip, shot, t, target, viewer, setViewer }) {
  const secs = list || [];
  const write = (n, commit = true) => set('secondaries', n, commit);
  const keyIndex = viewer && viewer.kind === 'key' && viewer.target === target ? viewer.index : -1;
  const showKey = (i, name) => setViewer({ kind: 'key', target, index: i, label: `Key: ${name}. Selected parts in color, the rest gray.` });
  const add = (name, q) => {
    const n = [...secs, { name, enabled: true, invert: false, ...clone(q), adjust: { ...SEC_NEUTRAL } }];
    write(n);
    showKey(n.length - 1, name);
  };
  const pick = (i) => {
    if (!clip) return;
    setViewer({ kind: 'pick', label: 'Click the color to key', onPick: async (x, y) => {
      try {
        const s = await api.get('/api/sample?' + qs({ clip: clip.id, t: t.toFixed(3), x: x.toFixed(4), y: y.toFixed(4), shot: shot && shot.id, target }));
        const q = s.suggested_qualifier;
        if (i < 0) add(`Picked ${s.color.split(' (')[0]}`, q);
        else {
          const n = clone(secs); const sec = { ...n[i] };
          delete sec.hue; delete sec.sat; delete sec.lum;
          n[i] = { ...sec, ...q };
          write(n);
          showKey(i, n[i].name);
        }
      } catch (e) { ctx.notify(e.message); setViewer(null); }
    } });
  };
  return html`<div>
    ${secs.map((sec, i) => html`<${Secondary} key=${i} sec=${sec} index=${i}
      onChange=${(s, commit) => { const n = clone(secs); n[i] = s; write(n, commit); }}
      onRemove=${() => { write(secs.filter((_, k) => k !== i)); if (keyIndex >= 0) setViewer(null); }}
      onPick=${() => pick(i)} keyOn=${keyIndex === i}
      onKey=${() => (keyIndex === i ? setViewer(null) : showKey(i, sec.name))} />`)}
    <div class="row wrap">
      <button class="small" onClick=${() => pick(-1)}>+ Pick a color</button>
      ${Object.entries(PRESETS).map(([name, q]) => html`<button class="small ghost" onClick=${() => add(name, q)}>+ ${name}</button>`)}
    </div>
    ${secs.length >= 8 ? html`<div class="dim">Up to 8 secondaries.</div>` : ''}
  </div>`;
}

// -------------------------------------------------------------- live sync
function controlsOf(src, neutral) {
  const out = clone(neutral);
  Object.keys(neutral).forEach((k) => { if (src && src[k] !== undefined && src[k] !== null) out[k] = clone(src[k]); });
  return out;
}

function useLiveControls({ target, shot, proj, neutral, onLive, ctx }) {
  const source = target === 'look' ? proj.look : ((shot && shot.grade && shot.grade.trim) || {});
  const [values, setValues] = useState(() => controlsOf(source, neutral));
  const pending = useRef({});
  const busy = useRef(false);
  const lastEdit = useRef(0);
  const timer = useRef(null);
  const idle = useRef(null);
  const url = target === 'look' ? '/api/grade/look' : (shot ? `/api/shots/${shot.id}/trim` : null);
  const urlRef = useRef(url); urlRef.current = url;
  const srcKey = JSON.stringify(controlsOf(source, neutral));
  useEffect(() => {   // follow outside changes (the MCP server, a rebuild) when nothing local is on its way
    if (!busy.current && !Object.keys(pending.current).length && Date.now() - lastEdit.current > 1500) setValues(JSON.parse(srcKey));
  }, [srcKey]);
  const flush = async (recheck = false) => {
    if (busy.current || !urlRef.current) return;
    const changes = pending.current;
    if (!Object.keys(changes).length && !recheck) return;
    pending.current = {};
    busy.current = true;
    try {
      await api.patch(urlRef.current, { changes, recheck, reasons: { '*': 'Set by hand in the Color tab.' } });
      onLive();
    } catch (e) { ctx.notify(e.message); }
    busy.current = false;
    if (Object.keys(pending.current).length) flush();
  };
  useEffect(() => () => { // leaving this target: send what is still waiting
    clearTimeout(timer.current); clearTimeout(idle.current);
    if (Object.keys(pending.current).length && urlRef.current) api.patch(urlRef.current, { changes: pending.current, recheck: true }).catch(() => {});
  }, []);
  const set = (key, value, commit = false) => {
    setValues((v) => ({ ...v, [key]: value }));
    pending.current[key] = value;
    lastEdit.current = Date.now();
    clearTimeout(timer.current);
    timer.current = setTimeout(() => flush(), commit ? 0 : 80);
    clearTimeout(idle.current);
    idle.current = setTimeout(async () => {   // quiet for a moment: check the quality rules and refresh the panels
      if (busy.current || Object.keys(pending.current).length) return;
      await flush(true);
      await ctx.refresh();
    }, 1400);
  };
  return [values, set];
}

// -------------------------------------------------------------------- panel
function Controls({ ctx, spec, target, shot, clip, t, onLive, viewer, setViewer }) {
  const proj = ctx.state.project;
  const neutral = spec.neutral;
  const [values, set] = useLiveControls({ target, shot, proj, neutral, onLive, ctx });
  const S = (key) => spec.controls.find((c) => c.key === key);
  const num = (key, extra = {}) => {
    const s = S(key);
    return html`<${Slider} label=${s.label} value=${values[key]} min=${s.min} max=${s.max} step=${s.step} neutral=${s.neutral}
      help=${s.help} unit=${{ stops: 'st', degrees: '°', points: 'pt' }[s.unit] || ''} onSet=${(v, c) => set(key, v, c)} ...${extra} />`;
  };
  const changed = (...keys) => keys.some((k) => !near(values[k], neutral[k]));
  const resetAll = () => { Object.keys(neutral).forEach((k) => { if (!near(values[k], neutral[k])) set(k, clone(neutral[k]), true); }); };
  const lg = values.log || neutral.log;
  const setLog = (part, v, c) => set('log', { ...lg, [part]: v }, c);
  const hs = values.hsv_sat || neutral.hsv_sat;
  const anyChanged = Object.keys(neutral).some((k) => !near(values[k], neutral[k]));
  return html`<div>
    <div class="row" style="justify-content:space-between;margin:2px 0 6px">
      <span class="dim" style="font-size:11.5px">${target === 'look'
        ? `Values of the look at full strength. The viewer shows it at ${proj.strength}%.`
        : `Only shot ${shot.index + 1}. Applied after its balance, before the look.`}</span>
      ${anyChanged ? html`<button class="small ghost" onClick=${() => { if (confirm('Put every color control of this target back to neutral?')) resetAll(); }}>Reset all</button>` : ''}
    </div>
    <${Group} title="Exposure and white balance" open=${true} changed=${changed('exposure', 'temp', 'tint')}>
      ${num('exposure')}${num('temp')}${num('tint')}
    <//>
    <${Group} title="Primaries: lift, gamma, gain, offset" open=${true} changed=${changed('lift', 'gamma', 'gain', 'offset', 'contrast', 'pivot')}
      note="Drag the ball for color (Shift for fine moves), the slider under it for level. Numbers are R, G, B.">
      <div class="wheels">
        <${Wheel} title="Lift" value=${values.lift} neutral=${neutral.lift} radius=${0.06} level=${[-0.1, 0.1]} step=${0.001} onSet=${(v, c) => set('lift', v, c)} />
        <${Wheel} title="Gamma" value=${values.gamma} neutral=${neutral.gamma} radius=${0.06} level=${[-0.1, 0.1]} step=${0.001} onSet=${(v, c) => set('gamma', v, c)} />
        <${Wheel} title="Gain" value=${values.gain} neutral=${neutral.gain} radius=${0.2} level=${[-0.5, 0.8]} step=${0.001} onSet=${(v, c) => set('gain', v, c)} />
        <${Wheel} title="Offset" value=${values.offset} neutral=${neutral.offset} radius=${4} level=${[-8, 8]} step=${0.05} onSet=${(v, c) => set('offset', v, c)} />
      </div>
      ${num('contrast')}${num('pivot')}
    <//>
    <${Group} title="Tone and color" open=${true} changed=${changed('saturation', 'hue', 'color_boost', 'shadows', 'highlights')}>
      ${num('saturation')}${num('color_boost')}${num('hue')}${num('shadows')}${num('highlights')}
    <//>
    <${Group} title="HDR zones" changed=${changed('hdr')} note=${S('hdr').help}>
      <${Zones} value=${values.hdr} spec=${S('hdr')} onSet=${(v, c) => set('hdr', v, c)} />
    <//>
    <${Group} title="Log wheels" changed=${changed('log')} note="Offset inside one range of brightness. Low Range and High Range set the borders.">
      <div class="wheels three">
        ${['shadow', 'midtone', 'highlight'].map((w) => html`<${Wheel} title=${w[0].toUpperCase() + w.slice(1)} value=${lg[w]} neutral=${[0, 0, 0]}
          radius=${0.03} level=${[-0.05, 0.05]} step=${0.001} onSet=${(v, c) => setLog(w, v, c)} />`)}
      </div>
      <${Slider} label="Low Range" value=${lg.low_range} min=${0.05} max=${0.6} step=${0.005} neutral=${0.333} onSet=${(v, c) => setLog('low_range', v, c)} />
      <${Slider} label="High Range" value=${lg.high_range} min=${0.3} max=${0.9} step=${0.005} neutral=${0.55} onSet=${(v, c) => setLog('high_range', v, c)} />
    <//>
    <${Group} title="Curves" changed=${changed('tone_curve', 'curves_rgb', 'hue_vs_hue', 'hue_vs_sat', 'hue_vs_lum', 'lum_vs_sat', 'sat_vs_sat', 'sat_vs_lum')}>
      <${Curves} values=${values} set=${set} />
    <//>
    <${Group} title="RGB mixer" changed=${changed('rgb_mixer')} note=${S('rgb_mixer').help}>
      <${Mixer} value=${values.rgb_mixer} onSet=${(v, c) => set('rgb_mixer', v, c)} />
      ${changed('rgb_mixer') ? html`<button class="small ghost" style="margin-top:4px" onClick=${() => set('rgb_mixer', clone(neutral.rgb_mixer), true)}>Reset</button>` : ''}
    <//>
    <${Group} title="Film-style saturation (HSV)" changed=${changed('hsv_sat')} note=${S('hsv_sat').help}>
      <${Slider} label="Gain" value=${hs.gain} min=${0.5} max=${1.6} step=${0.005} neutral=${1} onSet=${(v, c) => set('hsv_sat', { ...hs, gain: v }, c)} />
      <${Slider} label="Gamma" value=${hs.gamma} min=${-0.2} max=${0.2} step=${0.002} neutral=${0} onSet=${(v, c) => set('hsv_sat', { ...hs, gamma: v }, c)} />
    <//>
    <${Group} title="Secondaries (qualifiers)" open=${(values.secondaries || []).length > 0} changed=${changed('secondaries')}
      note="Key one color or area, such as the sky, skin or foliage, and correct only there.">
      <${Secondaries} list=${values.secondaries} set=${set} ctx=${ctx} clip=${clip} shot=${shot} t=${t} target=${target} viewer=${viewer} setViewer=${setViewer} />
    <//>
  </div>`;
}

export function ColorPanel({ ctx, clip, shot, t, onLive, viewer, setViewer }) {
  const spec = useSpec(ctx.notify);
  const [target, setTarget] = useState('look');
  const proj = ctx.state.project;
  useEffect(() => { setViewer(null); }, [target, shot && shot.id]);
  if (!proj.shots.length) return html`<div class="pad dim">Add and analyze clips first.</div>`;
  if (!spec) return html`<div class="pad dim">Loading controls…</div>`;
  return html`<div class="pad colors">
    <div class="row" style="margin-bottom:6px">
      <div class="seg">
        <button class=${target === 'look' ? 'on' : ''} onClick=${() => setTarget('look')}>Look: every shot</button>
        <button class=${target === 'shot' ? 'on' : ''} disabled=${!shot} onClick=${() => setTarget('shot')}>This shot only</button>
      </div>
    </div>
    <${Controls} key=${target + ':' + (target === 'shot' && shot ? shot.id : '')} ctx=${ctx} spec=${spec} target=${target} shot=${shot}
      clip=${clip} t=${t} onLive=${onLive} viewer=${viewer} setViewer=${setViewer} />
  </div>`;
}
