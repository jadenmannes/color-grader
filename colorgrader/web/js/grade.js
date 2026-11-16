import { html, useState, useEffect, useRef } from '/static/vendor/htm-preact-standalone.js';
import { api } from './api.js';
import { RecipePanel, ExportPanel } from './recipe.js';
import { LooksPanel } from './looks.js';

export function stageChoices(proj) {
  return proj.graded ? ['converted', 'balanced', 'graded'] : ['converted'];
}

const fmt = (v, d = 3) => (typeof v === 'number' ? v.toFixed(d) : v);
const rgb = (list, d = 3) => list.map((v) => fmt(v, d)).join('  ');
const STATUS = { pass: ['ok', 'Pass'], warn: ['warn', 'Check'], fail: ['bad', 'Fail'] };

export function ReferenceCard({ ctx }) {
  const { state, refresh, notify, runJob } = ctx;
  const ref = state.project.reference;
  const [path, setPath] = useState('');
  const [busy, setBusy] = useState(false);
  const hasShots = state.project.shots.length > 0;
  const setRef = async (p) => {
    if (!p) return;
    setBusy(true);
    try {
      await api.post('/api/reference', { path: p });
      setPath('');
      await refresh();
      if (hasShots) await runJob(() => api.post('/api/grade/build', {}), 'Building the grade');
    } catch (e) { notify(e.message); }
    setBusy(false);
  };
  const pick = async () => {
    const r = await api.post('/api/pick', { kind: 'reference' });
    if (r.error) return notify(r.error);
    setRef(r.paths && r.paths[0]);
  };
  const drop = async (e) => {
    e.preventDefault(); e.stopPropagation();
    const f = (e.dataTransfer.files || [])[0];
    if (!f) return;
    try { setRef((await api.upload(f)).path); } catch (err) { notify(err.message); }
  };
  return html`
    <div class="pad" style="border-top:1px solid var(--line);flex:none" onDragOver=${(e) => e.preventDefault()} onDrop=${drop}>
      <h3 style="margin-top:0">Reference</h3>
      ${!ref && html`<div>
        <div class="row"><button disabled=${busy} onClick=${pick}>Add reference</button><span class="dim">still, screenshot or clip</span></div>
        <input type="text" style="margin-top:8px" placeholder="or paste a file path" value=${path} onInput=${(e) => setPath(e.target.value)}
          onKeyDown=${(e) => { if (e.key === 'Enter' && path.trim()) setRef(path.trim().replace(/^"|"$/g, '')); }} />
      </div>`}
      ${ref && html`<div>
        <img class="refthumb" src=${`/api/reference/frame?v=${state.project.stamp || 0}`} />
        <div class="clip-name" style="margin-top:6px" title=${ref.path}>${ref.name}</div>
        <p class="dim" style="margin-top:4px">${ref.description}</p>
        <div class="row"><button class="small" onClick=${pick}>Replace</button>
          <button class="small ghost" onClick=${async () => { await api.del('/api/reference'); refresh(); }}>Remove</button></div>
      </div>`}
    </div>`;
}

function Rules({ rules }) {
  if (!rules || !rules.length) return null;
  return html`<table class="plain">${rules.map((r) => html`<tr>
    <td style="width:54px"><span class=${'badge ' + STATUS[r.status][0]}>${STATUS[r.status][1]}</span></td>
    <td><div>${r.name}</div><div class="dim">${r.detail}</div></td></tr>`)}</table>`;
}

function Points({ pts, unit = '' }) {
  if (!pts || !pts.length) return html`<span class="dim">none</span>`;
  return html`<span class="mono">${pts.map(([a, b]) => `${a}${unit} → ${b}`).join('   ')}</span>`;
}

function LookTable({ look }) {
  const r = look.reasons || {};
  const row = (name, value, why) => html`<tr><td style="width:96px">${name}</td><td><div class="mono">${value}</div>${why ? html`<div class="dim">${why}</div>` : ''}</td></tr>`;
  const skin = look.skin || {};
  return html`<table class="plain">
    ${row('Contrast', `${fmt(look.contrast)}   pivot ${fmt(look.pivot)}`, r.contrast)}
    ${row('Lift R G B', rgb(look.lift), r.lift)}
    ${row('Gamma R G B', rgb(look.gamma), r.gamma)}
    ${row('Gain R G B', rgb(look.gain), r.gain)}
    ${(look.tone_curve || []).length ? row('Film curve', html`<${Points} pts=${look.tone_curve} />`, r.tone_curve) : ''}
    ${row('Saturation', fmt(look.saturation, 1), r.saturation)}
    ${(look.sat_vs_lum || []).length ? row('Density', html`<${Points} pts=${look.sat_vs_lum.filter((p) => p[1] !== 1)} />`, r.sat_vs_lum) : ''}
    ${(look.lum_vs_sat || []).length ? row('Lum vs Sat', html`<${Points} pts=${look.lum_vs_sat} />`, r.lum_vs_sat) : ''}
    ${(look.hue_vs_sat || []).length ? row('Hue vs Sat', html`<${Points} pts=${look.hue_vs_sat.filter((p) => p[1] !== 1)} unit="°" />`, r.hue_vs_sat) : ''}
    ${(look.hue_vs_hue || []).length ? row('Hue vs Hue', html`<${Points} pts=${look.hue_vs_hue.filter((p) => p[1] !== 0)} unit="°" />`, r.hue_vs_hue) : ''}
    ${((skin.hue_vs_hue || []).length || (skin.hue_vs_sat || []).length) ? row('Skin', html`<div>hue <${Points} pts=${skin.hue_vs_hue} unit="°" /></div><div>sat <${Points} pts=${skin.hue_vs_sat} unit="°" /></div>`, r.skin) : (r.skin ? row('Skin', 'no correction', r.skin) : '')}
    ${(look.sat_vs_sat || []).length ? row('Sat vs Sat', html`<${Points} pts=${look.sat_vs_sat} />`, r.sat_vs_sat) : ''}
  </table>`;
}

function GradeTab({ ctx, shot, onLive }) {
  const { state, refresh, notify, runJob } = ctx;
  const proj = state.project;
  const [strength, setStrength] = useState(proj.strength);
  const timer = useRef(null);
  useEffect(() => { setStrength(proj.strength); }, [proj.strength]);
  const build = () => runJob(() => api.post('/api/grade/build', {}), 'Building the grade');
  const [bright, setBright] = useState(proj.brightness === undefined ? 100 : proj.brightness);
  useEffect(() => { setBright(proj.brightness === undefined ? 100 : proj.brightness); }, [proj.brightness]);
  const [polish, setPolish] = useState(proj.polish === undefined ? 70 : proj.polish);
  useEffect(() => { setPolish(proj.polish === undefined ? 70 : proj.polish); }, [proj.polish]);
  const changePolish = async (v) => {
    try {
      await api.put('/api/grade/polish', { polish: v });
      const done = proj.reference ? await build() : await runJob(() => api.post('/api/suggestions', {}), 'Finding looks for your footage');
      if (done) { await refresh(); onLive(); }
    } catch (e) { notify(e.message); }
  };
  const changeBright = async (v) => {
    try { await api.put('/api/grade/brightness', { brightness: v }); const done = await build(); if (done) { await refresh(); onLive(); } } catch (e) { notify(e.message); }
  };
  const changeStrength = (v) => {
    setStrength(v);
    clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      try { await api.put('/api/grade/strength', { strength: v }); await refresh(); onLive(); } catch (e) { notify(e.message); }
    }, 180);
  };
  const setFinish = async (patch) => {
    const f = JSON.parse(JSON.stringify(proj.finish));
    Object.keys(patch).forEach((k) => { f[k] = Object.assign(f[k] || {}, patch[k]); });
    try { await api.put('/api/grade/finish', { finish: f }); await refresh(); onLive(); } catch (e) { notify(e.message); }
  };
  if (!proj.shots.length) return html`<div class="pad dim">Add and analyze clips first.</div>`;
  const rep = proj.report;
  const look = proj.applied_look || proj.look;
  const g = shot && shot.grade;
  const reasons = (g && g.reasons) || {};
  const fin = proj.finish;
  return html`
    <div class="pad">
      <div class="row">
        <button class="primary" onClick=${build}>${proj.graded ? 'Rebuild grade' : 'Build grade'}</button>
        <span class="dim">${proj.reference ? 'Balance, match, then the look of your reference.' : 'Balance and match. Add a reference or pick a look for the rest.'}</span>
      </div>
      ${rep && html`<div style="margin-top:12px">
        <div class="row"><span class=${'badge ' + (rep.totals.failed ? 'bad' : 'ok')}>${rep.totals.failed ? rep.totals.failed + ' rule failures' : 'All quality rules pass'}</span>
          ${rep.totals.warned ? html`<span class="badge warn">${rep.totals.warned} to check</span>` : ''}
          <span class="dim">${rep.scenes} scene${rep.scenes === 1 ? '' : 's'}</span></div>
        ${(rep.notes || []).map((n) => html`<div class="notice" style="margin:8px 0 0">${n}</div>`)}
        ${(rep.fixes || []).length ? html`<h3>Fixed during the build</h3><ul style="margin:0;padding-left:18px">${rep.fixes.map((f) => html`<li>${f}</li>`)}</ul>` : ''}
      </div>`}
      ${proj.reference && html`<div>
        <h3>Reference brightness: ${bright}%</h3>
        <input type="range" min="0" max="100" step="5" value=${bright} onInput=${(e) => setBright(parseInt(e.target.value, 10))}
          onChange=${(e) => changeBright(parseInt(e.target.value, 10))} />
        <div class="dim">100%: your footage takes on the brightness of the reference. 0%: your own exposure stays and the grade copies contrast and color only. Release the slider to rebuild.</div>
      </div>`}
      <div>
        <h3>Punch: ${polish}%</h3>
        <input type="range" min="0" max="100" step="5" value=${polish} onInput=${(e) => setPolish(parseInt(e.target.value, 10))}
          onChange=${(e) => changePolish(parseInt(e.target.value, 10))} />
        <div class="dim">The finishing pass: deep blacks, bright top end, more midtone contrast. 0% leaves the tone exactly as the reference or look sets it. Release the slider to rebuild${proj.reference ? '' : ' the suggested looks'}.</div>
      </div>
      <${LooksPanel} ctx=${ctx} onLive=${onLive} />
      ${proj.graded && html`
        <h3>Look strength: ${strength}%</h3>
        <input type="range" min="0" max="100" value=${strength} onInput=${(e) => changeStrength(parseInt(e.target.value, 10))} />
        <h3>${proj.look.name && proj.look.name !== 'None' ? proj.look.name : 'Look'} at ${strength}%</h3>
        ${proj.look.description ? html`<p class="dim">${proj.look.description}</p>` : ''}
        <${LookTable} look=${look} />
        <h3>Finish</h3>
        <div class="row"><input type="checkbox" checked=${fin.vignette.enabled} onChange=${(e) => setFinish({ vignette: { enabled: e.target.checked } })} />
          <span style="width:70px">Vignette</span>
          <input type="range" min="0.05" max="0.8" step="0.05" value=${fin.vignette.amount} onChange=${(e) => setFinish({ vignette: { amount: parseFloat(e.target.value) } })} />
          <span class="mono" style="width:74px">${fin.vignette.amount.toFixed(2)} stops</span></div>
        ${[['denoise', 'Noise reduction', 'Cleans camera noise before the grade. Log footage needs it most.'],
           ['sharpen', 'Sharpen', 'Fine edge contrast. The crisp part of a finished picture.'],
           ['clarity', 'Clarity', 'Wide local contrast. Gives faces and buildings shape.']].map(([key, label, tip]) => html`
          <div class="row" title=${tip}><span style="width:112px;margin-left:22px">${label}</span>
            <input type="range" min="0" max="100" step="5" value=${(fin.detail || {})[key] || 0}
              onChange=${(e) => setFinish({ detail: { [key]: parseInt(e.target.value, 10) } })} />
            <span class="mono" style="width:44px">${(fin.detail || {})[key] || 0}</span></div>`)}
        <div class="dim" style="margin:2px 0 6px 22px">Judge noise and sharpness at full size: Play graded or the export. The viewer scales the picture down.</div>
        <div class="row"><input type="checkbox" checked=${fin.grain.enabled} onChange=${(e) => setFinish({ grain: { enabled: e.target.checked } })} />
          <span style="width:70px">Grain</span>
          <input type="range" min="0.1" max="1.2" step="0.05" value=${fin.grain.amount} onChange=${(e) => setFinish({ grain: { amount: parseFloat(e.target.value) } })} />
          <span class="mono" style="width:74px">${fin.grain.amount.toFixed(2)}%</span></div>
      `}
      ${proj.graded && shot && g && html`
        <h3>Shot ${shot.index + 1}: balance and match</h3>
        <div class="row" style="margin-bottom:6px"><span>Scene</span>
          <input type="number" min="1" style="width:64px" value=${shot.scene} onChange=${async (e) => { await api.put(`/api/shots/${shot.id}/scene`, { scene: parseInt(e.target.value, 10) || 1 }); refresh(); }} />
          ${shot.is_lead ? html`<span class="badge">lead shot of its scene</span>` : ''}
          <span class="dim">Change the number, then rebuild.</span></div>
        <table class="plain">
          <tr><td style="width:96px">Gain R G B</td><td><div class="mono">${rgb(g.balance.gain)}</div><div class="dim">${reasons.gain || ''}</div></td></tr>
          <tr><td>Offset R G B</td><td><div class="mono">${rgb(g.balance.offset, 2)}</div><div class="dim">${reasons.offset || ''}</div></td></tr>
          ${(g.match.contrast !== 1 || g.match.saturation !== 50) ? html`<tr><td>Match</td><td><div class="mono">contrast ${fmt(g.match.contrast)}   saturation ${fmt(g.match.saturation, 1)}</div><div class="dim">${reasons.match || ''}</div></td></tr>` : ''}
          ${(g.match.hue_vs_hue || []).length ? html`<tr><td>Skin trim</td><td><${Points} pts=${g.match.hue_vs_hue} unit="°" /><div class="dim">${reasons.skin || ''}</div></td></tr>` : ''}
        </table>
        <h3>Quality rules for this shot</h3>
        <${Rules} rules=${shot.rules} />
      `}
    </div>`;
}

export function GradePanel({ ctx, tab, clip, shot, t, onLive }) {
  if (tab === 'recipe') return html`<${RecipePanel} ctx=${ctx} shot=${shot} />`;
  if (tab === 'export') return html`<${ExportPanel} ctx=${ctx} clip=${clip} shot=${shot} />`;
  return html`<${GradeTab} ctx=${ctx} shot=${shot} onLive=${onLive} />`;
}
