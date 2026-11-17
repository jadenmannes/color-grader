import { html, useState, useEffect } from '/static/vendor/htm-preact-standalone.js';
import { api } from './api.js';

function Steps({ steps }) {
  if (!steps || !steps.length) return null;
  return html`<details class="lr-steps" onClick=${(e) => e.stopPropagation()}>
    <summary>Steps and values</summary>
    ${steps.map((st) => html`<div class="lr-step"><b>${st.step}</b> <span class="mono">${st.value}</span><div class="dim">${st.why}</div></div>`)}
  </details>`;
}

// Three looks designed from the selected shot.
export function SuggestedLooks({ ctx, onLive, shot }) {
  const { state, runJob, notify, refresh } = ctx;
  const proj = state.project;
  const list = proj.suggestions || [];
  const read = proj.scene_read;
  const target = shot || null;
  const suggest = () => runJob(() => api.post('/api/suggestions', { shot_id: target ? target.id : null }),
    target ? `Designing looks for shot ${target.index + 1}` : 'Designing looks for your footage');
  const apply = async (s) => {
    const done = await runJob(() => api.post(`/api/suggestions/${s.id}/apply`, {}), `Applying ${s.name}`);
    if (done) { await refresh(); onLive(); }
  };
  if (!proj.shots.length) return null;
  const current = proj.look && proj.look.source === 'suggestion' ? proj.look.name : null;
  return html`<div>
    <h3>Suggested looks</h3>
    ${read && list.length > 0 && html`<div class="scene-read">
      <div class="dim" style="margin-bottom:3px">Read from shot ${read.shot_index + 1} and the shots of its scene:</div>
      ${read.lines.map((l) => html`<div>${l}</div>`)}</div>`}
    ${list.length === 0 && html`<div>
      <p class="dim">${proj.reference ? 'Your reference drives the look. Suggestions offer three alternatives built from your shot.' : 'No reference loaded. The app reads the selected shot and designs three looks for it.'}</p>
      <button onClick=${suggest}>${target ? `Design three looks for shot ${target.index + 1}` : 'Design three looks'}</button></div>`}
    ${list.length > 0 && html`<div>
      ${list.map((s) => html`
        <div class=${'look-row' + (current === s.name ? ' on' : '')} onClick=${() => apply(s)}>
          <img src=${`/api/suggestions/${s.id}/thumb.jpg?v=${s.stamp || 0}`} loading="lazy" />
          <div class="lr-body"><div class="lr-name">${s.name}${current === s.name ? html` <span class="badge ok">applied</span>` : ''}</div>
            <div class="lr-mood">${s.mood}</div><div class="lr-why">${s.why}</div><${Steps} steps=${s.steps} /></div>
        </div>`)}
      <div class="row" style="margin-top:6px"><span class="dim" style="flex:1">Click a look to apply it to every shot. Each shot keeps its own white balance, exposure and black point; every look passes the quality rules.</span>
        <button class="small ghost" onClick=${suggest}>${target && read && target.id !== read.shot_id ? `Design for shot ${target.index + 1}` : 'Design again'}</button></div>
    </div>`}
  </div>`;
}

// Looks saved in the user folder.
export function SavedLooks({ ctx, onLive }) {
  const { state, runJob, notify, refresh } = ctx;
  const proj = state.project;
  const [looks, setLooks] = useState([]);
  const [name, setName] = useState('');
  const load = () => api.get('/api/looks').then(setLooks).catch(() => setLooks([]));
  useEffect(() => { load(); }, []);
  const save = async () => {
    try { await api.post('/api/looks', { name }); setName(''); await load(); notify('Look saved.', 'info'); } catch (e) { notify(e.message); }
  };
  const apply = async (l) => {
    try {
      const r = await api.post(`/api/looks/${l.id}/apply`, {});
      if (r.job) await runJob(() => Promise.resolve({ job: r.job }), `Applying ${l.name}`);
      await refresh();
      onLive();
    } catch (e) { notify(e.message); }
  };
  const remove = async (l) => {
    if (!confirm(`Delete the saved look "${l.name}"?`)) return;
    try { await api.del(`/api/looks/${l.id}`); await load(); } catch (e) { notify(e.message); }
  };
  return html`<div>
    <h3>Saved looks</h3>
    ${looks.length === 0 && html`<p class="dim">No saved looks yet. Save the current look to reuse it on other footage.</p>`}
    ${looks.map((l) => html`<div class="saved-look">
      ${l.has_thumb ? html`<img src=${`/api/looks/${l.id}/thumb.jpg?v=${Math.round(l.created || 0)}`} />` : html`<div style="width:64px"></div>`}
      <div style="flex:1;min-width:0"><div class="clip-name">${l.name}</div>
        <div class="dim" style="font-size:11.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title=${l.description}>${l.description || 'saved look'}</div></div>
      <button class="small" disabled=${!proj.shots.length} onClick=${() => apply(l)}>Apply</button>
      <button class="small ghost" title="Delete" onClick=${() => remove(l)}>✕</button>
    </div>`)}
    ${proj.graded && html`<div class="row" style="margin-top:8px">
      <input type="text" placeholder="Name for the current look" value=${name} onInput=${(e) => setName(e.target.value)}
        onKeyDown=${(e) => { if (e.key === 'Enter' && name.trim()) save(); }} />
      <button class="small" disabled=${!name.trim()} onClick=${save}>Save look</button></div>`}
  </div>`;
}

export function LooksPanel({ ctx, onLive, shot }) {
  const withSuggestions = !!ctx.state.features && ctx.state.features.suggestions;
  return html`<div>${withSuggestions ? html`<${SuggestedLooks} ctx=${ctx} onLive=${onLive} shot=${shot} />` : ''}<${SavedLooks} ctx=${ctx} onLive=${onLive} /></div>`;
}
