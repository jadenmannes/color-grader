import { html, useState, useEffect, useRef } from '/static/vendor/htm-preact-standalone.js';
import { api } from './api.js';

const BLANK_PROFILE = () => ({
  name: 'Default', input: { type: 'curve', curve: '', gamut: '' }, bit_depth: 10, data_range: 'auto',
  detect: { bit_depth: [], transfer: [], primaries: [], image: 'any', metadata: [] },
  corrections: { exposure: 0, temperature: 0, tint: 0, saturation: 1 },
});
const BLANK_CAMERA = () => ({ name: '', maker: '', metadata_names: [], notes: '', profiles: [BLANK_PROFILE()], definitions: { curves: {}, gamuts: {} } });
const clone = (o) => JSON.parse(JSON.stringify(o));
const csv = (list) => (list || []).join(', ');
const uncsv = (text) => text.split(',').map((s) => s.trim()).filter(Boolean);
const coreOf = (cam) => JSON.stringify(cam.profiles.map((p) => [p.name, p.input, p.data_range]));

function ProfileEditor({ ctx, profile, index, open, onToggle, onChange, onRemove, canRemove, localDefs }) {
  const { lib, notify } = ctx;
  const set = (patch) => onChange({ ...profile, ...patch });
  const setInput = (patch) => set({ input: { ...profile.input, ...patch } });
  const setDetect = (patch) => set({ detect: { ...profile.detect, ...patch } });
  const setCorr = (k, v) => set({ corrections: { ...profile.corrections, [k]: v } });
  const curves = [...lib.curves, ...Object.values(localDefs.curves || {}).filter((c) => !lib.curves.some((x) => x.id === c.id))];
  const gamuts = [...lib.gamuts, ...Object.values(localDefs.gamuts || {}).filter((g) => !lib.gamuts.some((x) => x.id === g.id))];
  const isLut = profile.input.type === 'lut';
  const curveInfo = curves.find((c) => c.id === (isLut ? (profile.input.lut_output || {}).curve : profile.input.curve));
  const pickLut = async () => {
    const r = await api.post('/api/pick', { kind: 'lut' });
    if (r.error) return notify(r.error);
    if (r.paths && r.paths[0]) loadLut(r.paths[0]);
  };
  const loadLut = async (path) => {
    try {
      const res = await api.post('/api/luts', { path });
      setInput({ lut_file: res.lut_file });
      notify(`LUT loaded: ${res.size} points per side.`, 'info');
    } catch (e) { notify(e.message); }
  };
  const group = (kind, label) => {
    const items = curves.filter((c) => c.kind === kind);
    return items.length ? html`<optgroup label=${label}>${items.map((c) => html`<option value=${c.id}>${c.name}</option>`)}</optgroup>` : '';
  };
  const curveSelect = (value, onPick) => html`
    <select value=${value || ''} onChange=${(e) => onPick(e.target.value)}>
      <option value="">Pick a curve</option>
      ${group('scene', 'Log and scene curves')}${group('display', 'Finished pictures')}${group('pq', 'HDR')}
    </select>`;
  const gamutSelect = (value, onPick) => html`
    <select value=${value || ''} onChange=${(e) => onPick(e.target.value)}>
      <option value="">Pick a gamut</option>
      ${gamuts.map((g) => html`<option value=${g.id}>${g.name}</option>`)}
    </select>`;
  return html`
    <div class="profile">
      <div class="profile-head" onClick=${onToggle}>
        <span>${open ? '▾' : '▸'}</span>
        <strong style="flex:1">${profile.name || 'Unnamed profile'}</strong>
        <span class="dim">${isLut ? 'Maker LUT' : ((curves.find((c) => c.id === profile.input.curve) || {}).name || 'no curve')}</span>
        ${canRemove && html`<button class="small ghost danger" onClick=${(e) => { e.stopPropagation(); onRemove(); }}>Remove</button>`}
      </div>
      ${open && html`<div class="profile-body">
        <div class="grid3" style="margin-top:8px">
          <div><label>Profile name</label><input type="text" value=${profile.name} onInput=${(e) => set({ name: e.target.value })} /></div>
          <div><label>Bit depth</label><select value=${profile.bit_depth} onChange=${(e) => set({ bit_depth: parseInt(e.target.value, 10) })}>
            ${[8, 10, 12, 14, 16].map((b) => html`<option value=${b}>${b}-bit</option>`)}</select></div>
          <div><label>Data range</label><select value=${profile.data_range} onChange=${(e) => set({ data_range: e.target.value })}>
            <option value="auto">Auto (follow the file flag)</option><option value="video">Video (16 to 235)</option><option value="full">Full (0 to 255)</option></select></div>
        </div>
        <h3>Input conversion</h3>
        <div class="seg" style="margin-bottom:8px">
          <button class=${!isLut ? 'on' : ''} onClick=${() => setInput({ type: 'curve' })}>Published curve and gamut</button>
          <button class=${isLut ? 'on' : ''} onClick=${() => setInput({ type: 'lut', lut_output: profile.input.lut_output || { curve: 'rec709-gamma24', gamut: 'rec709' }, lut_input: profile.input.lut_input || 'absolute' })}>Maker's conversion LUT</button>
        </div>
        ${!isLut && html`<div class="grid2">
          <div><label>Curve</label>${curveSelect(profile.input.curve, (v) => setInput({ curve: v }))}</div>
          <div><label>Color gamut</label>${gamutSelect(profile.input.gamut, (v) => setInput({ gamut: v }))}</div>
        </div>`}
        ${isLut && html`<div>
          <div class="row"><button onClick=${pickLut}>Load .cube file</button>
            <input type="text" placeholder="or paste the path of a .cube file, then press Enter" onKeyDown=${(e) => { if (e.key === 'Enter' && e.target.value.trim()) loadLut(e.target.value.trim().replace(/^"|"$/g, '')); }} />
          </div>
          <div class="hint">${profile.input.lut_file ? `Loaded: ${profile.input.lut_file}` : 'Use this when the maker publishes a LUT and no curve math.'}</div>
          <div class="grid3" style="margin-top:8px">
            <div><label>The LUT outputs this curve</label>${curveSelect((profile.input.lut_output || {}).curve, (v) => setInput({ lut_output: { ...(profile.input.lut_output || {}), curve: v } }))}</div>
            <div><label>The LUT outputs this gamut</label>${gamutSelect((profile.input.lut_output || {}).gamut, (v) => setInput({ lut_output: { ...(profile.input.lut_output || {}), gamut: v } }))}</div>
            <div><label>The LUT expects</label><select value=${profile.input.lut_input || 'absolute'} onChange=${(e) => setInput({ lut_input: e.target.value })}>
              <option value="absolute">Full-scale code values</option><option value="signal">Range-normalized signal</option></select></div>
          </div></div>`}
        ${curveInfo && html`<div class="hint">${curveInfo.source}${curveInfo.notes ? ' ' + curveInfo.notes : ''}</div>`}
        <h3>How to recognize this profile</h3>
        <div class="grid3">
          <div><label>Transfer tags in the file</label><input type="text" value=${csv(profile.detect.transfer)} placeholder="any" onInput=${(e) => setDetect({ transfer: uncsv(e.target.value) })} /></div>
          <div><label>Primaries tags in the file</label><input type="text" value=${csv(profile.detect.primaries)} placeholder="any" onInput=${(e) => setDetect({ primaries: uncsv(e.target.value) })} /></div>
          <div><label>The picture looks</label><select value=${profile.detect.image} onChange=${(e) => setDetect({ image: e.target.value })}>
            <option value="any">Either way</option><option value="log">Flat (log)</option><option value="display">Finished</option></select></div>
        </div>
        <label>Metadata tags (any one settles the profile)</label>
        <input type="text" value=${csv(profile.detect.metadata || [])} placeholder="xml.capturegammaequation=value"
          onInput=${(e) => setDetect({ metadata: uncsv(e.target.value) })} />
        <div class="hint">key=value checks one tag exactly. Plain text matches anywhere in the metadata. Camera XML settings appear as xml.* tags in the clip's metadata list.</div>
        <h3>Default corrections for every clip from this profile</h3>
        <div class="grid4">
          ${[['exposure', 'Exposure (stops)', 0.05], ['temperature', 'Temperature (+ warm)', 1], ['tint', 'Tint (+ magenta)', 1], ['saturation', 'Saturation (1 = none)', 0.01]].map(([k, name, step]) => html`
            <div><label>${name}</label><input type="number" step=${step} value=${profile.corrections[k]} onInput=${(e) => setCorr(k, parseFloat(e.target.value) || 0)} /></div>`)}
        </div>
      </div>`}
    </div>`;
}

export function CameraForm({ ctx, initial, existingId, samplePath, onSaved, onDeleted }) {
  const { notify, reloadLib } = ctx;
  const [cam, setCam] = useState(() => clone(initial));
  const [open, setOpen] = useState(0);
  const [sample, setSample] = useState(samplePath || '');
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { setCam(clone(initial)); setPreview(null); setOpen(0); }, [existingId, initial]);
  const set = (patch) => setCam({ ...cam, ...patch });
  const isNew = !existingId;
  const previewValid = preview && preview.core === coreOf(cam);
  const convert = async () => {
    setBusy(true);
    try {
      const res = await api.post('/api/cameras/preview', { camera: cam, profile_index: open < 0 ? 0 : open, sample_path: sample });
      setPreview({ ...res, core: coreOf(cam) });
    } catch (e) { notify(e.message); setPreview(null); }
    setBusy(false);
  };
  const pickSample = async () => {
    const r = await api.post('/api/pick', { kind: 'clips' });
    if (r.error) return notify(r.error);
    if (r.paths && r.paths[0]) setSample(r.paths[0]);
  };
  const save = async () => {
    setBusy(true);
    try {
      const saved = await api.post('/api/cameras', { camera: cam, replace_id: existingId || undefined, preview_token: preview ? preview.token : undefined });
      await reloadLib();
      notify(`Saved ${saved.name}.`, 'info');
      onSaved(saved);
    } catch (e) { notify(e.message); }
    setBusy(false);
  };
  const del = async () => {
    if (!confirm(`Delete ${cam.name}? Clips keep their files.`)) return;
    await api.del('/api/cameras/' + existingId);
    await reloadLib();
    onDeleted();
  };
  return html`
    <div>
      <div class="grid3">
        <div><label>Camera name</label><input type="text" value=${cam.name} onInput=${(e) => set({ name: e.target.value })} /></div>
        <div><label>Maker</label><input type="text" value=${cam.maker} onInput=${(e) => set({ maker: e.target.value })} /></div>
        <div><label>Name in file metadata (comma separated)</label><input type="text" value=${csv(cam.metadata_names)} onInput=${(e) => set({ metadata_names: uncsv(e.target.value) })} /></div>
      </div>
      <div class="field" style="margin-top:8px"><label>Notes</label><input type="text" value=${cam.notes || ''} onInput=${(e) => set({ notes: e.target.value })} /></div>
      <div class="row" style="margin-top:14px"><h3 style="flex:1;margin:0">Recording profiles</h3>
        <button class="small" onClick=${() => { set({ profiles: [...cam.profiles, { ...BLANK_PROFILE(), name: 'Profile ' + (cam.profiles.length + 1) }] }); setOpen(cam.profiles.length); }}>Add profile</button></div>
      <div style="margin-top:8px">
        ${cam.profiles.map((p, i) => html`<${ProfileEditor} key=${i} ctx=${ctx} profile=${p} index=${i} open=${open === i} localDefs=${cam.definitions || {}}
          onToggle=${() => setOpen(open === i ? -1 : i)} canRemove=${cam.profiles.length > 1}
          onChange=${(np) => set({ profiles: cam.profiles.map((x, j) => (j === i ? np : x)) })}
          onRemove=${() => { set({ profiles: cam.profiles.filter((_, j) => j !== i) }); setOpen(0); }} />`)}
      </div>
      <h3>Check the conversion on a sample clip</h3>
      <div class="row">
        <input type="text" placeholder="Path of a sample clip from this camera" value=${sample} onInput=${(e) => setSample(e.target.value)} />
        <button onClick=${pickSample}>Browse</button>
        <button class="primary" disabled=${busy || !sample} onClick=${convert}>Convert sample</button>
      </div>
      <div class="hint">The open profile converts the sample. ${isNew ? 'A new camera saves after you check this result.' : ''}</div>
      ${preview && html`<div>
        <div class="compare">
          <figure><img src=${`/api/preview/${preview.token}/original.jpg`} /><figcaption>Original, as recorded (${preview.sample.pix_fmt}, ${preview.sample.color_range || 'no range flag'})</figcaption></figure>
          <figure><img src=${`/api/preview/${preview.token}/converted.jpg`} /><figcaption>Converted to Rec.709. Black ${preview.stats.black}%, midtones ${preview.stats.mid}%, white ${preview.stats.white}%.</figcaption></figure>
        </div>
        ${!previewValid && html`<div class="hint" style="color:var(--warn)">Settings changed since this conversion. Convert again.</div>`}
      </div>`}
      <div class="row" style="margin-top:16px">
        <button class="primary" disabled=${busy || (isNew && !previewValid)} onClick=${save}>${isNew ? 'Save camera' : 'Save changes'}</button>
        ${!isNew && html`<a href=${`/api/cameras/${existingId}/export`} download><button>Export file</button></a>`}
        ${!isNew && html`<button class="danger" onClick=${del}>Delete</button>`}
      </div>
    </div>`;
}

export function CameraModal({ ctx, start, onClose }) {
  const draft = { ...BLANK_CAMERA(), ...(start.draft || {}) };
  return html`
    <div class="modal-back">
      <div class="modal">
        <div class="modal-head"><h2 style="margin:0;flex:1">Add camera</h2><button class="ghost" onClick=${() => onClose(false)}>Close</button></div>
        <div class="pad" style="padding:16px 20px 20px">
          <div class="notice">This clip comes from a camera missing from your library. The form holds what the file metadata shows. Pick the curve and gamut, convert the sample, then save.${draft.notes ? ' ' + draft.notes : ''}</div>
          <${CameraForm} ctx=${ctx} initial=${draft} existingId=${null} samplePath=${start.samplePath} onSaved=${() => onClose(true)} onDeleted=${() => onClose(false)} />
        </div>
      </div>
    </div>`;
}

export function CamerasPage({ ctx }) {
  const { cams, refresh, notify, reloadLib } = ctx;
  const [sel, setSel] = useState(cams[0] ? cams[0].id : null);
  const [creating, setCreating] = useState(false);
  const fileRef = useRef(null);
  const cur = cams.find((c) => c.id === sel);
  const blank = useRef(BLANK_CAMERA());
  const importFile = async (e) => {
    const f = e.target.files[0];
    e.target.value = '';
    if (!f) return;
    try {
      const data = JSON.parse(await f.text());
      let saved;
      try { saved = await api.post('/api/cameras/import', { camera: data }); }
      catch (err) {
        if (!/exists/.test(err.message) || !confirm(err.message + '\nReplace it?')) throw err;
        saved = await api.post('/api/cameras/import', { camera: data, overwrite: true });
      }
      await reloadLib(); await refresh(); setSel(saved.id); setCreating(false);
      notify(`Imported ${saved.name}.`, 'info');
    } catch (err) { notify(err.message); }
  };
  return html`
    <div class="form-page">
      <div class="col list">
        <div class="pad row wrap" style="border-bottom:1px solid var(--line)">
          <button class="primary" onClick=${() => { blank.current = BLANK_CAMERA(); setCreating(true); setSel(null); }}>Add camera</button>
          <button onClick=${() => fileRef.current.click()}>Import</button>
          <input type="file" accept=".json" ref=${fileRef} style="display:none" onChange=${importFile} />
        </div>
        <div class="scroll">
          ${cams.map((c) => html`<div class=${'list-item' + (sel === c.id && !creating ? ' sel' : '')} onClick=${() => { setSel(c.id); setCreating(false); }}>
            <strong>${c.name}</strong><small>${c.profiles.map((p) => p.name.split(' (')[0]).join(', ')}</small></div>`)}
        </div>
        <div class="pad" style="border-top:1px solid var(--line)">
          <button class="small ghost" onClick=${async () => { if (confirm('Write the shipped cameras again? Edited copies of them get replaced.')) { await api.post('/api/cameras/restore-defaults'); await reloadLib(); refresh(); } }}>Restore shipped cameras</button>
          <div class="hint">Cameras live as files in ${ctx.state.home}/cameras.</div>
        </div>
      </div>
      <div class="form">
        ${creating && html`<h2>New camera</h2><${CameraForm} ctx=${ctx} initial=${blank.current} existingId=${null}
          onSaved=${async (c) => { await refresh(); setCreating(false); setSel(c.id); }} onDeleted=${() => {}} />`}
        ${!creating && cur && html`<h2>${cur.name}</h2><${CameraForm} ctx=${ctx} initial=${cur} existingId=${cur.id}
          onSaved=${async (c) => { await refresh(); setSel(c.id); }} onDeleted=${async () => { await refresh(); setSel(null); }} />`}
        ${!creating && !cur && html`<div class="empty">Pick a camera, or add one.</div>`}
      </div>
    </div>`;
}
