import { html, useState, useEffect } from '/static/vendor/htm-preact-standalone.js';
import { api } from './api.js';

export function SettingsPage({ ctx }) {
  const { state, lib, refresh, notify } = ctx;
  const [s, setS] = useState(state.settings);
  const [formats, setFormats] = useState([]);
  const [editors, setEditors] = useState([]);
  useEffect(() => { setS(state.settings); }, [state.settings]);
  useEffect(() => { api.get('/api/export/formats').then(setFormats).catch(() => setFormats([])); }, []);
  useEffect(() => { api.get('/api/recipe/editors').then(setEditors).catch(() => setEditors([])); }, []);
  const save = async (patch) => {
    const next = { ...s, ...patch };
    setS(next);
    try { await api.put('/api/settings', patch); await refresh(); } catch (e) { notify(e.message); }
  };
  const pickFolder = async () => {
    const r = await api.post('/api/pick', { kind: 'folder' });
    if (r.error) return notify(r.error);
    if (r.paths && r.paths[0]) save({ export_folder: r.paths[0] });
  };
  return html`
    <div class="form-page"><div class="form">
      <h2>Settings</h2>
      <div class="card">
        <h3 style="margin-top:0">Editor</h3>
        <div class="grid2">
          <div><label>Target editor</label><select value=${s.target_editor} onChange=${(e) => save({ target_editor: e.target.value })}>
            ${editors.length ? editors.map((e) => html`<option value=${e.id}>${e.name}</option>`) : html`<option value=${s.target_editor}>${s.target_editor}</option>`}</select>
            <div class="hint">Each editor is one recipe module. A new editor arrives as one more module.</div></div>
          <div><label>Resolve edition</label><select value=${s.resolve_edition} onChange=${(e) => save({ resolve_edition: e.target.value })}>
            <option value="free">Free</option><option value="studio">Studio</option></select>
            <div class="hint">The recipe marks steps that need Studio.</div></div>
        </div>
      </div>
      <div class="card">
        <h3 style="margin-top:0">Color</h3>
        <div class="grid2">
          <div><label>Output color space</label><select value=${s.output_space} onChange=${(e) => save({ output_space: e.target.value })}>
            ${lib.output_spaces.map((o) => html`<option value=${o.id}>${o.name}</option>`)}</select>
            <div class="hint">Exports carry Rec.709 tags, so QuickTime, phones and YouTube read them the same way.</div></div>
          <div><label>Highlight range (max input nits of the tone mapping)</label>
            <input type="number" min="200" max="10000" step="100" value=${s.max_input_nits} onChange=${(e) => save({ max_input_nits: parseInt(e.target.value, 10) })} />
            <div class="hint">10000 is Resolve's default for DaVinci Intermediate: highlights roll off and never clip. The recipe uses the same number.</div></div>
          <div><label>Default look strength: ${s.look_strength}%</label>
            <input type="range" min="0" max="100" value=${s.look_strength} onChange=${(e) => save({ look_strength: parseInt(e.target.value, 10) })} /></div>
          <div><label>Analysis frames per second</label>
            <input type="number" min="1" max="8" step="0.5" value=${s.analysis_fps} onChange=${(e) => save({ analysis_fps: parseFloat(e.target.value) })} /></div>
        </div>
      </div>
      <div class="card">
        <h3 style="margin-top:0">LUT and export</h3>
        <div class="grid2">
          <div><label>LUT size</label><select value=${s.lut_size} onChange=${(e) => save({ lut_size: parseInt(e.target.value, 10) })}>
            ${[17, 33, 65].map((n) => html`<option value=${n}>${n} points</option>`)}</select></div>
          <div><label>Export format</label><select value=${s.export_format} onChange=${(e) => save({ export_format: e.target.value })}>
            ${formats.length ? formats.map((f) => html`<option value=${f.id}>${f.name}</option>`) : html`<option value=${s.export_format}>${s.export_format}</option>`}</select></div>
        </div>
        <div class="field" style="margin-top:8px"><label>Export folder</label>
          <div class="row"><input type="text" value=${s.export_folder} placeholder=${state.home + '/exports'} onChange=${(e) => save({ export_folder: e.target.value })} />
          <button onClick=${pickFolder}>Browse</button></div></div>
      </div>
      <p class="dim">Cameras, looks and settings live in ${state.home}. App updates leave this folder alone.</p>
    </div></div>`;
}
