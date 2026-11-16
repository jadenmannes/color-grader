import { html, useState, useEffect } from '/static/vendor/htm-preact-standalone.js';
import { api, qs } from './api.js';

function SettingsTable({ items }) {
  if (!items || !items.length) return null;
  return html`<table class="plain recipe-table">${items.map((i) => html`<tr>
    <td class="rc">${i.control}</td>
    <td><div class="mono rv">${i.value}</div>${i.reason ? html`<div class="dim">${i.reason}</div>` : ''}</td></tr>`)}</table>`;
}

function CurveTable({ c }) {
  return html`<div class="curve">
    <div class="curve-name">${c.curve}</div>
    ${c.how ? html`<div class="dim" style="margin-bottom:4px">${c.how}</div>` : ''}
    <table class="plain"><tr>${c.columns.map((h) => html`<th>${h}</th>`)}</tr>
      ${c.rows.map((r) => html`<tr>${r.map((v, i) => html`<td class=${i > 0 && i < 3 ? 'mono' : ''}>${v}</td>`)}</tr>`)}</table>
    ${c.reason ? html`<div class="dim" style="margin-top:4px">${c.reason}</div>` : ''}
  </div>`;
}

function Node({ n }) {
  return html`<div class="node">
    <div class="node-head"><span class="node-num">${n.number}</span><span class="node-label">${n.label}</span><span>${n.title}</span></div>
    <div class="node-tool"><span class="dim">Tool:</span> ${n.tool}${n.how ? html`<span class="dim"> · ${n.how}</span>` : ''}</div>
    <${SettingsTable} items=${n.settings} />
    ${(n.curves || []).map((c) => html`<${CurveTable} c=${c} />`)}
    ${(n.notes || []).map((t) => html`<div class="node-note">${t}</div>`)}
  </div>`;
}

export function RecipeView({ recipe }) {
  if (!recipe) return null;
  return html`<div class="recipe">
    <h2>${recipe.title}</h2>
    ${recipe.intro.map((t) => html`<p class="dim">${t}</p>`)}
    ${recipe.sections.map((sec) => html`<details open>
      <summary>${sec.title}</summary>
      ${sec.where ? html`<p class="where">${sec.where}</p>` : ''}
      ${(sec.notes || []).map((t) => html`<p class="dim">${t}</p>`)}
      <${SettingsTable} items=${sec.items} />
      ${(sec.nodes || []).map((n) => html`<${Node} n=${n} />`)}
      ${(sec.groups || []).map((g) => html`<div class="group">
        <div class="group-title">${g.title}</div>
        ${g.subtitle ? html`<div class="dim">${g.subtitle}</div>` : ''}
        ${(g.notes || []).map((t) => html`<div class="node-note">${t}</div>`)}
        ${(g.nodes || []).map((n) => html`<${Node} n=${n} />`)}
      </div>`)}
    </details>`)}
  </div>`;
}

export function RecipePanel({ ctx }) {
  const { state, notify, refresh } = ctx;
  const proj = state.project;
  const [recipe, setRecipe] = useState(null);
  const [editors, setEditors] = useState([]);
  const [wide, setWide] = useState(false);
  const editor = state.settings.target_editor;
  const edition = state.settings.resolve_edition;
  useEffect(() => { api.get('/api/recipe/editors').then(setEditors).catch(() => setEditors([])); }, []);
  useEffect(() => {
    let live = true;
    api.get('/api/recipe?' + qs({ editor, edition })).then((r) => { if (live) setRecipe(r); }).catch((e) => notify(e.message));
    return () => { live = false; };
  }, [proj.stamp, proj.strength, editor, edition, state.settings.output_space, state.settings.max_input_nits, state.settings.lut_size]);
  const save = async (patch) => { try { await api.put('/api/settings', patch); await refresh(); } catch (e) { notify(e.message); } };
  const copy = async () => {
    try { await navigator.clipboard.writeText(recipe.text); notify('Recipe copied as text.', 'info'); } catch (e) { notify('Copy failed. Use Download.'); }
  };
  if (!proj.shots.length) return html`<div class="pad dim">Add and analyze clips first.</div>`;
  const cur = editors.find((e) => e.id === editor);
  const bar = html`<div class="row wrap" style="margin-bottom:10px">
    <select style="width:auto" value=${editor} onChange=${(e) => save({ target_editor: e.target.value })}>
      ${editors.map((e) => html`<option value=${e.id}>${e.name}</option>`)}</select>
    ${cur && cur.editions.length ? html`<select style="width:auto" value=${edition} onChange=${(e) => save({ resolve_edition: e.target.value })}>
      ${cur.editions.map((e) => html`<option value=${e.id}>${e.name}</option>`)}</select>` : ''}
    <button class="small" onClick=${copy} disabled=${!recipe}>Copy as text</button>
    <a class="btn small" href=${'/api/recipe.txt?' + qs({ editor, edition })}>Download</a>
    <button class="small" onClick=${() => setWide(!wide)}>${wide ? 'Close large view' : 'Large view'}</button>
  </div>`;
  if (wide) {
    return html`<div class="pad">${bar}<div class="dim">The recipe is open in the large view.</div>
      <div class="modal-back" onClick=${(e) => { if (e.target.classList.contains('modal-back')) setWide(false); }}>
        <div class="modal"><div class="modal-head"><h2 style="margin:0;flex:1">Recipe</h2>${bar}</div>
          <div class="pad"><${RecipeView} recipe=${recipe} /></div></div></div></div>`;
  }
  return html`<div class="pad">${bar}
    ${!proj.graded ? html`<div class="notice">No grade yet. Build the grade on the Grade tab. The recipe then holds real values.</div>` : ''}
    ${recipe ? html`<${RecipeView} recipe=${recipe} />` : html`<div class="dim">Loading…</div>`}</div>`;
}

export function ExportPanel({ ctx, clip }) {
  const { state, notify, runJob } = ctx;
  const proj = state.project;
  const [formats, setFormats] = useState([]);
  const [format, setFormat] = useState(state.settings.export_format);
  const [luts, setLuts] = useState(null);
  const [videos, setVideos] = useState(null);
  useEffect(() => { api.get('/api/export/formats').then(setFormats).catch(() => setFormats([])); }, []);
  useEffect(() => { setFormat(state.settings.export_format); }, [state.settings.export_format]);
  if (!proj.shots.length) return html`<div class="pad dim">Add and analyze clips first.</div>`;
  const exportLuts = async () => {
    const done = await runJob(() => api.post('/api/export/luts', {}), 'Exporting the LUT set');
    if (done) setLuts(done.result);
  };
  const exportVideo = async (ids) => {
    const done = await runJob(() => api.post('/api/export/video', { clip_ids: ids, format }), 'Exporting graded clips');
    if (done) setVideos(done.result);
  };
  const open = async (path) => {
    try { const r = await api.post('/api/open-folder', { path }); if (!r.ok) notify(`Folder: ${r.path}`, 'info'); } catch (e) { notify(e.message); }
  };
  const videoClips = proj.clips.filter((c) => !c.info.is_image);
  const isVideo = clip && !clip.info.is_image;
  return html`<div class="pad">
    ${!proj.graded ? html`<div class="notice">No grade yet. Exports hold the input conversion only until you build the grade.</div>` : ''}
    <h3 style="margin-top:0">LUT set</h3>
    <p class="dim">${state.settings.lut_size}-point .cube files: the look for Resolve, the look with output conversion, and one camera LUT per recording profile for any editor. A text file in the folder says where each file goes.</p>
    <button class="primary" onClick=${exportLuts}>Export LUT set</button>
    ${luts && html`<div class="card" style="margin-top:10px">
      <div class="row"><b style="flex:1">${luts.files.length} LUTs written</b><button class="small" onClick=${() => open(luts.folder)}>Open folder</button></div>
      <div class="mono dim" style="margin:4px 0 8px;word-break:break-all">${luts.folder}</div>
      ${luts.files.map((f) => html`<div style="margin-bottom:6px"><div class="mono" style="word-break:break-all">${f.name}</div><div class="dim">${f.where}</div></div>`)}
    </div>`}
    <h3>Graded clip</h3>
    <p class="dim">Renders at source resolution with the full grade, noise reduction, sharpening, vignette and grain included. The file carries Rec.709 tags, so QuickTime, phones and YouTube show the same picture.</p>
    <label>Format</label>
    <select value=${format} onChange=${(e) => setFormat(e.target.value)}>
      ${formats.map((f) => html`<option value=${f.id}>${f.name}</option>`)}</select>
    <div class="row wrap" style="margin-top:8px">
      <button class="primary" disabled=${!isVideo} onClick=${() => exportVideo([clip.id])}>Export this clip</button>
      <button disabled=${videoClips.length < 2} onClick=${() => exportVideo(videoClips.map((c) => c.id))}>Export all ${videoClips.length} clips</button>
    </div>
    ${videos && html`<div class="card" style="margin-top:10px">
      <div class="row"><b style="flex:1">${videos.files.length} file${videos.files.length === 1 ? '' : 's'} written</b><button class="small" onClick=${() => open(videos.folder)}>Open folder</button></div>
      <div class="mono dim" style="margin:4px 0 8px;word-break:break-all">${videos.folder}</div>
      ${videos.files.map((f) => html`<div style="margin-bottom:6px"><div class="mono">${f.name}</div>
        <div class="dim">${f.format_name} · ${f.width}×${f.height} · ${f.pix_fmt} · tags ${f.tags.primaries}/${f.tags.transfer}/${f.tags.matrix} · ${f.seconds} s to render</div></div>`)}
    </div>`}
    <h3>Recipe</h3>
    <p class="dim">The Recipe tab lists every node and value for your editor. Download it as text from there.</p>
  </div>`;
}
