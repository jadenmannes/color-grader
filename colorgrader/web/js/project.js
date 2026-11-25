import { html, useState, useEffect, useRef, useMemo, useCallback } from '/static/vendor/htm-preact-standalone.js';
import { api, qs, fmtTime } from './api.js';
import { GradePanel, ReferenceCard, stageChoices } from './grade.js';
import { createPlayer, AssistantFeed } from './assistant.js';
import { StillView } from './still.js';

const STAGE_NAMES = { source: 'Source', converted: 'Converted', balanced: 'Balanced', graded: 'Graded' };
const SCOPE_KINDS = [['waveform', 'Waveform'], ['parade', 'RGB parade'], ['vectorscope', 'Vectorscope'], ['histogram', 'Histogram']];

function useDebounced(value, ms) {
  const [v, setV] = useState(value);
  useEffect(() => { const id = setTimeout(() => setV(value), ms); return () => clearTimeout(id); }, [value, ms]);
  return v;
}

function ClipRow({ ctx, clip, shots, sel, onSelect, onAnalyze }) {
  const { cams, refresh, notify } = ctx;
  const [edit, setEdit] = useState(false);
  const [camId, setCamId] = useState(clip.camera_id || '');
  const [profId, setProfId] = useState(clip.profile_id || '');
  useEffect(() => { setCamId(clip.camera_id || ''); setProfId(clip.profile_id || ''); }, [clip.camera_id, clip.profile_id]);
  const cam = cams.find((c) => c.id === camId);
  const curCam = cams.find((c) => c.id === clip.camera_id);
  const curProf = curCam && curCam.profiles.find((p) => p.id === clip.profile_id);
  const i = clip.info;
  const apply = async () => {
    try {
      await api.post(`/api/clips/${clip.id}/profile`, { camera_id: camId, profile_id: profId });
      setEdit(false);
      await refresh();
      onAnalyze([clip.id]);
    } catch (e) { notify(e.message); }
  };
  const remove = async () => { await api.del(`/api/clips/${clip.id}`); refresh(); };
  const needsCamera = !clip.camera_id;
  const needsConfirm = clip.camera_id && !clip.confirmed;
  return html`
    <div class="clip">
      <div class="row">
        <div class="clip-name" style="flex:1" title=${clip.path}>${clip.name}</div>
        <button class="small ghost" title="Remove clip" onClick=${remove}>✕</button>
      </div>
      <div class="clip-meta">${i.codec} ${i.bit_depth}-bit ${i.chroma} · ${i.width}×${i.height} · ${i.fps ? i.fps.toFixed(2) + ' fps · ' : ''}${i.is_image ? 'still' : fmtTime(i.duration)}</div>
      ${needsCamera && html`<div class="confirm">
        <div>Unknown camera. ${clip.detect.message}</div>
        <div class="row" style="margin-top:6px">
          <button class="small primary" onClick=${() => ctx.openCameraDraft(clip.detect.draft, clip.path, clip.id)}>Add this camera</button>
          <button class="small" onClick=${() => setEdit(true)}>Pick existing</button>
        </div></div>`}
      ${needsConfirm && !edit && html`<div class="confirm">
        <div>${clip.detect.message}</div>
        <div class="row" style="margin-top:6px">
          <select value=${profId} onChange=${(e) => setProfId(e.target.value)}>
            ${(curCam ? curCam.profiles : []).map((p) => html`<option value=${p.id}>${p.name}</option>`)}
          </select>
          <button class="small primary" onClick=${apply}>Confirm</button>
        </div></div>`}
      ${!needsCamera && !needsConfirm && !edit && html`<div class="row" style="margin-bottom:6px">
        <span class="badge ok" title=${clip.detect.message}>${curCam ? curCam.name : '?'} · ${curProf ? curProf.name : '?'}</span>
        <button class="small ghost" onClick=${() => setEdit(true)}>Change</button>
      </div>`}
      ${edit && html`<div class="confirm">
        <label>Camera</label>
        <select value=${camId} onChange=${(e) => { setCamId(e.target.value); const c = cams.find((x) => x.id === e.target.value); setProfId(c ? c.profiles[0].id : ''); }}>
          <option value="">Pick a camera</option>
          ${cams.map((c) => html`<option value=${c.id}>${c.name}</option>`)}
        </select>
        <label style="margin-top:6px">Recording profile</label>
        <select value=${profId} onChange=${(e) => setProfId(e.target.value)}>
          ${(cam ? cam.profiles : []).map((p) => html`<option value=${p.id}>${p.name}</option>`)}
        </select>
        <div class="row" style="margin-top:8px">
          <button class="small primary" disabled=${!camId || !profId} onClick=${apply}>Apply</button>
          <button class="small" onClick=${() => setEdit(false)}>Cancel</button>
        </div></div>`}
      ${shots.length === 0 && clip.camera_id && clip.confirmed && html`<button class="small" onClick=${() => onAnalyze([clip.id])}>Analyze</button>`}
      ${shots.map((s) => html`
        <div class=${'shot' + (sel === s.id ? ' sel' : '')} data-shot=${s.id} onClick=${() => onSelect(s)}>
          <img src=${`/api/thumb?shot=${s.id}&v=${s.signature || ''}`} loading="lazy" />
          <div class="shot-info">
            <div>Shot ${s.index + 1}${s.has_faces ? ' · faces' : ''}</div>
            <div class="dim">${fmtTime(s.start)} – ${fmtTime(s.end)}</div>
            ${s.report && s.report.failed > 0 ? html`<span class="badge bad">${s.report.failed} rule${s.report.failed > 1 ? 's' : ''} failed</span>` : ''}
          </div>
        </div>`)}
    </div>`;
}

// Preview sizes the server renders. The viewer asks for the smallest one that fills its screen pixels, so the picture
// stays sharp on large and high-DPI screens and stays fast on small ones.
const PREVIEW_STEPS = [640, 960, 1280, 1600, 1920, 2560, 3200, 3840, 5120, 7680];

function usePreviewFit(clip) {
  // The viewer's size in CSS pixels. From it: the box the picture fills (exact aspect, never cropped) and the preview
  // width to ask the server for (the screen pixels of that box, rounded up to a step, never more than the source).
  const [box, setBox] = useState([1280, 720]);
  const ro = useRef(null);
  const ref = useCallback((el) => {
    if (ro.current) { ro.current.disconnect(); ro.current = null; }
    if (!el || typeof ResizeObserver === 'undefined') return;
    ro.current = new ResizeObserver(() => setBox([el.clientWidth, el.clientHeight]));
    ro.current.observe(el);
  }, []);
  if (!clip) return [ref, 1280, null];
  const info = clip.info;
  const aspect = info.display_width / Math.max(1, info.display_height);
  const w = Math.max(16, Math.min(box[0], box[1] * aspect));
  const fit = { width: `${Math.floor(w)}px`, height: `${Math.floor(w / aspect)}px` };
  const need = w * (window.devicePixelRatio || 1);
  return [ref, Math.min(info.display_width, PREVIEW_STEPS.find((s) => s >= need) || PREVIEW_STEPS[PREVIEW_STEPS.length - 1]), fit];
}

function Viewer({ clip, shot, t, before, after, version, reference, play, mode, feed, onFull, wrapRef, isFs, onFs }) {
  const [split, setSplit] = useState(0.5);
  const box = useRef(null);
  const drag = useRef(false);
  const [sizeRef, previewW, fit] = usePreviewFit(clip);
  const move = (e) => {
    if (!drag.current || !box.current) return;
    const r = box.current.getBoundingClientRect();
    setSplit(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)));
  };
  useEffect(() => {
    const up = () => { drag.current = false; };
    window.addEventListener('pointerup', up);
    window.addEventListener('pointermove', move);
    return () => { window.removeEventListener('pointerup', up); window.removeEventListener('pointermove', move); };
  }, []);
  const exitFs = isFs && html`<button class="small fs-exit" onClick=${onFs}>Exit full screen (Esc)</button>`;
  if (!clip) return html`<div class="viewer-wrap" ref=${wrapRef}><div class="empty" style="margin:auto">Add clips to begin. The app reads them in place. Nothing leaves this computer.</div></div>`;
  const base = { clip: clip.id, t: t.toFixed(3), w: previewW, shot: shot ? shot.id : undefined, v: version };
  const urlA = '/api/frame?' + qs({ ...base, stage: before });
  const urlB = '/api/frame?' + qs({ ...base, stage: after });
  const refBox = reference && html`<div class="refbox"><img src=${`/api/reference/frame?v=${version}`} /><div class="tag l">Reference</div></div>`;
  if (mode && mode.kind === 'key') {
    const keyUrl = '/api/key?' + qs({ clip: clip.id, t: t.toFixed(3), w: Math.min(previewW, 1920), shot: shot ? shot.id : undefined, target: mode.target, index: mode.index, v: version });
    return html`
      <div class="viewer-wrap" ref=${wrapRef}>
        <div class="viewer" ref=${sizeRef}><div class="stage" style=${fit}>
          <img src=${keyUrl} draggable=${false} />
          <div class="tag l">${mode.label}</div>
        </div></div>
        ${refBox}
        <${AssistantFeed} items=${feed} />
        ${exitFs}
      </div>`;
  }
  const picking = mode && mode.kind === 'pick';
  const pickAt = (e) => {
    const r = box.current.getBoundingClientRect();
    mode.onPick(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)));
  };
  if (play) {
    return html`
      <div class="viewer-wrap" ref=${wrapRef}>
        <div class="viewer" ref=${sizeRef}><div class="stage" style=${fit}>
          <video src=${play.url} controls autoplay loop playsinline muted
            onLoadedMetadata=${(e) => { if (shot && shot.start > 0) e.target.currentTime = shot.start; }}></video>
          <div class="tag l">Graded playback, preview quality</div>
        </div></div>
        ${refBox}
        <${AssistantFeed} items=${feed} />
        ${exitFs}
      </div>`;
  }
  return html`
    <div class="viewer-wrap" ref=${wrapRef}>
      <div class="viewer" ref=${sizeRef}>
        <div class=${'stage' + (picking ? ' picking' : '')} ref=${box} style=${fit} title="Double-click: full-quality still"
          onDblClick=${() => { if (!picking && onFull) onFull(); }}
          onPointerDown=${(e) => { if (picking) { pickAt(e); return; } drag.current = true; move(e); }}>
          <img src=${urlA} draggable=${false} />
          <img class="after" src=${urlB} draggable=${false} style=${{ clipPath: `inset(0 0 0 ${split * 100}%)` }} />
          <div class="handle" style=${{ left: `calc(${split * 100}% - 1px)` }}></div>
          <div class="tag l">${picking ? mode.label : STAGE_NAMES[before]}</div>
          <div class="tag r">${STAGE_NAMES[after]}</div>
        </div>
      </div>
      ${refBox}
      <${AssistantFeed} items=${feed} />
      ${exitFs}
    </div>`;
}

function Scopes({ clip, shot, t, before, after, version, reference }) {
  const [kind, setKind] = useState('waveform');
  if (!clip) return null;
  const base = { clip: clip.id, t: t.toFixed(3), kind, shot: shot ? shot.id : undefined, v: version };
  const cells = [[STAGE_NAMES[before], '/api/scope?' + qs({ ...base, stage: before })], [STAGE_NAMES[after], '/api/scope?' + qs({ ...base, stage: after })]];
  if (reference) cells.push(['Reference', '/api/reference/scope?' + qs({ kind, v: version })]);
  return html`
    <div class="scopes">
      <div class="row" style="margin-bottom:8px">
        <div class="seg">${SCOPE_KINDS.map(([id, name]) => html`<button class=${kind === id ? 'on' : ''} onClick=${() => setKind(id)}>${name}</button>`)}</div>
        <span class="dim">Scopes show the display signal. The line on the vectorscope marks the skin tone line.</span>
      </div>
      <div class="scope-row" style=${{ gridTemplateColumns: `repeat(${cells.length}, 1fr)` }}>
        ${cells.map(([name, url]) => html`<div class="scope-cell"><img class=${kind === 'vectorscope' ? 'square' : ''} src=${url} /><div class="cap">${name}</div></div>`)}
      </div>
    </div>`;
}

export function Measurements({ m, summary }) {
  if (!m) return html`<p class="dim">Not analyzed yet.</p>`;
  return html`
    <div class="lines">${(summary || []).map((l) => html`<p>${l}</p>`)}</div>
    <h3>Main colors</h3>
    <div class="swatches">${m.palette.map((c) => html`<div title=${`${c.name} ${Math.round(c.share * 100)}%`} style=${{ background: c.hex, width: (c.share * 100) + '%' }}></div>`)}</div>
    <div class="dim" style="font-size:11.5px">${m.palette.slice(0, 4).map((c) => `${c.name} ${Math.round(c.share * 100)}%`).join(' · ')}</div>`;
}

function AnalysisPanel({ ctx, clip, shot }) {
  if (!clip) return html`<div class="pad dim">Select a shot.</div>`;
  const i = clip.info;
  return html`
    <div class="pad">
      <h2>${shot ? `Shot ${shot.index + 1}` : clip.name}</h2>
      ${shot ? html`<${Measurements} m=${shot.measure} summary=${shot.summary} />` : html`<p class="dim">Analyze the clip to see measurements.</p>`}
      <h3>File</h3>
      <div class="kv">
        <div>Codec</div><div>${i.codec} ${i.profile || ''}</div>
        <div>Pixel format</div><div class="mono">${i.pix_fmt}</div>
        <div>Range flag</div><div>${i.color_range || 'not set'}${i.full_range ? ' (full)' : ' (video)'}</div>
        <div>Matrix / transfer / primaries</div><div class="mono">${i.color_space || '?'} / ${i.color_transfer || '?'} / ${i.color_primaries || '?'}</div>
        <div>Camera metadata</div><div>${[i.make, i.model].filter(Boolean).join(' ') || 'none'}</div>
        <div>Detection</div><div>${clip.detect.method === 'metadata' ? 'From metadata' : clip.detect.method === 'image' ? 'Estimated from the picture' : 'No match'}${clip.confirmed ? ', confirmed' : ''}</div>
        ${clip.analysis && clip.analysis.seconds ? html`<div>Analysis time</div><div>${clip.analysis.seconds} s for ${fmtTime(i.duration)}</div>` : ''}
      </div>
    </div>`;
}

export function ProjectPage({ ctx }) {
  const { state, refresh, notify, runJob } = ctx;
  const proj = state.project;
  const [sel, setSel] = useState(null);
  const [t, setT] = useState(0);
  const stages = stageChoices(proj);
  const [before, setBefore] = useState('source');
  const [after, setAfter] = useState(stages[stages.length - 1]);
  const [rightTab, setRightTab] = useState('analysis');
  const [pathText, setPathText] = useState('');
  const [liveVersion, setLiveVersion] = useState(0);
  const [viewerMode, setViewerMode] = useState(null);   // color picking or a secondary's key, from the Color tab
  const [colorTarget, setColorTarget] = useState('look'); // the Color tab edits the look or this shot's trim
  const [feed, setFeed] = useState([]);                   // the assistant's last steps, shown on the viewer
  const [still, setStill] = useState(false);              // full-quality still
  const wrapEl = useRef(null);                            // the viewer, for full screen
  const [isFs, setIsFs] = useState(false);
  useEffect(() => {
    const on = () => setIsFs(!!document.fullscreenElement && document.fullscreenElement === wrapEl.current);
    document.addEventListener('fullscreenchange', on);
    return () => document.removeEventListener('fullscreenchange', on);
  }, []);
  const toggleFs = () => {
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else if (wrapEl.current && wrapEl.current.requestFullscreen) wrapEl.current.requestFullscreen().catch(() => {});
  };
  const stillRef = useRef(false); stillRef.current = still;
  const [watch, setWatch] = useState(() => { try { return localStorage.getItem('cg-watch') !== 'off'; } catch (e) { return true; } });
  const watchRef = useRef(watch); watchRef.current = watch;
  const keyByAssistant = useRef(false);
  const tDeb = useDebounced(t, 120);
  useEffect(() => { if (rightTab !== 'color') { setViewerMode(null); keyByAssistant.current = false; } }, [rightTab]);

  const shot = proj.shots.find((s) => s.id === sel) || proj.shots[0] || null;
  const clip = shot ? proj.clips.find((c) => c.id === shot.clip_id) : (proj.clips[0] || null);
  useEffect(() => { if (shot && (t < shot.start || t > shot.end)) setT(shot.times && shot.times.length ? shot.times[Math.floor(shot.times.length / 2)] : (shot.start + shot.end) / 2); }, [shot && shot.id]);
  useEffect(() => { if (!stages.includes(after)) setAfter(stages[stages.length - 1]); }, [stages.join(',')]);
  // Live: an outside tool (the MCP server) may change the grade or move to another shot. Follow it, and show each
  // step it reports with the assistant cursor.
  const lastFocus = useRef(-1);
  const lastActivity = useRef(-1);
  const stampRef = useRef(proj.stamp || 0);
  stampRef.current = proj.stamp || 0;
  const uiRef = useRef(null);
  uiRef.current = {
    select: (id, time) => { setSel(id); if (time != null) setT(time); },
    openColor: (target) => { setRightTab('color'); setColorTarget(target === 'shot' ? 'shot' : 'look'); },
    openGrade: () => setRightTab('grade'),
    showKey: (k) => { keyByAssistant.current = true; setRightTab('color'); setViewerMode({ kind: 'key', target: k.target, index: k.index, label: `Key: ${k.name || 'secondary'}. Selected parts in color, the rest gray.` }); },
    clearKey: () => { if (keyByAssistant.current) { keyByAssistant.current = false; setViewerMode(null); } },
    feed: (ev) => setFeed((f) => [...f.slice(-2), { ...ev, seen: Date.now() }]),
    covered: () => stillRef.current,
    refresh: async () => { await refresh(); setLiveVersion((v) => v + 1); },
  };
  const player = useRef(null);
  if (!player.current) player.current = createPlayer(() => uiRef.current);
  useEffect(() => {
    let live = true;
    const tick = async () => {
      try {
        const st = await api.get('/api/stamp');
        if (!live) return;
        if (lastFocus.current < 0) lastFocus.current = st.focus.n;
        if (st.focus.n !== lastFocus.current) {
          lastFocus.current = st.focus.n;
          if (st.focus.shot_id) { setSel(st.focus.shot_id); if (st.focus.t != null) setT(st.focus.t); if (!watchRef.current) setRightTab('grade'); }
        }
        if (lastActivity.current < 0) lastActivity.current = st.activity || 0;
        if ((st.activity || 0) !== lastActivity.current) {
          const res = await api.get('/api/activity?since=' + lastActivity.current);
          lastActivity.current = res.n;
          if (watchRef.current && res.events.length) player.current.push(res.events);
        }
        if (st.stamp !== stampRef.current && st.jobs === 0 && !player.current.busy()) { await refresh(); setLiveVersion((v) => v + 1); }
      } catch (e) { /* the app may be restarting */ }
    };
    const id = setInterval(tick, 600);
    return () => { live = false; clearInterval(id); };
  }, []);
  useEffect(() => { if (proj.graded) setAfter('graded'); }, [proj.graded]);
  useEffect(() => {   // F opens the full-quality still of the frame on screen
    const key = (e) => {
      if (stillRef.current || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
      if (e.key === 'f' || e.key === 'F') { e.preventDefault(); setStill(true); }
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, []);

  const version = useMemo(() => JSON.stringify([proj.look, proj.strength, proj.finish, shot && shot.grade, clip && clip.profile_id, state.settings.output_space, state.settings.max_input_nits, liveVersion]).length + ':' + liveVersion + ':' + (proj.stamp || 0), [proj, shot, clip, state.settings, liveVersion]);

  const analyze = (ids) => runJob(() => api.post('/api/analyze', { clip_ids: ids }), 'Analyzing footage');

  // graded playback: a small preview file rendered by ffmpeg with the full grade
  const [play, setPlay] = useState(null);
  useEffect(() => { setPlay(null); }, [clip && clip.id, version]);
  const startPlay = async () => {
    if (!clip) return;
    try {
      let st = await api.get('/api/playback/status?' + qs({ clip: clip.id, stage: 'graded' }));
      if (!st.ready) {
        const done = await runJob(() => api.post('/api/playback/build', { clip_id: clip.id, stage: 'graded' }), 'Rendering playback');
        if (!done) return;
        st = await api.get('/api/playback/status?' + qs({ clip: clip.id, stage: 'graded' }));
      }
      if (st.ready) setPlay({ url: st.url });
    } catch (e) { notify(e.message); }
  };

  const addPaths = async (paths) => {
    if (!paths || !paths.length) return;
    try {
      const res = await api.post('/api/clips', { paths });
      res.errors.forEach((e) => notify(`${e.path}: ${e.error}`));
      (res.skipped || []).forEach((d) => notify(`${d.path.split(/[\\/]/).pop()} is already in the project as ${d.name}. Skipped.`, 'info'));
      const s = await refresh();
      const ready = res.added.filter((c) => c.camera_id && c.confirmed && !(s && s.project.shots.some((x) => x.clip_id === c.id))).map((c) => c.id);
      const unknown = res.added.find((c) => !c.camera_id);
      if (unknown) ctx.openCameraDraft(unknown.detect.draft, unknown.path, unknown.id);
      if (ready.length) analyze(ready);
    } catch (e) { notify(e.message); }
  };
  const pick = async () => {
    const r = await api.post('/api/pick', { kind: 'clips' });
    if (r.error) notify(r.error);
    addPaths(r.paths);
  };
  const drop = async (e) => {
    e.preventDefault();
    const files = [...(e.dataTransfer.files || [])];
    const paths = [];
    for (const f of files) { try { paths.push((await api.upload(f)).path); } catch (err) { notify(err.message); } }
    addPaths(paths);
  };

  return html`
    <div class="page">
      <div class="col left" onDragOver=${(e) => e.preventDefault()} onDrop=${drop}>
        <div class="pad" style="border-bottom:1px solid var(--line)">
          <div class="row">
            <button class="primary" onClick=${pick}>Add clips</button>
            <button class="ghost small" title="Start an empty project" onClick=${async () => { if (confirm('Clear the project? Your files stay untouched.')) { await api.post('/api/project/reset'); refresh(); } }}>New project</button>
          </div>
          <div class="row" style="margin-top:8px">
            <input type="text" placeholder="or paste a file path" value=${pathText} onInput=${(e) => setPathText(e.target.value)}
              onKeyDown=${(e) => { if (e.key === 'Enter' && pathText.trim()) { addPaths([pathText.trim().replace(/^"|"$/g, '')]); setPathText(''); } }} />
          </div>
        </div>
        <div class="scroll">
          ${proj.clips.length === 0 && html`<div class="empty">No clips yet.<br/>Add clips, or drop files here.</div>`}
          ${proj.clips.map((c) => html`<${ClipRow} key=${c.id} ctx=${ctx} clip=${c} shots=${proj.shots.filter((s) => s.clip_id === c.id)}
            sel=${shot && shot.id} onSelect=${(s) => setSel(s.id)} onAnalyze=${analyze} />`)}
        </div>
        <${ReferenceCard} ctx=${ctx} />
      </div>
      <div class="col center">
        <${Viewer} clip=${clip} shot=${shot} t=${tDeb} before=${before} after=${after} version=${version} reference=${proj.reference} play=${play} mode=${viewerMode} feed=${watch ? feed : []}
          onFull=${() => setStill(true)} wrapRef=${wrapEl} isFs=${isFs} onFs=${toggleFs} />
        ${still && clip && html`<${StillView} clip=${clip} shot=${shot} t=${tDeb} before=${before} after=${after} names=${STAGE_NAMES} version=${version}
          onClose=${() => setStill(false)} denoise=${((proj.finish || {}).detail || {}).denoise || 0}
          onDenoiseOff=${async () => {
            const f = JSON.parse(JSON.stringify(proj.finish));
            f.detail = { ...(f.detail || {}), denoise: 0 };
            try { await api.put('/api/grade/finish', { finish: f }); await refresh(); setLiveVersion((v) => v + 1); } catch (e) { notify(e.message); }
          }} />`}
        ${clip && html`<div class="transport">
          ${!clip.info.is_image && proj.graded ? (play
            ? html`<button class="small" onClick=${() => setPlay(null)}>Back to frames</button>`
            : html`<button class="small" title="Renders a small preview file of the whole clip with the grade, then plays it" onClick=${startPlay}>▶ Play graded</button>`) : ''}
          <button class="small" title="The viewer on the whole screen, with the before and after slider. Esc to leave." onClick=${toggleFs}>⛶ Full screen</button>
          <button class="small" title="This frame at the source's full resolution with the full grade (F or double-click the picture)" onClick=${() => setStill(true)}>Full quality</button>
          <span class="mono">${fmtTime(t)}</span>
          <input type="range" min=${shot ? shot.start : 0} max=${shot ? Math.max(shot.start, shot.end - 0.04) : clip.info.duration} step="0.04" value=${t}
            onInput=${(e) => setT(parseFloat(e.target.value))} />
          <label class="row" style="margin:0;gap:4px;white-space:nowrap" title="Show what an AI assistant does through the MCP server: a cursor on each control it changes">
            <input type="checkbox" checked=${watch} onChange=${(e) => { setWatch(e.target.checked); try { localStorage.setItem('cg-watch', e.target.checked ? 'on' : 'off'); } catch (err) { /* private window */ } }} />Show assistant</label>
          <label style="margin:0">Left</label>
          <select value=${before} onChange=${(e) => setBefore(e.target.value)}>${['source', ...stages].map((s) => html`<option value=${s}>${STAGE_NAMES[s]}</option>`)}</select>
          <label style="margin:0">Right</label>
          <select value=${after} onChange=${(e) => setAfter(e.target.value)}>${stages.map((s) => html`<option value=${s}>${STAGE_NAMES[s]}</option>`)}</select>
        </div>`}
        <${Scopes} clip=${clip} shot=${shot} t=${tDeb} before=${before} after=${after} version=${version} reference=${proj.reference} />
      </div>
      <div class="col right">
        <div class="tabs" style="border-bottom:1px solid var(--line);padding-left:6px">
          ${[['analysis', 'Analysis'], ['grade', 'Grade'], ['color', 'Color'], ['recipe', 'Recipe'], ['export', 'Export']].map(([id, name]) =>
            html`<button class=${'tab' + (rightTab === id ? ' active' : '')} onClick=${() => setRightTab(id)}>${name}</button>`)}
        </div>
        <div class="scroll">
          ${rightTab === 'analysis' ? html`<${AnalysisPanel} ctx=${ctx} clip=${clip} shot=${shot} />`
            : html`<${GradePanel} ctx=${ctx} tab=${rightTab} clip=${clip} shot=${shot} t=${tDeb} onLive=${() => setLiveVersion((v) => v + 1)}
                viewer=${viewerMode} setViewer=${setViewerMode} colorTarget=${colorTarget} setColorTarget=${setColorTarget} />`}
        </div>
      </div>
    </div>`;
}
