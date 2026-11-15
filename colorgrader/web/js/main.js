import { html, render, useState, useEffect, useCallback, useRef } from '/static/vendor/htm-preact-standalone.js';
import { api } from './api.js';
import { ProjectPage } from './project.js';
import { CamerasPage, CameraModal } from './cameras.js';
import { SettingsPage } from './settings.js';

function App() {
  const [tab, setTab] = useState('project');
  const [state, setState] = useState(null);
  const [lib, setLib] = useState(null);
  const [cams, setCams] = useState([]);
  const [job, setJob] = useState(null);
  const [toast, setToast] = useState(null);
  const [cameraDraft, setCameraDraft] = useState(null); // {draft, samplePath, clipId}
  const toastTimer = useRef(null);

  const notify = useCallback((msg, kind = 'error') => {
    setToast({ msg: String(msg), kind });
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), kind === 'error' ? 9000 : 4000);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [s, c] = await Promise.all([api.get('/api/state'), api.get('/api/cameras')]);
      setState(s);
      setCams(c);
      return s;
    } catch (e) { notify(e.message); }
    return null;
  }, [notify]);

  useEffect(() => {
    refresh();
    api.get('/api/library').then(setLib).catch((e) => notify(e.message));
  }, []);

  // Run a server job and show progress in the top bar.
  const runJob = useCallback(async (startFn, label) => {
    try {
      setJob({ label, progress: 0, message: '' });
      const { job: id } = await startFn();
      const done = await api.job(id, (j) => setJob({ label, progress: j.progress, message: j.message }));
      setJob(null);
      await refresh();
      return done;
    } catch (e) {
      setJob(null);
      notify(e.message);
      await refresh();
      return null;
    }
  }, [refresh, notify]);

  const reloadLib = useCallback(() => api.get('/api/library').then(setLib), []);
  const ctx = { state, lib, cams, refresh, notify, runJob, setTab, reloadLib,
    openCameraDraft: (draft, samplePath, clipId) => setCameraDraft({ draft, samplePath, clipId }) };

  if (!state || !lib) return html`<div class="empty">Loading…</div>`;
  const tabs = [['project', 'Project'], ['cameras', 'Cameras'], ['settings', 'Settings']];
  return html`
    <div class="topbar">
      <div class="brand">Color Grader</div>
      <div class="tabs">
        ${tabs.map(([id, name]) => html`<button class=${'tab' + (tab === id ? ' active' : '')} onClick=${() => setTab(id)}>${name}</button>`)}
      </div>
      <div class="spacer"></div>
      ${job && html`<div class="jobbar"><span>${job.message || job.label}</span>
        <div class="progress"><div style=${{ width: Math.round(job.progress * 100) + '%' }}></div></div></div>`}
    </div>
    ${tab === 'project' && html`<${ProjectPage} ctx=${ctx} />`}
    ${tab === 'cameras' && html`<${CamerasPage} ctx=${ctx} />`}
    ${tab === 'settings' && html`<${SettingsPage} ctx=${ctx} />`}
    ${cameraDraft && html`<${CameraModal} ctx=${ctx} start=${cameraDraft} onClose=${async (saved) => {
      const clipId = cameraDraft.clipId;
      setCameraDraft(null);
      await reloadLib();
      if (saved && clipId) { try { await api.post('/api/clips/' + clipId + '/redetect'); } catch (e) { notify(e.message); } }
      await refresh();
    }} />`}
    ${toast && html`<div class=${'toast ' + (toast.kind === 'error' ? '' : 'info')} onClick=${() => setToast(null)}>${toast.msg}</div>`}
  `;
}

render(html`<${App} />`, document.getElementById('app'));
