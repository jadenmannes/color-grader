// Shows an AI assistant at work. The MCP server reports every step it takes (/api/activity). For each step a cursor
// moves to the control, drags it from the old value to the new one, clicks where a color gets sampled, and a short
// feed lists the step with its reason. The grade itself changes through the MCP server, never through this cursor.
import { html } from '/static/vendor/htm-preact-standalone.js';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const SQ3 = Math.sqrt(3);
const CURVE_KEYS = ['tone_curve', 'curves_rgb', 'lum_vs_sat', 'hue_vs_hue', 'hue_vs_sat', 'hue_vs_lum', 'sat_vs_sat', 'sat_vs_lum'];

let cursor = null;
let label = null;
let hideTimer = null;

function ensure() {
  if (cursor) return;
  cursor = document.createElement('div');
  cursor.className = 'ghost-cursor';
  cursor.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24"><path d="M3 2 L3 19.5 L7.8 15 L11.2 22.4 L14.6 20.9 L11.2 13.6 L17.8 13.6 Z" '
    + 'fill="#e8a33d" stroke="#1a1305" stroke-width="1.4" stroke-linejoin="round"/></svg><div class="ghost-label"></div>';
  document.body.appendChild(cursor);
  label = cursor.querySelector('.ghost-label');
  cursor.style.left = `${Math.round(window.innerWidth * 0.45)}px`;
  cursor.style.top = `${Math.round(window.innerHeight * 0.35)}px`;
}

function say(title, reason) {
  label.textContent = title || '';
  if (reason) {
    const small = document.createElement('small');
    small.textContent = reason;
    label.appendChild(small);
  }
}

async function moveTo(x, y, ms = 450) {
  cursor.style.transitionDuration = `${ms}ms, ${ms}ms, 300ms`;
  cursor.style.left = `${Math.round(x)}px`;
  cursor.style.top = `${Math.round(y)}px`;
  cursor.classList.toggle('flip', x > window.innerWidth - 380);
  await sleep(ms + 30);
}

function ripple() {
  const r = document.createElement('span');
  r.className = 'ghost-ripple';
  r.style.left = cursor.style.left;
  r.style.top = cursor.style.top;
  document.body.appendChild(r);
  setTimeout(() => r.remove(), 700);
}

async function click() {
  cursor.classList.add('down');
  ripple();
  await sleep(160);
  cursor.classList.remove('down');
}

async function waitFor(selector, ms = 1800) {
  const end = Date.now() + ms;
  for (;;) {
    const el = document.querySelector(selector);
    if (el && el.getBoundingClientRect().width > 0) return el;
    if (Date.now() > end) return null;
    await sleep(60);
  }
}

async function reveal(el) {
  const r = el.getBoundingClientRect();
  if (r.top < 60 || r.bottom > window.innerHeight - 20) {
    el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    await sleep(380);
  }
}

async function toEl(el, ms, fx = 0.5, fy = 0.5) {
  await reveal(el);
  const r = el.getBoundingClientRect();
  await moveTo(r.left + r.width * fx, r.top + r.height * fy, ms);
}

function ballAt(pad, value) {
  const radius = parseFloat(pad.dataset.radius);
  const neutral = JSON.parse(pad.dataset.neutral);
  const d = value.map((v, i) => v - neutral[i]);
  let bx = (d[0] - (d[1] + d[2]) / 2) / radius;
  let by = ((SQ3 / 2) * (d[1] - d[2])) / radius;
  const len = Math.hypot(bx, by);
  if (len > 1) { bx /= len; by /= len; }
  const r = pad.getBoundingClientRect();
  return [r.left + r.width * (0.5 + bx * 0.46), r.top + r.height * (0.5 - by * 0.46)];
}

async function playControl(c, T) {
  const el = await waitFor(CURVE_KEYS.includes(c.key) ? '[data-ctl="curves"]' : `[data-ctl="${c.key}"]`, 1500)
    || await waitFor(`details:has([data-ctl="${c.key}"]) > summary`, 300);
  if (!el) return;
  const det = el.closest('details');
  if (det && !det.open) {
    const sum = det.querySelector('summary');
    await toEl(sum, 380 * T, 0.2);
    await click();
    det.open = true;
    await sleep(120);
  }
  await reveal(el);
  el.classList.add('ghost-hit');
  const range = el.querySelector('input[type=range]');
  const pad = el.querySelector('.wheel-pad');
  if (range && typeof c.after === 'number') {
    const r = range.getBoundingClientRect();
    const lo = parseFloat(range.min); const hi = parseFloat(range.max);
    const px = (v) => r.left + clamp((v - lo) / (hi - lo), 0, 1) * r.width;
    const from = typeof c.before === 'number' ? c.before : parseFloat(range.value);
    await moveTo(px(from), r.top + r.height / 2, 420 * T);
    cursor.classList.add('down');
    await moveTo(px(c.after), r.top + r.height / 2, 700 * T);
    cursor.classList.remove('down');
  } else if (pad && Array.isArray(c.after)) {
    const [x0, y0] = ballAt(pad, Array.isArray(c.before) ? c.before : JSON.parse(pad.dataset.neutral));
    const [x1, y1] = ballAt(pad, c.after);
    await moveTo(x0, y0, 420 * T);
    cursor.classList.add('down');
    await moveTo(x1, y1, 700 * T);
    cursor.classList.remove('down');
  } else {
    await toEl(el, 450 * T, 0.5, 0.35);
    await click();
  }
  setTimeout(() => el.classList.remove('ghost-hit'), 1400);
}

async function playOne(ev, ui, backlog) {
  ensure();
  clearTimeout(hideTimer);
  cursor.classList.add('on');
  const T = backlog > 2 ? 0.4 : 1;          // falling behind the assistant: shorter moves
  say(ev.title, ev.reason);
  ui.feed(ev);
  if (!ev.key) ui.clearKey();
  if (ev.shot_id) {
    const row = document.querySelector(`[data-shot="${ev.shot_id}"]`);
    if (row && !row.classList.contains('sel')) {
      await toEl(row, 450 * T, 0.3);
      await click();
    }
    ui.select(ev.shot_id, ev.t);
    await sleep(120);
  }
  if (ev.point) {
    const stage = await waitFor('.viewer .stage');
    if (stage) {
      const r = stage.getBoundingClientRect();
      await moveTo(r.left + r.width * ev.point.x, r.top + r.height * ev.point.y, 550 * T);
      await click();
      await sleep(250 * T);
    }
  }
  if (ev.controls && ev.controls.length) {
    ui.openColor(ev.target || 'look');
    await sleep(160);
    const many = ev.controls.length > 3 ? 0.6 : 1;
    for (const c of ev.controls) await playControl(c, T * many);
  }
  if (ev.key) {
    ui.showKey(ev.key);
    const stage = await waitFor('.viewer .stage');
    if (stage) await toEl(stage, 500 * T, 0.5, 0.5);
  }
  if (ev.area) {
    if (ev.area === 'viewer') {
      const stage = await waitFor('.viewer .stage');
      if (stage) await toEl(stage, 500 * T, 0.55, 0.5);
    } else {
      ui.openGrade();
      const el = await waitFor(`[data-ctl="${ev.area}"]`, 1500);
      if (el) { await toEl(el, 450 * T, 0.3); await click(); }
    }
  }
  await sleep(350 * T);
  await ui.refresh();                         // the picture changes after the gesture, the way a person sees it
  await sleep(300 * T);
  hideTimer = setTimeout(() => cursor && cursor.classList.remove('on'), 6000);
}

export function createPlayer(getUi) {
  const queue = [];
  let busy = false;
  const run = async () => {
    if (busy) return;
    busy = true;
    while (queue.length) {
      const ev = queue.shift();
      try { await playOne(ev, getUi(), queue.length); } catch (e) { /* a step the page cannot show: keep going */ }
    }
    busy = false;
  };
  return { push: (evs) => { queue.push(...evs); run(); }, busy: () => busy };
}

export function AssistantFeed({ items }) {
  if (!items || !items.length) return null;
  const now = Date.now();
  return html`<div class="ghost-feed">${items.filter((e) => now - e.seen < 15000).map((e) => html`<div key=${e.n}>
    <span>${e.title}</span>${e.reason ? html`<small>${e.reason}</small>` : ''}</div>`)}</div>`;
}
