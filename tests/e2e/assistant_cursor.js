// Browser test of the assistant cursor: steps reported the way the MCP server reports them show up on the page.
// Run through tests/e2e/run.sh after color_panel.js, on the same graded project.
const { chromium } = require('playwright');

const [url, shotDir] = process.argv.slice(2);
const assert = (cond, msg) => { if (!cond) { console.error('FAIL: ' + msg); process.exit(1); } };

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const state = async () => (await (await page.request.get(url + 'api/state')).json()).project;
  const step = (ev) => page.request.post(url + 'api/activity', { data: ev });
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForSelector('.shot');
  await page.waitForTimeout(800);
  const proj = await state();
  const shot = proj.shots[1];
  const before = proj.look.temp;

  // 1. a look change: the cursor opens the Color tab, drags Temperature, then the panel shows the new value
  await page.request.fetch(url + 'api/grade/look', { method: 'PATCH', data: { changes: { temp: 0.6 }, recheck: false } });
  await step({ tool: 'adjust_look', title: 'Look: temperature +0.6', reason: 'Warmer autumn light', target: 'look',
    controls: [{ key: 'temp', before, after: 0.6 }] });
  await page.waitForSelector('.ghost-cursor.on', { timeout: 4000 });
  assert((await page.textContent('.ghost-label')).includes('temperature +0.6'), 'the cursor says what it does');
  await page.waitForSelector('[data-ctl="temp"].ghost-hit', { timeout: 4000 });
  assert(await page.isVisible('.col.right button.tab.active:has-text("Color")'), 'the Color tab opens');
  await page.waitForSelector('.ghost-cursor.down', { timeout: 3000 });
  if (shotDir) await page.screenshot({ path: shotDir + '/e2e_assistant_drag.png' });
  await page.waitForFunction(() => {
    const n = document.querySelector('[data-ctl="temp"] input.ctl-num');
    return n && Math.abs(parseFloat(n.value) - 0.6) < 1e-6;
  }, null, { timeout: 6000 });
  assert(await page.isVisible('.ghost-feed >> text=Warmer autumn light'), 'the feed lists the step with its reason');

  // 2. a color sample on another shot: the cursor selects the shot and clicks the point
  await step({ tool: 'sample_color', title: 'Sampling the color here: sky', shot_id: shot.id, t: (shot.start + shot.end) / 2, point: { x: 0.5, y: 0.15 } });
  await page.waitForSelector('.ghost-ripple', { timeout: 6000 });
  await page.waitForSelector(`.shot.sel[data-shot="${shot.id}"]`, { timeout: 4000 });

  // 3. a key check: the viewer shows what the secondary selects
  await step({ tool: 'view_key', title: 'Checking what Sky selects', shot_id: shot.id, key: { target: 'look', index: 0, name: 'Sky' } });
  await page.waitForSelector('.viewer img[src*="/api/key"]', { timeout: 6000 });
  await page.waitForSelector('.viewer .tag:has-text("Key: Sky")');
  if (shotDir) { await page.waitForTimeout(600); await page.screenshot({ path: shotDir + '/e2e_assistant_key.png' }); }

  // 4. switched off: steps no longer move anything
  await page.uncheck('.transport input[type=checkbox]');
  await page.waitForTimeout(6800);
  await step({ tool: 'adjust_look', title: 'Look: contrast 1.1', target: 'look', controls: [{ key: 'contrast', before: 1, after: 1.1 }] });
  await page.waitForTimeout(1500);
  assert(!(await page.isVisible('.ghost-cursor.on')), 'no cursor with Show assistant off');
  assert(errors.length === 0, 'no page errors: ' + errors.join(' | '));
  console.log('PASS: the assistant cursor shows each MCP step on the page');
  await browser.close();
})().catch((e) => { console.error('FAIL: ' + e.message); process.exit(1); });
