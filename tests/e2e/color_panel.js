// Browser test of the Color tab: every edit reaches the app's grade live.
// Run through tests/e2e/run.sh, which starts the server on a graded project.
const { chromium } = require('playwright');

const [url, shotDir] = process.argv.slice(2);
const assert = (cond, msg) => { if (!cond) { console.error('FAIL: ' + msg); process.exit(1); } };

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const state = async () => (await (await page.request.get(url + 'api/state')).json()).project;
  const settle = async (ms = 900) => { await page.waitForTimeout(ms); await page.waitForLoadState('networkidle'); };
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForSelector('.shot');
  await page.click('.shot >> nth=0');
  await page.click('.col.right button.tab:has-text("Color")');
  await page.waitForSelector('.cgroup summary:has-text("Exposure and white balance")');

  // 1. a number typed into a control lands in the look
  const temp = page.locator('.ctl:has(.ctl-label:text-is("Temperature")) input.ctl-num');
  await temp.fill('0.3');
  await temp.press('Enter');
  await settle();
  assert((await state()).look.temp === 0.3, 'temperature reaches the look');

  // 2. dragging the Lift ball to the right moves shadows toward red
  const pad = page.locator('.wheel:has(.wheel-title:text-is("Lift")) .wheel-pad');
  await pad.scrollIntoViewIfNeeded();
  const box = await pad.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 25, box.y + box.height / 2, { steps: 5 });
  await page.mouse.up();
  await settle();
  const lift = (await state()).look.lift;
  assert(lift[0] > 0.005 && lift[0] > lift[1] && lift[0] > lift[2], 'lift moves toward red: ' + JSON.stringify(lift));

  // 3. a curve point added and dragged in Hue vs Sat
  await page.click('.cgroup summary:has-text("Curves")');
  await page.selectOption('.cgroup select', 'hue_vs_sat');
  const svg = page.locator('svg.curve-svg');
  await svg.scrollIntoViewIfNeeded();
  const sb = await svg.boundingBox();
  await page.mouse.move(sb.x + sb.width * (214 / 360), sb.y + sb.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(sb.x + sb.width * (214 / 360), sb.y + sb.height * 0.3, { steps: 5 });
  await page.mouse.up();
  await settle();
  const hvs = (await state()).look.hue_vs_sat;
  assert(hvs.some(([h, v]) => Math.abs(h - 214) < 3 && v > 1.2), 'hue vs sat point lifts blue: ' + JSON.stringify(hvs));

  // 4. a sky secondary from the preset, with its key on the viewer
  await page.click('.cgroup summary:has-text("Secondaries")');
  await page.click('button:has-text("+ Sky")');
  await settle(1500);
  let look = (await state()).look;
  assert(look.secondaries.length === 1 && look.secondaries[0].name === 'Sky', 'sky secondary added');
  await page.waitForSelector('.viewer img[src*="/api/key"]');
  await page.waitForFunction(() => { const i = document.querySelector('.viewer img[src*="/api/key"]'); return i && i.complete && i.naturalWidth > 0; });
  if (shotDir) await page.screenshot({ path: shotDir + '/e2e_color_key.png' });
  const sat = page.locator('.sec-card .ctl:has(.ctl-label:text-is("Saturation")) input.ctl-num');
  await sat.fill('0.7');
  await sat.press('Enter');
  await settle();
  assert((await state()).look.secondaries[0].adjust.saturation === 0.7, 'secondary saturation reaches the look');

  // 5. pick a color from the viewer: the qualifier takes the picked hue
  await page.click('.sec-card button:has-text("Pick color")');
  await page.waitForSelector('.viewer .stage.picking');
  const st = await page.locator('.viewer .stage').boundingBox();
  await page.mouse.click(st.x + st.width * 0.3, st.y + st.height * 0.4);
  await settle(1500);
  look = (await state()).look;
  assert(look.secondaries[0].lum && look.secondaries[0].lum.length === 3, 'picked color sets the qualifier');
  await page.waitForSelector('.viewer img[src*="/api/key"]');

  // 6. this shot only: a trim that the other shots do not get
  await page.click('.colors .seg button:has-text("This shot only")');
  const exp = page.locator('.ctl:has(.ctl-label:text-is("Exposure")) input.ctl-num >> nth=0');
  await exp.fill('0.5');
  await exp.press('Enter');
  await settle(1800);
  const proj = await state();
  const shot = proj.shots[0];
  assert(shot.grade.trim.exposure === 0.5, 'shot trim exposure: ' + JSON.stringify(shot.grade.trim));
  assert(proj.look.exposure === 0, 'the look keeps its own exposure');
  if (shotDir) {
    await page.click('.colors .seg button:has-text("Look: every shot")');
    await page.waitForTimeout(700);
    await page.click('.cgroup summary:has-text("HDR zones")');
    await page.waitForTimeout(900);
    await page.screenshot({ path: shotDir + '/e2e_color_panel.png' });
  }
  // 7. the Grade tab lists the new controls and the shot's own trim
  await page.click('.col.right button.tab:has-text("Grade")');
  await page.waitForSelector('table.plain td:text-is("Temperature")');
  await page.waitForSelector('table.plain td:text-is("Own trim")');
  assert(await page.isVisible('text=1 secondary (Sky)'), 'the grade summary names the secondary');
  assert(errors.length === 0, 'no page errors: ' + errors.join(' | '));
  console.log('PASS: Color tab edits the look, a shot trim, curves and secondaries live');
  await browser.close();
})().catch((e) => { console.error('FAIL: ' + e.message); process.exit(1); });
