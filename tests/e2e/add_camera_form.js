// Browser test: add a camera through the real form, with no code edits.
// Run through tests/e2e/run.sh, which starts the server on a fresh user folder.
const { chromium } = require('playwright');

const [url, clipPath, shotDir] = process.argv.slice(2);
const assert = (cond, msg) => { if (!cond) { console.error('FAIL: ' + msg); process.exit(1); } };

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(url, { waitUntil: 'networkidle' });

  // 1. load the clip: the camera is unknown, so the add-camera form opens, pre-filled from metadata
  await page.fill('input[placeholder="or paste a file path"]', clipPath);
  await page.keyboard.press('Enter');
  await page.waitForSelector('.modal h2:has-text("Add camera")');
  const name = await page.inputValue('.modal input >> nth=0');
  assert(name === 'SONY ILCE-7SM3', `camera name pre-filled from metadata, got "${name}"`);
  const meta = await page.inputValue('.modal input >> nth=2');
  assert(meta === 'ILCE-7SM3', `metadata name pre-filled, got "${meta}"`);

  // 2. saving stays locked until a sample conversion is on screen
  assert(await page.isDisabled('.modal button:has-text("Save camera")'), 'save is locked before the sample conversion');

  // 3. pick curve and gamut from the lists
  await page.selectOption('.modal .profile-body select >> nth=2', { label: 'Sony S-Log3' });
  await page.selectOption('.modal .profile-body select >> nth=3', { label: 'Sony S-Gamut3.Cine' });
  await page.fill('.modal .profile-body input[type=text] >> nth=1', '');   // transfer tag: any

  // 4. convert the sample and look at original next to converted
  await page.click('.modal button:has-text("Convert sample")');
  await page.waitForSelector('.modal .compare img >> nth=1');
  await page.waitForTimeout(800);
  if (shotDir) await page.screenshot({ path: shotDir + '/e2e_camera_form.png' });
  assert(!(await page.isDisabled('.modal button:has-text("Save camera")')), 'save unlocks after the conversion');

  // 5. save: the clip matches the new camera and analysis starts
  await page.click('.modal button:has-text("Save camera")');
  await page.waitForSelector('.modal', { state: 'detached' });
  await page.waitForSelector('.clip .badge.ok:has-text("SONY ILCE-7SM3")', { timeout: 15000 });
  const analyze = page.locator('.clip button:has-text("Analyze")');
  if (await analyze.count()) await analyze.first().click();
  await page.waitForSelector('.shot', { timeout: 60000 });
  await page.waitForTimeout(1500);
  if (shotDir) await page.screenshot({ path: shotDir + '/e2e_camera_project.png' });

  // 6. the camera sits in the library as data
  await page.click('button.tab:has-text("Cameras")');
  await page.waitForSelector('.list-item:has-text("SONY ILCE-7SM3")');
  assert(errors.length === 0, 'no page errors: ' + errors.join(' | '));
  console.log('PASS: camera added through the form, clip converted and analyzed');
  await browser.close();
})().catch((e) => { console.error('FAIL: ' + e.message); process.exit(1); });
