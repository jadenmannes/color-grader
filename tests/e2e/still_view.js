// Browser test of the full-quality still: full resolution, before and after, split, 100% zoom, save, close.
// Run through tests/e2e/run.sh on the graded project (640 x 360 source).
const { chromium } = require('playwright');

const [url, shotDir] = process.argv.slice(2);
const assert = (cond, msg) => { if (!cond) { console.error('FAIL: ' + msg); process.exit(1); } };

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForSelector('.shot');
  await page.click('.shot >> nth=0');
  await page.waitForTimeout(600);

  // the preview asks for the size the viewer needs, never more than the source
  const src = await page.getAttribute('.viewer .stage img.after', 'src');
  assert(/[?&]w=640(&|$)/.test(src), 'preview width follows the viewer, capped at the source: ' + src);

  // the picture fits the viewer with the frame's own shape: never cropped, never stretched
  const fit = await page.evaluate(() => {
    const v = document.querySelector('.viewer').getBoundingClientRect();
    const i = document.querySelector('.viewer .stage img').getBoundingClientRect();
    return { vw: v.width, vh: v.height, iw: i.width, ih: i.height };
  });
  assert(fit.iw <= fit.vw + 0.5 && fit.ih <= fit.vh + 0.5, 'the picture fits the viewer: ' + JSON.stringify(fit));
  assert(Math.abs(fit.iw / fit.ih - 16 / 9) < 0.01, 'the picture keeps 16:9: ' + JSON.stringify(fit));

  // the Full screen button puts the viewer on the whole screen, with a way out
  await page.click('button:has-text("Full screen")');
  await page.waitForFunction(() => !!document.fullscreenElement);
  await page.waitForSelector('.fs-exit');
  await page.click('.fs-exit');
  await page.waitForFunction(() => !document.fullscreenElement);

  // 1. the button opens the still at the source's full resolution
  await page.click('button:has-text("Full quality")');
  await page.waitForSelector('.still .still-pic img');
  await page.waitForFunction(() => { const i = document.querySelector('.still .still-pic img'); return i && i.complete && i.naturalWidth > 0; });
  const nat = await page.evaluate(() => { const i = document.querySelector('.still .still-pic img'); return [i.naturalWidth, i.naturalHeight]; });
  assert(nat[0] === 640 && nat[1] === 360, 'full resolution: ' + nat);
  await page.waitForSelector('.still-title:has-text("640 × 360")');
  assert((await page.getAttribute('.still a:has-text("Save PNG")', 'href')).includes('download=1'), 'save link');

  // 2. before and after, then split
  await page.waitForSelector('.still .seg button:has-text("Split"):not([disabled])', { timeout: 15000 });
  const afterSrc = await page.getAttribute('.still .still-pic img', 'src');
  await page.keyboard.press('b');
  await page.waitForSelector('.still .seg button.on:has-text("Source")');
  assert((await page.getAttribute('.still .still-pic img', 'src')) !== afterSrc, 'B shows the source frame');
  await page.keyboard.press('b');
  await page.keyboard.press('s');
  await page.waitForSelector('.still .still-before');
  await page.waitForSelector('.still .still-handle');
  await page.waitForTimeout(300);
  if (shotDir) await page.screenshot({ path: shotDir + '/e2e_still_split.png' });

  // 3. 100%: one image pixel per screen pixel
  await page.keyboard.press('s');
  await page.keyboard.press('z');
  await page.waitForSelector('.still-stage.full');
  const w = await page.evaluate(() => document.querySelector('.still .still-pic img').getBoundingClientRect().width);
  assert(Math.abs(w - 640) < 1, '100% shows the frame pixel for pixel: ' + w);

  // 4. Esc closes; a double-click on the picture and the F key open it again
  await page.keyboard.press('Escape');
  if (await page.isVisible('.still')) await page.keyboard.press('Escape');   // the first one may only leave full screen
  await page.waitForSelector('.still', { state: 'detached' });
  await page.dblclick('.viewer .stage');
  await page.waitForSelector('.still .still-pic img');
  await page.keyboard.press('Escape');
  if (await page.isVisible('.still')) await page.keyboard.press('Escape');
  await page.waitForSelector('.still', { state: 'detached' });
  await page.keyboard.press('f');
  await page.waitForSelector('.still .still-pic img');
  assert(errors.length === 0, 'no page errors: ' + errors.join(' | '));
  console.log('PASS: full-quality still opens full size, compares before and after, zooms to 100% and closes');
  await browser.close();
})().catch((e) => { console.error('FAIL: ' + e.message); process.exit(1); });
