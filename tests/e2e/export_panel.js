// Browser test of the Export panel: an export shows its files, a failed export keeps its reason on screen,
// both stay after a page reload, and a reload during a running export brings its progress bar back.
// Run through tests/e2e/run.sh on the graded project (640 x 360 source).
const { chromium } = require('playwright');

const [url, shotDir] = process.argv.slice(2);
const assert = (cond, msg) => { if (!cond) { console.error('FAIL: ' + msg); process.exit(1); } };

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const openExport = async () => {
    await page.goto(url, { waitUntil: 'load' });
    await page.waitForSelector('.shot');
    await page.click('.shot >> nth=0');
    await page.click('.col.right button.tab:has-text("Export")');
    await page.waitForSelector('button:has-text("Export this clip")');
  };
  const idle = async () => {
    for (let i = 0; i < 240; i++) {
      const st = await (await page.request.get(url + 'api/state')).json();
      if (!st.jobs.length) return;
      await page.waitForTimeout(500);
    }
    assert(false, 'the jobs finish');
  };

  // 1. an export writes its file and the panel lists it, with the time it took
  await openExport();
  await page.click('button:has-text("Export this clip")');
  await page.waitForSelector('.card:has-text("1 file written")', { timeout: 180000 });
  assert(await page.isVisible('.card:has-text("to render")'), 'the file card says how long the render took');

  // 2. after a reload the panel still shows the newest export
  await openExport();
  await page.waitForSelector('.card:has-text("1 file written")');

  // 3. a failed export keeps its reason on the panel, also after a reload
  const bad = (await page.request.put(url + 'api/settings', { data: { export_folder: '/dev/null/exports' } })).ok();
  assert(bad, 'set an export folder that cannot exist');
  await page.click('button:has-text("Export this clip")');
  await page.waitForSelector('.export-error:has-text("The export failed")', { timeout: 60000 });
  await page.waitForSelector('.export-error:has-text("cannot be made")');
  await openExport();
  await page.waitForSelector('.export-error:has-text("The export failed")');
  if (shotDir) await page.screenshot({ path: shotDir + '/e2e_export_error.png' });
  await page.request.put(url + 'api/settings', { data: { export_folder: '' } });

  // 4. a reload during an export brings the progress bar back: two exports queue, the page reloads at once
  const st = (await (await page.request.get(url + 'api/state')).json()).project;
  const ids = [st.clips[0].id];
  for (let i = 0; i < 2; i++) await page.request.post(url + 'api/export/video', { data: { clip_ids: ids, format: 'h265-10bit-mp4' } });
  await openExport();
  await page.waitForSelector('.jobbar', { timeout: 10000 });
  const barW = await page.evaluate(() => document.querySelector('.jobbar .progress').getBoundingClientRect().width);
  assert(barW >= 100, 'the top progress bar keeps its width next to a long message: ' + barW);
  await page.waitForSelector('.export-run, .card:has-text("1 file written")', { timeout: 10000 });
  if (shotDir) await page.screenshot({ path: shotDir + '/e2e_export_running.png' });
  await idle();
  await page.waitForSelector('.jobbar', { state: 'detached', timeout: 20000 });
  await page.waitForSelector('.card:has-text("1 file written")', { timeout: 20000 });

  assert(errors.length === 0, 'no page errors: ' + errors.join(' | '));
  console.log('PASS: the Export panel lists files, keeps failures on screen, and follows an export across a reload');
  await browser.close();
})().catch((e) => { console.error('FAIL: ' + e.message); process.exit(1); });
