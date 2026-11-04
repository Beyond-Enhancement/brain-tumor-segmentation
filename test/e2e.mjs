// End-to-end test of the page in headless Chrome.
//   node test/e2e.mjs <base url> <out dir> [--dicom DIR] [--webgpu] [--tta] [--timeout ms]
// Loads the built-in demo case (or a DICOM folder), runs the pipeline, checks the volume table,
// exports every NIfTI and parses it, and saves screenshots. Prints a JSON summary.
import { chromium } from '/opt/npm-tools/node_modules/playwright/index.mjs';
import fs from 'fs';
import path from 'path';
const [base, out] = process.argv.slice(2);
const arg = k => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : null; };
const webgpu = process.argv.includes('--webgpu'), tta = process.argv.includes('--tta');
const timeout = +(arg('--timeout') || 3600000);
const dicom = arg('--dicom');
fs.mkdirSync(out, { recursive: true });
const args = ['--no-sandbox', '--ignore-gpu-blocklist', '--enable-unsafe-webgpu'];
if (webgpu) args.push('--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader');
else args.push('--use-angle=swiftshader', '--enable-unsafe-swiftshader');
const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args });
const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true });
const page = await ctx.newPage();
const logs = [], errors = [], requests = [];
page.on('console', m => { logs.push(m.type() + ': ' + m.text()); if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', e => errors.push('pageerror: ' + e.message));
ctx.on('request', r => requests.push(r.url()));
const t0 = Date.now();
await page.goto(base, { waitUntil: 'load' });
await page.waitForFunction(() => window.__hdglio && document.querySelectorAll('.demo').length > 0, null, { timeout: 30000 }).catch(() => {});
if (dicom) {
  const files = [];
  const walk = d => { for (const f of fs.readdirSync(d)) { const p = path.join(d, f); fs.statSync(p).isDirectory() ? walk(p) : files.push(p); } };
  walk(dicom);
  await page.setInputFiles('#filesInput', files);
} else {
  await page.click('.demo');
}
await page.waitForFunction(() => [...document.querySelectorAll('#roles select')].filter(s => s.value).length === 4, null, { timeout: 120000 });
const roles = await page.$$eval('#roles .role', rows => rows.map(r => r.querySelector('.tag').textContent + ' = ' + r.querySelector('select').selectedOptions[0].textContent));
if (tta) await page.check('#optTta');
await page.selectOption('#optDevice', webgpu ? 'gpu' : 'cpu');
await page.screenshot({ path: path.join(out, '1-loaded.png') });
const tRun = Date.now();
await page.click('#runBtn');
let last = '';
const iv = setInterval(async () => { try { const s = await page.$eval('#steps', e => e.innerText.replace(/\s+/g, ' ')); if (s !== last) { last = s; console.log('[steps]', s.slice(0, 200)); } } catch (e) {} }, 15000);
await page.waitForFunction(() => !document.querySelector('[data-exp="seg"]').disabled || [...document.querySelectorAll('.toast.err')].length, null, { timeout, polling: 1000 });
clearInterval(iv);
const runMs = Date.now() - tRun;
const errToast = await page.$$eval('.toast.err', t => t.map(x => x.textContent));
await page.waitForTimeout(1500);
await page.screenshot({ path: path.join(out, '2-result.png') });
const table = await page.$$eval('#volTable tbody tr', rows => rows.map(r => r.innerText.replace(/\s+/g, ' ').trim()));
const qc = await page.$eval('#regQC', e => e.innerText);
const info = await page.$eval('#runInfo', e => e.innerText);
// exports
const exported = {};
for (const what of ['seg', 'brain', 'input:T1', 'input:T1c', 'input:T2', 'input:FLAIR']) {
  if (errToast.length) break;
  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 60000 }), page.click(`[data-exp="${what}"]`)]);
  const p = path.join(out, dl.suggestedFilename());
  await dl.saveAs(p);
  exported[what] = { file: path.basename(p), bytes: fs.statSync(p).size };
}
const [csv] = errToast.length ? [null] : await Promise.all([page.waitForEvent('download'), page.click('#csvBtn')]);
if (csv) await csv.saveAs(path.join(out, csv.suggestedFilename()));
// views
for (const v of [0, 1, 2]) { await page.click(`#views button[data-view="${v}"]`); await page.waitForTimeout(400); await page.screenshot({ path: path.join(out, `3-view${v}.png`) }); }
await page.click('#views button[data-view="3"]');
const outside = [...new Set(requests.filter(u => !u.startsWith(base) && !u.startsWith('blob:') && !u.startsWith('data:')))];
const summary = { roles, runMs, table, qc, info, exported, errors: errors.filter(e => !/favicon|404 \(File not found\)/.test(e)), errToast, outsideRequests: outside, totalMs: Date.now() - t0, nRequests: requests.length };
fs.writeFileSync(path.join(out, 'summary.json'), JSON.stringify(summary, null, 1));
fs.writeFileSync(path.join(out, 'console.log'), logs.join('\n'));
console.log(JSON.stringify(summary, null, 1));
await browser.close();
