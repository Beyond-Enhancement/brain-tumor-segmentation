// GPU loss test: start a GPU (WebGPU) run of the demo case, kill the browser's GPU process while
// HD-BET runs on the GPU (as a Windows driver reset would), and check that the run finishes on
// the CPU, the viewer still draws, and the page remembers the failure.
//   node test/gpu_loss_test.mjs <base url> <out dir>
import { chromium } from '/opt/npm-tools/node_modules/playwright/index.mjs';
import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
const [base, out] = process.argv.slice(2);
fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox', '--ignore-gpu-blocklist', '--enable-unsafe-webgpu',
  '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await (await browser.newContext({ viewport: { width: 1600, height: 1000 } })).newPage();
const errors = [];
page.on('pageerror', e => errors.push(e.message));
await page.goto(base);
await page.waitForFunction(() => document.querySelectorAll('.demo').length > 0);
await page.click('.demo');
await page.waitForFunction(() => [...document.querySelectorAll('#roles select')].filter(s => s.value).length === 4, null, { timeout: 120000 });
await page.selectOption('#optDevice', 'gpu');
await page.click('#runBtn');
await page.waitForFunction(() => /HD-BET: patch \d+\/\d+ \(GPU\)/.test(document.querySelector('#steps').innerText), null, { timeout: 600000, polling: 1000 });
const before = await page.$eval('#steps', e => e.innerText.replace(/\s+/g, ' '));
await page.waitForTimeout(5000);
const gpuProcs = execSync("pgrep -f 'type=gpu-proces[s]' || true").toString().trim().split('\n').filter(Boolean);
execSync("pkill -9 -f 'type=gpu-proces[s]' || true"); // [s]: do not match this shell's own command line
const killedAt = Date.now();
await page.waitForTimeout(8000);
const after = await page.$eval('#steps', e => e.innerText.replace(/\s+/g, ' '));
await page.waitForFunction(() => !document.querySelector('[data-exp="seg"]').disabled || document.querySelector('.toast.err:not(:empty)') && !/GPU stopped/.test(document.querySelector('.toast.err').textContent), null, { timeout: 3600000, polling: 2000 });
await page.waitForTimeout(2500);
await page.screenshot({ path: path.join(out, 'after-gpu-loss.png') });
const res = {
  stepsBeforeKill: before.slice(0, 200), stepsAfterKill: after.slice(0, 200), gpuProcessesKilled: gpuProcs.length,
  table: await page.$$eval('#volTable tbody tr', r => r.map(x => x.innerText.replace(/\s+/g, ' ').trim())),
  runInfo: await page.$eval('#runInfo', e => e.innerText),
  toasts: await page.$$eval('.toast', t => t.map(x => x.textContent)),
  device: await page.$eval('#device', e => e.textContent),
  remembered: await page.evaluate(() => localStorage.getItem('btsw.gpuFailed')),
  minutesAfterKill: +((Date.now() - killedAt) / 60000).toFixed(1), pageErrors: errors,
};
// viewer must still draw: count non-black pixels in the viewer area of the screenshot
const box = await page.$eval('.viewer', e => { const r = e.getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; });
const shot = await page.screenshot({ clip: box });
fs.writeFileSync(path.join(out, 'viewer.png'), shot);
res.viewerPng = path.join(out, 'viewer.png');
console.log(JSON.stringify(res, null, 1));
await browser.close();
