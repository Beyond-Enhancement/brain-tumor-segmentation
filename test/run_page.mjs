// node test/run_page.mjs URL [--webgpu] [--timeout ms]: open a page in headless Chrome and print console until "DONE"
import { chromium } from '/opt/npm-tools/node_modules/playwright/index.mjs';
const url = process.argv[2];
const webgpu = process.argv.includes('--webgpu');
const ti = process.argv.indexOf('--timeout'); const timeout = ti > 0 ? +process.argv[ti + 1] : 600000;
const args = ['--no-sandbox', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist'];
if (webgpu) args.push('--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader');
const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args });
const page = await browser.newPage();
let done;
const finished = new Promise(r => (done = r));
page.on('console', m => { const t = m.text(); console.log('[page]', t); if (t.includes('DONE')) done(); });
page.on('pageerror', e => console.log('[pageerror]', e.message));
await page.goto(url);
await Promise.race([finished, new Promise(r => setTimeout(r, timeout))]);
await browser.close();
