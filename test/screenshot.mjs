// README screenshot: run the demo case with default settings in a fresh browser profile and
// capture the whole window once the result is shown (after the notice has gone).
//   node test/screenshot.mjs <base url> <out.png> [--model ID[,ID...]]
import { chromium } from '/opt/npm-tools/node_modules/playwright/index.mjs';
const [base, out] = process.argv.slice(2);
const mi = process.argv.indexOf('--model');
const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await (await browser.newContext({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 })).newPage();
await page.goto(base);
await page.waitForFunction(() => document.querySelectorAll('.demo').length > 0 && document.querySelectorAll('#modelList input').length > 0);
if (mi > 0) {
  // check exactly these models
  const want = process.argv[mi + 1].split(',');
  await page.waitForFunction(w => w.every(id => document.querySelector(`#modelList input[value="${id}"]`)), want, { timeout: 60000 });
  for (const box of await page.$$('#modelList input[name="model"]')) if ((await box.isChecked()) !== want.includes(await box.getAttribute('value'))) await box.click();
}
await page.click('.demo');
await page.waitForFunction(() => [...document.querySelectorAll('#roles select')].filter(s => s.value).length === 4, null, { timeout: 120000 });
await page.selectOption('#optDevice', 'cpu');
await page.click('#runBtn');
await page.waitForFunction(() => !document.querySelector('#copyBtn').disabled || document.querySelector('.toast.err'), null, { timeout: 3600000, polling: 2000 });
await page.waitForFunction(() => !document.querySelector('.toast'), null, { timeout: 30000 });
await page.waitForTimeout(800);
await page.screenshot({ path: out });
console.log(JSON.stringify({ table: await page.$$eval('#volTable tbody tr', r => r.map(x => x.innerText.replace(/\s+/g, ' ').trim())), err: await page.$$eval('.toast.err', t => t.map(x => x.textContent)) }));
await browser.close();
