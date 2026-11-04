// node test/unit/resize_test.mjs <in.raw> nx ny nz NX NY NZ ox oy oz <out.raw>
import fs from 'fs';
import { resizeVolume } from '../../app/js/lib/geom.js';
const [inp, nx, ny, nz, NX, NY, NZ, ox, oy, oz, outp] = process.argv.slice(2);
const b = fs.readFileSync(inp);
const d = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
const t = performance.now();
const r = resizeVolume(d, [+nx, +ny, +nz], [+NX, +NY, +NZ], [+ox, +oy, +oz], true);
console.error('js ms', (performance.now() - t).toFixed(0));
fs.writeFileSync(outp, Buffer.from(r.buffer));
