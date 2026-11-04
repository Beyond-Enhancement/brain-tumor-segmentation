import fs from 'fs';
import { resizeSeparateZ } from '../../app/js/lib/geom.js';
const [inp, nx, ny, nz, NX, NY, NZ, zAxis, order, orderZ, outp] = process.argv.slice(2);
const b = fs.readFileSync(inp);
const d = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
const r = resizeSeparateZ(d, [+nx, +ny, +nz], [+NX, +NY, +NZ], +zAxis, +order, +orderZ);
fs.writeFileSync(outp, Buffer.from(r.buffer));
