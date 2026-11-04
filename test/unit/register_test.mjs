// Registration recovery test: register a sequence to T1, then perturb its header by a known
// rigid transform and check the registration recovers it.
import fs from 'fs';
import { readNifti } from '../../app/js/lib/nifti.js';
import { mul4, percentile } from '../../app/js/lib/geom.js';
import { registerRigid, rigidMatrix } from '../../app/js/lib/register.js';
const D = process.argv[2], fixedName = process.argv[3], movName = process.argv[4];
const fixed = readNifti(fs.readFileSync(D + fixedName));
const mov = readNifti(fs.readFileSync(D + movName));
const thr = percentile(fixed.data, 99, null) * 0.1;
const mask = new Uint8Array(fixed.data.length); for (let i = 0; i < mask.length; i++) mask[i] = fixed.data[i] > thr ? 1 : 0;
let t = performance.now();
const r1 = registerRigid(fixed, mov, mask);
console.log('base   ', ((performance.now() - t) / 1000).toFixed(1) + 's', 'nmi', r1.nmi.toFixed(4), 't', r1.params.t.map(v => v.toFixed(2)), 'r', r1.params.rDeg.map(v => v.toFixed(2)));
const [nx, ny] = fixed.dims, A = fixed.affine;
const pts = []; for (let v = 0; v < mask.length; v += 37) if (mask[v]) { const i = v % nx, j = Math.floor(v / nx) % ny, k = Math.floor(v / (nx * ny)); pts.push([A[0]*i+A[1]*j+A[2]*k+A[3], A[4]*i+A[5]*j+A[6]*k+A[7], A[8]*i+A[9]*j+A[10]*k+A[11]]); }
const app = (M, p) => [0, 1, 2].map(r => M[4*r]*p[0] + M[4*r+1]*p[1] + M[4*r+2]*p[2] + M[4*r+3]);
const dist = (M1, M2) => { let s = 0, mx = 0; for (const p of pts) { const a = app(M1, p), b = app(M2, p); const d = Math.hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2]); s += d; if (d > mx) mx = d; } return [s / pts.length, mx]; };
for (const [tx, ty, tz, rx, ry, rz] of [[5, -8, 3, 6, 0, 4], [-10, 4, -6, -4, 5, -7], [2, 2, 12, 0, -8, 0]]) {
  const P = rigidMatrix([tx, ty, tz, rx * Math.PI / 180, ry * Math.PI / 180, rz * Math.PI / 180], r1.center);
  const movP = { ...mov, affine: mul4(P, mov.affine) };
  t = performance.now();
  const r2 = registerRigid(fixed, movP, mask);
  const [mean, max] = dist(r2.T, mul4(P, r1.T));
  console.log('perturb', [tx, ty, tz, rx, ry, rz].join(','), ((performance.now() - t) / 1000).toFixed(1) + 's', 'error mean ' + mean.toFixed(2) + ' mm, max ' + max.toFixed(2) + ' mm');
}
