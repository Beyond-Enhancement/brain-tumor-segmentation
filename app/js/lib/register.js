// Rigid (6 degrees of freedom) intra-subject registration by normalised mutual information,
// the in-browser counterpart of `flirt -dof 6` in the HD-GLIO preprocessing.
// Transform maps fixed-world (RAS mm) -> moving-world: x_m = R (x_f - c) + c + t.
import { mul4, inv4, spacing, isotropicGrid, resampleToGrid, percentile } from './geom.js';

export function rigidMatrix(p, c) {
  // p = [tx, ty, tz (mm), rx, ry, rz (radians)]
  const [tx, ty, tz, rx, ry, rz] = p;
  const cx = Math.cos(rx), sx = Math.sin(rx), cy = Math.cos(ry), sy = Math.sin(ry), cz = Math.cos(rz), sz = Math.sin(rz);
  // R = Rz * Ry * Rx
  const R = [
    cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
    sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
    -sy, cy * sx, cy * cx];
  const M = new Float64Array(16);
  for (let r = 0; r < 3; r++) {
    for (let k = 0; k < 3; k++) M[4 * r + k] = R[3 * r + k];
    M[4 * r + 3] = c[r] + [tx, ty, tz][r] - (R[3 * r] * c[0] + R[3 * r + 1] * c[1] + R[3 * r + 2] * c[2]);
  }
  M[15] = 1;
  return M;
}

function gaussianSmooth(data, dims, sigmaVox) {
  const out = new Float32Array(data);
  const [nx, ny, nz] = dims;
  const tmp = new Float32Array(Math.max(nx, ny, nz));
  for (let ax = 0; ax < 3; ax++) {
    const s = sigmaVox[ax];
    if (s < 0.3) continue;
    const rad = Math.ceil(3 * s);
    const w = []; let ws = 0;
    for (let i = -rad; i <= rad; i++) { const v = Math.exp(-0.5 * (i / s) ** 2); w.push(v); ws += v; }
    for (let i = 0; i < w.length; i++) w[i] /= ws;
    const n = dims[ax], st = ax === 0 ? 1 : ax === 1 ? nx : nx * ny;
    const o1 = (ax + 1) % 3, o2 = (ax + 2) % 3;
    const s1 = o1 === 0 ? 1 : o1 === 1 ? nx : nx * ny, s2 = o2 === 0 ? 1 : o2 === 1 ? nx : nx * ny;
    for (let b = 0; b < dims[o2]; b++) for (let a = 0; a < dims[o1]; a++) {
      const base = a * s1 + b * s2;
      for (let i = 0; i < n; i++) tmp[i] = out[base + i * st];
      for (let i = 0; i < n; i++) {
        let acc = 0;
        for (let k = -rad; k <= rad; k++) { let j = i + k; if (j < 0) j = 0; else if (j >= n) j = n - 1; acc += w[k + rad] * tmp[j]; }
        out[base + i * st] = acc;
      }
    }
  }
  return out;
}

export function downsample(vol, sp) {
  const g = isotropicGrid(vol, sp);
  const s = spacing(vol.affine);
  const sm = gaussianSmooth(vol.data, vol.dims, s.map(si => Math.max(0, (sp / si - 1) * 0.5)));
  return { dims: g.dims, affine: g.affine, data: resampleToGrid({ dims: vol.dims, affine: vol.affine, data: sm }, g.dims, g.affine, null, 'linear', 0) };
}

function binner(data, mask, nb) {
  const lo = percentile(data, 1, mask), hi = percentile(data, 99.5, mask);
  const sc = (nb - 1) / Math.max(1e-6, hi - lo);
  return v => { let b = (v - lo) * sc; return b < 0 ? 0 : b > nb - 1 ? nb - 1 : b; };
}

export class NMICost {
  constructor(fixed, fmask, moving, nb = 48, maxSamples = 120000, seed = 1) {
    this.nb = nb;
    const [nx, ny, nz] = fixed.dims;
    const idx = [];
    for (let i = 0; i < fixed.data.length; i++) if (fmask[i]) idx.push(i);
    // deterministic subsampling
    let s = seed >>> 0;
    const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    const keep = Math.min(idx.length, maxSamples);
    for (let i = idx.length - 1; i > 0 && idx.length > keep; i--) { const j = Math.floor(rnd() * (i + 1)); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    idx.length = keep;
    const fb = binner(fixed.data, fmask, nb);
    this.n = keep;
    this.pts = new Float64Array(3 * keep);
    this.fbin = new Uint8Array(keep);
    const A = fixed.affine;
    for (let q = 0; q < keep; q++) {
      const v = idx[q], i = v % nx, j = Math.floor(v / nx) % ny, k = Math.floor(v / (nx * ny));
      this.pts[3 * q] = A[0] * i + A[1] * j + A[2] * k + A[3];
      this.pts[3 * q + 1] = A[4] * i + A[5] * j + A[6] * k + A[7];
      this.pts[3 * q + 2] = A[8] * i + A[9] * j + A[10] * k + A[11];
      this.fbin[q] = Math.round(fb(fixed.data[v]));
    }
    // moving: pre-binned (fractional bins are not needed with PV interpolation)
    const mm = new Uint8Array(moving.data.length);
    const mmask = new Uint8Array(moving.data.length); for (let i = 0; i < mm.length; i++) mmask[i] = moving.data[i] > 0 ? 1 : 0;
    const mb = binner(moving.data, mmask, nb);
    for (let i = 0; i < mm.length; i++) mm[i] = Math.round(mb(moving.data[i]));
    this.mbin = mm; this.mdims = moving.dims; this.mInv = inv4(moving.affine);
    this.joint = new Float64Array(nb * nb);
  }
  eval(T) {
    const nb = this.nb, J = this.joint; J.fill(0);
    const M = mul4(this.mInv, T);
    const [mx, my, mz] = this.mdims, sxy = mx * my, mb = this.mbin, P = this.pts, fb = this.fbin;
    let tot = 0;
    for (let q = 0; q < this.n; q++) {
      const X = P[3 * q], Y = P[3 * q + 1], Zc = P[3 * q + 2];
      const x = M[0] * X + M[1] * Y + M[2] * Zc + M[3], y = M[4] * X + M[5] * Y + M[6] * Zc + M[7], z = M[8] * X + M[9] * Y + M[10] * Zc + M[11];
      if (x < 0 || y < 0 || z < 0 || x >= mx - 1 || y >= my - 1 || z >= mz - 1) continue;
      const i0 = x | 0, j0 = y | 0, k0 = z | 0, fx = x - i0, fy = y - j0, fz = z - k0;
      const b = i0 + mx * j0 + sxy * k0, row = fb[q] * nb;
      const w000 = (1 - fx) * (1 - fy) * (1 - fz), w100 = fx * (1 - fy) * (1 - fz), w010 = (1 - fx) * fy * (1 - fz), w110 = fx * fy * (1 - fz);
      const w001 = (1 - fx) * (1 - fy) * fz, w101 = fx * (1 - fy) * fz, w011 = (1 - fx) * fy * fz, w111 = fx * fy * fz;
      J[row + mb[b]] += w000; J[row + mb[b + 1]] += w100; J[row + mb[b + mx]] += w010; J[row + mb[b + mx + 1]] += w110;
      J[row + mb[b + sxy]] += w001; J[row + mb[b + sxy + 1]] += w101; J[row + mb[b + sxy + mx]] += w011; J[row + mb[b + sxy + mx + 1]] += w111;
      tot += 1;
    }
    if (tot < 0.25 * this.n) return 0; // too little overlap
    const pf = new Float64Array(nb), pm = new Float64Array(nb);
    let hj = 0;
    for (let a = 0; a < nb; a++) for (let c = 0; c < nb; c++) {
      const v = J[a * nb + c] / tot; if (v > 0) { hj -= v * Math.log(v); pf[a] += v; pm[c] += v; }
    }
    let hf = 0, hm = 0;
    for (let a = 0; a < nb; a++) { if (pf[a] > 0) hf -= pf[a] * Math.log(pf[a]); if (pm[a] > 0) hm -= pm[a] * Math.log(pm[a]); }
    return hj > 0 ? (hf + hm) / hj : 0;
  }
}

// Powell's method (minimisation) with golden-section line search.
function powell(f, x0, steps, { tol = 1e-3, maxIter = 6 } = {}) {
  const n = x0.length;
  let x = x0.slice(), fx = f(x);
  let dirs = steps.map((s, i) => { const d = new Array(n).fill(0); d[i] = s; return d; });
  const line = (x, d, f0) => {
    // bracket along d, then golden-section search on t in [-2, 2]
    const g = t => f(x.map((v, i) => v + t * d[i]));
    let a = -2, b = 2;
    const gr = (Math.sqrt(5) - 1) / 2;
    let c = b - gr * (b - a), e = a + gr * (b - a), fc = g(c), fe = g(e);
    for (let it = 0; it < 18 && Math.abs(b - a) > tol; it++) {
      if (fc < fe) { b = e; e = c; fe = fc; c = b - gr * (b - a); fc = g(c); }
      else { a = c; c = e; fc = fe; e = a + gr * (b - a); fe = g(e); }
    }
    const t = fc < fe ? c : e, ft = Math.min(fc, fe);
    if (ft < f0) return { x: x.map((v, i) => v + t * d[i]), f: ft, t };
    return { x, f: f0, t: 0 };
  };
  for (let iter = 0; iter < maxIter; iter++) {
    const xStart = x.slice(), fStart = fx;
    let bigDrop = 0, bigIdx = 0;
    for (let i = 0; i < n; i++) {
      const r = line(x, dirs[i], fx);
      if (fx - r.f > bigDrop) { bigDrop = fx - r.f; bigIdx = i; }
      x = r.x; fx = r.f;
    }
    if (fStart - fx < 1e-6 * Math.abs(fStart)) break;
    const dNew = x.map((v, i) => v - xStart[i]);
    const r = line(x, dNew, fx);
    if (r.t !== 0) { x = r.x; fx = r.f; dirs.splice(bigIdx, 1); dirs.push(dNew); }
  }
  return { x, f: fx };
}

// Register `moving` to `fixed`. fixedMask (Uint8Array on fixed grid) restricts the cost to the
// brain region. Returns { T, params, nmi, levels }.
export function registerRigid(fixed, moving, fixedMask, { levels = [6, 3, 1.5], onProgress } = {}) {
  // centre of rotation: centre of the fixed mask
  const [nx, ny] = fixed.dims;
  let cx = 0, cy = 0, cz = 0, cnt = 0;
  for (let v = 0; v < fixed.data.length; v++) if (fixedMask[v]) { cx += v % nx; cy += Math.floor(v / nx) % ny; cz += Math.floor(v / (nx * ny)); cnt++; }
  if (!cnt) throw new Error('Empty registration mask');
  const A = fixed.affine; cx /= cnt; cy /= cnt; cz /= cnt;
  const c = [A[0] * cx + A[1] * cy + A[2] * cz + A[3], A[4] * cx + A[5] * cy + A[6] * cz + A[7], A[8] * cx + A[9] * cy + A[10] * cz + A[11]];
  const R = 60; // mm: rotation parameters are scaled so 1 unit ~ 1 mm displacement at this radius
  const toMat = p => rigidMatrix([p[0], p[1], p[2], p[3] / R, p[4] / R, p[5] / R], c);
  const fMaskVol = { dims: fixed.dims, affine: fixed.affine, data: Float32Array.from(fixedMask) };
  const levelCost = sp => {
    const f = downsample(fixed, sp), m = downsample(moving, sp);
    const fm = resampleToGrid(fMaskVol, f.dims, f.affine, null, 'linear', 0);
    const fmu = new Uint8Array(fm.length); for (let i = 0; i < fm.length; i++) fmu[i] = fm[i] > 0.5 ? 1 : 0;
    const fine = sp <= levels[levels.length - 1];
    const cost = new NMICost(f, fmu, m, fine ? 64 : 48, fine ? 150000 : 60000, 7 + Math.round(sp * 10));
    return { f, m, fn: q => -cost.eval(toMat(q)) };
  };
  const com = vol => { let sx = 0, sy = 0, sz = 0, sw = 0; const [ax, ay] = vol.dims; for (let v = 0; v < vol.data.length; v++) { const w = vol.data[v]; if (w > 0) { sx += w * (v % ax); sy += w * (Math.floor(v / ax) % ay); sz += w * Math.floor(v / (ax * ay)); sw += w; } } const B = vol.affine; const i = sx / sw, j = sy / sw, k = sz / sw; return [B[0] * i + B[1] * j + B[2] * k + B[3], B[4] * i + B[5] * j + B[6] * k + B[7], B[8] * i + B[9] * j + B[10] * k + B[11]]; };

  // 1) Coarse search (as flirt does): rotations of -20..20 deg in 10 deg steps about each axis,
  //    for six translation starts (identity, centre-of-mass difference, +-10 and +-20 mm along
  //    the slice axis). Scored at 4 mm: coarser grids cannot tell basins apart for 5 mm 2D slices.
  const LG = (() => {
    const f = downsample(fixed, 4), m = downsample(moving, 4);
    const fm = resampleToGrid(fMaskVol, f.dims, f.affine, null, 'linear', 0);
    const fmu = Uint8Array.from(fm, v => (v > 0.5 ? 1 : 0));
    const cost = new NMICost(f, fmu, m, 32, 25000, 3);
    return { f, m, fn: q => -cost.eval(toMat(q)) };
  })();
  const cf = com(LG.f), cm = com(LG.m);
  const trans = [[0, 0, 0], [cm[0] - cf[0], cm[1] - cf[1], cm[2] - cf[2]], [0, 0, -10], [0, 0, 10], [0, 0, -20], [0, 0, 20]];
  const angs = [-20, -10, 0, 10, 20].map(a => a * R * Math.PI / 180);
  const cands = [];
  for (const t0 of trans) for (const ax of angs) for (const ay of angs) for (const az of angs) {
    const q = [t0[0], t0[1], t0[2], ax, ay, az];
    cands.push({ q, v: LG.fn(q) });
  }
  cands.sort((a, b) => a.v - b.v);
  const starts = [[0, 0, 0, 0, 0, 0]];
  for (const cnd of cands) {
    if (starts.length >= 5) break;
    if (!starts.some(o => Math.hypot(...o.map((v, i) => v - cnd.q[i])) < 12)) starts.push(cnd.q);
  }
  if (onProgress) onProgress(0.1, { spacing: 4, candidates: starts.length });

  // 2) Carry the starts through the pyramid. After the second-finest level keep the two best,
  //    and pick the winner at the finest level.
  const log = [];
  let sols = starts.map(x => ({ x, f: Infinity }));
  levels.forEach((sp, li) => {
    const L = levelCost(sp);
    const fine = li === levels.length - 1;
    sols = sols.map(s => powell(L.fn, s.x, new Array(6).fill(Math.max(1, sp) * 2), { tol: 0.02, maxIter: fine ? 4 : 6 }));
    sols.sort((a, b) => a.f - b.f);
    if (li === levels.length - 2) sols = sols.slice(0, 2);
    log.push({ spacing: sp, nmi: -sols[0].f, candidates: sols.length });
    if (onProgress) onProgress(0.1 + 0.9 * (li + 1) / levels.length, log[log.length - 1]);
  });
  const p = sols[0].x;
  return { T: toMat(p), params: { t: p.slice(0, 3), rDeg: p.slice(3).map(v => v / R * 180 / Math.PI) }, nmi: log[log.length - 1].nmi, levels: log, center: c };
}
