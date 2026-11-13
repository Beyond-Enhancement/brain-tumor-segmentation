// AURORA brain metastasis segmentation in JavaScript: a port of brainles_aurora 0.2.7 inference
// (DataHandler transforms and ModelHandler._sliding_window_inference, which use MONAI's
// ScaleIntensityRangePercentiles and SlidingWindowInferer) for its MONAI BasicUNet models.
//
// brainles_aurora runs on the image as it is given; its models were trained on brain-extracted
// images at 1 mm in BraTS (SRI-24) space. The workbench gives it the brain-extracted, registered
// images on a 1 mm grid with the BraTS axis order (i to the patient's left, j to posterior, k to
// superior), 240 x 240 x 155 voxels (larger if the brain needs it) centred on the brain. When
// the run grid is already 1 mm and axis-aligned, that grid is placed on the run grid's voxels,
// so the images are copied without interpolation (and when the run grid is itself such a
// 240 x 240 x 155 box, as BraTS-style images are, it is used as it is); otherwise they are
// resampled linearly. The
// probabilities are resampled linearly back to the run grid and thresholded there.
//
// Steps as brainles_aurora: per channel, intensities scaled from the 0.5th..99.5th percentile to
// 0..1 and clipped; sliding window 192 x 192 x 32 with overlap 0.5 and Gaussian weights
// (sigma = size / 8, clamped below at 0.001), replicate padding if the grid is smaller than the
// window; logits averaged; sigmoid >= 0.5; channel 0 the whole lesion (label 1, edema),
// channel 1 the metastasis (label 2, written over label 1).
// Test-time augmentation: brainles_aurora averages the logits of 13 passes: the plain one, 4 with
// Gaussian noise (std 0.001 on the 0..1 image) and 4 x 2 with that noisy image flipped on the
// first or the second axis. Without the noise those are 5 plain, 4 + 4 flipped passes, which is
// what is run here (3 passes, weighted 5 : 4 : 4).
//
// Run-grid volumes: index i + nx*(j + ny*k). Internal arrays in numpy C order (x, y, z) as the
// network sees them: index (x*N1 + y)*N2 + z.
import { mul4, inv4 } from './geom.js';

const AXES = { LPS: [[-1, 0, 0], [0, -1, 0], [0, 0, 1]] }; // world direction of internal axes i, j, k

// The internal grid: { dims, affine, snapped (on the run grid's voxels), sameFrame (the run grid's own box) }
export function auroraGrid(grid, brain, P) {
  const [nx, ny, nz] = grid.dims, A = grid.affine;
  let x0 = nx, y0 = ny, z0 = nz, x1 = -1, y1 = -1, z1 = -1;
  for (let k = 0, v = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++, v++) {
    if (brain[v]) { if (i < x0) x0 = i; if (i > x1) x1 = i; if (j < y0) y0 = j; if (j > y1) y1 = j; if (k < z0) z0 = k; if (k > z1) z1 = k; }
  }
  if (x1 < 0) throw new Error('AURORA: empty brain mask');
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const a of [x0, x1]) for (const b of [y0, y1]) for (const c of [z0, z1]) {
    for (let r = 0; r < 3; r++) {
      const w = a * A[4 * r] + b * A[4 * r + 1] + c * A[4 * r + 2] + A[4 * r + 3];
      if (w < lo[r]) lo[r] = w; if (w > hi[r]) hi[r] = w;
    }
  }
  const D = AXES[P.gridAxes], sp = P.gridSpacing;
  // internal axis a runs along world axis wa (the non-zero entry of D[a])
  const wAx = D.map(d => d.findIndex(v => v !== 0));
  const dims = [0, 1, 2].map(a => Math.max(P.gridDims[a], Math.ceil((hi[wAx[a]] - lo[wAx[a]]) / sp[a]) + 5));
  const centre = [0, 1, 2].map(r => (lo[r] + hi[r]) / 2);
  const origin = centre.slice();
  for (let a = 0; a < 3; a++) for (let r = 0; r < 3; r++) origin[r] -= D[a][r] * sp[a] * (dims[a] - 1) / 2;
  // run grid axis-aligned with the internal spacing: put the internal voxels on run-grid voxels
  let snapped = true;
  for (let c = 0; c < 3 && snapped; c++) {
    let n = 0;
    for (let r = 0; r < 3; r++) {
      const v = Math.abs(A[4 * r + c]);
      if (Math.abs(v - sp[wAx.indexOf(r)]) < 1e-4) n++; else if (v > 1e-4) snapped = false;
    }
    if (n !== 1) snapped = false;
  }
  let sameFrame = false;
  if (snapped) {
    for (let r = 0; r < 3; r++) { const s = sp[wAx.indexOf(r)]; origin[r] = A[4 * r + 3] + s * Math.round((origin[r] - A[4 * r + 3]) / s); }
    // a reference that already is a BraTS-style frame (as many voxels along each axis as the grid,
    // e.g. 240 x 240 x 155 at 1 mm): use that frame as it is, so that AURORA sees the image exactly
    // as its own code would (the sliding windows then fall on the same voxels)
    const ext = [0, 1, 2].map(r => {
      let a = Infinity, b = -Infinity;
      for (const i of [0, nx - 1]) for (const j of [0, ny - 1]) for (const k of [0, nz - 1]) {
        const w = i * A[4 * r] + j * A[4 * r + 1] + k * A[4 * r + 2] + A[4 * r + 3];
        if (w < a) a = w; if (w > b) b = w;
      }
      return [a, b];
    });
    if ([0, 1, 2].every(a => Math.round((ext[wAx[a]][1] - ext[wAx[a]][0]) / sp[a]) + 1 === dims[a])) {
      for (let a = 0; a < 3; a++) { const r = wAx[a]; origin[r] = D[a][r] < 0 ? ext[r][1] : ext[r][0]; }
      sameFrame = true;
    }
  }
  const affine = new Float64Array(16);
  for (let r = 0; r < 3; r++) { for (let a = 0; a < 3; a++) affine[4 * r + a] = D[a][r] * sp[a]; affine[4 * r + 3] = origin[r]; }
  affine[15] = 1;
  return { dims, affine, snapped, sameFrame };
}

const snap = c => { const r = Math.round(c); return Math.abs(c - r) < 1e-6 ? r : c; };

// trilinear sample of arr (element (i0,i1,i2) at base + i0*s0 + i1*s1 + i2*s2) at (c0,c1,c2);
// 0 outside [0, n-1] on any axis
function sampleLinear(arr, s0, s1, s2, n0, n1, n2, c0, c1, c2) {
  c0 = snap(c0); c1 = snap(c1); c2 = snap(c2);
  if (!(c0 >= 0 && c0 <= n0 - 1 && c1 >= 0 && c1 <= n1 - 1 && c2 >= 0 && c2 <= n2 - 1)) return 0;
  const i0 = Math.floor(c0), i1 = Math.floor(c1), i2 = Math.floor(c2);
  const b0 = c0 - i0, b1 = c1 - i1, b2 = c2 - i2;
  const p = i0 * s0 + i1 * s1 + i2 * s2;
  if (b0 === 0 && b1 === 0 && b2 === 0) return arr[p];
  const d0 = i0 + 1 < n0 ? s0 : 0, d1 = i1 + 1 < n1 ? s1 : 0, d2 = i2 + 1 < n2 ? s2 : 0;
  const a0 = 1 - b0, a1 = 1 - b1, a2 = 1 - b2;
  return ((arr[p] * a2 + arr[p + d2] * b2) * a1 + (arr[p + d1] * a2 + arr[p + d1 + d2] * b2) * b1) * a0 +
    ((arr[p + d0] * a2 + arr[p + d0 + d2] * b2) * a1 + (arr[p + d0 + d1] * a2 + arr[p + d0 + d1 + d2] * b2) * b1) * b0;
}

// numpy.percentile(a, q), 'linear' method, for sorted a
function percentileSorted(s, q) {
  const n = s.length, vi = (n - 1) * (q / 100);
  if (vi >= n - 1) return s[n - 1];
  const lo = Math.floor(vi), g = vi - lo, a = s[lo], b = s[lo + 1], d = b - a;
  return g >= 0.5 ? b - d * (1 - g) : a + d * g;
}

// MONAI compute_importance_map(mode='gaussian', sigma_scale), float32
function gaussianImportance(roi, sigmaScale) {
  const g = roi.map(n => {
    const s = n * sigmaScale, den = -2 * s * s, out = new Float32Array(n);
    for (let i = 0; i < n; i++) { const x = Math.fround(-(n - 1) / 2 + i); out[i] = Math.fround(Math.exp(Math.fround(Math.fround(x * x) / den))); }
    return out;
  });
  const m = new Float32Array(roi[0] * roi[1] * roi[2]);
  let mn = Infinity;
  for (let a = 0, p = 0; a < roi[0]; a++) for (let b = 0; b < roi[1]; b++) {
    const ab = Math.fround(g[0][a] * g[1][b]);
    for (let c = 0; c < roi[2]; c++, p++) { m[p] = Math.fround(ab * g[2][c]); if (m[p] < mn) mn = m[p]; }
  }
  const floor = Math.fround(Math.max(mn, 1e-3));
  for (let p = 0; p < m.length; p++) if (m[p] < floor) m[p] = floor;
  return m;
}

// MONAI dense_patch_slices starts for one axis
function windowStarts(size, roi, overlap) {
  const interval = roi === size ? roi : Math.max(1, Math.trunc(roi * (1 - overlap)));
  const num = Math.ceil(size / interval);
  let scan = 0;
  while (scan < num && !(scan * interval + roi >= size)) scan++;
  const count = scan < num ? scan + 1 : 1;
  const starts = [];
  for (let i = 0; i < count; i++) { const s = i * interval; starts.push(s - Math.max(s + roi - size, 0)); }
  return starts;
}

// vols: Float32Array per input channel on the run grid (brain-extracted, registered)
export async function predictAurora(vols, grid, brain, cfg, runPatch, opts = {}) {
  const P = cfg.aurora;
  const onProgress = opts.onProgress || (() => {});
  const shouldStop = opts.shouldStop || (() => false);
  const C = vols.length, K = P.outChannels.length;
  const [nx, ny, nz] = grid.dims;

  // 1) internal 1 mm grid and the images on it
  onProgress('preprocess', 0);
  const G = auroraGrid(grid, brain, P);
  const [N0, N1, N2] = G.dims, N = N0 * N1 * N2;
  const M = mul4(inv4(grid.affine), G.affine); // internal voxel -> run voxel
  const img = [];
  for (let ch = 0; ch < C; ch++) {
    const src = vols[ch], out = new Float32Array(N);
    for (let x = 0, p = 0; x < N0; x++) {
      if (x % 24 === 0) { onProgress('preprocess', (ch + 0.7 * x / N0) / C); if (shouldStop()) throw new Error('Cancelled'); }
      for (let y = 0; y < N1; y++) for (let z = 0; z < N2; z++, p++) {
        const c0 = M[3] + x * M[0] + y * M[1] + z * M[2];
        const c1 = M[7] + x * M[4] + y * M[5] + z * M[6];
        const c2 = M[11] + x * M[8] + y * M[9] + z * M[10];
        const v = sampleLinear(src, 1, nx, nx * ny, nx, ny, nz, c0, c1, c2);
        out[p] = v === v ? v : 0; // nan_to_num
      }
    }
    img.push(out);
  }

  if (opts.debug) opts.debug.raw = img.map(a => a.slice());

  // 2) ScaleIntensityRangePercentiles(lower, upper, b_min 0, b_max 1, clip, channel_wise), float32
  const pct = [];
  for (let ch = 0; ch < C; ch++) {
    const im = img[ch];
    let s = Float32Array.from(im).sort();
    const amin = Math.fround(percentileSorted(s, P.percentiles[0])), amax = Math.fround(percentileSorted(s, P.percentiles[1]));
    s = null;
    pct.push([amin, amax]);
    const den = Math.fround(amax - amin);
    if (den === 0) { for (let p = 0; p < N; p++) im[p] = Math.fround(im[p] - amin); }
    else for (let p = 0; p < N; p++) { const v = Math.fround(Math.fround(im[p] - amin) / den); im[p] = v < 0 ? 0 : v > 1 ? 1 : v; }
    onProgress('preprocess', (ch + 1) / C);
  }
  if (opts.debug) Object.assign(opts.debug, { grid: G, img: img.map(a => a.slice()), pct });

  // 3) sliding window (MONAI sliding_window_inference), once per pass
  const roi = P.roiSize, rn = roi[0] * roi[1] * roi[2];
  const Q = G.dims.map((n, d) => Math.max(n, roi[d]));            // padded size
  const padLo = G.dims.map((n, d) => Math.floor(Math.max(roi[d] - n, 0) / 2));
  const starts = [0, 1, 2].map(d => windowStarts(Q[d], roi[d], P.overlap));
  const wins = [];
  for (const a of starts[0]) for (const b of starts[1]) for (const c of starts[2]) wins.push([a, b, c]);
  const imp = gaussianImportance(roi, P.sigmaScale);
  const QN = Q[0] * Q[1] * Q[2];
  const cnt = new Float32Array(QN);
  for (const [a0, b0, c0] of wins) for (let a = 0, p = 0; a < roi[0]; a++) for (let b = 0; b < roi[1]; b++) {
    const q = ((a0 + a) * Q[1] + (b0 + b)) * Q[2] + c0;
    for (let c = 0; c < roi[2]; c++, p++) cnt[q + c] = Math.fround(cnt[q + c] + imp[p]);
  }
  const tta = opts.tta && P.tta;
  const passes = [{ flip: -1, w: tta ? P.tta.weights[0] : 1 }];
  if (tta) P.tta.flipAxes.forEach((ax, i) => passes.push({ flip: ax, w: P.tta.weights[i + 1] }));
  const wsum = passes.reduce((s, p) => s + p.w, 0);
  const total = new Float32Array(K * N);
  const agg = new Float32Array(K * QN);
  const patch = new Float32Array(C * rn);
  const nTot = wins.length * passes.length;
  let done = 0;
  for (let pi = 0; pi < passes.length; pi++) {
    const { flip, w } = passes[pi];
    // padded (and, for a flipped pass, flipped) coordinate -> internal index, per axis
    const map = [0, 1, 2].map(d => {
      const m = new Int32Array(Q[d]), n = G.dims[d];
      for (let q = 0; q < Q[d]; q++) { let i = Math.min(Math.max(q - padLo[d], 0), n - 1); if (d === flip) i = n - 1 - i; m[q] = i; }
      return m;
    });
    agg.fill(0);
    for (let t = 0; t < wins.length; t++) {
      if (shouldStop()) throw new Error('Cancelled');
      onProgress('network', done / nTot, { tile: t + 1, tiles: wins.length, mirror: pi + 1, mirrors: passes.length });
      const [a0, b0, c0] = wins[t];
      for (let ch = 0, p = 0; ch < C; ch++) {
        const im = img[ch];
        for (let a = 0; a < roi[0]; a++) {
          const ia = map[0][a0 + a] * N1;
          for (let b = 0; b < roi[1]; b++) {
            const base = (ia + map[1][b0 + b]) * N2, mz = map[2];
            for (let c = 0; c < roi[2]; c++, p++) patch[p] = im[base + mz[c0 + c]];
          }
        }
      }
      const out = await runPatch(patch, [1, C, ...roi]);
      for (let k = 0; k < K; k++) {
        const ko = k * rn, kq = k * QN;
        for (let a = 0, p = 0; a < roi[0]; a++) for (let b = 0; b < roi[1]; b++) {
          const q = kq + ((a0 + a) * Q[1] + (b0 + b)) * Q[2] + c0;
          for (let c = 0; c < roi[2]; c++, p++) agg[q + c] = Math.fround(agg[q + c] + Math.fround(out[ko + p] * imp[p]));
        }
      }
      done++;
    }
    // this pass's output (divided by the weight sum), unpadded and unflipped, added to the total
    for (let k = 0; k < K; k++) {
      const kq = k * QN, kn = k * N;
      for (let x = 0, p = 0; x < N0; x++) {
        const qx = (flip === 0 ? N0 - 1 - x : x) + padLo[0];
        for (let y = 0; y < N1; y++) {
          const qy = (flip === 1 ? N1 - 1 - y : y) + padLo[1];
          const qb = (qx * Q[1] + qy) * Q[2] + padLo[2];
          for (let z = 0; z < N2; z++, p++) {
            const q = qb + (flip === 2 ? N2 - 1 - z : z);
            const v = Math.fround(agg[kq + q] / cnt[q]);
            total[kn + p] = passes.length === 1 ? v : total[kn + p] + w * v;
          }
        }
      }
    }
  }
  onProgress('network', 1, { tile: wins.length, tiles: wins.length, mirror: passes.length, mirrors: passes.length });

  // 4) sigmoid of the (averaged) logits on the internal grid
  for (let p = 0; p < total.length; p++) {
    const v = passes.length === 1 ? total[p] : Math.fround(total[p] / wsum);
    total[p] = Math.fround(1 / (1 + Math.exp(-v)));
  }
  if (opts.debug) opts.debug.prob = total;

  // 5) back to the run grid (linear), threshold, labels: 1 whole lesion, 2 metastasis over it
  const M2 = mul4(inv4(G.affine), grid.affine); // run voxel -> internal voxel
  const thr = P.threshold, seg = new Uint8Array(nx * ny * nz);
  const pw = total.subarray(0, N), pm = total.subarray(N, 2 * N);
  let nOut = 0;
  for (let k = 0, v = 0; k < nz; k++) {
    if (k % 8 === 0) onProgress('post', k / nz);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++, v++) {
      const c0 = M2[3] + i * M2[0] + j * M2[1] + k * M2[2];
      const c1 = M2[7] + i * M2[4] + j * M2[5] + k * M2[6];
      const c2 = M2[11] + i * M2[8] + j * M2[9] + k * M2[10];
      if (c0 < -0.5 || c1 < -0.5 || c2 < -0.5 || c0 > N0 - 0.5 || c1 > N1 - 0.5 || c2 > N2 - 0.5) { if (brain[v]) nOut++; continue; }
      if (Math.fround(sampleLinear(pw, N1 * N2, N2, 1, N0, N1, N2, c0, c1, c2)) >= thr) seg[v] = 1;
      if (Math.fround(sampleLinear(pm, N1 * N2, N2, 1, N0, N1, N2, c0, c1, c2)) >= thr) seg[v] = 2;
    }
  }
  onProgress('post', 1);
  return { seg, info: { grid: G.dims, snapped: G.snapped, sameFrame: G.sameFrame, windows: wins.length, passes: passes.length, percentiles: pct, brainOutsideGrid: nOut } };
}
