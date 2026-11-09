// Raidionics segmentation in JavaScript: a port of raidionicsseg 1.5.2 fit.run_model (preprocessing,
// patch-wise inference, reconstruction) with the runtime settings the Raidionics app passes
// through raidionics_rads_lib (reconstruction 'thresholding', order 'resample_first', patch
// overlap 0, no test-time augmentation, first fold only), followed by raidionics_rads_lib's
// 'brain_overlap' refinement. Implemented for the configuration of MRI_FLAIRChanges
// (build/export_models.py checks it): background_cropping 'minimum', resampling to the output
// spacing with nibabel.processing.resample_to_output (linear), percentile clipping,
// 'zeromean_nonzero' normalization, one channels-first input, sigmoid output.
//
// Volumes are { dims:[nx,ny,nz], affine: Float64Array(16) row-major, data } with index
// i + nx*(j + ny*k). Arrays in nibabel's (X,Y,Z) C order, as the network sees them, are stored
// here with index (x*NY + y)*NZ + z.
import { mul4, inv4 } from './geom.js';

// scipy.ndimage.affine_transform(order=1, mode='constant', cval=0) at one point of an array whose
// element (i0,i1,i2) is arr[base + i0*s0 + i1*s1 + i2*s2]: coordinates outside [0, n-1] give 0;
// inside, the 8 neighbours are summed in C order with weights (1 - x, 1 - (1 - x)), as scipy's
// NI_GeometricTransform and get_spline_interpolation_weights do.
function sampleLinear(arr, base, s0, s1, s2, n0, n1, n2, c0, c1, c2) {
  if (!(c0 >= 0 && c0 <= n0 - 1 && c1 >= 0 && c1 <= n1 - 1 && c2 >= 0 && c2 <= n2 - 1)) return 0;
  const i0 = Math.floor(c0), i1 = Math.floor(c1), i2 = Math.floor(c2);
  const a0 = 1 - (c0 - i0), a1 = 1 - (c1 - i1), a2 = 1 - (c2 - i2);
  const b0 = 1 - a0, b1 = 1 - a1, b2 = 1 - a2;
  const p0 = base + i0 * s0, q0 = i0 + 1 < n0 ? p0 + s0 : p0;
  const d1 = i1 + 1 < n1 ? s1 : 0, d2 = i2 + 1 < n2 ? s2 : 0;
  const r00 = p0 + i1 * s1 + i2 * s2, r10 = q0 + i1 * s1 + i2 * s2;
  let t = arr[r00] * a0 * a1 * a2;
  t += arr[r00 + d2] * a0 * a1 * b2;
  t += arr[r00 + d1] * a0 * b1 * a2;
  t += arr[r00 + d1 + d2] * a0 * b1 * b2;
  t += arr[r10] * b0 * a1 * a2;
  t += arr[r10 + d2] * b0 * a1 * b2;
  t += arr[r10 + d1] * b0 * b1 * a2;
  t += arr[r10 + d1 + d2] * b0 * b1 * b2;
  return t;
}

// numpy.percentile(a, q) with the default 'linear' method, for sorted a
function percentileSorted(s, q) {
  const n = s.length;
  const vi = (n - 1) * (q / 100);
  if (vi >= n - 1) return s[n - 1];
  if (vi < 0) return s[0];
  const lo = Math.floor(vi), g = vi - lo;
  const a = s[lo], b = s[lo + 1], d = b - a;
  return g >= 0.5 ? b - d * (1 - g) : a + d * g;
}

export async function predictRaidionics(vol, brain, cfg, runPatch, opts = {}) {
  const P = cfg.raidionics;
  const onProgress = opts.onProgress || (() => {});
  const shouldStop = opts.shouldStop || (() => false);
  const [nx, ny, nz] = vol.dims;
  const A = vol.affine, src = vol.data;

  // 1) crop_background_minimum: bounding box of the voxels >= int(0.2 * (max - min))
  onProgress('preprocess', 0);
  let vmin = Infinity, vmax = -Infinity;
  for (let v = 0; v < src.length; v++) { const x = src[v]; if (x < vmin) vmin = x; if (x > vmax) vmax = x; }
  const limit = Math.trunc((vmax - vmin) * 0.2);
  let x0 = nx, y0 = ny, z0 = nz, x1 = -1, y1 = -1, z1 = -1;
  for (let k = 0, v = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++, v++) {
    if (src[v] >= limit) { if (i < x0) x0 = i; if (i > x1) x1 = i; if (j < y0) y0 = j; if (j > y1) y1 = j; if (k < z0) z0 = k; if (k > z1) z1 = k; }
  }
  if (x1 < 0) throw new Error('No foreground found in the FLAIR image');
  const cd = [x1 + 1 - x0, y1 + 1 - y0, z1 + 1 - z0];
  // raidionicsseg keeps the uncropped affine for the cropped array; doing the same keeps the
  // resampling grid identical (the result does not depend on it, the rounding does)
  const cropBase = x0 + nx * (y0 + ny * z0);

  // 2) nibabel.processing.resample_to_output(cropped, spacing, order=1)
  const vs = P.outputSpacing;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const a of [0, cd[0] - 1]) for (const b of [0, cd[1] - 1]) for (const c of [0, cd[2] - 1]) {
    for (let r = 0; r < 3; r++) {
      const w = a * A[4 * r] + b * A[4 * r + 1] + c * A[4 * r + 2] + A[4 * r + 3];
      if (w < lo[r]) lo[r] = w; if (w > hi[r]) hi[r] = w;
    }
  }
  const N = [0, 1, 2].map(r => Math.ceil((hi[r] - lo[r]) / vs[r]) + 1);
  const outAff = new Float64Array([vs[0], 0, 0, lo[0], 0, vs[1], 0, lo[1], 0, 0, vs[2], lo[2], 0, 0, 0, 1]);
  const M = mul4(inv4(A), outAff); // output voxel -> input voxel
  const [N0, N1, N2] = N;
  let data = new Float64Array(N0 * N1 * N2);
  for (let x = 0, p = 0; x < N0; x++) {
    if (x % 16 === 0) { onProgress('preprocess', 0.6 * x / N0); if (shouldStop()) throw new Error('Cancelled'); }
    for (let y = 0; y < N1; y++) for (let z = 0; z < N2; z++, p++) {
      const c0 = ((M[3] + x * M[0]) + y * M[1]) + z * M[2];
      const c1 = ((M[7] + x * M[4]) + y * M[5]) + z * M[6];
      const c2 = ((M[11] + x * M[8]) + y * M[9]) + z * M[10];
      data[p] = sampleLinear(src, cropBase, 1, nx, nx * ny, cd[0], cd[1], cd[2], c0, c1, c2);
    }
  }

  // 3) intensity_clipping (negatives to 0, then percentiles) and zeromean_nonzero normalization
  for (let p = 0; p < data.length; p++) if (data[p] < 0) data[p] = 0;
  let sorted = Float64Array.from(data).sort();
  const cl = percentileSorted(sorted, P.clipPercentile[0]), ch = percentileSorted(sorted, P.clipPercentile[1]);
  sorted = null;
  const img = new Float32Array(data.length);
  for (let p = 0; p < data.length; p++) img[p] = Math.min(Math.max(data[p], cl), ch);
  data = null;
  let n = 0, sum = 0;
  for (let p = 0; p < img.length; p++) if (img[p] !== 0) { sum += img[p]; n++; }
  const mean = sum / n;
  let ss = 0;
  for (let p = 0; p < img.length; p++) if (img[p] !== 0) { const d = img[p] - mean; ss += d * d; }
  const sd = Math.sqrt(ss / (n - 1));
  if (sd !== 0) for (let p = 0; p < img.length; p++) if (img[p] !== 0) img[p] = (img[p] - mean) / sd;
  if (opts.debug) Object.assign(opts.debug, { img, N, outAff, cropBox: [x0, y0, z0, x1 + 1, y1 + 1, z1 + 1] });
  onProgress('preprocess', 1);

  // 4) patch-wise inference (raidionicsseg __run_predictions_patch): edge padding of axes not
  // larger than the patch, patches on a grid with step patch - offset, the last one moved back
  // inside, overlaps resolved by the maximum, padding removed, then the sigmoid.
  const ps = P.patchSize, step = ps.map((s, a) => s - Math.trunc((P.overlapRatio || 0) * s));
  const pad = N.map((d, a) => (d <= ps[a] ? Math.ceil(Math.abs(d - ps[a]) / 2) : 0));
  const Q = N.map((d, a) => d + 2 * pad[a]);
  const bounds = [];
  for (let a = 0; a < Math.ceil(Q[0] / step[0]); a++) for (let b = 0; b < Math.ceil(Q[1] / step[1]); b++) for (let c = 0; c < Math.ceil(Q[2] / step[2]); c++) {
    const bb = [a * step[0], b * step[1], c * step[2]];
    let skip = false;
    for (let ax = 0; ax < 3; ax++) {
      if (bb[ax] + ps[ax] >= Q[ax]) { const s0 = bb[ax] - Math.abs(Q[ax] - (bb[ax] + ps[ax])); if (s0 < 0) { skip = true; break; } bb[ax] = s0; }
    }
    if (!skip) bounds.push(bb);
  }
  const res = new Float32Array(Q[0] * Q[1] * Q[2]).fill(-100);
  const patch = new Float32Array(ps[0] * ps[1] * ps[2]);
  const clampIdx = (q, a) => Math.min(Math.max(q - pad[a], 0), N[a] - 1);
  for (let t = 0; t < bounds.length; t++) {
    if (shouldStop()) throw new Error('Cancelled');
    onProgress('network', t / bounds.length, { tile: t + 1, tiles: bounds.length, mirror: 1, mirrors: 1 });
    const [bx, by, bz] = bounds[t];
    for (let x = 0, p = 0; x < ps[0]; x++) {
      const sx = clampIdx(bx + x, 0);
      for (let y = 0; y < ps[1]; y++) {
        const base = (sx * N1 + clampIdx(by + y, 1)) * N2;
        for (let z = 0; z < ps[2]; z++, p++) patch[p] = img[base + clampIdx(bz + z, 2)];
      }
    }
    const out = await runPatch(patch, [1, 1, ps[0], ps[1], ps[2]]);
    for (let x = 0, p = 0; x < ps[0]; x++) for (let y = 0; y < ps[1]; y++) {
      const base = ((bx + x) * Q[1] + (by + y)) * Q[2] + bz;
      for (let z = 0; z < ps[2]; z++, p++) if (out[p] > res[base + z]) res[base + z] = out[p];
    }
  }
  onProgress('network', 1, { tile: bounds.length, tiles: bounds.length, mirror: 1, mirrors: 1 });
  let prob = new Float32Array(N0 * N1 * N2);
  for (let x = 0, p = 0; x < N0; x++) for (let y = 0; y < N1; y++) {
    const base = ((x + pad[0]) * Q[1] + (y + pad[1])) * Q[2] + pad[2];
    for (let z = 0; z < N2; z++, p++) prob[p] = 1 / (1 + Math.exp(-res[base + z]));
  }

  // 5) reconstruction, 'resample_first': linear resampling of the probabilities onto the cropped
  // input grid (nibabel.processing.resample_from_to, float32 output), placed back into the
  // full grid, then thresholding
  const M2 = mul4(inv4(outAff), A);
  const thr = P.thresholds[0];
  const seg = new Uint8Array(src.length);
  for (let k = 0; k < cd[2]; k++) {
    if (k % 8 === 0) onProgress('post', k / cd[2]);
    for (let j = 0; j < cd[1]; j++) for (let i = 0; i < cd[0]; i++) {
      const c0 = ((M2[3] + i * M2[0]) + j * M2[1]) + k * M2[2];
      const c1 = ((M2[7] + i * M2[4]) + j * M2[5]) + k * M2[6];
      const c2 = ((M2[11] + i * M2[8]) + j * M2[9]) + k * M2[10];
      if (Math.fround(sampleLinear(prob, 0, N1 * N2, N2, 1, N0, N1, N2, c0, c1, c2)) >= thr) seg[(i + x0) + nx * ((j + y0) + ny * (k + z0))] = 1;
    }
  }

  if (opts.debug) opts.debug.prob = prob;
  prob = null;

  // 6) raidionics_rads_lib perform_brain_overlap_refinement: keep the connected components
  // (26-connectivity, as skimage.measure.label) that touch the brain mask
  const removed = P.refinement === 'brain_overlap' && brain ? keepComponentsTouching(seg, brain, vol.dims) : 0;
  onProgress('post', 1);
  return { seg, info: { cropBox: [x0, y0, z0, x1 + 1, y1 + 1, z1 + 1], resampledDims: N, patches: bounds.length, clip: [cl, ch], mean, sd, removedOutsideBrain: removed } };
}

// Zero the 26-connected components of seg (0/1) that have no voxel inside keep; returns the
// number of voxels removed. Works inside the bounding box of seg.
export function keepComponentsTouching(seg, keep, dims) {
  const [nx, ny, nz] = dims;
  let x0 = nx, y0 = ny, z0 = nz, x1 = -1, y1 = -1, z1 = -1;
  for (let k = 0, v = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++, v++) {
    if (seg[v]) { if (i < x0) x0 = i; if (i > x1) x1 = i; if (j < y0) y0 = j; if (j > y1) y1 = j; if (k < z0) z0 = k; if (k > z1) z1 = k; }
  }
  if (x1 < 0) return 0;
  const bx = x1 - x0 + 1, by = y1 - y0 + 1, bz = z1 - z0 + 1;
  const seen = new Uint8Array(bx * by * bz);
  const stack = new Int32Array(bx * by * bz);
  const comp = new Int32Array(bx * by * bz);
  let removed = 0;
  for (let k = 0; k < bz; k++) for (let j = 0; j < by; j++) for (let i = 0; i < bx; i++) {
    const b = i + bx * (j + by * k);
    if (seen[b] || !seg[(i + x0) + nx * ((j + y0) + ny * (k + z0))]) continue;
    let sp = 0, nc = 0, touches = false;
    stack[sp++] = b; seen[b] = 1;
    while (sp) {
      const c = stack[--sp];
      comp[nc++] = c;
      const ci = c % bx, cj = Math.floor(c / bx) % by, ck = Math.floor(c / (bx * by));
      if (keep[(ci + x0) + nx * ((cj + y0) + ny * (ck + z0))]) touches = true;
      for (let dk = -1; dk <= 1; dk++) {
        const kk = ck + dk; if (kk < 0 || kk >= bz) continue;
        for (let dj = -1; dj <= 1; dj++) {
          const jj = cj + dj; if (jj < 0 || jj >= by) continue;
          for (let di = -1; di <= 1; di++) {
            const ii = ci + di; if (ii < 0 || ii >= bx) continue;
            const nb = ii + bx * (jj + by * kk);
            if (seen[nb] || !seg[(ii + x0) + nx * ((jj + y0) + ny * (kk + z0))]) continue;
            seen[nb] = 1; stack[sp++] = nb;
          }
        }
      }
    }
    if (!touches) {
      for (let q = 0; q < nc; q++) { const c = comp[q]; const ci = c % bx, cj = Math.floor(c / bx) % by, ck = Math.floor(c / (bx * by)); seg[(ci + x0) + nx * ((cj + y0) + ny * (ck + z0))] = 0; }
      removed += nc;
    }
  }
  return removed;
}
