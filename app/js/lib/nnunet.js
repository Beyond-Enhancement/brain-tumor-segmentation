// nnU-Net inference (v1 and v2 code paths) in JavaScript, driven by a config exported from the
// trained model's plans (build/export_models.py). Arrays use numpy C-order conventions:
// shape [d0, d1, d2] with d2 fastest, which is the same memory layout as a NIfTI volume read by
// SimpleITK (d0 = k, d1 = j, d2 = i).
import { resizeVolume, resizeSeparateZ } from './geom.js';

// ---------- small array helpers (3D, C-order) ----------
const prod = a => a.reduce((x, y) => x * y, 1);

export function transpose3(data, shape, perm) {
  if (perm[0] === 0 && perm[1] === 1 && perm[2] === 2) return { data, shape: shape.slice() };
  const ns = perm.map(p => shape[p]);
  const out = new Float32Array(data.length);
  const st = [shape[1] * shape[2], shape[2], 1];
  const s0 = st[perm[0]], s1 = st[perm[1]], s2 = st[perm[2]];
  let o = 0;
  for (let a = 0; a < ns[0]; a++) for (let b = 0; b < ns[1]; b++) { let q = a * s0 + b * s1; for (let c = 0; c < ns[2]; c++, q += s2) out[o++] = data[q]; }
  return { data: out, shape: ns };
}

// scipy.ndimage.binary_fill_holes (default 6-connected structure for the background)
export function fillHoles(mask, shape) {
  const [d0, d1, d2] = shape, n = mask.length;
  const outside = new Uint8Array(n);
  const stack = new Int32Array(n);
  let sp = 0;
  const push = v => { if (!mask[v] && !outside[v]) { outside[v] = 1; stack[sp++] = v; } };
  for (let a = 0; a < d0; a++) for (let b = 0; b < d1; b++) for (let c = 0; c < d2; c++) {
    if (a === 0 || b === 0 || c === 0 || a === d0 - 1 || b === d1 - 1 || c === d2 - 1) push((a * d1 + b) * d2 + c);
  }
  const s0 = d1 * d2, s1 = d2;
  while (sp) {
    const v = stack[--sp];
    const c = v % d2, b = ((v - c) / d2) % d1, a = (v - c - b * d2) / s0;
    if (a > 0) push(v - s0); if (a < d0 - 1) push(v + s0);
    if (b > 0) push(v - s1); if (b < d1 - 1) push(v + s1);
    if (c > 0) push(v - 1); if (c < d2 - 1) push(v + 1);
  }
  const out = new Uint8Array(n);
  for (let v = 0; v < n; v++) out[v] = outside[v] ? 0 : 1;
  return out;
}

// Connected components. connectivity 6: scipy.ndimage.label with its default structure;
// 26: skimage.measure.label with full connectivity. Returns {labels:Int32Array, n, sizes}
export function labelComponents(mask, shape, connectivity = 6) {
  const [d0, d1, d2] = shape, n = mask.length, s0 = d1 * d2;
  const lab = new Int32Array(n), stack = new Int32Array(n);
  const full = connectivity === 26;
  let cur = 0; const sizes = [0];
  for (let v0 = 0; v0 < n; v0++) {
    if (!mask[v0] || lab[v0]) continue;
    cur++; let sp = 0, size = 0; lab[v0] = cur; stack[sp++] = v0;
    while (sp) {
      const v = stack[--sp]; size++;
      const c = v % d2, b = ((v - c) / d2) % d1, a = (v - c - b * d2) / s0;
      if (full) {
        for (let da = -1; da <= 1; da++) {
          const aa = a + da; if (aa < 0 || aa >= d0) continue;
          for (let db = -1; db <= 1; db++) {
            const bb = b + db; if (bb < 0 || bb >= d1) continue;
            for (let dc = -1; dc <= 1; dc++) {
              const cc = c + dc; if (cc < 0 || cc >= d2) continue;
              const w = (aa * d1 + bb) * d2 + cc;
              if (mask[w] && !lab[w]) { lab[w] = cur; stack[sp++] = w; }
            }
          }
        }
        continue;
      }
      const nb = [a > 0 ? v - s0 : -1, a < d0 - 1 ? v + s0 : -1, b > 0 ? v - d2 : -1, b < d1 - 1 ? v + d2 : -1, c > 0 ? v - 1 : -1, c < d2 - 1 ? v + 1 : -1];
      for (const w of nb) if (w >= 0 && mask[w] && !lab[w]) { lab[w] = cur; stack[sp++] = w; }
    }
    sizes.push(size);
  }
  return { labels: lab, n: cur, sizes };
}

// nnU-Net v1 remove_all_but_the_largest_connected_component (6-connected); with connectivity 26
// nnU-Net v2 remove_all_but_largest_component_from_segmentation (acvl_utils: skimage labels,
// every component as large as the largest is kept)
export function keepLargestComponents(seg, shape, forWhichClasses, voxelVolume, minValid, connectivity = 6) {
  for (const cls of forWhichClasses) {
    const set = Array.isArray(cls) ? cls : [cls];
    const mask = new Uint8Array(seg.length);
    for (let v = 0; v < seg.length; v++) if (set.includes(seg[v])) mask[v] = 1;
    const { labels, n, sizes } = labelComponents(mask, shape, connectivity);
    if (!n) continue;
    const vol = sizes.map(s => s * voxelVolume);
    const maxv = Math.max(...vol.slice(1));
    const key = Array.isArray(cls) ? String(cls) : String(cls);
    const remove = new Uint8Array(n + 1);
    for (let o = 1; o <= n; o++) if (vol[o] !== maxv) remove[o] = minValid && minValid[key] != null ? (vol[o] < minValid[key] ? 1 : 0) : 1;
    for (let v = 0; v < seg.length; v++) if (labels[v] && remove[labels[v]]) seg[v] = 0;
  }
  return seg;
}

function getDoSeparateZ(spacing, thr) { return Math.max(...spacing) / Math.min(...spacing) > thr; }
function lowresAxes(spacing) { const m = Math.max(...spacing); return spacing.map((s, i) => (s === m ? i : -1)).filter(i => i >= 0); }

function decideSeparateZ(curSp, newSp, thr) {
  let axes = null;
  if (getDoSeparateZ(curSp, thr)) axes = lowresAxes(curSp);
  else if (getDoSeparateZ(newSp, thr)) axes = lowresAxes(newSp);
  if (!axes || axes.length !== 1) return null;
  return axes[0];
}

// resample (C-order, numpy axes) one channel; zAxisNp is the numpy axis or null
function resampleNp(data, shapeNp, newShapeNp, order, orderZ, zAxisNp, Ctor = Float32Array) {
  // our geom resizers take x-fastest dims [nx,ny,nz] = reversed numpy shape
  const dims = shapeNp.slice().reverse(), nd = newShapeNp.slice().reverse();
  if (zAxisNp == null) return resizeVolume(data, dims, nd, [order, order, order], true, Ctor);
  return resizeSeparateZ(data, dims, nd, 2 - zAxisNp, order, orderZ, Ctor);
}

export function computeSteps(patch, image, step) {
  return patch.map((p, d) => {
    const target = p * step;
    const num = Math.ceil((image[d] - p) / target) + 1;
    const maxStep = image[d] - p;
    const actual = num > 1 ? maxStep / (num - 1) : 99999999999;
    return Array.from({ length: num }, (_, i) => Math.round(actual * i));
  });
}

export function gaussianMap(patch, sigmaScale = 1 / 8) {
  const g1 = patch.map(p => {
    const c = Math.floor(p / 2), s = p * sigmaScale, w = new Float64Array(p);
    for (let i = 0; i < p; i++) w[i] = Math.exp(-0.5 * ((i - c) / s) ** 2);
    return w;
  });
  const out = new Float32Array(prod(patch));
  let o = 0, mn = Infinity;
  for (let a = 0; a < patch[0]; a++) for (let b = 0; b < patch[1]; b++) for (let c = 0; c < patch[2]; c++) { const v = g1[0][a] * g1[1][b] * g1[2][c]; out[o++] = v; }
  for (let i = 0; i < out.length; i++) if (out[i] > 0 && out[i] < mn) mn = out[i];
  for (let i = 0; i < out.length; i++) if (out[i] === 0) out[i] = mn;
  return out;
}

function flipPatch(src, C, ps, axes, dst) {
  // flip spatial axes (subset of 0,1,2) of a [C, ps0, ps1, ps2] tensor
  const [p0, p1, p2] = ps, n = p0 * p1 * p2;
  const f0 = axes.includes(0), f1 = axes.includes(1), f2 = axes.includes(2);
  for (let ch = 0; ch < C; ch++) {
    const base = ch * n;
    for (let a = 0; a < p0; a++) {
      const aa = f0 ? p0 - 1 - a : a;
      for (let b = 0; b < p1; b++) {
        const bb = f1 ? p1 - 1 - b : b;
        let q = base + (aa * p1 + bb) * p2, o = base + (a * p1 + b) * p2;
        if (f2) { q += p2 - 1; for (let c = 0; c < p2; c++) dst[o + c] = src[q - c]; }
        else for (let c = 0; c < p2; c++) dst[o + c] = src[q + c];
      }
    }
  }
  return dst;
}

function mirrorCombos(axes, enabled) {
  // nnU-Net order: none, then all non-empty subsets
  const out = [[]];
  if (!enabled || !axes.length) return out;
  for (let k = 1; k <= axes.length; k++) {
    const rec = (start, cur) => { if (cur.length === k) { out.push(cur.slice()); return; } for (let i = start; i < axes.length; i++) { cur.push(axes[i]); rec(i + 1, cur); cur.pop(); } };
    rec(0, []);
  }
  return out;
}

// Sliding-window prediction over preprocessed data [C, d0, d1, d2] -> aggregated [K, d0, d1, d2]
async function slidingWindow(data, C, shape, cfg, runPatch, { tta, onProgress, shouldStop }) {
  const ps = cfg.patchSize, K = cfg.numClasses;
  // pad to at least patch size (constant 0), symmetric like nnU-Net pad_nd_image
  const padShape = shape.map((s, d) => Math.max(s, ps[d]));
  const padLo = shape.map((s, d) => Math.floor((padShape[d] - s) / 2));
  const nP = prod(padShape), n = prod(shape);
  let x = data;
  if (padShape.some((s, d) => s !== shape[d])) {
    x = new Float32Array(C * nP);
    for (let ch = 0; ch < C; ch++) for (let a = 0; a < shape[0]; a++) for (let b = 0; b < shape[1]; b++) {
      const src = ch * n + (a * shape[1] + b) * shape[2];
      const dst = ch * nP + ((a + padLo[0]) * padShape[1] + (b + padLo[1])) * padShape[2] + padLo[2];
      x.set(data.subarray(src, src + shape[2]), dst);
    }
  }
  const steps = computeSteps(ps, padShape, cfg.stepSize ?? 0.5);
  const g = gaussianMap(ps);
  const agg = new Float32Array(K * nP), wsum = new Float32Array(nP);
  const pn = prod(ps);
  const patch = new Float32Array(C * pn), flipped = new Float32Array(C * pn), outAcc = new Float32Array(K * pn), outFlip = new Float32Array(K * pn);
  const combos = mirrorCombos(cfg.mirrorAxes || [0, 1, 2], tta);
  const tiles = [];
  for (const s0 of steps[0]) for (const s1 of steps[1]) for (const s2 of steps[2]) tiles.push([s0, s1, s2]);
  const total = tiles.length * combos.length;
  let done = 0;
  for (let ti = 0; ti < tiles.length; ti++) {
    const [s0, s1, s2] = tiles[ti];
    for (let ch = 0; ch < C; ch++) for (let a = 0; a < ps[0]; a++) for (let b = 0; b < ps[1]; b++) {
      const src = ch * nP + ((s0 + a) * padShape[1] + (s1 + b)) * padShape[2] + s2;
      patch.set(x.subarray(src, src + ps[2]), ch * pn + (a * ps[1] + b) * ps[2]);
    }
    outAcc.fill(0);
    for (let mi = 0; mi < combos.length; mi++) {
      const axes = combos[mi];
      // report the pass about to run, so slow machines show which patch is in progress
      if (onProgress) onProgress(done / total, { tile: ti + 1, tiles: tiles.length, mirror: mi + 1, mirrors: combos.length });
      if (shouldStop && shouldStop()) throw new Error('Cancelled');
      const inp = axes.length ? flipPatch(patch, C, ps, axes, flipped) : patch;
      const y = await runPatch(inp, [1, C, ...ps]);
      const yy = axes.length ? flipPatch(y, K, ps, axes, outFlip) : y;
      for (let i = 0; i < outAcc.length; i++) outAcc[i] += yy[i];
      done++;
      if (onProgress) onProgress(done / total, { tile: ti + 1, tiles: tiles.length, mirror: mi + 1, mirrors: combos.length });
    }
    const inv = 1 / combos.length;
    for (let k = 0; k < K; k++) for (let a = 0; a < ps[0]; a++) for (let b = 0; b < ps[1]; b++) {
      const po = (a * ps[1] + b) * ps[2];
      const dst = k * nP + ((s0 + a) * padShape[1] + (s1 + b)) * padShape[2] + s2;
      const wdst = ((s0 + a) * padShape[1] + (s1 + b)) * padShape[2] + s2;
      for (let c = 0; c < ps[2]; c++) {
        const gw = g[po + c];
        agg[dst + c] += outAcc[k * pn + po + c] * inv * gw;
        if (k === 0) wsum[wdst + c] += gw;
      }
    }
  }
  // normalise and remove padding
  const out = new Float32Array(K * n);
  for (let k = 0; k < K; k++) for (let a = 0; a < shape[0]; a++) for (let b = 0; b < shape[1]; b++) {
    const src = ((a + padLo[0]) * padShape[1] + (b + padLo[1])) * padShape[2] + padLo[2];
    const dst = k * n + (a * shape[1] + b) * shape[2];
    for (let c = 0; c < shape[2]; c++) out[dst + c] = agg[k * nP + src + c] / wsum[src + c];
  }
  return out;
}

// Full prediction.
//   channels: array of Float32Array (numpy C-order, all with shape shapeNp)
//   spacingNp: voxel spacing in numpy axis order
//   runPatch(Float32Array, dims) -> Promise<Float32Array> (network output for one patch)
// Returns { seg: Uint8Array (shapeNp), info }
export async function predictCase(channels, shapeNp, spacingNp, cfg, runPatch, opts = {}) {
  const C = channels.length;
  if (C !== cfg.inputChannels.length) throw new Error('Expected ' + cfg.inputChannels.length + ' input channels');
  const report = (stage, frac, extra) => opts.onProgress && opts.onProgress(stage, frac, extra);
  const tf = cfg.transposeForward || [0, 1, 2], tb = cfg.transposeBackward || [0, 1, 2];
  // 1) transpose
  let shape = tf.map(p => shapeNp[p]);
  const spacing = tf.map(p => spacingNp[p]);
  let ch = channels.map(c => transpose3(c, shapeNp, tf).data);
  // 2) crop to nonzero (union over channels, holes filled)
  const n0 = prod(shape);
  let nz = new Uint8Array(n0);
  for (const c of ch) for (let i = 0; i < n0; i++) if (c[i] !== 0) nz[i] = 1;
  nz = fillHoles(nz, shape);
  const lo = [Infinity, Infinity, Infinity], hi = [-1, -1, -1];
  for (let a = 0; a < shape[0]; a++) for (let b = 0; b < shape[1]; b++) for (let c = 0; c < shape[2]; c++) {
    if (nz[(a * shape[1] + b) * shape[2] + c]) {
      if (a < lo[0]) lo[0] = a; if (a > hi[0]) hi[0] = a; if (b < lo[1]) lo[1] = b; if (b > hi[1]) hi[1] = b; if (c < lo[2]) lo[2] = c; if (c > hi[2]) hi[2] = c;
    }
  }
  if (hi[0] < 0) throw new Error('Input images are empty');
  const cs = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
  const crop = arr => {
    const out = new (arr.constructor)(prod(cs));
    let o = 0;
    for (let a = 0; a < cs[0]; a++) for (let b = 0; b < cs[1]; b++) {
      const src = ((a + lo[0]) * shape[1] + (b + lo[1])) * shape[2] + lo[2];
      out.set(arr.subarray(src, src + cs[2]), o); o += cs[2];
    }
    return out;
  };
  ch = ch.map(crop);
  const mask = crop(nz);
  // 3) target shape
  const target = cfg.targetSpacing;
  const newShape = cs.map((s, d) => Math.round(spacing[d] / target[d] * s));
  const zAx = decideSeparateZ(spacing, target, cfg.anisotropyThreshold ?? 3);
  const norm = (arrs, msk) => arrs.map((c, i) => {
    const nrm = cfg.normalization[i];
    const out = new Float32Array(c.length);
    let s = 0, s2 = 0, cnt = 0;
    const use = nrm.useMask;
    for (let v = 0; v < c.length; v++) if (!use || msk[v]) { s += c[v]; cnt++; }
    const mean = s / Math.max(1, cnt);
    for (let v = 0; v < c.length; v++) if (!use || msk[v]) { const d = c[v] - mean; s2 += d * d; }
    const std = Math.sqrt(s2 / Math.max(1, cnt));
    if (cfg.nnunet === 1) {
      const den = std + 1e-8;
      for (let v = 0; v < c.length; v++) out[v] = (!use || msk[v]) ? (c[v] - mean) / den : 0;
    } else {
      const den = Math.max(std, 1e-8);
      for (let v = 0; v < c.length; v++) out[v] = (!use || msk[v]) ? (c[v] - mean) / den : c[v];
    }
    return out;
  });
  report('preprocess', 0.2);
  const needResample = newShape.some((s, d) => s !== cs[d]);
  let pre;
  if (cfg.nnunet === 2) {
    const normed = norm(ch, mask);
    pre = needResample ? normed.map(c => resampleNp(c, cs, newShape, cfg.resampleData.order, cfg.resampleData.orderZ, zAx)) : normed;
  } else {
    const res = needResample ? ch.map(c => resampleNp(c, cs, newShape, cfg.resampleData.order, cfg.resampleData.orderZ, zAx)) : ch;
    let m2 = mask;
    if (needResample) {
      // float64 like skimage, so ties at 0.5 resolve the same way
      const mf = resampleNp(Float64Array.from(mask), cs, newShape, 1, 0, zAx, Float64Array);
      m2 = new Uint8Array(mf.length); for (let i = 0; i < mf.length; i++) m2[i] = mf[i] >= 0.5 ? 1 : 0;
    }
    pre = norm(res, m2);
  }
  report('preprocess', 1);
  // 4) sliding window
  const np = prod(newShape);
  const stacked = new Float32Array(C * np);
  pre.forEach((c, i) => stacked.set(c, i * np));
  pre = null; ch = null;
  if (opts.debug) opts.debug.pre = stacked.slice();
  const probs = await slidingWindow(stacked, C, newShape, cfg, runPatch, {
    tta: opts.tta, shouldStop: opts.shouldStop,
    onProgress: (f, e) => report('network', f, e),
  });
  if (opts.debug) opts.debug.probs = probs;
  // 5) back to cropped original shape (order 1, separate z as nnU-Net decides on export)
  const K = cfg.numClasses, nc = prod(cs);
  // nnU-Net v1 (export) looks at the original spacing first; v2 (determine_do_sep_z_and_axis with
  // the current = target spacing) at the target spacing first
  let zAxB = null;
  const [firstSp, secondSp] = cfg.nnunet === 2 ? [target, spacing] : [spacing, target];
  if (getDoSeparateZ(firstSp, cfg.anisotropyThreshold ?? 3)) { const ax = lowresAxes(firstSp); if (ax.length === 1) zAxB = ax[0]; }
  else if (getDoSeparateZ(secondSp, cfg.anisotropyThreshold ?? 3)) { const ax = lowresAxes(secondSp); if (ax.length === 1) zAxB = ax[0]; }
  const back = [];
  for (let k = 0; k < K; k++) {
    const pk = probs.subarray(k * np, (k + 1) * np);
    back.push(needResample ? resampleNp(pk, newShape, cs, cfg.resampleProb.order, cfg.resampleProb.orderZ, zAxB) : new Float32Array(pk));
  }
  // 6) labels (argmax; or for region-based models, nnU-Net's region_class_order: each region's
  //    sigmoid > 0.5 writes its label, later regions overwriting earlier ones), uncrop, transpose back
  const segC = new Uint8Array(nc);
  if (cfg.regionClassOrder) {
    cfg.regionClassOrder.forEach((lab, i) => { const r = back[i]; for (let v = 0; v < nc; v++) if (r[v] > 0.5) segC[v] = lab; });
  } else {
    for (let v = 0; v < nc; v++) { let best = 0, bv = back[0][v]; for (let k = 1; k < K; k++) if (back[k][v] > bv) { bv = back[k][v]; best = k; } segC[v] = best; }
  }
  const segT = new Float32Array(n0);
  for (let a = 0; a < cs[0]; a++) for (let b = 0; b < cs[1]; b++) for (let c = 0; c < cs[2]; c++) segT[((a + lo[0]) * shape[1] + (b + lo[1])) * shape[2] + c + lo[2]] = segC[(a * cs[1] + b) * cs[2] + c];
  const segBack = transpose3(segT, shape, tb).data;
  const seg = Uint8Array.from(segBack);
  // 7) postprocessing (connected components), on the original grid
  if (cfg.postprocessing && cfg.postprocessing.forWhichClasses && cfg.postprocessing.forWhichClasses.length) {
    keepLargestComponents(seg, shapeNp, cfg.postprocessing.forWhichClasses, spacingNp[0] * spacingNp[1] * spacingNp[2], cfg.postprocessing.minValidObjectSizes, cfg.postprocessing.connectivity || 6);
  }
  // 8) optional relabelling to the published convention (e.g. BraTS: 1 NCR, 2 ED, 4 ET)
  if (cfg.labelMap) {
    const lut = new Uint8Array(256); for (let i = 0; i < 256; i++) lut[i] = i;
    for (const [from, to] of Object.entries(cfg.labelMap)) lut[+from] = to;
    for (let v = 0; v < seg.length; v++) seg[v] = lut[seg[v]];
  }
  return { seg, info: { cropShape: cs, preShape: newShape, separateZ: zAx, separateZBack: zAxB, bbox: [lo, hi] } };
}
