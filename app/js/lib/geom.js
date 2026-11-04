// Geometry and resampling.
// A volume is { dims:[nx,ny,nz], affine: Float64Array(16) row-major voxel(i,j,k)->RAS mm, data: Float32Array }
// with index = i + nx*(j + ny*k).

export function mul4(A, B) {
  const C = new Float64Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    let s = 0; for (let k = 0; k < 4; k++) s += A[4 * r + k] * B[4 * k + c]; C[4 * r + c] = s;
  }
  return C;
}

export function inv4(m) {
  const inv = new Float64Array(16);
  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
  let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  if (Math.abs(det) < 1e-12) throw new Error('Singular matrix');
  det = 1 / det;
  for (let i = 0; i < 16; i++) inv[i] *= det;
  return inv;
}

export function spacing(A) {
  return [0, 1, 2].map(c => Math.hypot(A[c], A[4 + c], A[8 + c]));
}

export function voxelVolume(A) {
  return Math.abs(A[0] * (A[5] * A[10] - A[6] * A[9]) - A[1] * (A[4] * A[10] - A[6] * A[8]) + A[2] * (A[4] * A[9] - A[5] * A[8]));
}

export function cloneVol(v, data) {
  return { dims: v.dims.slice(), affine: new Float64Array(v.affine), data: data || new Float32Array(v.data) };
}

// ---------------------------------------------------------------------------------------
// Reorientation (equivalent to fslreorient2std: permute/flip voxel axes so that
// i runs towards Left, j towards Anterior, k towards Superior, i.e. MNI152 storage "LAS").

const PERMS = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
// For each voxel axis c: the world axis it is closest to and the sign. Uses the permutation of
// world axes that maximises the summed |cosine|, so oblique acquisitions map sensibly.
export function orientationOf(A) {
  const cols = [0, 1, 2].map(c => { const v = [A[c], A[4 + c], A[8 + c]]; const n = Math.hypot(...v) || 1; return v.map(x => x / n); });
  let best = null, bs = -1;
  for (const p of PERMS) { const sc = Math.abs(cols[0][p[0]]) + Math.abs(cols[1][p[1]]) + Math.abs(cols[2][p[2]]); if (sc > bs) { bs = sc; best = p; } }
  return [0, 1, 2].map(c => ({ world: best[c], sign: Math.sign(cols[c][best[c]]) || 1 }));
}

export function orientationCode(A) {
  const L = [['L', 'R'], ['P', 'A'], ['I', 'S']];
  return orientationOf(A).map(o => L[o.world][o.sign > 0 ? 1 : 0]).join('');
}

// Returns a plan { perm, flip, dims, affine } mapping input voxel axes to output axes such that
// output axis a corresponds to world axis a; target signs: x negative (towards L), y,z positive.
export function reorientPlan(vol) {
  const o = orientationOf(vol.affine);
  const targetSign = [-1, 1, 1];
  const perm = [0, 0, 0], flip = [false, false, false];
  for (let c = 0; c < 3; c++) { perm[o[c].world] = c; }
  for (let a = 0; a < 3; a++) flip[a] = o[perm[a]].sign !== targetSign[a];
  const dims = perm.map(c => vol.dims[c]);
  // new voxel index n (output) -> old index: old[perm[a]] = flip[a] ? dims[a]-1-n[a] : n[a]
  const M = new Float64Array(16); M[15] = 1; // old_index = M * new_index
  for (let a = 0; a < 3; a++) {
    const c = perm[a];
    if (flip[a]) { M[4 * c + a] = -1; M[4 * c + 3] = dims[a] - 1; } else M[4 * c + a] = 1;
  }
  const affine = mul4(vol.affine, M);
  return { perm, flip, dims, affine, M };
}

export function applyReorient(vol, plan) {
  const [nx, ny, nz] = plan.dims;
  const [ox, oy] = vol.dims;
  const out = new Float32Array(nx * ny * nz);
  const strideOld = [1, ox, ox * oy];
  // stride in old array when moving along each new axis
  const st = [0, 0, 0], base = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const c = plan.perm[a];
    st[a] = plan.flip[a] ? -strideOld[c] : strideOld[c];
    base[a] = plan.flip[a] ? (plan.dims[a] - 1) * strideOld[c] : 0;
  }
  const b0 = base[0] + base[1] + base[2];
  let p = 0;
  const src = vol.data;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) {
    let q = b0 + j * st[1] + k * st[2];
    for (let i = 0; i < nx; i++, q += st[0]) out[p++] = src[q];
  }
  return { dims: plan.dims.slice(), affine: new Float64Array(plan.affine), data: out };
}

// Inverse: take data in the reoriented grid back to the original voxel order.
export function undoReorient(dataNew, plan, origDims, Ctor = Float32Array) {
  const [ox, oy, oz] = origDims;
  const out = new Ctor(ox * oy * oz);
  const strideOld = [1, ox, ox * oy];
  const st = [0, 0, 0], base = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const c = plan.perm[a];
    st[a] = plan.flip[a] ? -strideOld[c] : strideOld[c];
    base[a] = plan.flip[a] ? (plan.dims[a] - 1) * strideOld[c] : 0;
  }
  const b0 = base[0] + base[1] + base[2];
  const [nx, ny, nz] = plan.dims;
  let p = 0;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) {
    let q = b0 + j * st[1] + k * st[2];
    for (let i = 0; i < nx; i++, q += st[0]) out[q] = dataNew[p++];
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Cubic B-spline (scipy.ndimage compatible: mode 'nearest' handled by padding 12 voxels with
// edge values and filtering with mirror boundary, as scipy >= 1.6 does).

const Z = Math.sqrt(3) - 2; // pole of cubic B-spline
const NPAD = 12;

// In-place 1D prefilter with mirror boundary conditions (scipy 'mirror').
function prefilter1D(c, n) {
  if (n === 1) return;
  const lambda = (1 - Z) * (1 - 1 / Z);
  for (let i = 0; i < n; i++) c[i] *= lambda;
  // causal init (mirror): sum_{k=0}^{n-1} z^k c[k] + ... use exact mirror init
  let zn = Z, iz = 1 / Z, z2n = Math.pow(Z, n - 1);
  let sum = c[0] + z2n * c[n - 1];
  z2n *= z2n * iz;
  for (let k = 1; k <= n - 2; k++) { sum += (zn + z2n) * c[k]; zn *= Z; z2n *= iz; }
  c[0] = sum / (1 - Math.pow(Z, 2 * (n - 1)));
  for (let k = 1; k < n; k++) c[k] += Z * c[k - 1];
  c[n - 1] = (Z / (Z * Z - 1)) * (Z * c[n - 2] + c[n - 1]);
  for (let k = n - 2; k >= 0; k--) c[k] = Z * (c[k + 1] - c[k]);
}

function bsplineW(t) { // weights for offsets -1,0,1,2 given fractional t in [0,1)
  const t2 = t * t, t3 = t2 * t, omt = 1 - t;
  return [omt * omt * omt / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6];
}
function mirrorIdx(i, n) { // scipy 'mirror' reflection for coefficient access
  if (n === 1) return 0;
  const p = 2 * n - 2;
  i = ((i % p) + p) % p;
  return i < n ? i : p - i;
}

// Resample a 1D line (length nIn, stride sIn in src) to nOut samples at coordinates coord(o).
// order: 0 nearest, 1 linear, 3 cubic B-spline. Writes to dst with stride sOut.
function resampleLine(src, offIn, sIn, nIn, dst, offOut, sOut, nOut, scale, order, buf) {
  // coordinate mapping (grid_mode=True / skimage resize): x_in = (o + 0.5) * scale - 0.5
  if (order === 0) {
    for (let o = 0; o < nOut; o++) {
      let x = (o + 0.5) * scale - 0.5;
      let i = Math.floor(x + 0.5); if (i < 0) i = 0; if (i > nIn - 1) i = nIn - 1;
      dst[offOut + o * sOut] = src[offIn + i * sIn];
    }
    return;
  }
  if (order === 1) {
    for (let o = 0; o < nOut; o++) {
      let x = (o + 0.5) * scale - 0.5;
      if (x < 0) x = 0; if (x > nIn - 1) x = nIn - 1;
      const i0 = Math.floor(x), f = x - i0, i1 = i0 + 1 < nIn ? i0 + 1 : i0;
      dst[offOut + o * sOut] = src[offIn + i0 * sIn] * (1 - f) + src[offIn + i1 * sIn] * f;
    }
    return;
  }
  // cubic: pad NPAD with edge values, prefilter (mirror), evaluate
  const np = nIn + 2 * NPAD;
  const c = buf;
  const first = src[offIn], last = src[offIn + (nIn - 1) * sIn];
  for (let i = 0; i < NPAD; i++) { c[i] = first; c[NPAD + nIn + i] = last; }
  for (let i = 0; i < nIn; i++) c[NPAD + i] = src[offIn + i * sIn];
  prefilter1D(c, np);
  for (let o = 0; o < nOut; o++) {
    let x = (o + 0.5) * scale - 0.5;
    // scipy 'nearest' mode: coordinates outside are clamped to the edge
    if (x < -0.5) x = -0.5; // beyond the padded region the padding (edge value) dominates anyway
    if (x > nIn - 0.5) x = nIn - 0.5;
    const xp = x + NPAD;
    const i0 = Math.floor(xp), t = xp - i0;
    const w = bsplineW(t);
    let v = 0;
    for (let k = -1; k <= 2; k++) v += w[k + 1] * c[mirrorIdx(i0 + k, np)];
    dst[offOut + o * sOut] = v;
  }
}

// skimage.transform.resize(img, newDims, order, mode='edge', anti_aliasing=False, clip=True)
// for a 3D volume stored x-fastest with dims [nx,ny,nz]. axesOrder lets callers resample
// some axes with a different order (orderPerAxis = [ox, oy, oz]).
export function resizeVolume(data, dims, newDims, orderPerAxis, clip = true, Ctor = Float32Array) {
  let cur = data, cd = dims.slice();
  let mn = Infinity, mx = -Infinity;
  if (clip && orderPerAxis.some(o => o > 1)) { for (let i = 0; i < data.length; i++) { const v = data[i]; if (v < mn) mn = v; if (v > mx) mx = v; } }
  const maxN = Math.max(...dims, ...newDims) + 2 * NPAD + 4;
  const buf = new Float64Array(maxN);
  for (let ax = 0; ax < 3; ax++) {
    if (newDims[ax] === cd[ax]) continue;
    const nd = cd.slice(); nd[ax] = newDims[ax];
    const out = new Ctor(nd[0] * nd[1] * nd[2]);
    const sIn = ax === 0 ? 1 : ax === 1 ? cd[0] : cd[0] * cd[1];
    const sOut = ax === 0 ? 1 : ax === 1 ? nd[0] : nd[0] * nd[1];
    const scale = cd[ax] / nd[ax];
    const o1 = (ax + 1) % 3, o2 = (ax + 2) % 3;
    const sIn1 = o1 === 0 ? 1 : o1 === 1 ? cd[0] : cd[0] * cd[1];
    const sIn2 = o2 === 0 ? 1 : o2 === 1 ? cd[0] : cd[0] * cd[1];
    const sOut1 = o1 === 0 ? 1 : o1 === 1 ? nd[0] : nd[0] * nd[1];
    const sOut2 = o2 === 0 ? 1 : o2 === 1 ? nd[0] : nd[0] * nd[1];
    for (let b = 0; b < cd[o2]; b++) for (let a = 0; a < cd[o1]; a++) {
      resampleLine(cur, a * sIn1 + b * sIn2, sIn, cd[ax], out, a * sOut1 + b * sOut2, sOut, nd[ax], scale, orderPerAxis[ax], buf);
    }
    cur = out; cd = nd;
  }
  if (cur === data) cur = new Ctor(data);
  if (clip && isFinite(mn)) for (let i = 0; i < cur.length; i++) { if (cur[i] < mn) cur[i] = mn; else if (cur[i] > mx) cur[i] = mx; }
  return cur;
}

// nnU-Net "separate z" resampling for anisotropic data: each slice perpendicular to zAxis is
// resized in 2D (order `order`, clipped to that slice's own range, as skimage does per slice),
// then the zAxis is resampled with `orderZ` (0 = nearest, 1 = linear), without clipping.
export function resizeSeparateZ(data, dims, newDims, zAxis, order, orderZ, Ctor = Float32Array) {
  const inPlane = [0, 1, 2].filter(a => a !== zAxis);
  const mid = dims.slice(); inPlane.forEach(a => { mid[a] = newDims[a]; });
  const midData = new Ctor(mid[0] * mid[1] * mid[2]);
  const sliceDims = dims.slice(); sliceDims[zAxis] = 1;
  const sliceNew = mid.slice(); sliceNew[zAxis] = 1;
  const nIn = sliceDims[0] * sliceDims[1] * sliceDims[2];
  const nOut = sliceNew[0] * sliceNew[1] * sliceNew[2];
  const tmpIn = new Ctor(nIn);
  const stride = a => (a === 0 ? 1 : a === 1 ? dims[0] : dims[0] * dims[1]);
  const strideM = a => (a === 0 ? 1 : a === 1 ? mid[0] : mid[0] * mid[1]);
  const [a0, a1] = inPlane;
  for (let z = 0; z < dims[zAxis]; z++) {
    let p = 0;
    for (let v = 0; v < dims[a1]; v++) for (let u = 0; u < dims[a0]; u++) tmpIn[p++] = data[u * stride(a0) + v * stride(a1) + z * stride(zAxis)];
    const d2 = [dims[a0], dims[a1], 1], n2 = [mid[a0], mid[a1], 1];
    const r = resizeVolume(tmpIn, d2, n2, [order, order, 0], true, Ctor);
    p = 0;
    for (let v = 0; v < mid[a1]; v++) for (let u = 0; u < mid[a0]; u++) midData[u * strideM(a0) + v * strideM(a1) + z * strideM(zAxis)] = r[p++];
  }
  if (mid[zAxis] === newDims[zAxis]) return midData;
  const ord = [0, 0, 0]; ord[zAxis] = orderZ;
  return resizeVolume(midData, mid, newDims, ord, false, Ctor);
}

// ---------------------------------------------------------------------------------------
// World-space resampling of a moving volume onto a target grid, through an optional rigid
// transform T (4x4, maps target-world -> moving-world).

export function bsplineCoefficients3D(vol) {
  // in-place style prefilter along each axis with mirror boundaries (used for 'cubic' world resampling)
  const [nx, ny, nz] = vol.dims;
  const c = new Float32Array(vol.data);
  const buf = new Float64Array(Math.max(nx, ny, nz));
  const run = (n, off, stride) => {
    for (let i = 0; i < n; i++) buf[i] = c[off + i * stride];
    prefilter1D(buf, n);
    for (let i = 0; i < n; i++) c[off + i * stride] = buf[i];
  };
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) run(nx, nx * (j + ny * k), 1);
  for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) run(ny, i + nx * ny * k, nx);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) run(nz, i + nx * j, nx * ny);
  return c;
}

export function resampleToGrid(mov, dims, affine, T, interp = 'linear', outside = 0) {
  const [nx, ny, nz] = dims;
  const out = new Float32Array(nx * ny * nz);
  // voxel(target) -> voxel(moving) = inv(Am) * T * At
  let M = affine;
  if (T) M = mul4(T, M);
  M = mul4(inv4(mov.affine), M);
  const [mx, my, mz] = mov.dims;
  const src = interp === 'cubic' ? bsplineCoefficients3D(mov) : mov.data;
  const sxy = mx * my;
  let p = 0;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) {
    let x = M[1] * j + M[2] * k + M[3], y = M[5] * j + M[6] * k + M[7], z = M[9] * j + M[10] * k + M[11];
    for (let i = 0; i < nx; i++, p++, x += M[0], y += M[4], z += M[8]) {
      if (x < -0.5 || y < -0.5 || z < -0.5 || x > mx - 0.5 || y > my - 0.5 || z > mz - 0.5) { out[p] = outside; continue; }
      if (interp === 'nearest') {
        const ii = Math.min(mx - 1, Math.max(0, Math.round(x))), jj = Math.min(my - 1, Math.max(0, Math.round(y))), kk = Math.min(mz - 1, Math.max(0, Math.round(z)));
        out[p] = src[ii + mx * jj + sxy * kk];
      } else if (interp === 'linear') {
        out[p] = trilinear(src, mx, my, mz, x, y, z);
      } else {
        out[p] = tricubic(src, mx, my, mz, x, y, z);
      }
    }
  }
  return out;
}

export function trilinear(src, mx, my, mz, x, y, z) {
  if (x < 0) x = 0; if (y < 0) y = 0; if (z < 0) z = 0;
  if (x > mx - 1) x = mx - 1; if (y > my - 1) y = my - 1; if (z > mz - 1) z = mz - 1;
  const i0 = Math.floor(x), j0 = Math.floor(y), k0 = Math.floor(z);
  const fx = x - i0, fy = y - j0, fz = z - k0;
  const i1 = i0 + 1 < mx ? 1 : 0, j1 = j0 + 1 < my ? mx : 0, k1 = k0 + 1 < mz ? mx * my : 0;
  const b = i0 + mx * (j0 + my * k0);
  const c00 = src[b] * (1 - fx) + src[b + i1] * fx;
  const c10 = src[b + j1] * (1 - fx) + src[b + j1 + i1] * fx;
  const c01 = src[b + k1] * (1 - fx) + src[b + k1 + i1] * fx;
  const c11 = src[b + j1 + k1] * (1 - fx) + src[b + j1 + k1 + i1] * fx;
  return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz;
}

function tricubic(c, mx, my, mz, x, y, z) {
  const i0 = Math.floor(x), j0 = Math.floor(y), k0 = Math.floor(z);
  const wx = bsplineW(x - i0), wy = bsplineW(y - j0), wz = bsplineW(z - k0);
  let v = 0;
  for (let c3 = 0; c3 < 4; c3++) {
    const kk = mirrorIdx(k0 + c3 - 1, mz) * mx * my;
    let vy = 0;
    for (let c2 = 0; c2 < 4; c2++) {
      const jj = mirrorIdx(j0 + c2 - 1, my) * mx + kk;
      let vx = 0;
      for (let c1 = 0; c1 < 4; c1++) vx += wx[c1] * c[mirrorIdx(i0 + c1 - 1, mx) + jj];
      vy += wy[c2] * vx;
    }
    v += wz[c3] * vy;
  }
  return v;
}

// Grid with isotropic spacing `sp` mm covering the same field of view as `vol`, axes aligned
// with vol's voxel axes.
export function isotropicGrid(vol, sp) {
  const s = spacing(vol.affine);
  const dims = [0, 1, 2].map(a => Math.max(1, Math.round(vol.dims[a] * s[a] / sp)));
  const f = [0, 1, 2].map(a => (vol.dims[a] * s[a] / dims[a]) / s[a]);
  const A = new Float64Array(vol.affine);
  for (let a = 0; a < 3; a++) for (let r = 0; r < 3; r++) A[4 * r + a] = vol.affine[4 * r + a] * f[a];
  // keep the outer edge of the field of view fixed
  for (let r = 0; r < 3; r++) {
    let shift = 0;
    for (let a = 0; a < 3; a++) shift += 0.5 * vol.affine[4 * r + a] * (f[a] - 1);
    A[4 * r + 3] = vol.affine[4 * r + 3] + shift;
  }
  return { dims, affine: A };
}

export function percentile(data, p, mask) {
  const vals = [];
  const step = Math.max(1, Math.floor(data.length / 400000));
  for (let i = 0; i < data.length; i += step) if (!mask || mask[i]) vals.push(data[i]);
  if (!vals.length) return 0;
  vals.sort((a, b) => a - b);
  return vals[Math.min(vals.length - 1, Math.max(0, Math.round(p / 100 * (vals.length - 1))))];
}
