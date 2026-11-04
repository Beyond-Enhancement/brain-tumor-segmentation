// NIfTI-1 / NIfTI-2 reader and NIfTI-1 writer.
// Volumes are { dims:[nx,ny,nz], affine: Float64Array(16) row-major voxel->RAS mm, data: Float32Array }
// with x varying fastest (NIfTI storage order).
import { gunzipSync, gzipSync } from '../../vendor/fflate.mjs';

const DT = { 2: 'u8', 4: 'i16', 8: 'i32', 16: 'f32', 64: 'f64', 256: 'i8', 512: 'u16', 768: 'u32', 1024: 'i64', 1280: 'u64' };

export function isGzip(u8) { return u8.length > 2 && u8[0] === 0x1f && u8[1] === 0x8b; }

export function readNifti(buf) {
  let u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  if (isGzip(u8)) u8 = gunzipSync(u8);
  const ab = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
  const dv = new DataView(ab);
  let le = true;
  let sizeof = dv.getInt32(0, true);
  if (sizeof !== 348 && sizeof !== 540) { le = false; sizeof = dv.getInt32(0, false); }
  if (sizeof !== 348 && sizeof !== 540) throw new Error('Not a NIfTI file (bad header size)');
  const v2 = sizeof === 540;
  const h = {};
  if (!v2) {
    const dim = []; for (let i = 0; i < 8; i++) dim.push(dv.getInt16(40 + 2 * i, le));
    const pix = []; for (let i = 0; i < 8; i++) pix.push(dv.getFloat32(76 + 4 * i, le));
    Object.assign(h, {
      dim, pixdim: pix, datatype: dv.getInt16(70, le), bitpix: dv.getInt16(72, le),
      vox_offset: dv.getFloat32(108, le), scl_slope: dv.getFloat32(112, le), scl_inter: dv.getFloat32(116, le),
      qform_code: dv.getInt16(252, le), sform_code: dv.getInt16(254, le),
      quatern_b: dv.getFloat32(256, le), quatern_c: dv.getFloat32(260, le), quatern_d: dv.getFloat32(264, le),
      qoffset_x: dv.getFloat32(268, le), qoffset_y: dv.getFloat32(272, le), qoffset_z: dv.getFloat32(276, le),
      srow: [0, 1, 2].map(r => [0, 1, 2, 3].map(c => dv.getFloat32(280 + 16 * r + 4 * c, le))),
    });
  } else {
    const dim = []; for (let i = 0; i < 8; i++) dim.push(Number(dv.getBigInt64(16 + 8 * i, le)));
    const pix = []; for (let i = 0; i < 8; i++) pix.push(dv.getFloat64(104 + 8 * i, le));
    Object.assign(h, {
      dim, pixdim: pix, datatype: dv.getInt16(12, le), bitpix: dv.getInt16(14, le),
      vox_offset: Number(dv.getBigInt64(168, le)), scl_slope: dv.getFloat64(176, le), scl_inter: dv.getFloat64(184, le),
      qform_code: dv.getInt32(344, le), sform_code: dv.getInt32(348, le),
      quatern_b: dv.getFloat64(352, le), quatern_c: dv.getFloat64(360, le), quatern_d: dv.getFloat64(368, le),
      qoffset_x: dv.getFloat64(376, le), qoffset_y: dv.getFloat64(384, le), qoffset_z: dv.getFloat64(392, le),
      srow: [0, 1, 2].map(r => [0, 1, 2, 3].map(c => dv.getFloat64(400 + 32 * r + 8 * c, le))),
    });
  }
  const nd = h.dim[0];
  const nx = h.dim[1], ny = nd > 1 ? h.dim[2] : 1, nz = nd > 2 ? h.dim[3] : 1;
  const nt = nd > 3 ? Math.max(1, h.dim[4]) : 1;
  const type = DT[h.datatype];
  if (!type) throw new Error('Unsupported NIfTI datatype ' + h.datatype);
  const n = nx * ny * nz;
  const bpv = { u8: 1, i8: 1, i16: 2, u16: 2, i32: 4, u32: 4, f32: 4, f64: 8, i64: 8, u64: 8 }[type];
  const off = Math.round(h.vox_offset);
  if (off + n * bpv > ab.byteLength) throw new Error('NIfTI file is truncated');
  const out = new Float32Array(n);
  let slope = h.scl_slope, inter = h.scl_inter;
  if (!isFinite(slope) || slope === 0) { slope = 1; inter = 0; }
  if (!isFinite(inter)) inter = 0;
  const rd = {
    u8: o => dv.getUint8(o), i8: o => dv.getInt8(o), i16: o => dv.getInt16(o, le), u16: o => dv.getUint16(o, le),
    i32: o => dv.getInt32(o, le), u32: o => dv.getUint32(o, le), f32: o => dv.getFloat32(o, le), f64: o => dv.getFloat64(o, le),
    i64: o => Number(dv.getBigInt64(o, le)), u64: o => Number(dv.getBigUint64(o, le)),
  }[type];
  // fast paths for common little-endian aligned types
  if (le && type === 'f32' && off % 4 === 0) out.set(new Float32Array(ab, off, n));
  else if (le && type === 'i16' && off % 2 === 0) { const s = new Int16Array(ab, off, n); for (let i = 0; i < n; i++) out[i] = s[i]; }
  else if (le && type === 'u16' && off % 2 === 0) { const s = new Uint16Array(ab, off, n); for (let i = 0; i < n; i++) out[i] = s[i]; }
  else if (type === 'u8') { const s = new Uint8Array(ab, off, n); for (let i = 0; i < n; i++) out[i] = s[i]; }
  else for (let i = 0; i < n; i++) out[i] = rd(off + i * bpv);
  if (slope !== 1 || inter !== 0) for (let i = 0; i < n; i++) out[i] = out[i] * slope + inter;
  return { dims: [nx, ny, nz], affine: headerAffine(h), data: out, nt, header: h };
}

function headerAffine(h) {
  const A = new Float64Array(16); A[15] = 1;
  if (h.sform_code > 0) {
    for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) A[4 * r + c] = h.srow[r][c];
    return A;
  }
  const dx = h.pixdim[1], dy = h.pixdim[2] || 1, dz = h.pixdim[3] || 1;
  if (h.qform_code > 0) {
    const b = h.quatern_b, c = h.quatern_c, d = h.quatern_d;
    let a = 1 - (b * b + c * c + d * d);
    if (a < 1e-7) { a = 1 / Math.sqrt(b * b + c * c + d * d); } else a = Math.sqrt(a);
    const qfac = h.pixdim[0] < 0 ? -1 : 1;
    const R = [
      [a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
      [2 * b * c + 2 * a * d, a * a + c * c - b * b - d * d, 2 * c * d - 2 * a * b],
      [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a * a + d * d - c * c - b * b]];
    const s = [dx, dy, dz * qfac];
    for (let r = 0; r < 3; r++) for (let k = 0; k < 3; k++) A[4 * r + k] = R[r][k] * s[k];
    A[3] = h.qoffset_x; A[7] = h.qoffset_y; A[11] = h.qoffset_z;
    return A;
  }
  // ANALYZE-style fallback
  A[0] = dx; A[5] = dy; A[10] = dz;
  return A;
}

// mat44 -> quaternion (port of nifti_mat44_to_quatern)
function affineToQuatern(A) {
  let r11 = A[0], r12 = A[1], r13 = A[2], r21 = A[4], r22 = A[5], r23 = A[6], r31 = A[8], r32 = A[9], r33 = A[10];
  let xd = Math.hypot(r11, r21, r31), yd = Math.hypot(r12, r22, r32), zd = Math.hypot(r13, r23, r33);
  if (xd === 0) { r11 = 1; r21 = r31 = 0; xd = 1; }
  if (yd === 0) { r22 = 1; r12 = r32 = 0; yd = 1; }
  if (zd === 0) { r33 = 1; r13 = r23 = 0; zd = 1; }
  r11 /= xd; r21 /= xd; r31 /= xd; r12 /= yd; r22 /= yd; r32 /= yd; r13 /= zd; r23 /= zd; r33 /= zd;
  // orthogonalise via polar decomposition (simple Gram-Schmidt is enough for our affines)
  const det = r11 * r22 * r33 - r11 * r32 * r23 - r21 * r12 * r33 + r21 * r32 * r13 + r31 * r12 * r23 - r31 * r22 * r13;
  let qfac = 1;
  if (det <= 0) { qfac = -1; r13 = -r13; r23 = -r23; r33 = -r33; }
  let a = r11 + r22 + r33 + 1, b, c, d;
  if (a > 0.5) {
    a = 0.5 * Math.sqrt(a); b = 0.25 * (r32 - r23) / a; c = 0.25 * (r13 - r31) / a; d = 0.25 * (r21 - r12) / a;
  } else {
    const xx = 1 + r11 - r22 - r33, yy = 1 - r11 + r22 - r33, zz = 1 - r11 - r22 + r33;
    if (xx > 1) { b = 0.5 * Math.sqrt(xx); c = 0.25 * (r12 + r21) / b; d = 0.25 * (r13 + r31) / b; a = 0.25 * (r32 - r23) / b; }
    else if (yy > 1) { c = 0.5 * Math.sqrt(yy); b = 0.25 * (r12 + r21) / c; d = 0.25 * (r23 + r32) / c; a = 0.25 * (r13 - r31) / c; }
    else { d = 0.5 * Math.sqrt(zz); b = 0.25 * (r13 + r31) / d; c = 0.25 * (r23 + r32) / d; a = 0.25 * (r21 - r12) / d; }
    if (a < 0) { b = -b; c = -c; d = -d; }
  }
  return { b, c, d, qfac, dx: xd, dy: yd, dz: zd };
}

// Write a NIfTI-1 file. dtype: 'u8' | 'i16' | 'f32'. Returns Uint8Array (gzipped when gz=true).
export function writeNifti(vol, { dtype = 'f32', gz = true, descrip = '' } = {}) {
  const [nx, ny, nz] = vol.dims;
  const n = nx * ny * nz;
  const code = { u8: 2, i16: 4, f32: 16 }[dtype];
  const bpv = { u8: 1, i16: 2, f32: 4 }[dtype];
  const off = 352;
  const ab = new ArrayBuffer(off + n * bpv);
  const dv = new DataView(ab);
  dv.setInt32(0, 348, true);
  const dim = [3, nx, ny, nz, 1, 1, 1, 1];
  dim.forEach((v, i) => dv.setInt16(40 + 2 * i, v, true));
  dv.setInt16(70, code, true); dv.setInt16(72, bpv * 8, true);
  const A = vol.affine;
  const q = affineToQuatern(A);
  const pix = [q.qfac, q.dx, q.dy, q.dz, 1, 1, 1, 1];
  pix.forEach((v, i) => dv.setFloat32(76 + 4 * i, v, true));
  dv.setFloat32(108, off, true);
  dv.setFloat32(112, 1, true); dv.setFloat32(116, 0, true);
  dv.setUint8(123, 10); // xyzt_units: mm + sec
  const d = String(descrip).slice(0, 79);
  for (let i = 0; i < d.length; i++) dv.setUint8(148 + i, d.charCodeAt(i) & 0x7f);
  dv.setInt16(252, 1, true); dv.setInt16(254, 1, true); // qform/sform: scanner anatomical
  dv.setFloat32(256, q.b, true); dv.setFloat32(260, q.c, true); dv.setFloat32(264, q.d, true);
  dv.setFloat32(268, A[3], true); dv.setFloat32(272, A[7], true); dv.setFloat32(276, A[11], true);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) dv.setFloat32(280 + 16 * r + 4 * c, A[4 * r + c], true);
  const magic = 'n+1\0';
  for (let i = 0; i < 4; i++) dv.setUint8(344 + i, magic.charCodeAt(i));
  const src = vol.data;
  if (dtype === 'u8') { const t = new Uint8Array(ab, off, n); for (let i = 0; i < n; i++) t[i] = src[i]; }
  else if (dtype === 'i16') { const t = new Int16Array(ab, off, n); for (let i = 0; i < n; i++) t[i] = Math.max(-32768, Math.min(32767, Math.round(src[i]))); }
  else { new Float32Array(ab, off, n).set(src); }
  const u8 = new Uint8Array(ab);
  return gz ? gzipSync(u8, { level: 6, mtime: 0 }) : u8;
}
