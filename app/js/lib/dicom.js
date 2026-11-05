// DICOM series discovery and volume assembly (runs in the worker).
// Supports classic single-frame and enhanced multi-frame MR images; uncompressed (implicit/explicit
// little endian, explicit big endian, deflated), RLE, JPEG lossless, JPEG-LS and JPEG 2000.
import dicomParser from '../../vendor/dicomParser.mjs';
import { inflateSync } from '../../vendor/fflate.mjs';

const TS = {
  IMPLICIT_LE: '1.2.840.10008.1.2', EXPLICIT_LE: '1.2.840.10008.1.2.1', DEFLATE: '1.2.840.10008.1.2.1.99',
  EXPLICIT_BE: '1.2.840.10008.1.2.2', RLE: '1.2.840.10008.1.2.5',
  JPEG_BASELINE: '1.2.840.10008.1.2.4.50', JPEG_EXT: '1.2.840.10008.1.2.4.51',
  JPEG_LOSSLESS: '1.2.840.10008.1.2.4.57', JPEG_LOSSLESS_SV1: '1.2.840.10008.1.2.4.70',
  JPEGLS_LOSSLESS: '1.2.840.10008.1.2.4.80', JPEGLS_NEAR: '1.2.840.10008.1.2.4.81',
  J2K_LOSSLESS: '1.2.840.10008.1.2.4.90', J2K: '1.2.840.10008.1.2.4.91',
  HTJ2K_LOSSLESS: '1.2.840.10008.1.2.4.201', HTJ2K_RPCL: '1.2.840.10008.1.2.4.202', HTJ2K: '1.2.840.10008.1.2.4.203',
};
const TS_NAMES = Object.fromEntries(Object.entries(TS).map(([k, v]) => [v, k]));

function inflater(byteArray, position) {
  const inflated = inflateSync(byteArray.subarray(position));
  const full = new Uint8Array(position + inflated.length);
  full.set(byteArray.subarray(0, position), 0);
  full.set(inflated, position);
  return full;
}

function str(ds, tag) { try { const v = ds.string(tag); return v == null ? '' : String(v).trim(); } catch (e) { return ''; } }
function num(ds, tag, i = 0) { try { const v = ds.floatString(tag, i); return v == null || isNaN(v) ? null : v; } catch (e) { return null; } }
function nums(ds, tag, n) { const out = []; for (let i = 0; i < n; i++) { const v = num(ds, tag, i); if (v == null) return null; out.push(v); } return out; }
function u16(ds, tag) { try { return ds.uint16(tag); } catch (e) { return undefined; } }
function seqItems(ds, tag) { const el = ds.elements[tag]; return el && el.items ? el.items.map(it => it.dataSet) : []; }

// Parse one file. Returns null if not an image DICOM.
export function parseFile(name, u8) {
  let ds;
  try {
    // Quick reject: DICM preamble or plausible group 0x0008 start
    ds = dicomParser.parseDicom(u8, { inflater });
  } catch (e) {
    return null;
  }
  if (!ds.elements.x7fe00010) return null;
  const rows = u16(ds, 'x00280010'), cols = u16(ds, 'x00280011');
  if (!rows || !cols) return null;
  return { name, ds, u8 };
}

// Build per-frame records (one per 2D image plane) from a parsed file.
function framesOf(f) {
  const ds = f.ds;
  const nFrames = parseInt(str(ds, 'x00280008') || '1', 10) || 1;
  const shared = seqItems(ds, 'x52009229')[0];
  const perFrame = seqItems(ds, 'x52009230');
  const base = {
    file: f, rows: u16(ds, 'x00280010'), cols: u16(ds, 'x00280011'),
    bits: u16(ds, 'x00280100') || 16, pixRep: u16(ds, 'x00280103') || 0, spp: u16(ds, 'x00280002') || 1,
  };
  const get = (fg, seqTag, fn) => {
    for (const src of [fg, shared]) {
      if (!src) continue;
      const it = seqItems(src, seqTag)[0];
      if (it) { const v = fn(it); if (v != null) return v; }
    }
    return null;
  };
  const out = [];
  for (let fi = 0; fi < nFrames; fi++) {
    const fg = perFrame[fi];
    const ipp = get(fg, 'x00209113', it => nums(it, 'x00200032', 3)) || nums(ds, 'x00200032', 3);
    const iop = get(fg, 'x00209116', it => nums(it, 'x00200037', 6)) || nums(ds, 'x00200037', 6);
    const ps = get(fg, 'x00289110', it => nums(it, 'x00280030', 2)) || nums(ds, 'x00280030', 2) || nums(ds, 'x00181164', 2);
    const slope = get(fg, 'x00289145', it => num(it, 'x00281053')) ?? num(ds, 'x00281053') ?? 1;
    const inter = get(fg, 'x00289145', it => num(it, 'x00281052')) ?? num(ds, 'x00281052') ?? 0;
    const echo = get(fg, 'x00189114', it => num(it, 'x00189082')) ?? num(ds, 'x00180086') ?? 1;
    const temporal = get(fg, 'x00209111', it => num(it, 'x00209128')) ?? num(ds, 'x00200100') ?? 0;
    const stack = get(fg, 'x00209111', it => str(it, 'x00209056') || null) || '';
    out.push({ ...base, frame: fi, nFrames, ipp, iop, ps, slope, inter, echo, temporal, stack,
      instance: num(ds, 'x00200013') ?? 0, acq: num(ds, 'x00200012') ?? 0 });
  }
  return out;
}

function seriesInfo(ds) {
  const imageType = str(ds, 'x00080008').toUpperCase();
  return {
    seriesUID: str(ds, 'x0020000e'), studyUID: str(ds, 'x0020000d'),
    description: str(ds, 'x0008103e'), protocol: str(ds, 'x00181030'), seriesNumber: num(ds, 'x00200011'),
    modality: str(ds, 'x00080060'), imageType,
    contrast: str(ds, 'x00180010'), contrastSeq: !!ds.elements.x00180012 && seqItems(ds, 'x00180012').length > 0,
    tr: num(ds, 'x00180080'), te: num(ds, 'x00180081'), ti: num(ds, 'x00180082'),
    scanSeq: str(ds, 'x00180020'), seqVariant: str(ds, 'x00180021'), seqName: str(ds, 'x00180024'),
    acqType: str(ds, 'x00180023'), sliceThickness: num(ds, 'x00180050'), fieldStrength: num(ds, 'x00180087'),
    manufacturer: str(ds, 'x00080070'), ts: str(ds, 'x00020010') || TS.IMPLICIT_LE,
  };
}

// Group parsed files into candidate volumes.
export function groupSeries(parsed) {
  const groups = new Map();
  for (const f of parsed) {
    const info = seriesInfo(f.ds);
    if (info.modality && info.modality !== 'MR' && info.modality !== 'OT') continue;
    for (const fr of framesOf(f)) {
      if (!fr.ipp || !fr.iop || !fr.ps) continue;
      const iopKey = fr.iop.map(v => v.toFixed(3)).join(',');
      const key = [info.seriesUID, iopKey, fr.rows, fr.cols, fr.echo, fr.stack].join('|');
      if (!groups.has(key)) groups.set(key, { info, frames: [] });
      groups.get(key).frames.push(fr);
    }
  }
  const vols = [];
  for (const [key, g] of groups) {
    const iop = g.frames[0].iop;
    const r = iop.slice(0, 3), c = iop.slice(3, 6);
    const n = [r[1] * c[2] - r[2] * c[1], r[2] * c[0] - r[0] * c[2], r[0] * c[1] - r[1] * c[0]];
    for (const fr of g.frames) fr.d = fr.ipp[0] * n[0] + fr.ipp[1] * n[1] + fr.ipp[2] * n[2];
    // split repeated positions (dynamics / multiple acquisitions) into separate volumes
    const byPos = new Map();
    for (const fr of g.frames) {
      const k = fr.d.toFixed(2);
      if (!byPos.has(k)) byPos.set(k, []);
      byPos.get(k).push(fr);
    }
    const counts = [...byPos.values()].map(a => a.length);
    const m = Math.min(...counts);
    let sets;
    if (counts.every(cn => cn === 1)) sets = [g.frames];
    else if (counts.every(cn => cn === m)) {
      sets = Array.from({ length: m }, () => []);
      for (const arr of byPos.values()) {
        arr.sort((a, b) => (a.temporal - b.temporal) || (a.acq - b.acq) || (a.instance - b.instance) || (a.frame - b.frame));
        arr.forEach((fr, i) => sets[i].push(fr));
      }
    } else {
      // inconsistent: keep the first frame at each position
      sets = [[...byPos.values()].map(a => a.sort((x, y) => x.instance - y.instance)[0])];
    }
    sets.forEach((frames, si) => {
      frames.sort((a, b) => a.d - b.d);
      if (frames.length < 3) return; // localisers etc.
      const d0 = frames[0].d, d1 = frames[frames.length - 1].d;
      const sl = (d1 - d0) / (frames.length - 1);
      const gaps = frames.slice(1).map((fr, i) => fr.d - frames[i].d);
      const irregular = gaps.some(gp => Math.abs(gp - sl) > Math.max(0.05 * Math.abs(sl), 0.02));
      vols.push({
        id: key + '#' + si, info: g.info, frames, iop, normal: n, sliceSpacing: Math.abs(sl), irregular,
        nTimepoints: sets.length, timepoint: si,
        rows: frames[0].rows, cols: frames[0].cols, ps: frames[0].ps,
      });
    });
  }
  vols.sort((a, b) => (a.info.seriesNumber ?? 1e9) - (b.info.seriesNumber ?? 1e9));
  return vols;
}

export function describe(v) {
  const i = v.info;
  return {
    id: v.id,
    label: (i.seriesNumber != null ? '#' + i.seriesNumber + ' ' : '') + (i.description || i.protocol || 'Series'),
    description: i.description, protocol: i.protocol, seriesNumber: i.seriesNumber,
    dims: [v.cols, v.rows, v.frames.length],
    spacing: [v.ps[1], v.ps[0], v.sliceSpacing],
    tr: i.tr, te: i.te, ti: i.ti, acqType: i.acqType, contrast: i.contrast || (i.contrastSeq ? 'yes' : ''),
    imageType: i.imageType, nTimepoints: v.nTimepoints, timepoint: v.timepoint, irregular: v.irregular,
    ts: TS_NAMES[i.ts] || i.ts,
  };
}

// ---------------------------------------------------------------------------------------
// Sequence role guessing. Returns scores for T1, T1c, T2, FLAIR (higher = more likely).

export function roleScores(d) {
  const t = ((d.description || '') + ' ' + (d.protocol || '')).toLowerCase();
  const s = { T1: 0, T1c: 0, T2: 0, FLAIR: 0 };
  const has = re => re.test(t);
  if (has(/loc|scout|survey|localiz|calibration|asset|cal_|aahead|mip|perf|dsc|dce|swi|swan|dwi|diff|adc|trace|tensor|dti|bold|fmri|mra|tof|angio|spect|ep2d/)) return s;
  if (d.nTimepoints > 3) return s;
  const post = has(/post|\+c|\bgad|\bgd\b|gado|contrast|kontrast|\bce\b|t1c|t1ce|t1gd|cet1|\+gd|\bkm\b|mit km|w\/c|w c\b|enh/) || !!d.contrast;
  const pre = has(/\bpre\b|nativ|precontrast|pre-contrast|w\/o|without/);
  const flair = has(/flair|dark.?fluid|tirm|t2.?fl|fl.?t2|\bfl3d?\b|t2f|cube.?fl|space.?fl/);
  const t2 = has(/t2|tse.*t2|propeller|blade/) && !has(/t2\*|t2star|gre|ffe|hemo|swi/);
  const t1 = has(/t1|mprage|mp-rage|mp2rage|spgr|bravo|tfl|vibe|flash|t1w|t1_|\bse\b.*t1|fspgr|ir.?fspgr|3d.?tfe|t1tfe/);
  if (flair) s.FLAIR += 10;
  if (t2 && !flair) s.T2 += 8;
  if (t1 && !flair) { if (post && !pre) s.T1c += 9; else s.T1 += 8; if (pre) s.T1 += 2; }
  // physics-based hints when the name is uninformative
  const tr = d.tr || 0, te = d.te || 0, ti = d.ti || 0;
  if (!flair && !t2 && !t1) {
    if (ti > 1500 && tr > 4000) s.FLAIR += 4;
    else if (tr > 2000 && te > 60) s.T2 += 4;
    else if (tr > 0 && tr < 1000 && te < 30) { if (d.contrast) s.T1c += 4; else s.T1 += 3; }
  }
  if (ti > 1500 && tr > 4000) s.FLAIR += 2;
  if (d.acqType === '3D') { s.T1c += 1; s.T1 += 0.5; }
  if (/DERIVED|SECONDARY/.test(d.imageType || '')) { for (const k in s) s[k] -= 1.5; }
  if (/MPR|REFORMAT|PROJECTION/.test(d.imageType || '') || has(/\bmpr\b|reformat|\bcor\b|\bsag\b/)) { for (const k in s) s[k] -= 1; }
  return s;
}

export function nameRoleScores(name) {
  const t = name.toLowerCase().replace(/\.nii(\.gz)?$/, '');
  const s = { T1: 0, T1c: 0, T2: 0, FLAIR: 0 };
  const m = t.match(/_000([0-3])$/);
  if (m) { s[['T1', 'T1c', 'T2', 'FLAIR'][+m[1]]] += 10; return s; }
  if (/flair|t2f\b|t2-?f$|[-_]t2f/.test(t)) s.FLAIR += 10;
  else if (/t1c|t1ce|t1gd|t1\+c|t1-?post|t1_post|post|gad|\bgd\b|cet1|ct1|contrast|[-_]t1c$/.test(t)) s.T1c += 10;
  else if (/t2/.test(t)) s.T2 += 9;
  else if (/t1|mprage|t1n\b|[-_]t1n/.test(t)) s.T1 += 9;
  return s;
}

export function assignRoles(items, scoreFn) {
  // items: [{id, scores}] -> {T1:id,...} greedy by best score
  const roles = ['T1', 'T1c', 'T2', 'FLAIR'];
  const cand = [];
  for (const it of items) for (const r of roles) if (it.scores[r] > 0) cand.push({ id: it.id, r, s: it.scores[r] + (it.bonus || 0) });
  cand.sort((a, b) => b.s - a.s);
  const out = {}, used = new Set();
  for (const c of cand) { if (out[c.r] || used.has(c.id)) continue; out[c.r] = c.id; used.add(c.id); }
  return out;
}

// By elimination: if three sequences are recognised and exactly one loaded series is left over
// (for example four NIfTI files, one with an uninformative name), it takes the remaining role.
// Returns that role, or null.
export function fillByElimination(guess, ids) {
  const roles = ['T1', 'T1c', 'T2', 'FLAIR'];
  const open = roles.filter(r => !guess[r]);
  const used = new Set(roles.map(r => guess[r]).filter(Boolean));
  const left = ids.filter(id => !used.has(id));
  if (open.length !== 1 || left.length !== 1) return null;
  guess[open[0]] = left[0];
  return open[0];
}

// ---------------------------------------------------------------------------------------
// Volume assembly

let codecs = null;
async function loadCodecs(base) {
  if (codecs) return codecs;
  const [oj, cl, jl] = await Promise.all([
    import(base + 'vendor/codecs/openjpeg-decode.mjs'), import(base + 'vendor/codecs/charls-decode.mjs'), import(base + 'vendor/codecs/jpeg-lossless.mjs'),
  ]);
  const [ojm, clm] = await Promise.all([
    oj.default({ locateFile: () => base + 'vendor/codecs/openjpeg-decode.wasm' }),
    cl.default({ locateFile: () => base + 'vendor/codecs/charls-decode.wasm' }),
  ]);
  codecs = { openjpeg: ojm, charls: clm, lossless: jl };
  return codecs;
}

function decodeRLE(frag, rows, cols, bytesPerSample) {
  const dv = new DataView(frag.buffer, frag.byteOffset, frag.byteLength);
  const nSeg = dv.getUint32(0, true);
  const out = new Uint8Array(rows * cols * bytesPerSample);
  const n = rows * cols;
  for (let s = 0; s < nSeg; s++) {
    const start = dv.getUint32(4 + 4 * s, true);
    const end = s + 1 < nSeg ? dv.getUint32(8 + 4 * s, true) : frag.byteLength;
    // segment s holds byte (bytesPerSample-1-s) of each sample (most significant first)
    const byteIdx = bytesPerSample - 1 - s;
    let p = start, o = 0;
    while (p < end && o < n) {
      const c = dv.getInt8(p++);
      if (c >= 0) { for (let i = 0; i <= c && o < n; i++) out[(o++) * bytesPerSample + byteIdx] = frag[p++]; }
      else if (c > -128) { const v = frag[p++]; for (let i = 0; i < 1 - c && o < n; i++) out[(o++) * bytesPerSample + byteIdx] = v; }
    }
  }
  return out; // little-endian samples
}

function framePixels(fr, base) {
  const ds = fr.file.ds, u8 = fr.file.u8;
  const el = ds.elements.x7fe00010;
  const ts = str(ds, 'x00020010') || TS.IMPLICIT_LE;
  const n = fr.rows * fr.cols;
  const bps = fr.bits <= 8 ? 1 : fr.bits <= 16 ? 2 : 4;
  if (fr.spp !== 1) throw new Error('Colour DICOM images are not supported');
  let bytes, le = true;
  if (ts === TS.IMPLICIT_LE || ts === TS.EXPLICIT_LE || ts === TS.DEFLATE || ts === TS.EXPLICIT_BE) {
    const data = ds.byteArray; // after inflate for deflated syntax
    const off = el.dataOffset + fr.frame * n * bps;
    bytes = data.subarray(off, off + n * bps);
    le = ts !== TS.EXPLICIT_BE;
  } else {
    let frag;
    if (el.encapsulatedPixelData) {
      if (fr.nFrames === 1) {
        frag = el.basicOffsetTable && el.basicOffsetTable.length > 1
          ? dicomParser.readEncapsulatedImageFrame(ds, el, 0)
          : dicomParser.readEncapsulatedPixelDataFromFragments(ds, el, 0, el.fragments.length);
      } else {
        frag = el.basicOffsetTable && el.basicOffsetTable.length
          ? dicomParser.readEncapsulatedImageFrame(ds, el, fr.frame)
          : dicomParser.readEncapsulatedImageFrame(ds, el, fr.frame, dicomParser.createJPEGBasicOffsetTable(ds, el));
      }
    } else throw new Error('Unexpected pixel data layout');
    if (ts === TS.RLE) bytes = decodeRLE(frag, fr.rows, fr.cols, bps);
    else if (ts === TS.JPEG_LOSSLESS || ts === TS.JPEG_LOSSLESS_SV1) {
      const dec = new codecs.lossless.Decoder();
      const outBuf = dec.decode(frag.buffer, frag.byteOffset, frag.byteLength, bps);
      bytes = new Uint8Array(outBuf.buffer || outBuf, outBuf.byteOffset || 0, n * bps);
    } else if (ts === TS.JPEGLS_LOSSLESS || ts === TS.JPEGLS_NEAR) {
      const d = new codecs.charls.JpegLSDecoder();
      d.getEncodedBuffer(frag.length).set(frag); d.decode();
      bytes = new Uint8Array(d.getDecodedBuffer()); d.delete();
    } else if ([TS.J2K_LOSSLESS, TS.J2K, TS.HTJ2K_LOSSLESS, TS.HTJ2K_RPCL, TS.HTJ2K].includes(ts)) {
      const d = new codecs.openjpeg.J2KDecoder();
      d.getEncodedBuffer(frag.length).set(frag); d.decode();
      bytes = new Uint8Array(d.getDecodedBuffer()); d.delete();
    } else {
      throw new Error('Unsupported DICOM compression (' + (TS_NAMES[ts] || ts) + '). Export the series uncompressed or as NIfTI.');
    }
  }
  const out = new Float32Array(n);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const signed = fr.pixRep === 1;
  if (bps === 1) for (let i = 0; i < n; i++) out[i] = signed ? dv.getInt8(i) : dv.getUint8(i);
  else if (bps === 2) {
    // mask to BitsStored for unsigned data with high bits set (rare)
    for (let i = 0; i < n; i++) out[i] = signed ? dv.getInt16(2 * i, le) : dv.getUint16(2 * i, le);
  } else for (let i = 0; i < n; i++) out[i] = signed ? dv.getInt32(4 * i, le) : dv.getUint32(4 * i, le);
  if (fr.slope !== 1 || fr.inter !== 0) for (let i = 0; i < n; i++) out[i] = out[i] * fr.slope + fr.inter;
  return out;
}

export function needsCodecs(v) {
  const ts = v.info.ts;
  return ![TS.IMPLICIT_LE, TS.EXPLICIT_LE, TS.DEFLATE, TS.EXPLICIT_BE, TS.RLE].includes(ts);
}

export async function buildVolume(v, base) {
  if (needsCodecs(v)) await loadCodecs(base);
  const nx = v.cols, ny = v.rows, nz = v.frames.length;
  const data = new Float32Array(nx * ny * nz);
  for (let k = 0; k < nz; k++) data.set(framePixels(v.frames[k], base), k * nx * ny);
  // affine in LPS then flipped to RAS
  const [rx, ry, rz, cx, cy, cz] = v.iop;
  const dc = v.ps[1], dr = v.ps[0]; // PixelSpacing = [row spacing, column spacing]
  const p0 = v.frames[0].ipp, p1 = v.frames[nz - 1].ipp;
  const sv = nz > 1 ? [(p1[0] - p0[0]) / (nz - 1), (p1[1] - p0[1]) / (nz - 1), (p1[2] - p0[2]) / (nz - 1)]
    : [v.normal[0] * (v.info.sliceThickness || 1), v.normal[1] * (v.info.sliceThickness || 1), v.normal[2] * (v.info.sliceThickness || 1)];
  const L = [rx * dc, cx * dr, sv[0], p0[0], ry * dc, cy * dr, sv[1], p0[1], rz * dc, cz * dr, sv[2], p0[2]];
  const A = new Float64Array(16);
  for (let i = 0; i < 12; i++) A[i] = L[i];
  for (let c = 0; c < 4; c++) { A[c] = -A[c]; A[4 + c] = -A[4 + c]; }
  A[15] = 1;
  return { dims: [nx, ny, nz], affine: A, data };
}
