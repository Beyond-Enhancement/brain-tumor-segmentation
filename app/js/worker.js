// Processing worker: input parsing, preprocessing (as in the HD-GLIO README), HD-BET brain
// extraction, rigid registration, segmentation (HD-GLIO or an add-on model) and NIfTI export. Nothing leaves the
// computer: the worker only talks to the page and fetches its own bundled files.
import { readNifti, writeNifti, isGzip } from './lib/nifti.js';
import * as G from './lib/geom.js';
import * as DCM from './lib/dicom.js';
import { registerRigid } from './lib/register.js';
import { predictCase, labelComponents } from './lib/nnunet.js';
import { Network, gpuInfo, loadOrt } from './lib/ortrun.js';
import { unzipSync, gunzipSync } from '../vendor/fflate.mjs';

const BASE = new URL('../', import.meta.url).href; // app root
const ROLES = ['T1', 'T1c', 'T2', 'FLAIR'];
const state = { series: new Map(), parsedDicom: [], result: null, cancel: null, models: {}, nets: {}, addonErrors: [] };

const post = (type, data, transfer) => self.postMessage({ type, ...data }, transfer || []);
const progress = (stage, frac, text) => post('progress', { stage, frac, text });
const now = () => performance.now();

self.onmessage = async ev => {
  const m = ev.data;
  try {
    if (m.type === 'init') {
      state.cancel = m.cancel ? new Int32Array(m.cancel) : null;
      post('ready', { gpu: await gpuInfo(), threads: navigator.hardwareConcurrency || 1 });
      await loadLauncherAddons();
      await postModels();
    }
    else if (m.type === 'addon') { try { await registerAddon(m.name, new Uint8Array(m.buffer)); } catch (e) { state.addonErrors.push(e.message); } await postModels(m.name); }
    else if (m.type === 'clear') { state.series.clear(); state.parsedDicom = []; state.result = null; post('cleared', {}); }
    else if (m.type === 'files') await loadFiles(m.files);
    else if (m.type === 'run') await run(m.assignment, m.options);
    else if (m.type === 'export') doExport(m.what, m.id);
  } catch (e) {
    post('error', { message: e && e.message ? e.message : String(e), stage: m.type });
  }
};

const cancelled = () => state.cancel && Atomics.load(state.cancel, 0) === 1;

// ---------------------------------------------------------------------------------------
// Input

function isNiftiName(n) { return /\.nii(\.gz)?$/i.test(n); }

async function loadFiles(files) {
  const t0 = now();
  const entries = [];
  const pushEntry = (name, u8) => {
    if (/\.zip$/i.test(name) || (u8[0] === 0x50 && u8[1] === 0x4b && u8[2] === 0x03 && u8[3] === 0x04)) {
      const z = unzipSync(u8);
      for (const [n, d] of Object.entries(z)) if (d.length && !n.endsWith('/')) pushEntry(n, d);
    } else entries.push({ name, u8 });
  };
  for (const f of files) pushEntry(f.name, new Uint8Array(f.buffer));
  let nDicom = 0, nSkipped = 0, nNifti = 0;
  for (let i = 0; i < entries.length; i++) {
    const { name, u8 } = entries[i];
    if (i % 50 === 0) progress('load', i / entries.length, 'Reading files ' + i + ' / ' + entries.length);
    const base = name.split('/').pop();
    if (isNiftiName(base) || (isGzip(u8) && !/\.dcm$/i.test(base))) {
      try {
        const vol = readNifti(u8);
        const id = 'nii:' + name;
        state.series.set(id, { id, kind: 'nifti', name: base, vol, desc: { id, label: base, dims: vol.dims, spacing: G.spacing(vol.affine).map(v => +v.toFixed(3)), nt: vol.nt, orientation: G.orientationCode(vol.affine) }, scores: DCM.nameRoleScores(base) });
        nNifti++;
        continue;
      } catch (e) { if (isNiftiName(base)) { nSkipped++; continue; } }
    }
    if (/^(dicomdir|\.ds_store|thumbs\.db)$/i.test(base) || /\.(txt|xml|json|jpg|png|pdf|htm|html|exe|ini|inf)$/i.test(base)) { nSkipped++; continue; }
    const f = DCM.parseFile(base, u8);
    if (f) { state.parsedDicom.push(f); nDicom++; } else nSkipped++;
  }
  // regroup all DICOM collected so far (files may arrive in several drops)
  for (const k of [...state.series.keys()]) if (k.startsWith('dcm:')) state.series.delete(k);
  const vols = DCM.groupSeries(state.parsedDicom);
  for (const v of vols) {
    const d = DCM.describe(v);
    const id = 'dcm:' + v.id;
    d.id = id;
    state.series.set(id, { id, kind: 'dicom', group: v, desc: d, scores: DCM.roleScores(d) });
  }
  sendSeries({ nDicom, nNifti, nSkipped, ms: now() - t0 });
}

function sendSeries(stats) {
  const list = [...state.series.values()].map(s => ({ ...s.desc, kind: s.kind, scores: s.scores }));
  const guess = DCM.assignRoles([...state.series.values()].map(s => ({ id: s.id, scores: s.scores, bonus: s.kind === 'dicom' && s.desc.acqType === '3D' ? 0.5 : 0 })));
  const inferred = DCM.fillByElimination(guess, [...state.series.keys()]);
  post('series', { list, guess, inferred, stats });
}

async function getVolume(id) {
  const s = state.series.get(id);
  if (!s) throw new Error('Series not found');
  if (s.kind === 'nifti') return s.vol;
  if (!s.vol) s.vol = await DCM.buildVolume(s.group, BASE);
  return s.vol;
}

// ---------------------------------------------------------------------------------------
// Helpers

function dilateBox(mask, dims, aff, mm) {
  const sp = G.spacing(aff), r = sp.map(s => Math.max(0, Math.round(mm / s)));
  let cur = mask;
  const [nx, ny, nz] = dims;
  for (let ax = 0; ax < 3; ax++) {
    if (!r[ax]) continue;
    const out = new Uint8Array(cur.length);
    const st = ax === 0 ? 1 : ax === 1 ? nx : nx * ny, n = dims[ax];
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const v = i + nx * (j + ny * k);
      if (!cur[v]) continue;
      const c = ax === 0 ? i : ax === 1 ? j : k;
      for (let d = Math.max(0, c - r[ax]); d <= Math.min(n - 1, c + r[ax]); d++) out[v + (d - c) * st] = 1;
    }
    cur = out;
  }
  return cur;
}

function largestComponent(mask, dims) {
  const shape = [dims[2], dims[1], dims[0]];
  const { labels, n, sizes } = labelComponents(mask, shape);
  if (n <= 1) return mask;
  let best = 1; for (let i = 2; i <= n; i++) if (sizes[i] > sizes[best]) best = i;
  const out = new Uint8Array(mask.length);
  for (let v = 0; v < mask.length; v++) out[v] = labels[v] === best ? 1 : 0;
  return out;
}

function sameGrid(a, b) {
  if (a.dims.some((d, i) => d !== b.dims[i])) return false;
  for (let i = 0; i < 12; i++) if (Math.abs(a.affine[i] - b.affine[i]) > 1e-3) return false;
  return true;
}

async function loadModelConfig(name) {
  if (state.models[name]) return state.models[name];
  const r = await fetch(BASE + 'models/' + name + '/config.json');
  if (!r.ok) throw new Error('Model ' + name + ' is not bundled with this build');
  const cfg = await r.json();
  cfg.dir = 'models/' + name + '/';
  cfg.id = name;
  state.models[name] = cfg;
  return cfg;
}

// ---------------------------------------------------------------------------------------
// Segmentation models: HD-GLIO is built in; other networks come as add-on files (*.btsmodel, a
// zip of config.json, model.onnx and weights.pack) placed next to the program, which the launcher
// lists, or opened with the page's "Add model file" button. Each is checked before it is offered.

const ADDON_FORMAT = 1;
const hex = buf => [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, '0')).join('');

async function registerAddon(name, u8) {
  let z;
  try { z = unzipSync(u8); } catch (e) { throw new Error(name + ': not a workbench model file'); }
  const get = n => z[n] || z[Object.keys(z).find(k => k.endsWith('/' + n))];
  const cj = get('config.json');
  if (!cj) throw new Error(name + ': not a workbench model file (no config.json)');
  const cfg = JSON.parse(new TextDecoder().decode(cj));
  if ((cfg.addonFormat || 0) !== ADDON_FORMAT) throw new Error(name + ': made for a different version of the workbench');
  if (!/^[a-z0-9_-]{2,40}$/.test(cfg.id || '') || cfg.id === 'hdglio' || cfg.id === 'hdbet') throw new Error(name + ': invalid model id');
  if (![1, 2].includes(cfg.nnunet) || !Array.isArray(cfg.inputChannels) || !cfg.inputChannels.every(c => ROLES.includes(c))) throw new Error(name + ': unsupported model');
  const onnx = get(cfg.onnx), pack = get(cfg.weights && cfg.weights.pack);
  if (!onnx || !pack) throw new Error(name + ': network files missing');
  const digest = hex(await crypto.subtle.digest('SHA-256', pack));
  if (digest !== cfg.weights.sha256) throw new Error(name + ': the file is damaged (checksum does not match)');
  cfg.inline = { [cfg.onnx]: onnx, [cfg.weights.pack]: pack };
  cfg.source = name;
  for (const k of Object.keys(state.nets)) if (k.startsWith(cfg.id + ':')) { await state.nets[k].release(); delete state.nets[k]; }
  state.models[cfg.id] = cfg;
  return cfg;
}

async function loadLauncherAddons() {
  let list;
  try {
    const r = await fetch(BASE + '__bts/addons', { cache: 'no-store' });
    if (!r.ok) return;
    list = await r.json();
  } catch (e) { return; }
  for (const a of list) {
    try {
      const r = await fetch(BASE + '__bts/addons/' + encodeURIComponent(a.name), { cache: 'no-store' });
      if (!r.ok) throw new Error(a.name + ': could not be read');
      await registerAddon(a.name, new Uint8Array(await r.arrayBuffer()));
    } catch (e) { state.addonErrors.push(e.message); }
  }
}

async function postModels(added) {
  try { await loadModelConfig('hdglio'); } catch (e) { /* reported when run */ }
  const list = Object.values(state.models).filter(c => c.id !== 'hdbet' && (c.id === 'hdglio' || c.source)).map(c => ({
    id: c.id, name: c.name, version: c.version, source: c.source || null, license: c.license || null, citation: c.citation || null,
    labels: c.labels, compartments: c.compartments, weights: c.weights.format,
  })).sort((a, b) => (a.id === 'hdglio' ? -1 : b.id === 'hdglio' ? 1 : a.name.localeCompare(b.name)));
  post('models', { list, errors: state.addonErrors.splice(0), added: added || null });
}

async function getNetwork(name, backend) {
  const key = name + ':' + backend;
  if (state.nets[key]) return state.nets[key];
  for (const k of Object.keys(state.nets)) { if (k.startsWith(name + ':')) { await state.nets[k].release(); delete state.nets[k]; } }
  const cfg = await loadModelConfig(name);
  const net = await Network.create(BASE, cfg, backend, t => progress('model', 0, t));
  state.nets[key] = net;
  return net;
}

// volume (x-fastest dims [nx,ny,nz]) -> numpy shape / spacing for nnU-Net
const npShape = dims => [dims[2], dims[1], dims[0]];
const npSpacing = aff => { const s = G.spacing(aff); return [s[2], s[1], s[0]]; };

// ---------------------------------------------------------------------------------------
// Pipeline

async function run(assignment, opt) {
  const T = { start: now() };
  const log = [];
  const note = s => { log.push(s); };
  for (const r of ROLES) if (!assignment[r]) throw new Error('Choose a series for ' + r);
  const ids = ROLES.map(r => assignment[r]);
  if (new Set(ids).size !== 4) throw new Error('Each sequence must be a different series');
  const refRole = opt.reference || 'T1';

  // 1) volumes, reoriented to standard (fslreorient2std)
  progress('prepare', 0, 'Assembling volumes');
  const raw = {}, plans = {}, reo = {};
  for (const r of ROLES) {
    raw[r] = await getVolume(assignment[r]);
    if (raw[r].nt > 1) note(r + ': 4D file, first volume used');
    plans[r] = G.reorientPlan(raw[r]);
    reo[r] = G.applyReorient(raw[r], plans[r]);
    progress('prepare', (ROLES.indexOf(r) + 1) / 4, 'Reorienting ' + r);
  }
  // reference grid
  let grid = { dims: reo[refRole].dims.slice(), affine: new Float64Array(reo[refRole].affine) };
  let gridIsNativeRef = true;
  if (opt.isotropic) {
    grid = G.isotropicGrid(reo[refRole], 1.0);
    gridIsNativeRef = false;
  }
  const onGrid = (vol, Tm, interp = 'cubic') => {
    if (!Tm && sameGrid(vol, grid)) return new Float32Array(vol.data);
    const d = G.resampleToGrid(vol, grid.dims, grid.affine, Tm, interp, 0);
    if (interp === 'cubic') { let mx = 0; for (const v of vol.data) if (v > mx) mx = v; for (let i = 0; i < d.length; i++) { if (d[i] < 0) d[i] = 0; else if (d[i] > mx) d[i] = mx; } }
    return d;
  };
  const refOnGrid = { dims: grid.dims, affine: grid.affine, data: onGrid(reo[refRole], null) };
  T.prepare = now();

  // backend
  const gpu = await gpuInfo();
  let backend = opt.device === 'cpu' ? 'wasm' : opt.device === 'gpu' ? 'webgpu' : (gpu && !gpu.software ? 'webgpu' : 'wasm');
  if (backend === 'webgpu' && !gpu) { backend = 'wasm'; note('No WebGPU adapter: using the CPU'); }
  if (backend === 'webgpu' && state.gpuBroken) { backend = 'wasm'; note('The GPU failed earlier in this session: using the CPU'); }
  else if (backend === 'webgpu' && opt.device === 'auto' && opt.avoidGpu) { backend = 'wasm'; note('The GPU failed on this computer before: Automatic uses the CPU (choose GPU to try it again)'); }
  const ctx = { backend, note, gpuFailed: false, usedGpu: false };
  const netProgress = (stage, name) => (st, f, e) => progress(stage, st === 'network' ? 0.1 + 0.85 * f : st === 'preprocess' ? 0.1 * f : 0.95,
    st === 'network' && e ? name + ': patch ' + e.tile + '/' + e.tiles + (e.mirrors > 1 ? ', mirror ' + e.mirror + '/' + e.mirrors : '') + (ctx.backend === 'webgpu' ? ' (GPU)' : '') : name + ' preprocessing');

  // 2) brain extraction (HD-BET) on the reference
  let brain;
  if (opt.skullStrip) {
    progress('hdbet', 0, 'HD-BET brain extraction on ' + refRole);
    const { res: r } = await runNetwork('hdbet', 'HD-BET', 'hdbet', [refOnGrid.data], npShape(grid.dims), npSpacing(grid.affine), !!opt.ttaBet, ctx, netProgress('hdbet', 'HD-BET'));
    brain = new Uint8Array(r.seg.length);
    for (let i = 0; i < brain.length; i++) brain[i] = r.seg[i] === 1 ? 1 : 0;
    if (opt.lcc !== false) brain = largestComponent(brain, grid.dims);
  } else {
    brain = new Uint8Array(refOnGrid.data.length);
    for (let i = 0; i < brain.length; i++) brain[i] = refOnGrid.data[i] > 0 ? 1 : 0;
  }
  let nBrain = 0; for (const v of brain) nBrain += v;
  if (nBrain < 1000) throw new Error('Brain extraction found no brain. Check the reference series.');
  T.hdbet = now();

  // 3) registration of the other sequences to the reference
  const regMask = dilateBox(brain, grid.dims, grid.affine, 6);
  const reg = {}, onRef = {};
  onRef[refRole] = refOnGrid.data;
  const others = ROLES.filter(r => r !== refRole);
  for (let oi = 0; oi < others.length; oi++) {
    const r = others[oi];
    if (cancelled()) throw new Error('Cancelled');
    if (opt.register) {
      progress('register', oi / others.length, 'Registering ' + r + ' to ' + refRole);
      const res = registerRigid(refOnGrid, reo[r], regMask, { onProgress: f => progress('register', (oi + f) / others.length, 'Registering ' + r + ' to ' + refRole) });
      reg[r] = { t: res.params.t.map(v => +v.toFixed(2)), rDeg: res.params.rDeg.map(v => +v.toFixed(2)), nmi: +res.nmi.toFixed(4) };
      const big = Math.max(...res.params.rDeg.map(Math.abs)) > 10 || Math.hypot(...res.params.t) > 20;
      reg[r].check = big;
      onRef[r] = onGrid(reo[r], res.T);
    } else {
      onRef[r] = onGrid(reo[r], null);
      reg[r] = { skipped: true };
    }
  }
  T.register = now();

  // 4) mask all sequences with the reference brain mask (fslmaths -mas)
  const masked = {};
  for (const r of ROLES) { const d = new Float32Array(onRef[r]); for (let i = 0; i < d.length; i++) if (!brain[i]) d[i] = 0; masked[r] = d; }

  // 5) tumor segmentation (HD-GLIO, or an add-on model)
  const modelId = opt.model || 'hdglio';
  if (modelId !== 'hdglio' && !state.models[modelId]) throw new Error('The selected model is not loaded. Add its model file again.');
  const gcfg = await loadModelConfig(modelId);
  const mname = gcfg.short || gcfg.name;
  progress('hdglio', 0, 'Loading ' + mname);
  const chans = gcfg.inputChannels.map(r => masked[r]);
  const { res } = await runNetwork(modelId, mname, 'hdglio', chans, npShape(grid.dims), npSpacing(grid.affine), !!opt.tta, ctx, netProgress('hdglio', mname));
  const seg = res.seg;
  backend = ctx.backend;
  T.hdglio = now();

  // 6) volumes
  const vv = G.voxelVolume(grid.affine);
  const counts = new Array(256).fill(0); // by label value (BraTS uses 1, 2, 4)
  for (const v of seg) counts[v]++;
  const comps = gcfg.compartments.map(c => {
    const n = c.labels.reduce((a, l) => a + counts[l], 0);
    return { key: c.key, name: c.name, labels: c.labels, voxels: n, cm3: n * vv / 1000 };
  });
  const brainCm3 = nBrain * vv / 1000;
  let tumorCenterMM = null;
  {
    const [nx, ny] = grid.dims; let si = 0, sj = 0, sk = 0, cnt = 0;
    for (let v = 0; v < seg.length; v++) if (seg[v]) { si += v % nx; sj += Math.floor(v / nx) % ny; sk += Math.floor(v / (nx * ny)); cnt++; }
    if (cnt) { const A = grid.affine, i = si / cnt, j = sj / cnt, k = sk / cnt; tumorCenterMM = [0, 1, 2].map(r => A[4 * r] * i + A[4 * r + 1] * j + A[4 * r + 2] * k + A[4 * r + 3]); }
  }

  state.result = { grid, gridIsNativeRef, refRole, plan: plans[refRole], rawRef: raw[refRole], onRef, masked, brain, seg, cfg: gcfg };
  const timing = { prepare: T.prepare - T.start, hdbet: T.hdbet - T.prepare, register: T.register - T.hdbet, hdglio: T.hdglio - T.register, total: T.hdglio - T.start };

  // 7) send display volumes (uncompressed NIfTI) to the page
  const vols = {};
  const transfer = [];
  for (const r of ROLES) { const b = writeNifti({ dims: grid.dims, affine: grid.affine, data: onRef[r] }, { dtype: 'f32', gz: false }); vols[r] = b.buffer; transfer.push(b.buffer); }
  const bm = writeNifti({ dims: grid.dims, affine: grid.affine, data: brain }, { dtype: 'u8', gz: false });
  const sg = writeNifti({ dims: grid.dims, affine: grid.affine, data: seg }, { dtype: 'u8', gz: false });
  vols.brain = bm.buffer; vols.seg = sg.buffer; transfer.push(bm.buffer, sg.buffer);
  post('result', {
    vols, comps, brainCm3, tumorCenterMM, voxelVolume: vv, voxelSize: G.spacing(grid.affine), dims: grid.dims, reg, refRole,
    backend, gpu, gpuFailed: ctx.gpuFailed, usedGpu: ctx.usedGpu && !ctx.gpuFailed, timing, notes: log, tta: !!opt.tta, labels: gcfg.labels, compartments: gcfg.compartments,
    model: { id: gcfg.id, name: gcfg.name, version: gcfg.version, weights: gcfg.weights.format, source: gcfg.source || null, citation: gcfg.citation || null, license: gcfg.license || null }, bet: opt.skullStrip ? (await loadModelConfig('hdbet')).version : null,
  }, transfer);
}

// A GPU failure while a network runs (device lost: on Windows usually the driver's 2-second
// timeout (TDR) on a slow or integrated GPU, or GPU memory exhausted) restarts that network on
// the CPU; the rest of the session stays on the CPU.
const isGpuFailure = e => /device.*lost|is lost|mapAsync|GPUBuffer|GPUDevice|webgpu|WGPU|out of memory|OrtRun/i.test(String((e && e.message) || e));

async function runNetwork(name, label, stage, channels, shape, spacing, tta, ctx, onProgress) {
  for (;;) {
    const net = await getNetworkSafe(name, ctx.backend, ctx.note);
    ctx.backend = net.backend;
    try {
      const res = await predictCase(channels, shape, spacing, net.cfg, (p, d) => net.run(p, d), { tta, shouldStop: cancelled, onProgress });
      if (net.backend === 'webgpu') ctx.usedGpu = true;
      return { cfg: net.cfg, res };
    } catch (e) {
      if (net.backend !== 'webgpu' || cancelled() || !isGpuFailure(e)) throw e;
      const msg = String((e && e.message) || e);
      const why = /lost/i.test(msg) ? 'the browser lost the GPU device' : msg.slice(0, 160);
      ctx.note('The GPU stopped responding during ' + label + ' (' + why + '). ' + label + ' was restarted and finished on the CPU.');
      ctx.gpuFailed = true;
      state.gpuBroken = true;
      for (const k of Object.keys(state.nets)) { if (k.endsWith(':webgpu')) { await state.nets[k].release(); delete state.nets[k]; } }
      ctx.backend = 'wasm';
      progress(stage, 0, label + ': GPU stopped responding, restarting on the CPU');
    }
  }
}

async function getNetworkSafe(name, backend, note) {
  try {
    return await getNetwork(name, backend);
  } catch (e) {
    if (backend === 'webgpu') {
      note('GPU unavailable for ' + name + ' (' + (e.message || e) + '); used the CPU');
      return await getNetwork(name, 'wasm');
    }
    throw e;
  }
}

// ---------------------------------------------------------------------------------------
// Export

function toOutputGrid(data, Ctor) {
  const R = state.result;
  if (R.gridIsNativeRef) {
    // undo the reorientation: identical grid and voxel order to the reference series as loaded
    return { dims: R.rawRef.dims.slice(), affine: new Float64Array(R.rawRef.affine), data: G.undoReorient(data, R.plan, R.rawRef.dims, Ctor) };
  }
  return { dims: R.grid.dims.slice(), affine: new Float64Array(R.grid.affine), data };
}

function doExport(what, reqId) {
  const R = state.result;
  if (!R) throw new Error('Nothing to export yet');
  let vol, dtype = 'f32', name, descrip;
  const prefix = (R.cfg && R.cfg.id) || 'hdglio';
  if (what === 'seg') { vol = toOutputGrid(R.seg, Uint8Array); dtype = 'u8'; name = prefix + '_segmentation.nii.gz'; descrip = (R.cfg.name || 'HD-GLIO').slice(0, 60) + ' segmentation'; }
  else if (what === 'brain') { vol = toOutputGrid(R.brain, Uint8Array); dtype = 'u8'; name = 'hdbet_brain_mask.nii.gz'; descrip = 'HD-BET brain mask'; }
  else if (what.startsWith('input:')) {
    const r = what.slice(6);
    vol = toOutputGrid(R.masked[r], Float32Array); name = prefix + '_input_' + r.toLowerCase() + '.nii.gz'; descrip = r + ' registered, brain-extracted';
  } else if (what.startsWith('reg:')) {
    const r = what.slice(4);
    vol = toOutputGrid(R.onRef[r], Float32Array); name = 'registered_' + r.toLowerCase() + '.nii.gz'; descrip = r + ' registered to ' + R.refRole;
  } else throw new Error('Unknown export ' + what);
  const bytes = writeNifti(vol, { dtype, gz: true, descrip });
  post('export', { id: reqId, name, bytes: bytes.buffer }, [bytes.buffer]);
}

loadOrt(BASE).catch(() => {});
