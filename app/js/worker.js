// Processing worker: input parsing, preprocessing (as in the HD-GLIO README), HD-BET brain
// extraction, rigid registration, segmentation (HD-GLIO or an add-on model: nnU-Net, Raidionics or
// AURORA, the last two with their own preprocessing) and NIfTI export. Nothing leaves the computer: the worker only talks
// to the page and fetches its own bundled files.
import { readNifti, writeNifti, isGzip } from './lib/nifti.js';
import * as G from './lib/geom.js';
import * as DCM from './lib/dicom.js';
import { registerRigid } from './lib/register.js';
import { predictCase, labelComponents } from './lib/nnunet.js';
import { predictRaidionics } from './lib/raidionics.js';
import { predictAurora } from './lib/aurora.js';
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
    else if (m.type === 'agreement') postAgreement(m.a, m.b, m.region, m.id);
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
  if (!Array.isArray(cfg.inputChannels) || !cfg.inputChannels.length || !cfg.inputChannels.every(c => ROLES.includes(c))) throw new Error(name + ': unsupported model');
  if (cfg.kind === 'raidionics') {
    const P = cfg.raidionics || {};
    const ok = P.crop === 'minimum' && P.normalization === 'zeromean_nonzero' && P.activation === 'sigmoid' && !P.activationIncluded &&
      P.channelsOrder === 'channels_first' && P.numClasses === 1 && P.numberInputs === 1 && cfg.inputChannels.length === 1 && !P.swapInput &&
      P.reconstructionOrder === 'resample_first' && Array.isArray(P.patchSize) && P.patchSize.length === 3 && Array.isArray(P.outputSpacing) && P.outputSpacing.length === 3;
    if (!ok) throw new Error(name + ': Raidionics configuration not supported by this version of the workbench');
  } else if (cfg.kind === 'aurora') {
    const P = cfg.aurora || {}, eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    const ok = eq(P.roiSize, [192, 192, 32]) && P.overlap === 0.5 && P.blend === 'gaussian' && P.sigmaScale === 0.125 && P.padding === 'replicate' &&
      eq(P.percentiles, [0.5, 99.5]) && P.threshold === 0.5 && eq(P.outChannels, ['whole', 'metastasis']) && eq(P.gridSpacing, [1, 1, 1]) &&
      P.gridAxes === 'LPS' && eq(P.gridDims, [240, 240, 155]) && (!P.tta || (eq(P.tta.flipAxes, [0, 1]) && P.tta.weights.length === 3));
    if (!ok) throw new Error(name + ': AURORA configuration not supported by this version of the workbench');
  } else if (![1, 2].includes(cfg.nnunet)) throw new Error(name + ': unsupported model');
  if (cfg.postprocessing && cfg.postprocessing.connectivity && ![6, 26].includes(cfg.postprocessing.connectivity)) throw new Error(name + ': unsupported postprocessing');
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
    inputChannels: c.inputChannels, kind: c.kind || 'nnunet', tta: c.tta !== false, tableNote: c.tableNote || null, short: c.short || null,
    usesBrain: usesBrain(c),
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

// Regions that mean the same thing in different models, for comparing them. A model's own
// compartments stay as they are; these are only used for the comparison table and maps.
const REGION_DEFS = [
  { key: 'whole', name: 'Whole abnormality', short: 'Whole' },
  { key: 'core', name: 'Tumor core', short: 'Core' },
  { key: 'enh', name: 'Enhancing tumor', short: 'Enh.' },
  { key: 'nonenh', name: 'Non-enhancing abnormality', short: 'Non-enh.' },
];
const MODEL_REGIONS = {
  hdglio: { whole: [1, 2], enh: [2], nonenh: [1] },
  brats2020: { whole: [1, 2, 4], core: [1, 4], enh: [4], nonenh: [1, 2] },
  'raidionics-flair': { whole: [1] },
  // enhancing tissue, necrosis and cysts (Bouget et al. 2026), as BraTS tumor core = NCR + ET.
  // HD-GLIO is left out: its labels do not say where necrosis goes.
  'raidionics-core': { core: [1] },
};
const regionsFor = cfg => cfg.compareRegions || MODEL_REGIONS[cfg.id] || { whole: cfg.labels.map(l => l.value) };
// Inputs: nnU-Net models get the brain-extracted images unless their config says otherwise (the
// vestibular schwannoma models were trained on whole-head images); AURORA gets them brain-extracted;
// Raidionics gets the image as it is and uses the brain mask afterwards.
const maskedInput = cfg => cfg.kind !== 'raidionics' && cfg.maskInput !== false;
const usesBrain = cfg => cfg.kind === 'raidionics' || maskedInput(cfg);
const labelLut = labels => { const t = new Uint8Array(256); for (const l of labels) t[l] = 1; return t; };

async function run(assignment, opt) {
  const T = { start: now() };
  const log = [];
  const note = s => { log.push(s); };
  // the selected models decide which sequences are needed; any others loaded are registered for viewing
  const modelIds = [...new Set(opt.models && opt.models.length ? opt.models : [opt.model || 'hdglio'])];
  const cfgs = [];
  for (const id of modelIds) {
    if (id !== 'hdglio' && !state.models[id]) throw new Error('A selected model is not loaded (' + id + '). Add its model file again.');
    cfgs.push(await loadModelConfig(id));
  }
  const mn = c => c.short || c.name;
  for (const c of cfgs) for (const r of c.inputChannels) if (!assignment[r]) throw new Error(mn(c) + ' needs a series for ' + r);
  const required = ROLES.filter(r => cfgs.some(c => c.inputChannels.includes(r)));
  const loaded = ROLES.filter(r => assignment[r]);
  const ids = loaded.map(r => assignment[r]);
  if (new Set(ids).size !== ids.length) throw new Error('Each sequence must be a different series');
  let refRole = opt.reference || 'T1';
  if (!loaded.includes(refRole)) { note(refRole + ' is not loaded: ' + required[0] + ' is the reference'); refRole = required[0]; }
  const multi = cfgs.length > 1;

  // 1) volumes, reoriented to standard (fslreorient2std)
  progress('prepare', 0, 'Assembling volumes');
  const raw = {}, plans = {}, reo = {};
  for (const r of loaded) {
    raw[r] = await getVolume(assignment[r]);
    if (raw[r].nt > 1) note(r + ': 4D file, first volume used');
    plans[r] = G.reorientPlan(raw[r]);
    reo[r] = G.applyReorient(raw[r], plans[r]);
    progress('prepare', (loaded.indexOf(r) + 1) / loaded.length, 'Reorienting ' + r);
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
  const netProgress = (stage, name) => (st, f, e) => progress(stage, st === 'network' ? 0.1 + 0.85 * f : st === 'preprocess' ? 0.1 * f : 0.95 + 0.05 * (f || 0),
    st === 'network' && e ? name + ': patch ' + e.tile + '/' + e.tiles + (e.mirrors > 1 ? ', mirror ' + e.mirror + '/' + e.mirrors : '') + (ctx.backend === 'webgpu' ? ' (GPU)' : '')
      : st === 'post' ? name + ': back to the reference grid' : name + ' preprocessing');

  // 2) brain extraction (HD-BET) on the reference, when a selected model uses the brain mask
  const needBrain = cfgs.some(usesBrain), doBet = !!opt.skullStrip && needBrain;
  let brain;
  if (doBet) {
    progress('hdbet', 0, 'HD-BET brain extraction on ' + refRole);
    const betProgress = netProgress('hdbet', 'HD-BET');
    const { res: r } = await runNetwork('hdbet', 'HD-BET', 'hdbet', ctx, net => predictCase([refOnGrid.data], npShape(grid.dims), npSpacing(grid.affine), net.cfg, (p, d) => net.run(p, d), { tta: !!opt.ttaBet, shouldStop: cancelled, onProgress: betProgress }));
    brain = new Uint8Array(r.seg.length);
    for (let i = 0; i < brain.length; i++) brain[i] = r.seg[i] === 1 ? 1 : 0;
    if (opt.lcc !== false) brain = largestComponent(brain, grid.dims);
    if (multi) await releaseNetwork('hdbet'); // several networks in one run: keep only one in memory
  } else {
    brain = new Uint8Array(refOnGrid.data.length);
    for (let i = 0; i < brain.length; i++) brain[i] = refOnGrid.data[i] > 0 ? 1 : 0;
  }
  let nBrain = 0; for (const v of brain) nBrain += v;
  if (nBrain < 1000) throw new Error(doBet ? 'Brain extraction found no brain. Check the reference series.' : 'The reference image is empty.');
  T.hdbet = now();

  // 3) registration of the other sequences to the reference
  const regMask = dilateBox(brain, grid.dims, grid.affine, 6);
  const reg = {}, onRef = {};
  onRef[refRole] = refOnGrid.data;
  const others = loaded.filter(r => r !== refRole);
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

  // 4) model inputs. nnU-Net models (HD-GLIO and add-ons) and AURORA: sequences masked with the
  // reference brain mask (fslmaths -mas), except models trained on whole-head images (maskInput
  // false: the vestibular schwannoma models). Raidionics: the image as it is (the model crops the
  // head itself); the brain mask is used afterwards to drop findings outside the brain.
  const masked = {};
  const getMasked = r => {
    if (!masked[r]) { const d = new Float32Array(onRef[r]); for (let i = 0; i < d.length; i++) if (!brain[i]) d[i] = 0; masked[r] = d; }
    return masked[r];
  };

  // 5) tumor segmentation, one model after the other on the same prepared images
  const vv = G.voxelVolume(grid.affine);
  const results = [];
  for (const c of cfgs) {
    const key = 'seg:' + c.id, name = mn(c), raid = c.kind === 'raidionics', aur = c.kind === 'aurora';
    const t0 = now();
    progress(key, 0, 'Loading ' + name);
    const pr = netProgress(key, name);
    const { res } = await runNetwork(c.id, name, key, ctx, raid
      ? net => predictRaidionics({ dims: grid.dims, affine: grid.affine, data: onRef[c.inputChannels[0]] }, doBet ? brain : null, net.cfg, (p, d) => net.run(p, d), { shouldStop: cancelled, onProgress: pr })
      : aur ? net => predictAurora(c.inputChannels.map(getMasked), grid, brain, net.cfg, (p, d) => net.run(p, d), { tta: !!opt.tta, shouldStop: cancelled, onProgress: pr })
      : net => predictCase(c.inputChannels.map(r => (maskedInput(c) ? getMasked(r) : onRef[r])), npShape(grid.dims), npSpacing(grid.affine), net.cfg, (p, d) => net.run(p, d), { tta: !!opt.tta, shouldStop: cancelled, onProgress: pr }));
    if (multi) await releaseNetwork(c.id);
    if (raid && res.info.removedOutsideBrain) note(name + ': ' + (res.info.removedOutsideBrain * vv / 1000).toFixed(2) + ' cm3 outside the brain mask removed (brain overlap refinement)');
    if (aur && res.info.brainOutsideGrid) note(name + ': ' + (res.info.brainOutsideGrid * vv / 1000).toFixed(2) + ' cm3 of the brain mask lay outside its 1 mm grid and was not segmented');
    const counts = new Array(256).fill(0); // by label value (BraTS uses 1, 2, 4)
    for (const v of res.seg) counts[v]++;
    const comps = c.compartments.map(cp => {
      const n = cp.labels.reduce((a, l) => a + counts[l], 0);
      return { key: cp.key, name: cp.name, labels: cp.labels, voxels: n, cm3: n * vv / 1000 };
    });
    results.push({ cfg: c, seg: res.seg, comps, segInfo: raid || aur ? res.info : null, ms: now() - t0, tta: !!opt.tta && c.tta !== false && !raid });
  }
  backend = ctx.backend;
  T.seg = now();

  // 6) brain volume, centre of the findings, comparison between models
  const brainCm3 = nBrain * vv / 1000;
  let tumorCenterMM = null;
  {
    const [nx, ny] = grid.dims; let si = 0, sj = 0, sk = 0, cnt = 0;
    const segs = results.map(r => r.seg);
    for (let v = 0; v < brain.length; v++) {
      let any = 0; for (const sg of segs) any |= sg[v];
      if (any) { si += v % nx; sj += Math.floor(v / nx) % ny; sk += Math.floor(v / (nx * ny)); cnt++; }
    }
    if (cnt) { const A = grid.affine, i = si / cnt, j = sj / cnt, k = sk / cnt; tumorCenterMM = [0, 1, 2].map(r => A[4 * r] * i + A[4 * r + 1] * j + A[4 * r + 2] * k + A[4 * r + 3]); }
  }
  const compare = compareModels(results, vv);

  // inputs as exported: brain-extracted when a model used them so, else as given to the model
  // (Raidionics, the vestibular schwannoma models)
  const inputs = {}, inputKind = {}, inputPrefix = {}, inputModel = {};
  for (const r of required) {
    const nn = cfgs.find(c => maskedInput(c) && c.inputChannels.includes(r));
    const c0 = nn || cfgs.find(c => c.inputChannels.includes(r));
    inputs[r] = nn ? getMasked(r) : onRef[r]; inputKind[r] = nn ? 'masked' : 'raw'; inputPrefix[r] = c0.id; inputModel[r] = mn(c0);
  }
  state.result = { grid, gridIsNativeRef, refRole, plan: plans[refRole], rawRef: raw[refRole], onRef, inputs, inputKind, inputPrefix, inputModel, brain, results };
  const timing = { prepare: T.prepare - T.start, hdbet: T.hdbet - T.prepare, register: T.register - T.hdbet, seg: T.seg - T.register, total: T.seg - T.start };

  // 7) send display volumes (uncompressed NIfTI) to the page
  const vols = {};
  const transfer = [];
  const add = (k, data, dtype) => { const b = writeNifti({ dims: grid.dims, affine: grid.affine, data }, { dtype, gz: false }); vols[k] = b.buffer; transfer.push(b.buffer); };
  for (const r of loaded) add(r, onRef[r], 'f32');
  add('brain', brain, 'u8');
  for (const r of results) add('seg:' + r.cfg.id, r.seg, 'u8');
  post('result', {
    vols, brainCm3, brainUsed: needBrain, tumorCenterMM, voxelVolume: vv, voxelSize: G.spacing(grid.affine), dims: grid.dims, reg, refRole, roles: loaded, inputs: required, inputKind, inputModel,
    backend, gpu, gpuFailed: ctx.gpuFailed, usedGpu: ctx.usedGpu && !ctx.gpuFailed, timing, notes: log, tta: results.some(r => r.tta),
    models: results.map(r => {
      const c = r.cfg;
      return { id: c.id, name: c.name, short: mn(c), version: c.version, weights: c.weights.format, source: c.source || null, citation: c.citation || null, license: c.license || null,
        kind: c.kind || 'nnunet', tableNote: c.tableNote || null, labels: c.labels, compartments: c.compartments, inputs: c.inputChannels, comps: r.comps, segInfo: r.segInfo, ms: r.ms, tta: r.tta };
    }),
    compare, bet: doBet ? (await loadModelConfig('hdbet')).version : null,
  }, transfer);
}

// Volumes of the shared regions per model, and for each pair of models and each region both have:
// Dice, and the volumes found by both, by the first only and by the second only.
function compareModels(results, vv) {
  const cm3 = n => n * vv / 1000;
  const regs = results.map(r => regionsFor(r.cfg));
  const regions = REGION_DEFS.filter(d => regs.some(g => g[d.key]));
  const volumes = results.map((r, i) => {
    const out = {};
    for (const d of regions) {
      if (!regs[i][d.key]) continue;
      const t = labelLut(regs[i][d.key]); let n = 0;
      for (const v of r.seg) n += t[v];
      out[d.key] = { labels: regs[i][d.key], voxels: n, cm3: cm3(n) };
    }
    return { id: r.cfg.id, regions: out };
  });
  const pairs = [];
  for (let i = 0; i < results.length; i++) for (let j = i + 1; j < results.length; j++) {
    for (const d of regions) {
      if (!regs[i][d.key] || !regs[j][d.key]) continue;
      const ta = labelLut(regs[i][d.key]), tb = labelLut(regs[j][d.key]);
      const A = results[i].seg, B = results[j].seg;
      let both = 0, ao = 0, bo = 0;
      for (let v = 0; v < A.length; v++) { const a = ta[A[v]], b = tb[B[v]]; if (a & b) both++; else if (a) ao++; else if (b) bo++; }
      const den = 2 * both + ao + bo;
      pairs.push({ a: results[i].cfg.id, b: results[j].cfg.id, region: d.key, dice: den ? 2 * both / den : null,
        both: cm3(both), aOnly: cm3(ao), bOnly: cm3(bo), voxels: { both, aOnly: ao, bOnly: bo } });
    }
  }
  return { regions: regions.map(d => ({ ...d, comparable: regs.filter(g => g[d.key]).length >= 2 })), volumes, pairs };
}

// Agreement map of one region between two models: 1 first model only, 2 second only, 3 both.
function agreementMap(a, b, region) {
  const R = state.result;
  const ra = R.results.find(r => r.cfg.id === a), rb = R.results.find(r => r.cfg.id === b);
  if (!ra || !rb) throw new Error('Model not in this run');
  const ga = regionsFor(ra.cfg)[region], gb = regionsFor(rb.cfg)[region];
  if (!ga || !gb) throw new Error('Region not available for both models');
  const ta = labelLut(ga), tb = labelLut(gb);
  const out = new Uint8Array(ra.seg.length);
  for (let v = 0; v < out.length; v++) out[v] = ta[ra.seg[v]] | (tb[rb.seg[v]] << 1);
  return out;
}

async function releaseNetwork(name) {
  for (const k of Object.keys(state.nets)) if (k.startsWith(name + ':')) { await state.nets[k].release(); delete state.nets[k]; }
}

// A GPU failure while a network runs (device lost: on Windows usually the driver's 2-second
// timeout (TDR) on a slow or integrated GPU, or GPU memory exhausted) restarts that network on
// the CPU; the rest of the session stays on the CPU.
const isGpuFailure = e => /device.*lost|is lost|mapAsync|GPUBuffer|GPUDevice|webgpu|WGPU|out of memory|OrtRun/i.test(String((e && e.message) || e));

async function runNetwork(name, label, stage, ctx, predict) {
  for (;;) {
    const net = await getNetworkSafe(name, ctx.backend, ctx.note);
    ctx.backend = net.backend;
    try {
      const res = await predict(net);
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
  const byId = id => { const r = id ? R.results.find(x => x.cfg.id === id) : R.results[0]; if (!r) throw new Error('Model not in this run'); return r; };
  if (what === 'seg' || what.startsWith('seg:')) {
    const r = byId(what.slice(4));
    vol = toOutputGrid(r.seg, Uint8Array); dtype = 'u8'; name = r.cfg.id + '_segmentation.nii.gz'; descrip = (r.cfg.name || 'HD-GLIO').slice(0, 60) + ' segmentation';
  } else if (what.startsWith('agree:')) {
    const [, a, b, region] = what.split(':');
    vol = toOutputGrid(agreementMap(a, b, region), Uint8Array); dtype = 'u8'; name = 'agreement_' + a + '_vs_' + b + '_' + region + '.nii.gz';
    descrip = '1 ' + a + ' only, 2 ' + b + ' only, 3 both (' + region + ')';
  } else if (what === 'brain') { vol = toOutputGrid(R.brain, Uint8Array); dtype = 'u8'; name = 'hdbet_brain_mask.nii.gz'; descrip = 'HD-BET brain mask'; }
  else if (what.startsWith('input:')) {
    const r = what.slice(6);
    if (R.inputs[r]) {
      const raw = R.inputKind[r] === 'raw';
      vol = toOutputGrid(R.inputs[r], Float32Array); name = R.inputPrefix[r] + '_input_' + r.toLowerCase() + '.nii.gz';
      descrip = raw ? (r + ' as given to ' + R.inputModel[r]).slice(0, 79) : r + ' registered, brain-extracted';
    } else if (R.onRef[r]) {
      vol = toOutputGrid(R.onRef[r], Float32Array); name = 'registered_' + r.toLowerCase() + '.nii.gz'; descrip = r + ' registered to ' + R.refRole;
    } else throw new Error(r + ' was not loaded');
  } else if (what.startsWith('reg:')) {
    const r = what.slice(4);
    vol = toOutputGrid(R.onRef[r], Float32Array); name = 'registered_' + r.toLowerCase() + '.nii.gz'; descrip = r + ' registered to ' + R.refRole;
  } else throw new Error('Unknown export ' + what);
  const bytes = writeNifti(vol, { dtype, gz: true, descrip });
  post('export', { id: reqId, name, bytes: bytes.buffer }, [bytes.buffer]);
}

function postAgreement(a, b, region, reqId) {
  const R = state.result;
  const bytes = writeNifti({ dims: R.grid.dims, affine: R.grid.affine, data: agreementMap(a, b, region) }, { dtype: 'u8', gz: false });
  post('agreement', { id: reqId, key: a + ':' + b + ':' + region, bytes: bytes.buffer }, [bytes.buffer]);
}

loadOrt(BASE).catch(() => {});
