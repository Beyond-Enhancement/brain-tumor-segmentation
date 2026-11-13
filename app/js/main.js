// Page controller: loading, sequence assignment, run, viewer, volumes, export.
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const ROLES = ['T1', 'T1c', 'T2', 'FLAIR'];
const ROLE_NAMES = { T1: 'T1 pre-contrast', T1c: 'T1 post-contrast', T2: 'T2', FLAIR: 'FLAIR' };
// run steps: shared preparation, then one segmentation step per selected model
const steps = () => [
  { key: 'prepare', name: 'Prepare and reorient' },
  { key: 'hdbet', name: 'Brain extraction (HD-BET)' },
  { key: 'register', name: 'Registration to ' + (ui.runRef || effectiveRef()) },
  ...(ui.runModels || selectedModels()).map(md => ({ key: 'seg:' + md.id, name: 'Tumor segmentation (' + shortName(md) + ')' })),
];
const COLORS = { 1: [255, 210, 63], 2: [255, 77, 77], 3: [80, 160, 255], 4: [120, 230, 140] };
// agreement map colours: first model only, second model only, both
const AGREE = { 1: [64, 156, 255], 2: [255, 92, 170], 3: [255, 210, 63] };

const ui = { series: [], assignment: {}, result: null, running: false, hidden: new Set(), demo: null, caseLabel: '' };
let cancelFlag = null;

// ---------------------------------------------------------------------------------------
// worker
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
if (self.crossOriginIsolated) cancelFlag = new Int32Array(new SharedArrayBuffer(4));
worker.postMessage({ type: 'init', cancel: cancelFlag ? cancelFlag.buffer : null });
const pendingExports = new Map();
worker.onmessage = ev => {
  const m = ev.data;
  if (m.type === 'ready') { ui.env = m; showDevice(); }
  else if (m.type === 'series') onSeries(m);
  else if (m.type === 'models') onModels(m);
  else if (m.type === 'progress') onProgress(m);
  else if (m.type === 'result') onResult(m);
  else if (m.type === 'export') { const f = pendingExports.get(m.id); pendingExports.delete(m.id); download(m.name, new Blob([m.bytes], { type: 'application/gzip' })); if (f) f(); }
  else if (m.type === 'agreement') onAgreement(m);
  else if (m.type === 'error') onError(m);
};
worker.onerror = e => toast('Worker error: ' + (e.message || 'unknown'), true);

function toast(msg, err) {
  const t = document.createElement('div');
  t.className = 'toast' + (err ? ' err' : '');
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), err ? 9000 : 3500);
}

// ---------------------------------------------------------------------------------------
// loading
async function readFileList(files) {
  const out = [];
  let i = 0;
  for (const f of files) {
    out.push({ name: f.webkitRelativePath || f.relPath || f.name, buffer: await f.arrayBuffer() });
    if (++i % 100 === 0) $('#loadStatus').textContent = 'Reading ' + i + ' / ' + files.length + ' files';
  }
  return out;
}

async function sendFiles(files, label) {
  if (!files.length) return;
  $('#loadStatus').textContent = 'Reading ' + files.length + ' file' + (files.length > 1 ? 's' : '') + ' ...';
  const list = await readFileList(files);
  ui.caseLabel = label || '';
  worker.postMessage({ type: 'files', files: list }, list.map(f => f.buffer));
}

function clearSeries() {
  worker.postMessage({ type: 'clear' });
  ui.series = []; ui.assignment = {};
  renderRoles();
  $('#loadStatus').textContent = '';
  $$('.demo').forEach(d => d.classList.remove('on'));
}

$('#pickFolder').onclick = () => $('#folderInput').click();
$('#pickFiles').onclick = () => $('#filesInput').click();
$('#clearBtn').onclick = clearSeries;
$('#folderInput').onchange = e => { const fs = [...e.target.files]; e.target.value = ''; sendFiles(fs, fs[0] ? (fs[0].webkitRelativePath || '').split('/')[0] : ''); };
$('#filesInput').onchange = e => { const fs = [...e.target.files]; e.target.value = ''; sendFiles(fs, fs.length === 1 ? fs[0].name : ''); };

const drop = $('#drop');
['dragenter', 'dragover'].forEach(t => document.addEventListener(t, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(t => document.addEventListener(t, e => { e.preventDefault(); if (t === 'drop' || e.target === document.documentElement) drop.classList.remove('over'); }));
document.addEventListener('drop', async e => {
  const items = [...(e.dataTransfer.items || [])];
  const entries = items.map(it => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
  const files = [];
  const walk = async (entry, path) => {
    if (entry.isFile) { const f = await new Promise((res, rej) => entry.file(res, rej)); f.relPath = path + f.name; files.push(f); }
    else if (entry.isDirectory) {
      const r = entry.createReader();
      let batch;
      do { batch = await new Promise((res, rej) => r.readEntries(res, rej)); for (const c of batch) await walk(c, path + entry.name + '/'); } while (batch.length);
    }
  };
  if (entries.length) { for (const en of entries) await walk(en, ''); }
  else files.push(...e.dataTransfer.files);
  const top = entries.length === 1 ? entries[0].name : '';
  sendFiles(files, top);
});

function onSeries(m) {
  ui.series = m.list;
  const s = m.stats;
  const parts = [];
  if (s.nDicom) parts.push(s.nDicom + ' DICOM image' + (s.nDicom > 1 ? 's' : ''));
  if (s.nNifti) parts.push(s.nNifti + ' NIfTI file' + (s.nNifti > 1 ? 's' : ''));
  if (s.nSkipped) parts.push(s.nSkipped + ' other file' + (s.nSkipped > 1 ? 's' : '') + ' ignored');
  $('#loadStatus').textContent = (parts.join(', ') || 'No images found') + '. ' + ui.series.length + ' series available.';
  ui.assignment = {};
  for (const r of ROLES) if (m.guess[r]) ui.assignment[r] = m.guess[r];
  ui.inferred = m.inferred ? { role: m.inferred, id: m.guess[m.inferred] } : null;
  autoAssignSingle();
  renderRoles();
  if (ui.caseLabel) $('#caseLabel').textContent = ui.caseLabel;
}

// The checked models need one sequence only, it is not assigned, and exactly one series is
// unassigned: assign that series to it (marked for checking), as for the fourth by elimination.
function autoAssignSingle() {
  const req = requiredRoles();
  if (req.length !== 1 || ui.assignment[req[0]]) return;
  const used = new Set(Object.values(ui.assignment).filter(Boolean));
  const free = ui.series.filter(x => !used.has(x.id));
  if (free.length !== 1) return;
  ui.assignment[req[0]] = free[0].id;
  ui.inferred = { role: req[0], id: free[0].id, single: true };
}

function seriesMeta(s) {
  if (!s) return '';
  const sp = s.spacing ? s.spacing.map(v => (+v).toFixed(2)).join(' x ') + ' mm' : '';
  const dm = s.dims ? s.dims.join(' x ') : '';
  const extra = [];
  if (s.tr) extra.push('TR ' + Math.round(s.tr));
  if (s.te) extra.push('TE ' + Math.round(s.te));
  if (s.ti) extra.push('TI ' + Math.round(s.ti));
  if (s.contrast) extra.push('contrast');
  if (s.nTimepoints > 1) extra.push('vol ' + (s.timepoint + 1) + '/' + s.nTimepoints);
  if (s.nt > 1) extra.push('4D: first volume');
  return [dm, sp, ...extra].filter(Boolean).join('  ');
}

const shortName = md => (md && (md.short || md.name)) || 'HD-GLIO';
const selectedModels = () => ui.models.filter(md => ui.selModels.has(md.id));
const requiredRoles = () => { const ms = selectedModels(); return ROLES.filter(r => ms.some(md => (md.inputChannels || ROLES).includes(r))); };
const loadedRoles = () => ROLES.filter(r => ui.assignment[r]);

function renderRoles() {
  const box = $('#roles');
  box.innerHTML = '';
  const req = requiredRoles();
  const names = selectedModels().map(shortName);
  const who = names.length ? names.join(' and ') : 'The selected model';
  $('#rolesHint').textContent = (req.length === 4 ? (names.length > 1 ? who + ' together need all four.' : who + ' needs all four.')
    : who + ' needs only ' + req.join(' and ') + '; the others are optional and, if chosen, are registered for viewing.') + ' Guessed from the series names and headers; check them.';
  for (const r of ROLES) {
    const row = document.createElement('div');
    row.className = 'role' + (req.includes(r) ? '' : ' optional');
    const sel = document.createElement('select');
    sel.innerHTML = '<option value="">- choose -</option>' + ui.series.map(s => `<option value="${esc(s.id)}">${esc(s.label)}</option>`).join('');
    sel.value = ui.assignment[r] || '';
    sel.onchange = () => { ui.assignment[r] = sel.value; renderRoles(); };
    sel.title = ROLE_NAMES[r];
    const meta = document.createElement('div');
    meta.className = 'meta';
    const s = ui.series.find(x => x.id === ui.assignment[r]);
    const byElim = ui.inferred && ui.inferred.role === r && ui.inferred.id === ui.assignment[r];
    meta.textContent = (byElim ? (ui.inferred.single ? 'only series left - check it.  ' : 'by elimination - check it.  ') : '') + seriesMeta(s);
    if (byElim) meta.classList.add('elim');
    meta.title = byElim ? (ui.inferred.single ? 'The selected models need only ' + r + ' and this was the only series not assigned, so it was assigned here.'
      : 'The other three sequences were recognised and this was the only series left, so it was assigned here.') : meta.textContent;
    row.innerHTML = `<span class="tag" title="${ROLE_NAMES[r]}${req.includes(r) ? '' : ' (optional for the selected models)'}">${r}</span>`;
    row.appendChild(sel); row.appendChild(meta);
    box.appendChild(row);
  }
  const ids = loadedRoles().map(r => ui.assignment[r]);
  const ok = req.length > 0 && req.every(r => ui.assignment[r]) && new Set(ids).size === ids.length;
  $('#runBtn').disabled = !ok || ui.running;
  // pre-processed inputs on one grid: default to skipping registration and brain extraction
  const chosen = loadedRoles().map(r => ui.series.find(x => x.id === ui.assignment[r])).filter(Boolean);
  if (chosen.length > 1 && chosen.every(s => s.kind === 'nifti') && ui.demo == null) {
    const same = chosen.every(s => s.dims.join() === chosen[0].dims.join() && s.spacing.join() === chosen[0].spacing.join());
    if (same && !ui.autoSet) { $('#optReg').checked = false; ui.autoSet = true; toast('All ' + (chosen.length === 4 ? 'four' : chosen.length) + ' images share one grid: registration switched off. Switch it back on if they are not co-registered.'); }
  }
  updateRefLabels();
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// segmentation models: HD-GLIO built in; add-on model files found next to the program or added
// here. Any combination can be checked; they run one after the other on the same prepared images.
ui.models = [{ id: 'hdglio', name: 'HD-GLIO', inputChannels: ROLES }];
ui.selModels = new Set(['hdglio']);
try {
  const saved = JSON.parse(localStorage.getItem('btsw.models') || 'null') || [localStorage.getItem('btsw.model') || 'hdglio'];
  if (Array.isArray(saved) && saved.length) ui.selModels = new Set(saved.map(String));
} catch (e) { /* storage unavailable */ }
const saveSelection = () => { try { localStorage.setItem('btsw.models', JSON.stringify([...ui.selModels])); } catch (e) { /* ignore */ } };
const MODEL_BLURB = {
  hdglio: 'Contrast-enhancing tumor and non-enhancing T2/FLAIR abnormality. T1, T1c, T2, FLAIR.',
  brats2020: 'Enhancing tumor, necrosis, edema (BraTS labels). T1, T1c, T2, FLAIR.',
  'raidionics-flair': 'FLAIR hyperintensity (tumor and edema). FLAIR only.',
  'raidionics-core': 'Tumor core: enhancing tissue, necrosis, cysts; gliomas, meningiomas, metastases. T1c only.',
  'vs-t1c': 'Vestibular schwannoma. T1c only; whole-head image, no brain extraction.',
  'vs-t2': 'Vestibular schwannoma. High-resolution T2 (e.g. CISS) only; whole-head image, no brain extraction.',
  'aurora-t1c': 'Brain metastases: metastasis (enhancing tumor and necrosis) and edema. T1c only.',
  'aurora-t1c-flair': 'Brain metastases: metastasis (enhancing tumor and necrosis) and edema. T1c and FLAIR.',
};
// models that use the HD-BET brain mask (all but those trained on whole-head images)
const needsBrain = ms => ms.some(md => md.usesBrain !== false);
function modelInfo(md) {
  if (md.id === 'hdglio') return 'HD-GLIO (Kickingereder et al. 2019): contrast-enhancing tumor and non-enhancing T2/FLAIR abnormality; no necrosis class. Built in.';
  const comps = (md.compartments || []).map(c => (c.name.match(/\(([^)]*)\)\s*$/) || [])[1] || c.name).filter(Boolean);
  return md.name + (md.version ? ', ' + md.version : '') + '. Compartments: ' + comps.join(', ') + '.' + (md.license ? ' License ' + md.license + '.' : '') + ' From ' + md.source + '.';
}
function onModels(m) {
  ui.models = m.list.length ? m.list : ui.models;
  if (m.added) { const md = ui.models.find(x => x.source === m.added); if (md) ui.selModels.add(md.id); }
  for (const id of [...ui.selModels]) if (!ui.models.some(md => md.id === id)) ui.selModels.delete(id);
  if (!ui.selModels.size) ui.selModels.add('hdglio');
  renderModelList();
  for (const e of m.errors || []) toast('Model file not used: ' + e, true);
  if (m.added && !m.errors.length) { const md = ui.models.find(x => x.source === m.added); toast('Model added: ' + (md ? md.name : m.added) + '.'); }
}
function renderModelList() {
  const box = $('#modelList');
  box.innerHTML = '';
  for (const md of ui.models) {
    const lab = document.createElement('label');
    lab.className = 'check';
    lab.title = modelInfo(md);
    lab.innerHTML = `<input type="checkbox" name="model" value="${esc(md.id)}"${ui.selModels.has(md.id) ? ' checked' : ''}> <span>${esc(md.name)}${md.source ? ' <span class="muted">(add-on)</span>' : ''}<small>${esc(MODEL_BLURB[md.id] || modelInfo(md))}</small></span>`;
    lab.querySelector('input').onchange = e => {
      e.target.checked ? ui.selModels.add(md.id) : ui.selModels.delete(md.id);
      saveSelection(); updateModelNote();
    };
    box.appendChild(lab);
  }
  updateModelNote();
}
function updateModelNote() {
  const ms = selectedModels();
  const ttaUsers = ms.filter(md => md.tta !== false);
  const tta = $('#optTta');
  tta.disabled = !ttaUsers.length;
  const aur = ttaUsers.filter(md => md.kind === 'aurora');
  $('#ttaNote').textContent = (!ttaUsers.length ? 'Not used by ' + (ms.length > 1 ? 'these models' : 'this model') + ', as in ' + (ms.length > 1 ? 'their' : 'its') + ' original software.'
    : ttaUsers.length < ms.length ? 'Applies to ' + ttaUsers.map(shortName).join(' and ') + ' only.' : '') +
    (aur.length ? (ttaUsers.length < ms.length || !ttaUsers.length ? ' ' : '') + aur.map(shortName).join(' and ') + ': 3 passes (as is and flipped on two axes), which is its own 13-pass augmentation without the small noise it adds.' : '');
  if (!ui.running) ui.runModels = null;
  if (ui.series.length) autoAssignSingle();
  renderRoles();
  $('#modelNote').textContent = !ms.length ? 'Check at least one model.'
    : ms.length === 1 ? (ms[0].id === 'hdglio' ? 'Check more than one model to run them on the same prepared images and compare the results.' : modelInfo(ms[0]))
    : ms.length + ' models: they run one after the other on the same prepared images (one brain extraction, one registration), then the results can be compared.';
  if (!ui.running && document.querySelector('#steps li')) renderSteps(null);
}
$('#addModelBtn').onclick = () => $('#modelInput').click();
$('#modelInput').onchange = async e => {
  const f = e.target.files[0]; e.target.value = '';
  if (!f) return;
  toast('Checking ' + f.name + ' ...');
  const buffer = await f.arrayBuffer();
  worker.postMessage({ type: 'addon', name: f.name, buffer }, [buffer]);
};

// reference sequence and grid: remembered in this browser
const refSeq = () => $('#optRefSeq').value;
// the reference actually used: the chosen one if it is loaded (or needed), else the model's first input
function effectiveRef() {
  const ref = refSeq(), req = requiredRoles();
  if (!req.length || req.length === 4 || ui.assignment[ref] || !ui.series.length) return ref;
  return req[0];
}
function updateRefLabels() {
  const ref = effectiveRef(), chosen = refSeq();
  const req = requiredRoles(), ms = selectedModels();
  const others = req.length === 4 ? ROLES.filter(r => r !== ref) : loadedRoles().filter(r => r !== ref);
  const bet = !ms.length || needsBrain(ms);
  $('#betLabel').textContent = bet ? 'Brain extraction with HD-BET on ' + ref + '.' : 'Brain extraction: not used by ' + (ms.length > 1 ? 'these models' : 'this model') + ' (trained on whole-head images).';
  $('#optBet').disabled = !bet;
  $('#betHint').hidden = !bet;
  $('#regLabel').textContent = others.length === 3 ? 'Rigid registration of ' + others.slice(0, 2).join(', ') + ' and ' + others[2] + ' to ' + ref + '.'
    : others.length ? 'Rigid registration of ' + others.join(' and ') + ' to ' + ref + '.' : 'Rigid registration of the other sequences, if any, to ' + ref + '.';
  const notes = [];
  if (ref !== chosen) notes.push(chosen + ' is not loaded, so ' + ref + ' is the reference.');
  if (!ms.length || ms.some(md => md.id === 'hdglio')) notes.push(ref === 'T1'
    ? 'HD-GLIO was published with everything registered to T1.'
    : 'Brain mask, registration and outputs on the ' + ref + ' grid. HD-GLIO was published with T1 as the reference; registering to a thinner-slice ' + ref + ' keeps more detail.');
  for (const md of ms.filter(x => x.kind === 'raidionics')) {
    const inp = (md.inputChannels || ['FLAIR'])[0];
    notes.push(ref === inp && $('#optGrid').value === 'native' ? 'Raidionics runs on the ' + inp + ' as acquired, as in its own software. The HD-BET mask only removes findings outside the brain.'
      : 'Raidionics was validated on the ' + inp + ' as acquired: register to ' + inp + ' with the grid as acquired to match it.');
  }
  for (const md of ms.filter(x => x.usesBrain === false)) {
    const inp = md.inputChannels[0];
    if (!(ref === inp && $('#optGrid').value === 'native')) notes.push(shortName(md) + ' was trained on the ' + inp + ' as acquired: register to ' + inp + ' with the grid as acquired to keep its resolution.');
  }
  $('#refNote').textContent = notes.join(' ');
  if (!ui.running && document.querySelector('#steps li')) renderSteps(null);
}
for (const [id, key] of [['#optRefSeq', 'btsw.refSeq'], ['#optGrid', 'btsw.grid']]) {
  try { const v = localStorage.getItem(key); if (v && [...$(id).options].some(o => o.value === v)) $(id).value = v; } catch (e) { /* storage unavailable */ }
  $(id).addEventListener('change', () => { try { localStorage.setItem(key, $(id).value); } catch (e) { /* ignore */ } updateRefLabels(); });
}
updateRefLabels();

// demo cases
fetch('demo/manifest.json').then(r => (r.ok ? r.json() : [])).then(list => {
  const box = $('#demoList');
  for (const d of list) {
    const b = document.createElement('button');
    b.className = 'demo';
    b.innerHTML = `<b>${esc(d.title)}</b><span>${esc(d.subtitle)}</span>`;
    b.onclick = async () => {
      clearSeries();
      $$('.demo').forEach(x => x.classList.remove('on')); b.classList.add('on');
      ui.demo = d.id;
      const files = [];
      for (const [role, path] of Object.entries(d.files)) {
        const r = await fetch('demo/' + path);
        files.push({ name: role.toLowerCase() + '.nii.gz', buffer: await r.arrayBuffer() });
      }
      ui.caseLabel = d.title + ' - ' + d.subtitle;
      worker.postMessage({ type: 'files', files }, files.map(f => f.buffer));
      $('#optReg').checked = true; $('#optBet').checked = true;
    };
    box.appendChild(b);
  }
}).catch(() => {});

// ---------------------------------------------------------------------------------------
// run
const stepState = {};
function renderSteps(active) {
  const ol = $('#steps');
  ol.innerHTML = '';
  for (const s of steps()) {
    const st = stepState[s.key] || {};
    if (st.skip) continue;
    const li = document.createElement('li');
    li.className = st.err ? 'err' : st.done ? 'done' : s.key === active ? 'active' : '';
    li.innerHTML = `<span>${esc(s.name)}</span><span class="t">${st.ms != null ? fmtMs(st.ms) : ''}</span>`;
    if (s.key === active && !st.done) {
      li.innerHTML += `<div class="bar"><i style="width:${Math.round(100 * (st.frac || 0))}%"></i></div><div class="sub">${esc(st.text || '')}</div>`;
    }
    ol.appendChild(li);
  }
}
const fmtMs = ms => (ms < 60000 ? (ms / 1000).toFixed(1) + ' s' : Math.floor(ms / 60000) + ' min ' + Math.round((ms % 60000) / 1000) + ' s');

let runStart = 0, curStep = null, stepStart = 0;
$('#runBtn').onclick = () => {
  if (ui.running) return;
  if (!selectedModels().length) { toast('Check at least one segmentation model.', true); return; }
  ui.running = true;
  ui.runModels = selectedModels();
  if (cancelFlag) Atomics.store(cancelFlag, 0, 0);
  for (const k of Object.keys(stepState)) delete stepState[k];
  if (!$('#optBet').checked || !needsBrain(ui.runModels)) stepState.hdbet = { skip: true };
  if (loadedRoles().length < 2) stepState.register = { skip: true };
  curStep = 'prepare'; stepStart = runStart = performance.now();
  renderSteps('prepare');
  $('#runBtn').disabled = true; $('#cancelBtn').hidden = !cancelFlag;
  const options = {
    models: ui.runModels.map(md => md.id), reference: (ui.runRef = effectiveRef()), isotropic: $('#optGrid').value === 'iso', skullStrip: $('#optBet').checked, register: $('#optReg').checked,
    tta: $('#optTta').checked && !$('#optTta').disabled, device: $('#optDevice').value, avoidGpu: gpuFailedBefore(),
  };
  worker.postMessage({ type: 'run', assignment: { ...ui.assignment }, options });
};
$('#cancelBtn').onclick = () => { if (cancelFlag) Atomics.store(cancelFlag, 0, 1); };

function onProgress(m) {
  if (m.stage === 'load') { $('#loadStatus').textContent = m.text; return; }
  let key = m.stage === 'model' ? curStep : m.stage;
  if (!steps().find(s => s.key === key)) return;
  if (key !== curStep) {
    const t = performance.now();
    if (curStep) stepState[curStep] = { ...(stepState[curStep] || {}), done: true, ms: t - stepStart };
    curStep = key; stepStart = t;
  }
  stepState[key] = { ...(stepState[key] || {}), frac: m.frac, text: m.text };
  renderSteps(key);
}

function onError(m) {
  ui.running = false;
  $('#cancelBtn').hidden = true;
  if (curStep && stepState[curStep]) stepState[curStep].err = true;
  else if (curStep) stepState[curStep] = { err: true };
  renderSteps(null);
  renderRoles();
  toast(m.message === 'Cancelled' ? 'Cancelled.' : m.message, m.message !== 'Cancelled');
  if (m.stage === 'files') $('#loadStatus').textContent = 'Could not read the files: ' + m.message;
}

// ---------------------------------------------------------------------------------------
// viewer
const NV_OPTS = {
  backColor: [0, 0, 0, 1], crosshairColor: [0.88, 0.63, 0.19, 0.9], crosshairWidth: 1, show3Dcrosshair: false,
  isRadiologicalConvention: true, multiplanarForceRender: false, dragAndDropEnabled: false, loadingText: '',
  textHeight: 0.035, isColorbar: false, isOrientCube: false, multiplanarShowRender: 0, logLevel: 'error',
  sagittalNoseLeft: true, // sagittal: anterior to the left of the screen
};
let nv = null, nvReady = null, viewerLost = false, view = 3;
const images = {};

// The viewer draws with WebGL. A GPU reset (for example the Windows driver timeout that also
// stops WebGPU) destroys that context, so the viewer is rebuilt on a fresh canvas when it is lost.
function makeViewer() {
  const canvas = document.createElement('canvas');
  canvas.id = 'gl';
  $('#gl').replaceWith(canvas);
  nv = new niivue.Niivue(NV_OPTS);
  nv.onLocationChange = onLocation;
  canvas.addEventListener('webglcontextlost', e => { e.preventDefault(); viewerLost = true; }, false);
  canvas.addEventListener('webglcontextrestored', () => { rebuildViewer(); }, false);
  nvReady = nv.attachToCanvas(canvas).then(() => { nv.setSliceType(view); nv.setRadiologicalConvention($('#optRadio').checked); nv.setCrosshairWidth($('#optCross').checked ? 1 : 0); });
  if (window.__hdglio) window.__hdglio.nv = nv;
}
async function rebuildViewer() {
  viewerLost = false;
  makeViewer();
  if (ui.result) { await showVolumes(); try { centreOnTumor(ui.result); } catch (e) { /* ignore */ } }
}
makeViewer();

$$('#views button').forEach(b => b.onclick = () => { $$('#views button').forEach(x => x.classList.remove('on')); b.classList.add('on'); view = +b.dataset.view; nv.setSliceType(view); });
$('#optRadio').onchange = e => nv.setRadiologicalConvention(e.target.checked);
// crosshairs on/off, remembered in this browser
try { if (localStorage.getItem('btsw.crosshairs') === '0') $('#optCross').checked = false; } catch (e) { /* storage unavailable */ }
$('#optCross').onchange = e => {
  nv.setCrosshairWidth(e.target.checked ? 1 : 0);
  try { localStorage.setItem('btsw.crosshairs', e.target.checked ? '1' : '0'); } catch (err) { /* ignore */ }
};
$('#opacity').oninput = e => { const ov = overlayImage(); if (ov) { ov.opacity = +e.target.value; nv.updateGLVolume(); } };
$('#optBrain').onchange = () => showVolumes();
$('#bgSel').onchange = () => showVolumes();
$('#segSel').onchange = () => { ui.overlay = $('#segSel').value; if (ui.overlay === 'cmp') showComparison(); else showVolumes(); renderLegend(); };

const modelById = id => ui.result && ui.result.models.find(md => md.id === id);
// overlay shown on the viewer: 'seg:<model id>' or 'cmp' (agreement map of two models)
const overlayImage = () => (ui.overlay === 'cmp' ? images.cmp : images[ui.overlay]) || null;

function labelLut(md) {
  const K = Math.max(...md.labels.map(l => l.value));
  const R = [0], Gc = [0], B = [0], A = [0], I = [0], names = ['background'];
  for (let v = 1; v <= K; v++) {
    const lab = md.labels.find(l => l.value === v) || { name: 'label ' + v };
    const c = lab.color || COLORS[v] || [200, 200, 200];
    R.push(c[0]); Gc.push(c[1]); B.push(c[2]); A.push(ui.hidden.has(md.id + ':' + v) ? 0 : 255); I.push(v); names.push(lab.name);
  }
  return { R, G: Gc, B, A, I, labels: names };
}
function agreeLut() {
  const [a, b] = ui.cmpPair || [];
  const R = [0], Gc = [0], B = [0], A = [0], I = [0], names = ['background'];
  const nm = ['', shortName(modelById(a)) + ' only', shortName(modelById(b)) + ' only', 'both'];
  for (let v = 1; v <= 3; v++) { const c = AGREE[v]; R.push(c[0]); Gc.push(c[1]); B.push(c[2]); A.push(ui.hidden.has('cmp:' + v) ? 0 : 255); I.push(v); names.push(nm[v]); }
  return { R, G: Gc, B, A, I, labels: names };
}

async function makeImage(buf, name, opts) {
  return niivue.NVImage.loadFromUrl({ url: name, name, buffer: buf, ...opts });
}

async function showVolumes() {
  if (!ui.result) return;
  if (viewerLost) { await rebuildViewer(); return; }
  await nvReady;
  const bg = images[$('#bgSel').value] || images[ui.result.refRole];
  while (nv.volumes.length) nv.removeVolume(nv.volumes[0]);
  nv.addVolume(bg);
  if ($('#optBrain').checked && images.brain) nv.addVolume(images.brain);
  const ov = overlayImage();
  if (ov) {
    ov.setColormapLabel(ui.overlay === 'cmp' ? agreeLut() : labelLut(modelById(ui.overlay.slice(4))));
    ov.opacity = +$('#opacity').value;
    nv.addVolume(ov);
  }
  nv.updateGLVolume();
}

function onLocation(d) {
  if (!d || !d.mm) return;
  const segv = (d.values || []).find(v => v.name === 'seg.nii' || v.name === 'cmp.nii');
  let lab = null;
  if (segv && ui.result && Math.round(segv.value)) {
    const v = Math.round(segv.value);
    if (segv.name === 'cmp.nii') lab = { name: agreeLut().labels[v] };
    else { const md = modelById(ui.overlay.slice(4)); lab = md ? md.labels.find(l => l.value === v) : null; }
  }
  const bgv = (d.values || [])[0];
  const ax = (v, p, n) => (v >= 0 ? p : n) + ' ' + Math.abs(v).toFixed(1);
  $('#cursor').textContent = `${ax(d.mm[0], 'R', 'L')}  ${ax(d.mm[1], 'A', 'P')}  ${ax(d.mm[2], 'S', 'I')} mm` + (bgv ? `   ${Math.round(bgv.value)}` : '') + (lab ? `   ${lab.short || lab.name}` : '');
}

async function onResult(m) {
  ui.running = false;
  $('#cancelBtn').hidden = true;
  const t = performance.now();
  if (curStep) stepState[curStep] = { ...(stepState[curStep] || {}), done: true, ms: t - stepStart };
  curStep = null;
  renderSteps(null);
  renderRoles();
  ui.result = m;
  ui.hidden.clear();
  // viewer volumes: images, brain mask, one label volume per model (the agreement map on request)
  for (const k of Object.keys(images)) delete images[k];
  for (const r of m.roles) images[r] = await makeImage(m.vols[r], r.toLowerCase() + '.nii', { colormap: 'gray' });
  images.brain = await makeImage(m.vols.brain, 'brain.nii', { colormap: 'gray', opacity: 0.18 });
  images.brain.setColormapLabel({ R: [0, 90], G: [0, 200], B: [0, 255], A: [0, 120], I: [0, 1], labels: ['', 'brain'] });
  for (const md of m.models) images['seg:' + md.id] = await makeImage(m.vols['seg:' + md.id], 'seg.nii', { opacity: +$('#opacity').value });
  ui.overlay = 'seg:' + m.models[0].id;
  // default comparison: the first pair of models that shares a region
  ui.cmpPair = m.compare && m.compare.pairs.length ? [m.compare.pairs[0].a, m.compare.pairs[0].b] : null;
  const sel = $('#bgSel');
  sel.innerHTML = m.roles.map(r => `<option value="${r}">${r === m.refRole ? r + ' (reference)' : r + ' (registered)'}</option>`).join('');
  const allIn = m.models.flatMap(md => md.inputs);
  sel.value = allIn.includes('T1c') ? 'T1c' : allIn[0]; sel.disabled = false;
  renderOverlaySelect();
  $('#empty').hidden = true;
  await showVolumes();
  // jump to the tumor centre
  try { centreOnTumor(m); } catch (e) { /* ignore */ }
  renderVolumes();
  renderComparison();
  renderQC();
  renderExports();
  $('#copyBtn').disabled = false; $('#csvBtn').disabled = false;
  if (m.gpuFailed) { setGpuFailed(true); showDevice(); }
  else if (m.usedGpu) { setGpuFailed(false); showDevice(); }
  toast(m.gpuFailed ? 'The GPU stopped responding, so the run finished on the CPU (' + fmtMs(m.timing.total) + '). Automatic will use the CPU on this computer from now on.'
    : 'Segmentation finished in ' + fmtMs(m.timing.total) + '.', m.gpuFailed);
}

function renderOverlaySelect() {
  const m = ui.result, sel = $('#segSel');
  const opts = m.models.map(md => `<option value="seg:${esc(md.id)}">${esc(shortName(md))}</option>`);
  if (m.models.length > 1 && ui.cmpPair) opts.push(`<option value="cmp" title="${esc(shortName(modelById(ui.cmpPair[0])))} vs ${esc(shortName(modelById(ui.cmpPair[1])))}">Agreement map</option>`);
  sel.innerHTML = opts.join('');
  sel.value = ui.overlay; sel.disabled = m.models.length < 2;
}

function centreOnTumor(m) {
  if (!m.tumorCenterMM) return;
  nv.scene.crosshairPos = nv.mm2frac(m.tumorCenterMM);
  nv.updateGLVolume();
}

function fmtVol(v) { return v < 0.05 && v > 0 ? v.toFixed(3) : v.toFixed(2); }

function renderVolumes() {
  const m = ui.result;
  const tb = $('#volTable tbody');
  tb.innerHTML = '';
  const multi = m.models.length > 1;
  for (const md of m.models) {
    if (multi) { const g = document.createElement('tr'); g.className = 'grp'; g.innerHTML = `<td colspan="3">${esc(md.name)}</td>`; tb.appendChild(g); }
    for (const c of md.comps) {
      const tr = document.createElement('tr');
      const single = c.labels.length === 1 ? c.labels[0] : null;
      if (!single) tr.className = 'total';
      const labc = single ? md.labels.find(l => l.value === single) : null;
      const col = single ? (labc && labc.color) || COLORS[single] : null;
      const hk = md.id + ':' + single;
      tr.innerHTML = `<td class="sw">${col ? `<span class="swatch${ui.hidden.has(hk) ? ' off' : ''}" data-h="${esc(hk)}" title="Show or hide" style="background:rgb(${col.join(',')})"></span>` : ''}</td><td>${esc(c.name)}</td><td class="num">${fmtVol(c.cm3)}</td>`;
      tb.appendChild(tr);
    }
  }
  if (m.brainUsed) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td></td><td class="muted">${m.bet ? 'Brain (HD-BET mask)' : 'Brain (nonzero voxels of the input)'}</td><td class="num muted">${m.brainCm3.toFixed(0)}</td>`;
    tb.appendChild(tr);
  }
  $$('#volTable .swatch').forEach(s => s.onclick = () => {
    const h = s.dataset.h; ui.hidden.has(h) ? ui.hidden.delete(h) : ui.hidden.add(h);
    const id = h.split(':')[0];
    if (ui.overlay !== 'seg:' + id) { ui.overlay = 'seg:' + id; $('#segSel').value = ui.overlay; renderLegend(); }
    renderVolumes(); showVolumes();
  });
  const vs = m.voxelSize.map(v => v.toFixed(2)).join(' x ');
  $('#volMeta').textContent = `Counted on the ${m.refRole} grid: ${m.dims.join(' x ')} voxels of ${vs} mm (${m.voxelVolume.toFixed(3)} mm3 each).` + (multi ? ' Click a colour to show that model on the viewer and hide or show the label.' : '');
  const labs = m.models.map(md => (multi ? shortName(md) + ': ' : '') + md.labels.map(l => l.value + ' ' + l.name).join(', ')).join('. ');
  const raw = m.inputs.filter(r => m.inputKind[r] === 'raw');
  const inNote = !raw.length ? 'Inputs are the registered, brain-extracted images given to ' + m.models.filter(md => md.masked).map(shortName).join(' and ') + '.'
    : raw.length === m.inputs.length ? 'The ' + raw.join(', ') + (raw.length > 1 ? ' inputs are the images' : ' input is the image') + ' given to ' + [...new Set(raw.map(r => m.inputModel[r]))].join(' and ') + ' (not brain-extracted).'
    : 'Inputs are the registered, brain-extracted images; ' + raw.map(r => m.inputModel[r] + ' got the ' + r).join(', ') + ' without brain extraction.';
  $('#expNote').textContent = 'Saved on the grid of the reference ' + m.refRole + ' as loaded. Labels: 0 background, ' + labs + '. ' +
    (multi ? 'Agreement map: 1 first model only, 2 second only, 3 both, for the region and models chosen under Model comparison. ' : '') + inNote +
    (m.roles.some(r => !m.inputs.includes(r)) ? ' Other sequences are exported as registered.' : '');
  $('#tableNote').textContent = m.models.map(md => md.id === 'hdglio' ? 'HD-GLIO has no separate necrosis class.' : md.tableNote).filter(Boolean).join(' ');
}

// Model comparison: volumes of the shared regions per model, then Dice and overlap for every pair
// of models in the chosen region; the chosen pair can be shown as an agreement map on the viewer.
function renderComparison() {
  const m = ui.result, C = m.compare;
  const show = m.models.length > 1 && C;
  $('#cmpSection').hidden = !show;
  if (!show) return;
  const regs = C.regions;
  // models that share no region: volumes only, no overlap
  const shared = C.pairs.length > 0;
  $('#cmpPairs').hidden = !shared;
  $('#cmpVolTable thead').innerHTML = `<tr><th>Model</th>${regs.map(r => `<th class="num" title="${esc(r.name)}">${esc(r.short)}</th>`).join('')}</tr>`;
  $('#cmpVolTable tbody').innerHTML = m.models.map(md => {
    const v = C.volumes.find(x => x.id === md.id).regions;
    return `<tr><td>${esc(shortName(md))}</td>${regs.map(r => (v[r.key] ? `<td class="num">${fmtVol(v[r.key].cm3)}</td>` : '<td class="num na">-</td>')).join('')}</tr>`;
  }).join('');
  if (!shared) {
    $('#cmpNote').textContent = 'These models share no region, so no overlap is computed. ' + m.models.map(md => shortName(md) + ': ' + Object.keys(C.volumes.find(x => x.id === md.id).regions).map(k => (regs.find(r => r.key === k) || { name: k }).name.toLowerCase()).join(', ')).join('; ') + '.';
    renderLegend();
    return;
  }
  const rsel = $('#cmpRegion');
  const comparable = regs.filter(r => r.comparable);
  const keep = rsel.value;
  rsel.innerHTML = comparable.map(r => `<option value="${r.key}">${esc(r.name)}</option>`).join('');
  rsel.value = comparable.some(r => r.key === keep) ? keep : comparable[0].key;
  const pickA = $('#cmpA'), pickB = $('#cmpB');
  const opts = m.models.map(md => `<option value="${esc(md.id)}">${esc(shortName(md))}</option>`).join('');
  pickA.innerHTML = opts; pickB.innerHTML = opts;
  pickA.value = ui.cmpPair[0]; pickB.value = ui.cmpPair[1];
  renderPairs();
  const defs = { whole: 'Whole abnormality: all labels of each model', core: 'Tumor core: BraTS NCR + ET, Raidionics tumor core (enhancing tissue, necrosis, cysts); HD-GLIO is not included, as its labels do not say where necrosis goes', enh: 'Enhancing tumor: HD-GLIO contrast-enhancing, BraTS ET', nonenh: 'Non-enhancing abnormality: whole minus enhancing (HD-GLIO non-enhancing, BraTS NCR + ED)' };
  $('#cmpNote').textContent = regs.map(r => defs[r.key] || r.name).join('. ') + '. Regions a model does not have are left out of its comparisons.';
  renderLegend();
}
function renderPairs() {
  const m = ui.result, C = m.compare, region = $('#cmpRegion').value;
  const rows = C.pairs.filter(p => p.region === region);
  $('#cmpPairTable tbody').innerHTML = rows.map(p => `<tr class="pair"><td colspan="4">${esc(shortName(modelById(p.a)))} vs ${esc(shortName(modelById(p.b)))}</td></tr>` +
    `<tr><td class="num">${p.dice == null ? '-' : p.dice.toFixed(3)}</td><td class="num">${fmtVol(p.both)}</td><td class="num">${fmtVol(p.aOnly)}</td><td class="num">${fmtVol(p.bOnly)}</td></tr>`).join('')
    || '<tr><td colspan="4" class="muted">No two models share this region.</td></tr>';
  const a = $('#cmpA').value, b = $('#cmpB').value;
  $('#cmpShow').disabled = a === b || !C.pairs.some(p => p.region === region && ((p.a === a && p.b === b) || (p.a === b && p.b === a)));
}
$('#cmpRegion').onchange = () => { renderPairs(); if (ui.overlay === 'cmp') showComparison(); renderLegend(); };
$('#cmpA').onchange = $('#cmpB').onchange = () => { renderPairs(); };
$('#cmpShow').onclick = () => {
  const a = $('#cmpA').value, b = $('#cmpB').value;
  if (a === b) return;
  ui.cmpPair = [a, b]; ui.overlay = 'cmp';
  renderOverlaySelect(); showComparison(); renderLegend();
};
let agreeReq = 0;
const agreeWait = new Map();
function showComparison() {
  const [a, b] = ui.cmpPair, region = $('#cmpRegion').value;
  const key = a + ':' + b + ':' + region;
  if (images.cmp && images.cmpKey === key) { showVolumes(); return; }
  const id = ++agreeReq;
  agreeWait.set(id, key);
  worker.postMessage({ type: 'agreement', a, b, region, id });
}
async function onAgreement(m) {
  if (!agreeWait.has(m.id)) return;
  agreeWait.delete(m.id);
  images.cmp = await makeImage(m.bytes, 'cmp.nii', { opacity: +$('#opacity').value });
  images.cmpKey = m.key;
  if (ui.overlay === 'cmp') showVolumes();
}
function renderLegend() {
  const box = $('#cmpLegend');
  if (!ui.result || ui.result.models.length < 2 || !ui.cmpPair) { box.innerHTML = ''; return; }
  const [a, b] = ui.cmpPair, rn = $('#cmpRegion').selectedOptions[0];
  const sw = v => `<i style="background:rgb(${AGREE[v].join(',')})"></i>`;
  box.innerHTML = (ui.overlay === 'cmp' ? '<b>On the viewer' + (rn ? ' (' + esc(rn.textContent) + ')' : '') + ':</b> ' : '<b>Show on viewer gives:</b> ') +
    `<span>${sw(1)}${esc(shortName(modelById(a)))} only</span><span>${sw(2)}${esc(shortName(modelById(b)))} only</span><span>${sw(3)}both</span>`;
}

function renderQC() {
  const m = ui.result;
  const rows = [];
  for (const [r, q] of Object.entries(m.reg)) {
    if (q.skipped) { rows.push(`<span>${r}</span><span>not registered (option off)</span>`); continue; }
    const cls = q.check ? ' class="bad"' : '';
    rows.push(`<span>${r}</span><span${cls}>rot ${q.rDeg.map(v => v.toFixed(1)).join('/')} deg, shift ${q.t.map(v => v.toFixed(1)).join('/')} mm, NMI ${q.nmi.toFixed(3)}${q.check ? ' - large: check alignment' : ''}</span>`);
  }
  $('#regQC').innerHTML = rows.length ? `<div class="qc">${rows.join('')}</div><p>Switch the background to each registered sequence to check the alignment.</p>` : 'Only ' + m.refRole + ' was loaded: nothing to register.';
  const dev = m.backend === 'webgpu' ? 'GPU (WebGPU' + (m.gpu && m.gpu.vendor ? ', ' + m.gpu.vendor : '') + ')' : 'CPU (' + (ui.env ? ui.env.threads : '?') + ' threads)';
  const tm = m.timing;
  const wf = w => ({ q8: '8-bit weights', f16: '16-bit weights', f32: '32-bit weights' }[w] || '');
  const aug = md => (md.tta ? (md.kind === 'aurora' ? '3 passes (AURORA without its noise)' : '8 mirrored passes') : md.kind === 'raidionics' ? 'none (as in Raidionics)' : 'off');
  $('#runInfo').innerHTML = '<div class="qc">' + m.models.map(md => `<span>${m.models.length > 1 ? esc(shortName(md)) : 'Model'}</span><span>${esc(md.name)} ${esc(md.version || '')}${wf(md.weights) ? ', ' + wf(md.weights) : ''}; augmentation ${aug(md)}` +
      (md.segInfo && md.kind === 'raidionics' ? `; head crop and 1 mm grid ${md.segInfo.resampledDims.join(' x ')}, ${md.segInfo.patches} patches` : '') +
      (md.segInfo && md.kind === 'aurora' ? `; 1 mm grid ${md.segInfo.grid.join(' x ')}${md.segInfo.sameFrame ? ', the reference as it is' : md.segInfo.snapped ? ' on the reference voxels' : ' (resampled)'}, ${md.segInfo.windows} windows` : '') + `; ${fmtMs(md.ms)}</span>`).join('') +
    (m.bet ? `<span>Brain mask</span><span>HD-BET ${esc(m.bet)}</span>` : '') +
    `<span>Compute</span><span>${dev}</span>` +
    `<span>Time</span><span>${fmtMs(tm.total)} (prepare ${fmtMs(tm.prepare)}, ${m.bet ? 'HD-BET ' + fmtMs(tm.hdbet) + ', ' : ''}registration ${fmtMs(tm.register)}, segmentation ${fmtMs(tm.seg)})</span></div>` +
    (m.notes.length ? '<p class="warn">' + m.notes.map(esc).join('<br>') + '</p>' : '');
}

// ---------------------------------------------------------------------------------------
// export
let expId = 0;
// export buttons: one segmentation button per model, the agreement map for the chosen pair and region
function renderExports() {
  const m = ui.result, box = $('.exports');
  box.querySelectorAll('[data-exp^="seg"]').forEach(b => b.remove());
  const first = box.firstElementChild;
  for (const md of m.models) {
    const b = document.createElement('button');
    b.dataset.exp = 'seg:' + md.id; b.className = 'primary';
    b.textContent = m.models.length > 1 ? 'Segmentation: ' + shortName(md) : 'Segmentation';
    box.insertBefore(b, first);
  }
  // no brain mask to offer when no model used one (the schwannoma models alone)
  box.querySelector('[data-exp="brain"]').hidden = !m.brainUsed;
  $('#optBrain').closest('label').hidden = !m.brainUsed;
  if (!m.brainUsed) $('#optBrain').checked = false;
  const ag = box.querySelector('[data-exp="agree"]');
  ag.hidden = !ui.cmpPair; ag.disabled = !ui.cmpPair;
  ag.title = 'The agreement map of the two models and region chosen under Model comparison';
  for (const b of $$('[data-exp]')) {
    const r = b.dataset.exp.startsWith('input:') ? b.dataset.exp.slice(6) : null;
    if (r) { b.disabled = !m.roles.includes(r); b.textContent = r + (m.roles.includes(r) && !m.inputs.includes(r) ? ' (registered)' : ' input'); }
    else if (b.dataset.exp !== 'agree') b.disabled = false;
  }
}
$('.exports').addEventListener('click', e => {
  const b = e.target.closest('[data-exp]');
  if (!b || b.disabled) return;
  let what = b.dataset.exp;
  if (what === 'agree') {
    const a = $('#cmpA').value, bb = $('#cmpB').value;
    if (a === bb) { toast('Choose two different models under Model comparison.', true); return; }
    what = 'agree:' + a + ':' + bb + ':' + $('#cmpRegion').value;
  }
  const id = ++expId;
  b.disabled = true;
  pendingExports.set(id, () => (b.disabled = false));
  worker.postMessage({ type: 'export', what, id });
});

// one table for copy and CSV: compartment volumes per model, shared-region volumes, and the
// agreement of every pair of models
function tableRows() {
  const m = ui.result, C = m.compare;
  const rows = [['table', 'model', 'model_b', 'compartment_or_region', 'labels', 'volume_cm3', 'voxels', 'dice', 'both_cm3', 'model_only_cm3', 'model_b_only_cm3']];
  for (const md of m.models) for (const c of md.comps) rows.push(['volume', md.id, '', c.name, c.labels.join('+'), c.cm3.toFixed(3), c.voxels, '', '', '', '']);
  if (m.brainUsed) rows.push(['volume', m.bet ? 'hdbet' : 'nonzero', '', 'brain', 'mask', m.brainCm3.toFixed(1), '', '', '', '', '']);
  if (m.models.length > 1 && C) {
    const rn = k => (C.regions.find(r => r.key === k) || { name: k }).name;
    for (const v of C.volumes) for (const [k, r] of Object.entries(v.regions)) rows.push(['region', v.id, '', rn(k), r.labels.join('+'), r.cm3.toFixed(3), r.voxels, '', '', '', '']);
    for (const p of C.pairs) rows.push(['agreement', p.a, p.b, rn(p.region), '', '', '', p.dice == null ? '' : p.dice.toFixed(4), p.both.toFixed(3), p.aOnly.toFixed(3), p.bOnly.toFixed(3)]);
  }
  return rows;
}
$('#copyBtn').onclick = async () => {
  const txt = tableRows().map(r => r.join('\t')).join('\n');
  try { await navigator.clipboard.writeText(txt); toast('Copied.'); } catch (e) { toast('Copy failed: ' + e.message, true); }
};
$('#csvBtn').onclick = () => {
  const m = ui.result;
  const meta = [
    ...m.models.map(md => ['# model', md.id + ': ' + md.name + ' ' + (md.version || '') + (md.tta ? ', test-time augmentation' : '')]), ['# brain_mask', m.bet ? 'HD-BET ' + m.bet : 'input nonzero voxels'],
    ['# reference_grid', m.refRole + ' ' + m.dims.join('x') + ' voxels, ' + m.voxelSize.map(v => v.toFixed(3)).join('x') + ' mm'],
    ['# analysed', new Date().toISOString().slice(0, 10)],
  ];
  const csv = [...meta, ...tableRows()].map(r => r.map(v => /[",\n]/.test(String(v)) ? '"' + String(v).replace(/"/g, '""') + '"' : v).join(',')).join('\n') + '\n';
  download((m.models.length > 1 ? 'model_comparison' : m.models[0].id) + '_volumes.csv', new Blob([csv], { type: 'text/csv' }));
};

function download(name, blob) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

// Remember (in this browser only) that the GPU failed on this computer, so Automatic uses the CPU.
const GPU_KEY = 'btsw.gpuFailed';
function gpuFailedBefore() { try { return localStorage.getItem(GPU_KEY) === '1'; } catch (e) { return false; } }
function setGpuFailed(v) { try { v ? localStorage.setItem(GPU_KEY, '1') : localStorage.removeItem(GPU_KEY); } catch (e) { /* storage unavailable */ } }

function showDevice() {
  if (!ui.env) return;
  const g = ui.env.gpu;
  $('#device').textContent = !(g && !g.software) ? 'No usable GPU in this browser: runs on the CPU (' + ui.env.threads + ' threads).'
    : gpuFailedBefore() ? 'The GPU (' + (g.vendor || 'WebGPU') + ') stopped responding in an earlier run, so Automatic uses the CPU (' + ui.env.threads + ' threads). Choose GPU to try it again.'
    : 'GPU available (' + (g.vendor || 'WebGPU') + '). ' + ui.env.threads + ' CPU threads.';
}
window.__hdglio = { ui, nv, worker, rebuildViewer };

// When served by the standalone launcher, hold a keep-alive stream open so the program exits
// shortly after the last workbench window closes.
fetch('__bts/ping', { cache: 'no-store' }).then(r => (r.ok ? r.text() : '')).then(t => {
  if (t.trim() === 'brain-tumor-segmentation-workbench') new EventSource('__bts/events');
}).catch(() => {});
