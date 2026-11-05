// Page controller: loading, sequence assignment, run, viewer, volumes, export.
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const ROLES = ['T1', 'T1c', 'T2', 'FLAIR'];
const ROLE_NAMES = { T1: 'T1 pre-contrast', T1c: 'T1 post-contrast', T2: 'T2', FLAIR: 'FLAIR' };
const STEPS = [
  { key: 'prepare', name: 'Prepare and reorient' },
  { key: 'hdbet', name: 'Brain extraction (HD-BET)' },
  { key: 'register', name: 'Registration', dynamic: () => 'Registration to ' + (ui.runRef || refSeq()) },
  { key: 'hdglio', name: 'Tumor segmentation (HD-GLIO)' },
];
const COLORS = { 1: [255, 210, 63], 2: [255, 77, 77], 3: [80, 160, 255], 4: [120, 230, 140] };

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
  else if (m.type === 'progress') onProgress(m);
  else if (m.type === 'result') onResult(m);
  else if (m.type === 'export') { const f = pendingExports.get(m.id); pendingExports.delete(m.id); download(m.name, new Blob([m.bytes], { type: 'application/gzip' })); if (f) f(); }
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
  renderRoles();
  if (ui.caseLabel) $('#caseLabel').textContent = ui.caseLabel;
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

function renderRoles() {
  const box = $('#roles');
  box.innerHTML = '';
  for (const r of ROLES) {
    const row = document.createElement('div');
    row.className = 'role';
    const sel = document.createElement('select');
    sel.innerHTML = '<option value="">- choose -</option>' + ui.series.map(s => `<option value="${esc(s.id)}">${esc(s.label)}</option>`).join('');
    sel.value = ui.assignment[r] || '';
    sel.onchange = () => { ui.assignment[r] = sel.value; renderRoles(); };
    sel.title = ROLE_NAMES[r];
    const meta = document.createElement('div');
    meta.className = 'meta';
    const s = ui.series.find(x => x.id === ui.assignment[r]);
    const byElim = ui.inferred && ui.inferred.role === r && ui.inferred.id === ui.assignment[r];
    meta.textContent = (byElim ? 'by elimination - check it.  ' : '') + seriesMeta(s);
    if (byElim) meta.classList.add('elim');
    meta.title = byElim ? 'The other three sequences were recognised and this was the only series left, so it was assigned here.' : meta.textContent;
    row.innerHTML = `<span class="tag" title="${ROLE_NAMES[r]}">${r}</span>`;
    row.appendChild(sel); row.appendChild(meta);
    box.appendChild(row);
  }
  const ok = ROLES.every(r => ui.assignment[r]) && new Set(ROLES.map(r => ui.assignment[r])).size === 4;
  $('#runBtn').disabled = !ok || ui.running;
  // pre-processed inputs on one grid: default to skipping registration and brain extraction
  const chosen = ROLES.map(r => ui.series.find(x => x.id === ui.assignment[r])).filter(Boolean);
  if (chosen.length === 4 && chosen.every(s => s.kind === 'nifti') && ui.demo == null) {
    const same = chosen.every(s => s.dims.join() === chosen[0].dims.join() && s.spacing.join() === chosen[0].spacing.join());
    if (same && !ui.autoSet) { $('#optReg').checked = false; ui.autoSet = true; toast('All four images share one grid: registration switched off. Switch it back on if they are not co-registered.'); }
  }
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// reference sequence and grid: remembered in this browser
const refSeq = () => $('#optRefSeq').value;
function updateRefLabels() {
  const ref = refSeq();
  const others = ROLES.filter(r => r !== ref);
  $('#betLabel').textContent = 'Brain extraction with HD-BET on ' + ref + '.';
  $('#regLabel').textContent = 'Rigid registration of ' + others.slice(0, 2).join(', ') + ' and ' + others[2] + ' to ' + ref + '.';
  $('#refNote').textContent = ref === 'T1'
    ? 'HD-GLIO was published with everything registered to T1.'
    : 'Brain mask, registration and outputs on the ' + ref + ' grid. HD-GLIO was published with T1 as the reference; registering to a thinner-slice ' + ref + ' keeps more detail.';
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
  for (const s of STEPS) {
    const st = stepState[s.key] || {};
    if (st.skip) continue;
    const li = document.createElement('li');
    li.className = st.err ? 'err' : st.done ? 'done' : s.key === active ? 'active' : '';
    li.innerHTML = `<span>${s.dynamic ? s.dynamic() : s.name}</span><span class="t">${st.ms != null ? fmtMs(st.ms) : ''}</span>`;
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
  ui.running = true;
  if (cancelFlag) Atomics.store(cancelFlag, 0, 0);
  for (const k of Object.keys(stepState)) delete stepState[k];
  if (!$('#optBet').checked) stepState.hdbet = { skip: true };
  curStep = 'prepare'; stepStart = runStart = performance.now();
  renderSteps('prepare');
  $('#runBtn').disabled = true; $('#cancelBtn').hidden = !cancelFlag;
  const options = {
    reference: (ui.runRef = refSeq()), isotropic: $('#optGrid').value === 'iso', skullStrip: $('#optBet').checked, register: $('#optReg').checked,
    tta: $('#optTta').checked, device: $('#optDevice').value, avoidGpu: gpuFailedBefore(),
  };
  worker.postMessage({ type: 'run', assignment: { ...ui.assignment }, options });
};
$('#cancelBtn').onclick = () => { if (cancelFlag) Atomics.store(cancelFlag, 0, 1); };

function onProgress(m) {
  if (m.stage === 'load') { $('#loadStatus').textContent = m.text; return; }
  let key = m.stage === 'model' ? curStep : m.stage;
  if (!STEPS.find(s => s.key === key)) return;
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
$('#opacity').oninput = e => { if (images.seg) { images.seg.opacity = +e.target.value; nv.updateGLVolume(); } };
$('#optBrain').onchange = () => showVolumes();
$('#bgSel').onchange = () => showVolumes();

function labelLut() {
  const cfg = ui.result;
  const K = cfg ? Math.max(...cfg.labels.map(l => l.value)) : 2;
  const R = [0], Gc = [0], B = [0], A = [0], I = [0], names = ['background'];
  for (let v = 1; v <= K; v++) {
    const lab = cfg.labels.find(l => l.value === v) || { name: 'label ' + v };
    const c = COLORS[v] || [200, 200, 200];
    R.push(c[0]); Gc.push(c[1]); B.push(c[2]); A.push(ui.hidden.has(v) ? 0 : 255); I.push(v); names.push(lab.name);
  }
  return { R, G: Gc, B, A, I, labels: names };
}

async function makeImage(buf, name, opts) {
  return niivue.NVImage.loadFromUrl({ url: name, name, buffer: buf, ...opts });
}

async function showVolumes() {
  if (!ui.result) return;
  if (viewerLost) { await rebuildViewer(); return; }
  await nvReady;
  const bg = images[$('#bgSel').value] || images.T1c;
  while (nv.volumes.length) nv.removeVolume(nv.volumes[0]);
  nv.addVolume(bg);
  if ($('#optBrain').checked && images.brain) nv.addVolume(images.brain);
  images.seg.setColormapLabel(labelLut());
  images.seg.opacity = +$('#opacity').value;
  nv.addVolume(images.seg);
  nv.updateGLVolume();
}

function onLocation(d) {
  if (!d || !d.mm) return;
  const segv = (d.values || []).find(v => v.name === 'seg.nii');
  const lab = segv && ui.result ? ui.result.labels.find(l => l.value === Math.round(segv.value)) : null;
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
  // viewer volumes
  for (const k of Object.keys(images)) delete images[k];
  for (const r of ROLES) images[r] = await makeImage(m.vols[r], r.toLowerCase() + '.nii', { colormap: 'gray' });
  images.brain = await makeImage(m.vols.brain, 'brain.nii', { colormap: 'gray', opacity: 0.18 });
  images.brain.setColormapLabel({ R: [0, 90], G: [0, 200], B: [0, 255], A: [0, 120], I: [0, 1], labels: ['', 'brain'] });
  images.seg = await makeImage(m.vols.seg, 'seg.nii', { opacity: +$('#opacity').value });
  const sel = $('#bgSel');
  sel.innerHTML = ROLES.map(r => `<option value="${r}">${r === m.refRole ? r + ' (reference)' : r + ' (registered)'}</option>`).join('');
  sel.value = 'T1c'; sel.disabled = false;
  $('#empty').hidden = true;
  await showVolumes();
  // jump to the tumor centre
  try { centreOnTumor(m); } catch (e) { /* ignore */ }
  renderVolumes();
  renderQC();
  $$('[data-exp]').forEach(b => (b.disabled = false));
  $('#copyBtn').disabled = false; $('#csvBtn').disabled = false;
  if (m.gpuFailed) { setGpuFailed(true); showDevice(); }
  else if (m.usedGpu) { setGpuFailed(false); showDevice(); }
  toast(m.gpuFailed ? 'The GPU stopped responding, so the run finished on the CPU (' + fmtMs(m.timing.total) + '). Automatic will use the CPU on this computer from now on.'
    : 'Segmentation finished in ' + fmtMs(m.timing.total) + '.', m.gpuFailed);
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
  for (const c of m.comps) {
    const tr = document.createElement('tr');
    const single = c.labels.length === 1 ? c.labels[0] : null;
    if (!single) tr.className = 'total';
    const col = single ? COLORS[single] : null;
    tr.innerHTML = `<td class="sw">${col ? `<span class="swatch${ui.hidden.has(single) ? ' off' : ''}" data-l="${single}" title="Show or hide" style="background:rgb(${col.join(',')})"></span>` : ''}</td><td>${esc(c.name)}</td><td class="num">${fmtVol(c.cm3)}</td>`;
    tb.appendChild(tr);
  }
  const tr = document.createElement('tr');
  tr.innerHTML = `<td></td><td class="muted">Brain (HD-BET mask)</td><td class="num muted">${m.brainCm3.toFixed(0)}</td>`;
  if (stepState.hdbet && stepState.hdbet.skip) tr.innerHTML = `<td></td><td class="muted">Brain (nonzero voxels of the input)</td><td class="num muted">${m.brainCm3.toFixed(0)}</td>`;
  tb.appendChild(tr);
  $$('.swatch').forEach(s => s.onclick = () => { const l = +s.dataset.l; ui.hidden.has(l) ? ui.hidden.delete(l) : ui.hidden.add(l); renderVolumes(); showVolumes(); });
  const vs = m.voxelSize.map(v => v.toFixed(2)).join(' x ');
  $('#volMeta').textContent = `Counted on the ${m.refRole} grid: ${m.dims.join(' x ')} voxels of ${vs} mm (${m.voxelVolume.toFixed(3)} mm3 each).`;
  $('#expNote').textContent = 'Saved on the grid of the reference ' + m.refRole + ' as loaded. Labels: 0 background, ' + m.labels.map(l => l.value + ' ' + l.name).join(', ') + '. Inputs are the registered, brain-extracted images given to HD-GLIO.';
}

function renderQC() {
  const m = ui.result;
  const rows = [];
  for (const [r, q] of Object.entries(m.reg)) {
    if (q.skipped) { rows.push(`<span>${r}</span><span>not registered (option off)</span>`); continue; }
    const cls = q.check ? ' class="bad"' : '';
    rows.push(`<span>${r}</span><span${cls}>rot ${q.rDeg.map(v => v.toFixed(1)).join('/')} deg, shift ${q.t.map(v => v.toFixed(1)).join('/')} mm, NMI ${q.nmi.toFixed(3)}${q.check ? ' - large: check alignment' : ''}</span>`);
  }
  $('#regQC').innerHTML = `<div class="qc">${rows.join('')}</div><p>Switch the background to each registered sequence to check the alignment.</p>`;
  const dev = m.backend === 'webgpu' ? 'GPU (WebGPU' + (m.gpu && m.gpu.vendor ? ', ' + m.gpu.vendor : '') + ')' : 'CPU (' + (ui.env ? ui.env.threads : '?') + ' threads)';
  const tm = m.timing;
  const wf = { q8: '8-bit weights', f16: '16-bit weights', f32: '32-bit weights' }[m.model.weights] || '';
  $('#runInfo').innerHTML = `<div class="qc"><span>Model</span><span>${esc(m.model.name)} ${esc(m.model.version || '')}${wf ? ', ' + wf : ''}</span>` +
    (m.bet ? `<span>Brain mask</span><span>HD-BET ${esc(m.bet)}</span>` : '') +
    `<span>Compute</span><span>${dev}</span><span>Augmentation</span><span>${m.tta ? '8 mirrored passes' : 'off'}</span>` +
    `<span>Time</span><span>${fmtMs(tm.total)} (prepare ${fmtMs(tm.prepare)}, HD-BET ${fmtMs(tm.hdbet)}, registration ${fmtMs(tm.register)}, HD-GLIO ${fmtMs(tm.hdglio)})</span></div>` +
    (m.notes.length ? '<p class="warn">' + m.notes.map(esc).join('<br>') + '</p>' : '');
}

// ---------------------------------------------------------------------------------------
// export
let expId = 0;
$$('[data-exp]').forEach(b => b.onclick = () => {
  const id = ++expId;
  b.disabled = true;
  pendingExports.set(id, () => (b.disabled = false));
  worker.postMessage({ type: 'export', what: b.dataset.exp, id });
});

function tableRows() {
  const m = ui.result;
  const rows = [['compartment', 'labels', 'volume_cm3', 'voxels']];
  for (const c of m.comps) rows.push([c.name, c.labels.join('+'), c.cm3.toFixed(3), c.voxels]);
  rows.push(['brain', 'mask', m.brainCm3.toFixed(1), '']);
  return rows;
}
$('#copyBtn').onclick = async () => {
  const txt = tableRows().map(r => r.join('\t')).join('\n');
  try { await navigator.clipboard.writeText(txt); toast('Copied.'); } catch (e) { toast('Copy failed: ' + e.message, true); }
};
$('#csvBtn').onclick = () => {
  const m = ui.result;
  const meta = [
    ['# model', m.model.name + ' ' + (m.model.version || '')], ['# brain_mask', m.bet ? 'HD-BET ' + m.bet : 'input nonzero voxels'],
    ['# reference_grid', m.refRole + ' ' + m.dims.join('x') + ' voxels, ' + m.voxelSize.map(v => v.toFixed(3)).join('x') + ' mm'],
    ['# test_time_augmentation', m.tta ? 'on' : 'off'], ['# analysed', new Date().toISOString().slice(0, 10)],
  ];
  const csv = [...meta, ...tableRows()].map(r => r.map(v => /[",\n]/.test(String(v)) ? '"' + String(v).replace(/"/g, '""') + '"' : v).join(',')).join('\n') + '\n';
  download('hdglio_volumes.csv', new Blob([csv], { type: 'text/csv' }));
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
