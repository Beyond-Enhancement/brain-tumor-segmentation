// Run the JavaScript AURORA pipeline (app/js/lib/aurora.js) with onnxruntime-node and write its
// outputs for test/validate/compare_aurora.py.
//   node test/unit/aurora_test.mjs <exported add-on folder> <brain mask .nii.gz> <out dir> <input .nii.gz>... [--tta] [--onnx FILE]
// The inputs (in the order of the model's inputChannels) are the brain-extracted images on one
// grid, e.g. the workbench's aurora-*_input_*.nii.gz exports, with its hdbet_brain_mask.nii.gz.
// Writes: aurora_grid_<role>.nii.gz (the images on AURORA's internal 1 mm grid, before
// intensity scaling: what brainles_aurora is then given), js_prob.raw (sigmoid, 2 channels, on
// the internal grid, numpy C order), js_meta.json, js_seg.nii.gz (labels on the input grid).
// --onnx runs that network file instead of the packed weights.
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { readNifti, writeNifti } from '../../app/js/lib/nifti.js';
import { predictAurora } from '../../app/js/lib/aurora.js';
import { unpackWeights } from '../../app/js/lib/ortrun.js';
const require = createRequire('/home/claude/jsdeps/');
const ort = require('onnxruntime-node');
const args = process.argv.slice(2).filter((a, i, all) => !a.startsWith('--') && !(i > 0 && all[i - 1] === '--onnx'));
const [dir, maskFile, out, ...inputs] = args;
const tta = process.argv.includes('--tta');
const onnxArg = process.argv.includes('--onnx') ? process.argv[process.argv.indexOf('--onnx') + 1] : null;
fs.mkdirSync(out, { recursive: true });
const cfg = JSON.parse(fs.readFileSync(path.join(dir, 'config.json')));
if (inputs.length !== cfg.inputChannels.length) throw new Error('give ' + cfg.inputChannels.join(', '));
let modelPath = onnxArg;
if (!modelPath) {
  const tmp = fs.mkdtempSync(path.join(out, 'ort-'));
  fs.copyFileSync(path.join(dir, cfg.onnx), path.join(tmp, 'model.onnx'));
  fs.writeFileSync(path.join(tmp, cfg.weights.externalName), unpackWeights(new Uint8Array(fs.readFileSync(path.join(dir, cfg.weights.pack))), cfg.weights));
  modelPath = path.join(tmp, 'model.onnx');
}
const sess = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'], intraOpNumThreads: 4 });
const runPatch = async (p, dims) => (await sess.run({ [sess.inputNames[0]]: new ort.Tensor('float32', p, dims) }))[sess.outputNames[0]].data;
const vols = inputs.map(f => readNifti(fs.readFileSync(f)));
const brainV = readNifti(fs.readFileSync(maskFile));
const brain = new Uint8Array(brainV.data.length); for (let i = 0; i < brain.length; i++) brain[i] = brainV.data[i] > 0 ? 1 : 0;
const grid = { dims: vols[0].dims, affine: vols[0].affine };
const debug = {};
const t = performance.now();
const res = await predictAurora(vols.map(v => Float32Array.from(v.data)), grid, brain, cfg, runPatch,
  { tta, debug, onProgress: (st, f, e) => { if (e && e.tile) process.stdout.write('\rpass ' + e.mirror + '/' + e.mirrors + ' window ' + e.tile + '/' + e.tiles + '   '); } });
console.log('\nJS pipeline', ((performance.now() - t) / 1000).toFixed(1) + ' s', JSON.stringify(res.info));
const G = debug.grid;
// internal grid images as NIfTI: our arrays are numpy C order (x, y, z); NIfTI wants x fastest
const toNifti = a => {
  const [N0, N1, N2] = G.dims, o = new Float32Array(a.length);
  for (let x = 0, p = 0; x < N0; x++) for (let y = 0; y < N1; y++) for (let z = 0; z < N2; z++, p++) o[x + N0 * (y + N1 * z)] = a[p];
  return o;
};
cfg.inputChannels.forEach((r, i) => fs.writeFileSync(path.join(out, 'aurora_grid_' + r.toLowerCase() + '.nii.gz'),
  writeNifti({ dims: G.dims, affine: G.affine, data: toNifti(debug.raw[i]) }, { dtype: 'f32', gz: true })));
fs.writeFileSync(path.join(out, 'js_prob.raw'), Buffer.from(debug.prob.buffer, debug.prob.byteOffset, debug.prob.byteLength));
fs.writeFileSync(path.join(out, 'js_meta.json'), JSON.stringify({ dims: G.dims, affine: [...G.affine], snapped: G.snapped, pct: debug.pct, tta, info: res.info }));
fs.writeFileSync(path.join(out, 'js_seg.nii.gz'), writeNifti({ dims: grid.dims, affine: grid.affine, data: res.seg }, { dtype: 'u8', gz: true }));
