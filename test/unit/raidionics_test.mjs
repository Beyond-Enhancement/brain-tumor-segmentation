// Run the JavaScript Raidionics pipeline (app/js/lib/raidionics.js) with onnxruntime-node and
// write its outputs for test/validate/compare_raidionics.py.
//   node test/unit/raidionics_test.mjs <FLAIR .nii.gz> <brain mask .nii.gz> <exported add-on folder> <out dir> [--onnx FILE]
// --onnx runs that network file (e.g. the original fp32 model.onnx) instead of the packed weights,
// to separate the implementation from the weight compression.
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { readNifti, writeNifti } from '../../app/js/lib/nifti.js';
import { predictRaidionics } from '../../app/js/lib/raidionics.js';
import { unpackWeights } from '../../app/js/lib/ortrun.js';
const require = createRequire('/home/claude/jsdeps/');
const ort = require('onnxruntime-node');
const [flairFile, maskFile, dir, out] = process.argv.slice(2);
const onnxArg = process.argv.includes('--onnx') ? process.argv[process.argv.indexOf('--onnx') + 1] : null;
fs.mkdirSync(out, { recursive: true });
const cfg = JSON.parse(fs.readFileSync(path.join(dir, 'config.json')));
let modelPath = onnxArg;
if (!modelPath) {
  const tmp = fs.mkdtempSync(path.join(out, 'ort-'));
  fs.copyFileSync(path.join(dir, cfg.onnx), path.join(tmp, 'model.onnx'));
  fs.writeFileSync(path.join(tmp, cfg.weights.externalName), unpackWeights(new Uint8Array(fs.readFileSync(path.join(dir, cfg.weights.pack))), cfg.weights));
  modelPath = path.join(tmp, 'model.onnx');
}
const sess = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'], intraOpNumThreads: 2 });
const runPatch = async (p, dims) => (await sess.run({ [sess.inputNames[0]]: new ort.Tensor('float32', p, dims) }))[sess.outputNames[0]].data;
const flair = readNifti(fs.readFileSync(flairFile));
const brain = readNifti(fs.readFileSync(maskFile));
const mask = new Uint8Array(brain.data.length); for (let i = 0; i < mask.length; i++) mask[i] = brain.data[i] > 0 ? 1 : 0;
const debug = {};
const t = performance.now();
const res = await predictRaidionics({ dims: flair.dims, affine: flair.affine, data: flair.data }, mask, cfg, runPatch,
  { debug, onProgress: (st, f, e) => { if (e && e.tile) process.stdout.write('\rpatch ' + e.tile + '/' + e.tiles + '   '); } });
console.log('\nJS pipeline', ((performance.now() - t) / 1000).toFixed(1) + ' s', JSON.stringify({ ...res.info, N: debug.N }));
fs.writeFileSync(path.join(out, 'js_pre.raw'), Buffer.from(debug.img.buffer));
fs.writeFileSync(path.join(out, 'js_prob.raw'), Buffer.from(debug.prob.buffer));
fs.writeFileSync(path.join(out, 'js_meta.json'), JSON.stringify({ N: debug.N, cropBox: debug.cropBox, outAff: [...debug.outAff], info: res.info }));
fs.writeFileSync(path.join(out, 'js_seg.nii.gz'), writeNifti({ dims: flair.dims, affine: flair.affine, data: res.seg }, { dtype: 'u8', gz: true }));
