// Run the JavaScript nnU-Net engine (app/js/lib/nnunet.js) with onnxruntime-node on the inputs
// written by test/validate/engine_v1_reference.py and compare with nnU-Net's own output.
// node test/unit/engine_test.mjs <dir> [--pp]
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { readNifti } from '../../app/js/lib/nifti.js';
import { spacing } from '../../app/js/lib/geom.js';
import { predictCase } from '../../app/js/lib/nnunet.js';
import { unpackWeights } from '../../app/js/lib/ortrun.js';
const require = createRequire('/home/claude/jsdeps/');
const ort = require('onnxruntime-node');
const dir = process.argv[2], pp = process.argv.includes('--pp');
const cfg = JSON.parse(fs.readFileSync(path.join(dir, 'config.json')));
if (pp) cfg.postprocessing = { forWhichClasses: [[1, 2], 2] };
const pack = new Uint8Array(fs.readFileSync(path.join(dir, cfg.weights.pack)));
const tmp = fs.mkdtempSync('/tmp/claude-0/-home-claude/6206506b-0c74-5d1e-bd38-d9a96d8d4346/scratchpad/ort-');
fs.copyFileSync(path.join(dir, cfg.onnx), path.join(tmp, 'model.onnx'));
fs.writeFileSync(path.join(tmp, cfg.weights.externalName), unpackWeights(pack, cfg.weights));
const sess = await ort.InferenceSession.create(path.join(tmp, 'model.onnx'), { executionProviders: ['cpu'], intraOpNumThreads: 2 });
const runPatch = async (p, dims) => (await sess.run({ [sess.inputNames[0]]: new ort.Tensor('float32', p, dims) }))[sess.outputNames[0]].data;
const vols = ['t1', 't1c', 't2', 'flair'].map(r => readNifti(fs.readFileSync(path.join(dir, 'in_' + r + '.nii.gz'))));
const d = vols[0].dims, sp = spacing(vols[0].affine);
const t = performance.now();
const debug = {};
const res = await predictCase(vols.map(v => v.data), [d[2], d[1], d[0]], [sp[2], sp[1], sp[0]], cfg, runPatch, { tta: cfg.tta, debug });
fs.writeFileSync(path.join(dir, 'js_pre.raw'), Buffer.from(debug.pre.buffer));
fs.writeFileSync(path.join(dir, 'js_probs.raw'), Buffer.from(debug.probs.buffer));
console.log('JS engine', ((performance.now() - t) / 1000).toFixed(1) + ' s', JSON.stringify(res.info));
const ref = readNifti(fs.readFileSync(path.join(dir, pp ? 'ref_seg_pp.nii.gz' : 'ref_seg.nii.gz'))).data;
let agree = 0; const n = ref.length;
const dice = {};
for (const c of [1, 2]) { let a = 0, b = 0, ab = 0; for (let i = 0; i < n; i++) { const x = res.seg[i] === c, y = ref[i] === c; a += x; b += y; ab += x && y; } dice[c] = { js: a, ref: b, dice: (2 * ab / (a + b)).toFixed(5) }; }
for (let i = 0; i < n; i++) agree += res.seg[i] === ref[i];
console.log('voxel agreement', (agree / n * 100).toFixed(4) + '%', 'differing voxels', n - agree, JSON.stringify(dice));
