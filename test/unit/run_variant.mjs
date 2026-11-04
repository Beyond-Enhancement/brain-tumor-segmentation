// Run an exported network variant (any app/models-style folder) with the JavaScript engine on
// given inputs and write the segmentation. Used to measure the effect of weight storage
// precision (fp16 vs int8) on identical inputs.
//   node test/unit/run_variant.mjs <model dir> <out.nii.gz> <in1.nii.gz> [in2 ...] [--tta]
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { readNifti, writeNifti } from '../../app/js/lib/nifti.js';
import { spacing } from '../../app/js/lib/geom.js';
import { predictCase } from '../../app/js/lib/nnunet.js';
import { unpackWeights } from '../../app/js/lib/ortrun.js';
const require = createRequire('/home/claude/jsdeps/');
const useWasm = process.argv.includes('--wasm');
const ort = useWasm ? require('onnxruntime-web') : require('onnxruntime-node');
if (useWasm) { ort.env.wasm.numThreads = 2; }
const a = process.argv.slice(2).filter(x => !x.startsWith('--'));
const [dir, out, ...inputs] = a;
const tta = process.argv.includes('--tta');
const cfg = JSON.parse(fs.readFileSync(path.join(dir, 'config.json')));
const tmp = fs.mkdtempSync('/tmp/claude-0/-home-claude/6206506b-0c74-5d1e-bd38-d9a96d8d4346/scratchpad/ort-');
fs.copyFileSync(path.join(dir, cfg.onnx), path.join(tmp, 'model.onnx'));
fs.writeFileSync(path.join(tmp, cfg.weights.externalName), unpackWeights(new Uint8Array(fs.readFileSync(path.join(dir, cfg.weights.pack))), cfg.weights));
const sess = useWasm
  ? await ort.InferenceSession.create(new Uint8Array(fs.readFileSync(path.join(tmp, 'model.onnx'))), { executionProviders: ['wasm'], externalData: [{ path: cfg.weights.externalName, data: new Uint8Array(fs.readFileSync(path.join(tmp, cfg.weights.externalName))) }] })
  : await ort.InferenceSession.create(path.join(tmp, 'model.onnx'), { executionProviders: ['cpu'], intraOpNumThreads: 2 });
fs.rmSync(tmp, { recursive: true });
const runPatch = async (p, dims) => (await sess.run({ [sess.inputNames[0]]: new ort.Tensor('float32', p, dims) }))[sess.outputNames[0]].data;
const vols = inputs.map(f => readNifti(fs.readFileSync(f)));
const d = vols[0].dims, sp = spacing(vols[0].affine);
const t = performance.now();
const res = await predictCase(vols.map(v => v.data), [d[2], d[1], d[0]], [sp[2], sp[1], sp[0]], cfg, runPatch, { tta });
fs.writeFileSync(out, writeNifti({ dims: d, affine: vols[0].affine, data: res.seg }, { dtype: 'u8' }));
console.log(path.basename(dir), useWasm ? 'wasm' : 'native', cfg.weights.format, tta ? 'tta' : 'no-tta', ((performance.now() - t) / 1000).toFixed(1) + ' s', JSON.stringify(res.info));
