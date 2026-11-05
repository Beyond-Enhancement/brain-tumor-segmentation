// Assemble the DICOM variants (JPEG lossless series, enhanced multi-frame file) with
// app/js/lib/dicom.js and compare voxel by voxel with the NIfTI they were written from.
//   node test/unit/dicom_variants_test.mjs <dicom dir> <demo dir>
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
// the emscripten codec builds take their Node.js branch here and expect CommonJS globals
globalThis.require = createRequire(import.meta.url);
globalThis.__dirname = new URL('../../app/vendor/codecs/', import.meta.url).pathname;
const DCM = await import('../../app/js/lib/dicom.js');
import { readNifti } from '../../app/js/lib/nifti.js';
import { reorientPlan, applyReorient } from '../../app/js/lib/geom.js';
const std = v => applyReorient(v, reorientPlan(v)); // compare in one voxel order (LAS)
const [dir, demo] = process.argv.slice(2);
const files = [];
const walk = d => { for (const f of fs.readdirSync(d)) { const p = path.join(d, f); fs.statSync(p).isDirectory() ? walk(p) : files.push(p); } };
walk(dir);
const parsed = files.map(f => DCM.parseFile(path.basename(f), new Uint8Array(fs.readFileSync(f)))).filter(Boolean);
const vols = DCM.groupSeries(parsed);
const guess = DCM.assignRoles(vols.map(v => ({ id: v.id, scores: DCM.roleScores(DCM.describe(v)) })));
for (const v of vols) {
  const d = DCM.describe(v);
  const role = Object.keys(guess).find(r => guess[r] === v.id);
  if (!role) { console.log(`${d.label}: not assigned (${d.dims.join('x')})`); continue; }
  const vol = std(await DCM.buildVolume(v, new URL('../../app/', import.meta.url).href));
  const ref = std(readNifti(fs.readFileSync(path.join(demo, role.toLowerCase() + '.nii.gz'))));
  let maxd = 0; for (let i = 0; i < ref.data.length; i++) maxd = Math.max(maxd, Math.abs(Math.max(0, Math.round(ref.data[i])) - vol.data[i]));
  let maxa = 0; for (let i = 0; i < 12; i++) maxa = Math.max(maxa, Math.abs(ref.affine[i] - vol.affine[i]));
  console.log(`${d.label} -> ${role}: ${d.ts}, ${d.dims.join('x')}, voxel max diff ${maxd}, affine max diff ${maxa.toExponential(1)} mm`);
}
