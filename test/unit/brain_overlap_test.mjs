// keepComponentsTouching (app/js/lib/raidionics.js) against raidionics_rads_lib's
// perform_brain_overlap_refinement (skimage.measure.label, full connectivity) on random blobs.
//   node test/unit/brain_overlap_test.mjs <seg .nii.gz> <mask .nii.gz> <out .nii.gz>
import fs from 'fs';
import { readNifti, writeNifti } from '../../app/js/lib/nifti.js';
import { keepComponentsTouching } from '../../app/js/lib/raidionics.js';
const [segF, maskF, outF] = process.argv.slice(2);
const s = readNifti(fs.readFileSync(segF)), m = readNifti(fs.readFileSync(maskF));
const seg = Uint8Array.from(s.data, v => (v > 0 ? 1 : 0)), mask = Uint8Array.from(m.data, v => (v > 0 ? 1 : 0));
const removed = keepComponentsTouching(seg, mask, s.dims);
fs.writeFileSync(outF, writeNifti({ dims: s.dims, affine: s.affine, data: seg }, { dtype: 'u8', gz: true }));
console.log('removed', removed);
