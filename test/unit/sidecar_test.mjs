// Sidecar written next to each segmentation for BrainTumIQ (docs: BrainTumIQ docs/mask-sidecar.md).
//   node test/unit/sidecar_test.mjs
import assert from 'assert/strict';
import { maskSidecar, sidecarName, BRAINTUMIQ_LABELS } from '../../app/js/lib/sidecar.js';
import { APP_VERSION } from '../../app/js/version.js';

const affine = new Float64Array([-1, 0, 0, 90, 0, 1, 0, -126, 0, 0, 1, -72, 0, 0, 0, 1]);
const vol = { dims: [182, 218, 182], affine };

// HD-GLIO, registered to T1, brain mask used
let sc = maskSidecar({ model: { id: 'hdglio', name: 'HD-GLIO', version: '2 (Zenodo 4014850)' }, vol, refRole: 'T1', brainMask: true });
assert.equal(sc.schema, 'tumor-mask-sidecar');
assert.equal(sc.schema_version, '1.0');
assert.deepEqual(sc.labels, { 1: 'ED', 2: 'ET' });
assert.equal(sc.source_sequence, 't1');
assert.deepEqual(sc.geometry.shape, [182, 218, 182]);
assert.deepEqual(sc.geometry.affine, [[-1, 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]]);
assert.equal(sc.brain_mask, 'hdbet_brain_mask.nii.gz');
assert.deepEqual(sc.generator, { name: 'Brain Tumor Segmentation Workbench', version: APP_VERSION });
assert.deepEqual(sc.model, { name: 'HD-GLIO', version: '2 (Zenodo 4014850)' });
assert.equal(sc.tumor_type, 'glioma');
assert.equal(sc.grid, undefined);
assert.ok(!Object.is(sc.geometry.affine[0][1], -0));

// BraTS convention; FLAIR reference
sc = maskSidecar({ model: { id: 'brats2020', name: 'nnU-Net BraTS 2020' }, vol, refRole: 'FLAIR' });
assert.deepEqual(sc.labels, { 1: 'NCR', 2: 'ED', 4: 'ET' });
assert.equal(sc.source_sequence, 'flair');
assert.equal(sc.brain_mask, undefined);

// a tumor-core / whole-lesion model: no label meanings (chosen when importing), the rest still there
sc = maskSidecar({ model: { id: 'aurora-t1c', name: 'AURORA T1c' }, vol, refRole: 'T1c', onReferenceGrid: false, brainMask: true });
assert.equal(sc.labels, undefined);
assert.equal(sc.source_sequence, 't1c');
assert.equal(sc.tumor_type, 'brain metastasis');
assert.match(sc.grid, /resampled/);
for (const id of ['raidionics-core', 'vs-t1c', 'vs-t2']) assert.equal(BRAINTUMIQ_LABELS[id], undefined, id);

// the sidecar sits next to the NIfTI, same base name
assert.equal(sidecarName('hdglio_segmentation.nii.gz'), 'hdglio_segmentation.json');
assert.equal(sidecarName('x.nii'), 'x.json');

// every value a mapped model can write is mapped (BrainTumIQ refuses unlisted values)
const cfg = JSON.parse((await import('fs')).readFileSync(new URL('../../app/models/hdglio/config.json', import.meta.url)));
assert.deepEqual(cfg.labels.map(l => String(l.value)).sort(), Object.keys(BRAINTUMIQ_LABELS.hdglio).sort());
console.log('sidecar_test: ok');
