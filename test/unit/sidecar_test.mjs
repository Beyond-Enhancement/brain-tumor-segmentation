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

// a metastasis model: the metastasis (enhancing tumor with necrosis) is BrainTumIQ's undivided core
sc = maskSidecar({ model: { id: 'aurora-t1c', name: 'AURORA T1c' }, vol, refRole: 'T1c', onReferenceGrid: false, brainMask: true });
assert.deepEqual(sc.labels, { 1: 'ED', 2: 'CORE' });
assert.equal(sc.source_sequence, 't1c');
assert.equal(sc.tumor_type, 'brain metastasis');
assert.match(sc.grid, /resampled/);
for (const id of ['raidionics-core', 'vs-t1c', 'vs-t2']) assert.deepEqual(BRAINTUMIQ_LABELS[id], { 1: 'CORE' }, id);
// a model the table does not know: no labels, the rest still there
sc = maskSidecar({ model: { id: 'something-new', name: 'X' }, vol, refRole: 'T2' });
assert.equal(sc.labels, undefined);
assert.equal(sc.source_sequence, 't2');

// the sidecar sits next to the NIfTI, same base name
assert.equal(sidecarName('hdglio_segmentation.nii.gz'), 'hdglio_segmentation.json');
assert.equal(sidecarName('x.nii'), 'x.json');

// every model the workbench ships (built-in and add-on files) is mapped, value for value
// (BrainTumIQ refuses a value the sidecar does not list)
const fs = await import('fs');
const zlib = await import('zlib');
// the built-in model's id is its folder name
const configs = [{ id: 'hdglio', ...JSON.parse(fs.readFileSync(new URL('../../app/models/hdglio/config.json', import.meta.url))) }];
const dl = new URL('../../downloads/', import.meta.url);
for (const f of fs.readdirSync(dl).filter(f => f.endsWith('.btsmodel'))) {
  const z = fs.readFileSync(new URL(f, dl));                    // zip; config.json is the first entry
  assert.equal(z.readUInt32LE(0), 0x04034b50, f);
  const n = z.readUInt16LE(26), m = z.readUInt16LE(28), size = z.readUInt32LE(18);
  assert.equal(z.toString('latin1', 30, 30 + n), 'config.json', f);
  configs.push(JSON.parse(zlib.inflateRawSync(z.subarray(30 + n + m, 30 + n + m + size)).toString('utf8')));
}
assert.ok(configs.length >= 8, `found ${configs.length} models`); // HD-GLIO and the 7 add-ons
for (const cfg of configs) {
  const map = BRAINTUMIQ_LABELS[cfg.id];
  assert.ok(map, `no BrainTumIQ labels for ${cfg.id}`);
  assert.deepEqual(cfg.labels.map(l => String(l.value)).sort(), Object.keys(map).sort(), cfg.id);
}
console.log(`sidecar_test: ok (${configs.length} models)`);
