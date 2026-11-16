// Tumor-mask sidecar (version 1): a small JSON file written next to each exported segmentation,
// read by BrainTumIQ so that it imports the label map without guessing what the values mean or
// which image the mask is on. Format: https://github.com/Beyond-Enhancement/BrainTumIQ, docs/mask-sidecar.md
import { APP_VERSION } from '../version.js';

// Workbench label value -> BrainTumIQ compartment: NCR necrosis / non-enhancing core, ED edema /
// FLAIR hyperintensity, ET enhancing tumor, CORE undivided tumor core (enhancing tissue and
// necrosis labelled as one, or a whole lesion; counted in BrainTumIQ's whole tumor and tumor core
// but never as enhancing tumor or necrosis). Every model of the workbench is listed; a model
// missing here would get a sidecar without labels, and its convention would be chosen at import.
export const BRAINTUMIQ_LABELS = {
  hdglio: { 1: 'ED', 2: 'ET' },              // 1 non-enhancing T2/FLAIR abnormality, 2 contrast-enhancing
  brats2020: { 1: 'NCR', 2: 'ED', 4: 'ET' }, // BraTS convention
  'raidionics-flair': { 1: 'ED' },           // FLAIR hyperintensity (tumor and edema)
  'raidionics-core': { 1: 'CORE' },          // tumor core: enhancing tissue, necrosis, cysts
  'aurora-t1c': { 1: 'ED', 2: 'CORE' },      // 2 metastasis: enhancing tumor and necrosis
  'aurora-t1c-flair': { 1: 'ED', 2: 'CORE' },
  'vs-t1c': { 1: 'CORE' },                   // whole schwannoma
  'vs-t2': { 1: 'CORE' },
};

export const TUMOR_TYPES = {
  hdglio: 'glioma',
  brats2020: 'glioma',
  'aurora-t1c': 'brain metastasis',
  'aurora-t1c-flair': 'brain metastasis',
  'vs-t1c': 'vestibular schwannoma',
  'vs-t2': 'vestibular schwannoma',
};

const ROLE_KEYS = { T1: 't1', T1c: 't1c', T2: 't2', FLAIR: 'flair' };
export const BRAIN_MASK_FILE = 'hdbet_brain_mask.nii.gz';

const round6 = v => Math.round(v * 1e6) / 1e6 + 0; // + 0 turns -0 into 0

// model: { id, name, version }; vol: the exported volume { dims, affine (row-major 16) };
// refRole: the reference sequence (T1, T1c, T2, FLAIR); onReferenceGrid: false when the run used
// the 1 mm grid (the mask is then in the reference's space but not on its voxels);
// brainMask: whether the model used the brain mask (it is exported as BRAIN_MASK_FILE).
export function maskSidecar({ model, vol, refRole, onReferenceGrid = true, brainMask = false }) {
  const A = vol.affine;
  const sc = { schema: 'tumor-mask-sidecar', schema_version: '1.0', content: 'labels' };
  const labels = BRAINTUMIQ_LABELS[model.id];
  if (labels) sc.labels = Object.fromEntries(Object.entries(labels).map(([k, v]) => [String(k), v]));
  if (ROLE_KEYS[refRole]) sc.source_sequence = ROLE_KEYS[refRole];
  sc.geometry = {
    shape: vol.dims.slice(0, 3),
    affine: [0, 1, 2, 3].map(r => [0, 1, 2, 3].map(c => round6(r < 3 ? A[4 * r + c] : (c === 3 ? 1 : 0)))),
  };
  if (brainMask) sc.brain_mask = BRAIN_MASK_FILE;
  sc.generator = { name: 'Brain Tumor Segmentation Workbench', version: APP_VERSION };
  sc.model = { name: String(model.name || model.id), ...(model.version ? { version: String(model.version) } : {}) };
  if (TUMOR_TYPES[model.id]) sc.tumor_type = TUMOR_TYPES[model.id];
  if (!onReferenceGrid) sc.grid = 'resampled to 1 mm in the space of the reference sequence';
  return sc;
}

export function sidecarName(niftiName) {
  return niftiName.replace(/\.nii(\.gz)?$/i, '') + '.json';
}
