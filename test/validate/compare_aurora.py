#!/usr/bin/env python3
"""Compare the workbench's AURORA add-on with the original brainles_aurora (0.2.7, MONAI, PyTorch,
fp32 weights, CPU) on identical inputs: the images on AURORA's internal 1 mm grid that
test/unit/aurora_test.mjs wrote (aurora_grid_<role>.nii.gz), given to AuroraInferer as files.

Usage: compare_aurora.py <aurora_test.mjs out dir> <weights folder (holds <combination>_best.tar)> [--tta] [--noise]
  (no option)  AURORA with tta=False; compare with aurora_test.mjs run without --tta
  --tta        AURORA's test-time augmentation with its Gaussian noise left out (std 0), which is the
               workbench's 3 weighted passes; compare with aurora_test.mjs --tta
  --tta --noise  AURORA's test-time augmentation as published (random noise, std 0.001)
The weights are read from the given folder; nothing is downloaded. Compares the sigmoid
probabilities and the labels on the internal grid, and the labels on the input grid
(js_seg.nii.gz) with AURORA's mapped back by voxel (the grids share voxel positions when the
internal grid is placed on the input grid's voxels, js_meta.json 'snapped')."""
import os, sys, json, numpy as np, nibabel as nib, torch
D, W = sys.argv[1], sys.argv[2]
tta, noise = '--tta' in sys.argv, '--noise' in sys.argv
meta = json.load(open(os.path.join(D, 'js_meta.json')))
N = tuple(meta['dims'])
if bool(meta['tta']) != tta:
    sys.exit('compare_aurora: aurora_test.mjs was run with tta=%s' % meta['tta'])
import brainles_aurora.inferer.model as M
from brainles_aurora.inferer import AuroraInferer, AuroraInfererConfig
from brainles_aurora.inferer.constants import Device
M.check_model_weights = lambda: W  # local weights only
captured = {}
_post = M.ModelHandler._post_process
def post(self, onehot_model_outputs_CHWD):
    captured['logits'] = onehot_model_outputs_CHWD[0].detach().cpu().numpy().astype(np.float32)
    return _post(self, onehot_model_outputs_CHWD)
M.ModelHandler._post_process = post
if tta and not noise:
    M.RandGaussianNoised = lambda keys, prob, std: (lambda d: d)
torch.set_num_threads(os.cpu_count())
files = {r: os.path.join(D, 'aurora_grid_%s.nii.gz' % r) for r in ('t1', 't1c', 't2', 'flair')}
kw = {k: files[r] for k, r in (('t1', 't1'), ('t1c', 't1c'), ('t2', 't2'), ('fla', 'flair')) if os.path.isfile(files[r])}
inf = AuroraInferer(AuroraInfererConfig(tta=tta, device=Device.CPU, log_level=40))
out = inf.infer(**kw, log_file=os.path.join(D, 'aurora_reference.log'))
ref_seg = np.asarray(out['segmentation'])
ref_prob = 1 / (1 + np.exp(-captured['logits'].astype(np.float64)))
js_prob = np.fromfile(os.path.join(D, 'js_prob.raw'), dtype='<f4').reshape((2,) + N)
js_seg = np.zeros(N, np.uint8); js_seg[js_prob[0] >= 0.5] = 1; js_seg[js_prob[1] >= 0.5] = 2
dice = lambda a, b: round(float(2 * (a & b).sum() / max(1, a.sum() + b.sum())), 5)
res = {'inputs': sorted(kw), 'tta': tta, 'noise': noise, 'grid': list(N), 'snapped': meta['snapped'],
       'max_abs_prob_diff': [round(float(np.abs(js_prob[k] - ref_prob[k]).max()), 6) for k in range(2)],
       'internal_grid': {'differing_voxels': int((js_seg != ref_seg).sum()), 'of': int(ref_seg.size)}}
for name, labs in (('metastasis (2)', [2]), ('edema (1)', [1]), ('whole (1+2)', [1, 2])):
    a, b = np.isin(js_seg, labs), np.isin(ref_seg, labs)
    res['internal_grid'][name] = {'workbench_cm3': round(float(a.sum()) / 1000, 3), 'aurora_cm3': round(float(b.sum()) / 1000, 3), 'dice': dice(a, b)}
# labels on the input grid: AURORA's internal-grid labels taken at each input voxel's position
seg_img = nib.load(os.path.join(D, 'js_seg.nii.gz'))
js_run = np.asarray(seg_img.dataobj)
G = np.array(meta['affine']).reshape(4, 4)
ijk = np.indices(js_run.shape).reshape(3, -1)
q = (np.linalg.inv(G) @ seg_img.affine @ np.vstack([ijk, np.ones((1, ijk.shape[1]))]))[:3]
qi = np.rint(q).astype(int)
inside = np.all((qi >= 0) & (qi < np.array(N)[:, None]), 0)
ref_run = np.zeros(js_run.size, np.uint8)
ref_run[inside] = ref_seg[qi[0, inside], qi[1, inside], qi[2, inside]]
ref_run = ref_run.reshape(js_run.shape)
if meta['snapped']:
    assert np.abs(q - qi).max() < 1e-4
vv = float(abs(np.linalg.det(seg_img.affine[:3, :3]))) / 1000
res['input_grid'] = {'differing_voxels': int((js_run != ref_run).sum()), 'of': int(js_run.size),
                     'note': 'exact voxel correspondence' if meta['snapped'] else 'nearest internal voxel (the workbench interpolates linearly)'}
for name, labs in (('metastasis (2)', [2]), ('edema (1)', [1]), ('whole (1+2)', [1, 2])):
    a, b = np.isin(js_run, labs), np.isin(ref_run, labs)
    res['input_grid'][name] = {'workbench_cm3': round(float(a.sum() * vv), 3), 'aurora_cm3': round(float(b.sum() * vv), 3), 'dice': dice(a, b)}
tag = ('_tta' + ('_noise' if noise else '')) if tta else ''
nib.save(nib.Nifti1Image(ref_run, seg_img.affine), os.path.join(D, 'aurora_reference_seg%s.nii.gz' % tag))
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(D, 'compare_aurora%s.json' % tag), 'w'), indent=1)
