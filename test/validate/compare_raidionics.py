#!/usr/bin/env python3
"""Compare the workbench's Raidionics FLAIR result with raidionicsseg's (raidionics_reference.py).
Usage: compare_raidionics.py <reference out dir> <workbench segmentation .nii.gz> [<js debug dir>]
The optional debug dir (from test/unit/raidionics_test.mjs) adds the network input and the
probability maps to the comparison."""
import os, sys, json, numpy as np, nibabel as nib
R, seg_file = sys.argv[1], sys.argv[2]
J = sys.argv[3] if len(sys.argv) > 3 else None
ref_img = nib.load(os.path.join(R, 'ref_seg.nii.gz'))
ref = ref_img.get_fdata() > 0
js_img = nib.load(seg_file)
js = js_img.get_fdata() > 0
assert ref.shape == js.shape, (ref.shape, js.shape)
assert np.allclose(ref_img.affine, js_img.affine, atol=1e-3), (ref_img.affine, js_img.affine)
vv = float(abs(np.linalg.det(ref_img.affine[:3, :3]))) / 1000
res = {'workbench_file': os.path.basename(seg_file), 'grid': list(ref.shape),
       'raidionicsseg_cm3': round(float(ref.sum() * vv), 3), 'workbench_cm3': round(float(js.sum() * vv), 3),
       'dice': round(float(2 * (ref & js).sum() / max(1, ref.sum() + js.sum())), 5),
       'differing_voxels': int((ref != js).sum()), 'of': int(ref.size)}
if J:
    meta = json.load(open(os.path.join(J, 'js_meta.json')))
    N = tuple(meta['N'])
    pre_r = np.load(os.path.join(R, 'ref_pre.npy')); prob_r = np.load(os.path.join(R, 'ref_prob.npy'))
    pre_j = np.fromfile(os.path.join(J, 'js_pre.raw'), dtype='<f4').reshape(N)
    prob_j = np.fromfile(os.path.join(J, 'js_prob.raw'), dtype='<f4').reshape(N)
    res['resampled_grid'] = {'raidionicsseg': list(pre_r.shape), 'workbench': list(N)}
    if pre_r.shape == N:
        d = np.abs(pre_r.astype(np.float64) - pre_j)
        res['network_input'] = {'max_abs_diff': float(d.max()), 'mean_abs_diff': float(d.mean()), 'identical_pct': float((d == 0).mean() * 100),
                                'value_range': [float(pre_r.min()), float(pre_r.max())]}
        dp = np.abs(prob_r.astype(np.float64) - prob_j)
        res['probabilities'] = {'max_abs_diff': float(dp.max()), 'mean_abs_diff': float(dp.mean()),
                                'voxels_crossing_0.5': int(((prob_r >= 0.5) != (prob_j >= 0.5)).sum())}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(seg_file)), 'compare_raidionics.json'), 'w'), indent=1)
