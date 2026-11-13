#!/usr/bin/env python3
"""End-to-end check of an AURORA add-on run: the original brainles_aurora (0.2.7, fp32 weights, CPU)
on the files as given, against the workbench's segmentation of the same files.

Usage: aurora_end_to_end.py <workbench aurora-*_segmentation.nii.gz> <weights folder (holds <combination>_best.tar)>
                            --t1c FILE [--fla FILE] [--tta [--noise]]
The files must be on one grid, the grid the workbench's segmentation is on (for example four
co-registered BraTS-style files, with the workbench's reference among them). AURORA runs on them
as they are: no brain extraction, no resampling. --tta without --noise leaves out AURORA's Gaussian
noise, as the workbench does. Nothing is downloaded."""
import os, sys, json, numpy as np, nibabel as nib, torch
SEG, W = sys.argv[1], sys.argv[2]
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
tta, noise = '--tta' in sys.argv, '--noise' in sys.argv
import brainles_aurora.inferer.model as M
from brainles_aurora.inferer import AuroraInferer, AuroraInfererConfig
from brainles_aurora.inferer.constants import Device
M.check_model_weights = lambda: W
if tta and not noise:
    M.RandGaussianNoised = lambda keys, prob, std: (lambda d: d)
torch.set_num_threads(os.cpu_count())
kw = {k: arg('--' + k) for k in ('t1c', 'fla') if arg('--' + k)}
seg_img = nib.load(SEG)
for f in kw.values():
    im = nib.load(f)
    assert im.shape[:3] == seg_img.shape and np.allclose(im.affine, seg_img.affine, atol=1e-3), (f, im.shape, seg_img.shape)
out = AuroraInferer(AuroraInfererConfig(tta=tta, device=Device.CPU, log_level=40)).infer(**kw, log_file=os.path.join(os.path.dirname(SEG), 'aurora_end_to_end.log'))
ref = np.asarray(out['segmentation']).astype(np.uint8)
js = np.asarray(seg_img.dataobj).astype(np.uint8)
vv = float(abs(np.linalg.det(seg_img.affine[:3, :3]))) / 1000
dice = lambda a, b: round(float(2 * (a & b).sum() / max(1, a.sum() + b.sum())), 5)
res = {'inputs': sorted(kw), 'tta': tta, 'noise': noise, 'differing_voxels': int((js != ref).sum()), 'of': int(js.size)}
for name, labs in (('metastasis (2)', [2]), ('edema (1)', [1]), ('whole (1+2)', [1, 2])):
    a, b = np.isin(js, labs), np.isin(ref, labs)
    res[name] = {'workbench_cm3': round(float(a.sum() * vv), 3), 'aurora_cm3': round(float(b.sum() * vv), 3), 'dice': dice(a, b)}
tag = '_'.join(sorted(kw)) + ('_tta' + ('_noise' if noise else '') if tta else '')
nib.save(nib.Nifti1Image(ref, seg_img.affine), os.path.join(os.path.dirname(SEG), 'aurora_original_%s.nii.gz' % tag))
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(os.path.dirname(SEG), 'aurora_end_to_end_%s.json' % tag), 'w'), indent=1)
