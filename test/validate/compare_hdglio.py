#!/usr/bin/env python3
"""Compare the workbench's HD-GLIO output with the original PyTorch HD-GLIO on identical inputs.

Runs nnU-Net v1 predict_cases exactly as hd_glio_predict does (fold 0, model_final_checkpoint,
fp32 on CPU) on the inputs the workbench gave the network (its "T1/T1c/T2/FLAIR input" exports),
and compares with the workbench segmentation.
Usage: compare_hdglio.py <params folder> <dir with workbench exports> [--tta] [--seg FILE] [--bg-current]
  --tta         run PyTorch HD-GLIO with 8x mirroring; the workbench output must then also be a
                TTA run: give it with --seg (refuses to compare against the default no-TTA export)
  --seg FILE    workbench segmentation to compare (default: <dir>/hdglio_segmentation.nii.gz)
  --bg-current  use the installed batchgenerators resize_segmentation (default: the 2019
                behaviour HD-GLIO was built with; see engine_v1_reference.py)"""
import os, sys, json, tempfile, shutil, numpy as np, SimpleITK as sitk, torch
P, D = sys.argv[1], sys.argv[2]
tta = '--tta' in sys.argv
seg_file = sys.argv[sys.argv.index('--seg') + 1] if '--seg' in sys.argv else os.path.join(D, 'hdglio_segmentation.nii.gz')
summ = os.path.join(D, 'summary.json')
if os.path.isfile(summ) and os.path.dirname(os.path.abspath(seg_file)) == os.path.abspath(D):
    run_tta = 'Augmentation\n8 mirrored' in json.load(open(summ)).get('info', '')
    if run_tta != tta:
        sys.exit('compare_hdglio: the workbench run in %s was made %s test-time augmentation; match --tta' % (D, 'with' if run_tta else 'without'))
import numpy.core.multiarray as ma
torch.serialization.add_safe_globals([(ma.scalar, 'numpy.core.multiarray.scalar'), np.dtype] + [type(np.dtype(t)) for t in ('float64', 'float32', 'int64', 'int32', 'bool', 'uint8')])
if '--bg-current' not in sys.argv:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import nnunet.preprocessing.preprocessing as _pp
    from skimage.transform import resize as _resize
    def resize_segmentation_2019(segmentation, new_shape, order=3, cval=0):
        tpe = segmentation.dtype
        if order == 0:
            return _resize(segmentation.astype(float), new_shape, order, mode='edge', clip=True, anti_aliasing=False).astype(tpe)
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)
        for c in np.unique(segmentation):
            m = _resize((segmentation == c).astype(float), new_shape, order, mode='edge', clip=True, anti_aliasing=False)
            reshaped[m >= 0.5] = c
        return reshaped
    _pp.resize_segmentation = resize_segmentation_2019
from nnunet.inference.predict import predict_cases
ins = [os.path.join(D, 'hdglio_input_%s.nii.gz' % r) for r in ('t1', 't1c', 't2', 'flair')]
out = os.path.join(D, 'pytorch_hdglio%s.nii.gz' % ('_tta' if tta else ''))
torch.set_num_threads(os.cpu_count())
predict_cases(P, [ins], [out], (0,), False, 1, 1, None, tta, False, True, checkpoint_name='model_final_checkpoint')
ref_img = sitk.ReadImage(out)
ref = sitk.GetArrayFromImage(ref_img)
js = sitk.GetArrayFromImage(sitk.ReadImage(seg_file))
vv = float(np.prod(ref_img.GetSpacing())) / 1000.0
res = {'tta': tta, 'workbench_file': os.path.basename(seg_file), 'voxel_agreement_pct': float((ref == js).mean() * 100), 'differing_voxels': int((ref != js).sum())}
for name, labs in (('CE (2)', [2]), ('NE (1)', [1]), ('total (1+2)', [1, 2])):
    a, b = np.isin(js, labs), np.isin(ref, labs)
    res[name] = {'workbench_cm3': round(float(a.sum() * vv), 3), 'pytorch_cm3': round(float(b.sum() * vv), 3),
                 'dice': round(float(2 * (a & b).sum() / max(1, a.sum() + b.sum())), 5)}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(D, 'compare_hdglio%s.json' % ('_tta' if tta else '')), 'w'), indent=1)
