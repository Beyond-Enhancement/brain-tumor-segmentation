#!/usr/bin/env python3
"""Compare the workbench's nnU-Net BraTS 2020 add-on with the original PyTorch model on identical
inputs.

Runs nnU-Net v1 predict_cases (the same fold, fp32, CPU) on the inputs the workbench gave the
network (its "T1/T1c/T2/FLAIR input" exports, brats2020_input_*.nii.gz), converts the labels to
the BraTS convention as nnU-Net's convert_labels_back_to_BraTS does, and compares per label and
per BraTS region (ET, TC, WT).
Usage: compare_brats.py <trained model folder> <dir with workbench exports> [--fold 0] [--tta] [--seg FILE]
  --tta       run PyTorch with 8x mirroring; give the workbench's TTA output with --seg
  --seg FILE  workbench segmentation (default <dir>/brats2020_segmentation.nii.gz)
Uses the 2019 batchgenerators resize_segmentation behaviour, as engine_v1_reference.py explains."""
import os, sys, json, numpy as np, SimpleITK as sitk, torch
P, D = sys.argv[1], sys.argv[2]
tta = '--tta' in sys.argv
fold = int(sys.argv[sys.argv.index('--fold') + 1]) if '--fold' in sys.argv else 0
seg_file = sys.argv[sys.argv.index('--seg') + 1] if '--seg' in sys.argv else os.path.join(D, 'brats2020_segmentation.nii.gz')
if tta and '--seg' not in sys.argv:
    sys.exit('compare_brats: with --tta give the workbench TTA segmentation with --seg')
import numpy.core.multiarray as ma
torch.serialization.add_safe_globals([(ma.scalar, 'numpy.core.multiarray.scalar'), np.dtype] + [type(np.dtype(t)) for t in ('float64', 'float32', 'int64', 'int32', 'bool', 'uint8')])
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
from nnunet.dataset_conversion.Task032_BraTS_2018 import convert_labels_back_to_BraTS

ins = [os.path.join(D, 'brats2020_input_%s.nii.gz' % r) for r in ('t1', 't1c', 't2', 'flair')]
out = os.path.join(D, 'pytorch_brats2020%s.nii.gz' % ('_tta' if tta else ''))
torch.set_num_threads(os.cpu_count())
predict_cases(P, [ins], [out], (fold,), False, 1, 1, None, tta, False, True, checkpoint_name='model_final_checkpoint')
ref_img = sitk.ReadImage(out)
ref = convert_labels_back_to_BraTS(sitk.GetArrayFromImage(ref_img))
js = sitk.GetArrayFromImage(sitk.ReadImage(seg_file))
vv = float(np.prod(ref_img.GetSpacing())) / 1000.0
res = {'tta': tta, 'fold': fold, 'workbench_file': os.path.basename(seg_file), 'voxel_agreement_pct': float((ref == js).mean() * 100),
       'differing_voxels': int((ref != js).sum()), 'of': int(ref.size)}
for name, labs in (('ET (4)', [4]), ('NCR (1)', [1]), ('ED (2)', [2]), ('TC (1+4)', [1, 4]), ('WT (1+2+4)', [1, 2, 4])):
    a, b = np.isin(js, labs), np.isin(ref, labs)
    res[name] = {'workbench_cm3': round(float(a.sum() * vv), 3), 'pytorch_cm3': round(float(b.sum() * vv), 3),
                 'dice': round(float(2 * (a & b).sum() / max(1, a.sum() + b.sum())), 5)}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(D, 'compare_brats2020%s.json' % ('_tta' if tta else '')), 'w'), indent=1)
