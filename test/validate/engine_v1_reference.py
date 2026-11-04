#!/usr/bin/env python3
"""Reference for the JavaScript nnU-Net v1 engine: run a small random Generic_UNet through
nnU-Net v1's own preprocessing, sliding-window prediction (Gaussian, mirroring) and export, on a
real anisotropic case, and export the same network for the JavaScript engine.
Usage: engine_v1_reference.py <demo dir> <out dir> [--tta]"""
import sys, os, json, numpy as np, SimpleITK as sitk, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build'))
from export_models import export_onnx, pack_weights
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.preprocessing.preprocessing import GenericPreprocessor
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.postprocessing.connected_components import load_remove_save

import nnunet.preprocessing.preprocessing as _pp
from skimage.transform import resize as _resize

def resize_segmentation_2019(segmentation, new_shape, order=3, cval=0):
    # batchgenerators <= 0.21 behaviour (used by nnU-Net when HD-GLIO was trained): labels are
    # thresholded at 0.5 in order, so the later label wins ties. Current batchgenerators breaks
    # ties by nearest neighbour instead, which differs at a handful of boundary voxels.
    tpe = segmentation.dtype
    if order == 0:
        return _resize(segmentation.astype(float), new_shape, order, mode='edge', clip=True, anti_aliasing=False).astype(tpe)
    reshaped = np.zeros(new_shape, dtype=segmentation.dtype)
    for c in np.unique(segmentation):
        m = _resize((segmentation == c).astype(float), new_shape, order, mode='edge', clip=True, anti_aliasing=False)
        reshaped[m >= 0.5] = c
    return reshaped
_pp.resize_segmentation = resize_segmentation_2019

D, OUT = sys.argv[1], sys.argv[2]
tta = '--tta' in sys.argv
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(3)
# inputs: four sequences on the T1 grid, zero outside a crude head mask (stands in for HD-BET)
t1 = sitk.Cast(sitk.ReadImage(D + 'fg_t1.nii.gz'), sitk.sitkFloat32)
a1 = sitk.GetArrayFromImage(t1)
mask = a1 > np.percentile(a1, 99) * 0.12
files = []
for r, f in [('t1', 'fg_t1'), ('t1c', 'fg_t1c'), ('t2', 'fg_t2'), ('flair', 'fg_flair')]:
    im = sitk.Cast(sitk.ReadImage(D + f + '.nii.gz'), sitk.sitkFloat32)
    im = sitk.Resample(im, t1, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    a = sitk.GetArrayFromImage(im); a[~mask] = 0
    o = sitk.GetImageFromArray(a); o.CopyInformation(t1)
    p = os.path.join(OUT, 'in_%s.nii.gz' % r); sitk.WriteImage(o, p); files.append(p)

net = Generic_UNet(4, 8, 3, 4, 2, 2, nn.Conv3d, nn.InstanceNorm3d, {'eps': 1e-5, 'affine': True}, nn.Dropout3d, {'p': 0, 'inplace': True},
                   nn.LeakyReLU, {'negative_slope': 1e-2, 'inplace': True}, False, False, lambda x: x, InitWeights_He(1e-2),
                   [[2, 2, 2]] * 4, [[3, 3, 3]] * 5, False, True, True)
with torch.no_grad():
    for m in net.modules():
        if isinstance(m, nn.InstanceNorm3d): m.weight.uniform_(0.5, 1.5); m.bias.uniform_(-0.3, 0.3)

    for p in net.parameters(): p.copy_(p.half().float())  # same precision as the fp16 pack
net.eval(); net.inference_apply_nonlin = softmax_helper; net.do_ds = False
patch, target = [64, 64, 64], [1.5, 1.5, 1.5]
pre = GenericPreprocessor({0: 'nonCT', 1: 'nonCT', 2: 'nonCT', 3: 'nonCT'}, {0: True, 1: True, 2: True, 3: True}, [0, 1, 2], None)
data, seg, props = pre.preprocess_test_case(files, target)
print('preprocessed', data.shape, 'from', props['original_size_of_raw_data'], props['original_spacing'])
with torch.no_grad():
    softmax = net.predict_3D(data, do_mirroring=tta, mirror_axes=(0, 1, 2), use_sliding_window=True, step_size=0.5, patch_size=patch,
                             regions_class_order=None, use_gaussian=True, pad_border_mode='constant', pad_kwargs={'constant_values': 0},
                             all_in_gpu=False, verbose=False, mixed_precision=False)[1]
ref = os.path.join(OUT, 'ref_seg.nii.gz')
save_segmentation_nifti_from_softmax(softmax, ref, props, 1, None, None, None, None, None, None, 0)
load_remove_save(ref, os.path.join(OUT, 'ref_seg_pp.nii.gz'), [(1, 2), 2], None)
np.save(os.path.join(OUT, 'ref_pre.npy'), data.astype(np.float32)); np.save(os.path.join(OUT, 'ref_probs.npy'), softmax.astype(np.float32))
export_onnx(net, 4, patch, os.path.join(OUT, 'full.onnx'), softmax=True)
w = pack_weights(os.path.join(OUT, 'full.onnx'), OUT, q8=False)
os.remove(os.path.join(OUT, 'full.onnx'))
cfg = {'name': 'engine-test', 'nnunet': 1, 'inputChannels': ['T1', 'T1c', 'T2', 'FLAIR'], 'numClasses': 3,
       'transposeForward': [0, 1, 2], 'transposeBackward': [0, 1, 2], 'targetSpacing': target, 'patchSize': patch,
       'normalization': [{'scheme': 'nonCT', 'useMask': True}] * 4, 'mirrorAxes': [0, 1, 2], 'stepSize': 0.5, 'aggregate': 'softmax',
       'anisotropyThreshold': 3, 'resampleData': {'order': 3, 'orderZ': 0}, 'resampleProb': {'order': 1, 'orderZ': 0},
       'postprocessing': None, 'onnx': 'model.onnx', 'weights': w, 'tta': tta}
json.dump(cfg, open(os.path.join(OUT, 'config.json'), 'w'), indent=1)
r = sitk.GetArrayFromImage(sitk.ReadImage(ref))
print('reference classes', {int(k): int(v) for k, v in zip(*np.unique(r, return_counts=True))})
