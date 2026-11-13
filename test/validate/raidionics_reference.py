#!/usr/bin/env python3
"""Reference run of the original Raidionics code for the workbench's Raidionics add-ons (FLAIR, tumor core).

Runs raidionicsseg (1.5.2, pip package) fit.run_model on one FLAIR image with the runtime settings
that the Raidionics app passes through raidionics_rads_lib (gpu -1, batch 1, overlapping_ratio 0,
reconstruction thresholding / resample_first, no stripped data, folds_ensembling False, no
test-time augmentation), then raidionics_rads_lib's perform_brain_overlap_refinement (copied
below, from raidionicsrads/Processing/brain_processing.py) with the given brain mask.

Usage: raidionics_reference.py <input .nii.gz> <brain mask .nii.gz> <model folder (t2f or t1c)> <out dir>
                               [--onnx FILE]   run this network file instead of <model>/0/model.onnx
Writes <out>/labels_<class>.nii.gz (thresholded, before refinement), <out>/ref_seg.nii.gz
(after refinement), <out>/ref_pre.npy (network input, X,Y,Z) and <out>/ref_prob.npy (sigmoid
output on the resampled grid)."""
import os, sys, shutil, configparser, tempfile
import numpy as np, nibabel as nib
from skimage import measure

flair, mask, model_dir, out = sys.argv[1:5]
onnx_file = sys.argv[sys.argv.index('--onnx') + 1] if '--onnx' in sys.argv else None
os.makedirs(out, exist_ok=True)
work = tempfile.mkdtemp(prefix='raid-', dir=out)
os.makedirs(os.path.join(work, 'inputs')); os.makedirs(os.path.join(work, 'outputs'))
shutil.copyfile(flair, os.path.join(work, 'inputs', 'input0.nii.gz'))

# model folder as raidionicsseg expects it: pre_processing.ini + <fold>/model.onnx
mdir = os.path.join(work, 'model')
os.makedirs(os.path.join(mdir, '0'))
shutil.copyfile(os.path.join(model_dir, 'pre_processing.ini'), os.path.join(mdir, 'pre_processing.ini'))
shutil.copyfile(onnx_file or os.path.join(model_dir, '0', 'model.onnx'), os.path.join(mdir, '0', 'model.onnx'))

cfg = configparser.ConfigParser()
cfg['System'] = {'gpu_id': '-1', 'acceleration': 'cpu', 'inputs_folder': os.path.join(work, 'inputs'),
                 'output_folder': os.path.join(work, 'outputs'), 'model_folder': mdir}
cfg['Runtime'] = {'batch_size': '1', 'overlapping_ratio': '0.0', 'reconstruction_method': 'thresholding',
                  'reconstruction_order': 'resample_first', 'use_preprocessed_data': 'False', 'folds_ensembling': 'False',
                  'ensembling_strategy': 'average', 'test_time_augmentation_iteration': '0',
                  'test_time_augmentation_fusion_mode': 'average'}
cfg['Neuro'] = {'brain_segmentation_filename': mask}
cfg_file = os.path.join(work, 'seg_config.ini')
with open(cfg_file, 'w') as f:
    cfg.write(f)

import raidionicsseg.fit as fit
_pre, _pred = fit.prepare_pre_processing, fit.run_predictions
def pre(*a, **k):
    r = _pre(*a, **k)
    np.save(os.path.join(out, 'ref_pre.npy'), r[3][0, ..., 0])
    return r
def pred(*a, **k):
    r = _pred(*a, **k)
    np.save(os.path.join(out, 'ref_prob.npy'), r[..., 0])
    return r
fit.prepare_pre_processing, fit.run_predictions = pre, pred
fit.run_model(cfg_file)
import glob
found = glob.glob(os.path.join(work, 'outputs', 'labels_*.nii.gz'))
assert len(found) == 1, found
labels = found[0]
labels_out = os.path.join(out, os.path.basename(labels))
shutil.copyfile(labels, labels_out)


def perform_brain_overlap_refinement(predictions_filepath, brain_mask_filepath):
    # raidionics_rads_lib 1.3.3, raidionicsrads/Processing/brain_processing.py
    pred_nib = nib.load(predictions_filepath)
    brain_mask_nib = nib.load(brain_mask_filepath)
    pred = pred_nib.get_fdata()[:]
    brain_mask = brain_mask_nib.get_fdata()[:].astype('uint8')
    pred_binary = np.zeros(pred.shape, dtype='uint8')
    pred_binary[pred > 1e-3] = 1
    cc_pred_bin = measure.label(pred_binary)
    obj_labels = np.unique(cc_pred_bin)[1:]
    final_pred = np.zeros(pred.shape, dtype='float32')
    for l in range(0, len(obj_labels)):
        obj_pred = np.zeros(pred_binary.shape, dtype='uint8')
        obj_pred[cc_pred_bin == (l + 1)] = 1
        overlap = np.count_nonzero(obj_pred & brain_mask) > 0
        if overlap:
            label_pred = np.where(cc_pred_bin == (l + 1), pred, 0).astype("float32")
            final_pred = final_pred + label_pred
    final_pred_nib = nib.Nifti1Image(final_pred, affine=pred_nib.affine, header=pred_nib.header)
    nib.save(final_pred_nib, predictions_filepath)


refined = os.path.join(out, 'ref_seg.nii.gz')
shutil.copyfile(labels, refined)
perform_brain_overlap_refinement(refined, mask)
shutil.rmtree(work)
a = nib.load(labels_out).get_fdata() > 0
b = nib.load(refined).get_fdata() > 0
vv = float(np.prod(nib.load(refined).header.get_zooms()[:3])) / 1000
print('raidionicsseg: %d voxels (%.3f cm3) thresholded, %d (%.3f cm3) after brain overlap refinement' % (a.sum(), a.sum() * vv, b.sum(), b.sum() * vv))
