#!/usr/bin/env python3
"""Compare the workbench's HD-BET brain mask with the original PyTorch HD-BET (nnU-Net v2
predictor, as the hd-bet command runs it) on the same reference image, reoriented like the
workbench does (fslreorient2std: LAS).
Usage: compare_hdbet.py <hd-bet params folder> <reference T1 nii.gz> <workbench brain mask nii.gz> [--tta]"""
import os, sys, json, tempfile, numpy as np, nibabel as nib, torch
from scipy.ndimage import label
def main():
    P, T1, MASK = sys.argv[1], sys.argv[2], sys.argv[3]
    tta = '--tta' in sys.argv
    os.environ['nnUNet_compile'] = 'F'
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    im = nib.load(T1)
    ornt = nib.orientations.ornt_transform(nib.orientations.io_orientation(im.affine), nib.orientations.axcodes2ornt(('L', 'A', 'S')))
    reo = im.as_reoriented(ornt)
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, 'ref.nii.gz'); nib.save(nib.Nifti1Image(np.asarray(reo.dataobj, dtype=np.float32), reo.affine), src)
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=tta, perform_everything_on_device=False, device=torch.device('cpu'), verbose=False)
    pred.initialize_from_trained_model_folder(P, 'all')
    torch.set_num_threads(os.cpu_count())
    out = os.path.join(tmp, 'mask.nii.gz')
    pred.predict_from_files([[src]], [out], save_probabilities=False, overwrite=True, num_processes_preprocessing=1, num_processes_segmentation_export=1)
    ref_r = nib.load(out)
    import shutil; shutil.copy(out, os.path.join(os.path.dirname(MASK), 'pytorch_hdbet%s.nii.gz' % ('_tta' if tta else '')))
    ref = np.asarray(ref_r.as_reoriented(nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(('L', 'A', 'S')), nib.orientations.io_orientation(im.affine))).dataobj) > 0
    lab, n = label(ref); ref_lcc = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1) if n else ref
    js = np.asarray(nib.load(MASK).dataobj) > 0
    vv = float(np.prod(im.header.get_zooms()[:3])) / 1000
    d = lambda a, b: round(float(2 * (a & b).sum() / max(1, a.sum() + b.sum())), 5)
    res = {'tta': tta, 'pytorch_cm3': round(float(ref.sum() * vv), 1), 'pytorch_lcc_cm3': round(float(ref_lcc.sum() * vv), 1), 'workbench_cm3': round(float(js.sum() * vv), 1),
           'dice_vs_pytorch': d(js, ref), 'dice_vs_pytorch_lcc': d(js, ref_lcc), 'differing_voxels_vs_lcc': int((js != ref_lcc).sum())}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(os.path.dirname(MASK), 'compare_hdbet%s.json' % ('_tta' if tta else '')), 'w'), indent=1)


if __name__ == '__main__':
    main()
