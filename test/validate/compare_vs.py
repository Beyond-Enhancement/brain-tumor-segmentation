#!/usr/bin/env python3
"""Compare the workbench's vestibular schwannoma add-on (vs-t1c or vs-t2) with the original nnU-Net v2
predictor (nnunetv2, PyTorch, fp32 checkpoint, CPU) on the same input.

Usage: compare_vs.py <nnUNetTrainer__nnUNetPlans__3d_fullres folder> <workbench input nii.gz> <workbench segmentation nii.gz>
                     [--tta] [--as-stored]
The input is the workbench's export of the image it gave the model (vs-t1c_input_t1c.nii.gz or
vs-t2_input_t2.nii.gz). It is reoriented as the workbench does (fslreorient2std: LAS), predicted by
nnUNetPredictor with the network of fold 0 (--tta: with nnU-Net's mirroring, as the workbench's
test-time augmentation; else without), without nnU-Net's postprocessing (the model record says to
omit it, and the workbench does), and the result is taken back to the input's voxel order and
compared voxel by voxel.
--as-stored also predicts the image in its stored voxel order (what nnUNetv2_predict does with
the file), to show the effect of the reorientation."""
import os, sys, json, tempfile, numpy as np, nibabel as nib, torch


def main():
    FOLDER, INP, SEG = sys.argv[1:4]
    tta, as_stored = '--tta' in sys.argv, '--as-stored' in sys.argv
    os.environ['nnUNet_compile'] = 'F'
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    torch.set_num_threads(os.cpu_count())
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=tta, perform_everything_on_device=False, device=torch.device('cpu'), verbose=False)
    pred.initialize_from_trained_model_folder(FOLDER, (0,), 'checkpoint_final.pth')
    im = nib.load(INP)
    tmp = tempfile.mkdtemp()

    def predict(img, tag):
        src, out = os.path.join(tmp, tag + '_0000.nii.gz'), os.path.join(tmp, tag + '.nii.gz')
        nib.save(nib.Nifti1Image(np.asarray(img.dataobj, dtype=np.float32), img.affine), src)
        pred.predict_from_files([[src]], [out], save_probabilities=False, overwrite=True, num_processes_preprocessing=1, num_processes_segmentation_export=1)
        seg = nib.load(out)
        return nib.Nifti1Image(np.asarray(seg.dataobj).astype(np.uint8), seg.affine)

    to_las = nib.orientations.ornt_transform(nib.orientations.io_orientation(im.affine), nib.orientations.axcodes2ornt(('L', 'A', 'S')))
    back = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(('L', 'A', 'S')), nib.orientations.io_orientation(im.affine))
    ref = np.asarray(predict(im.as_reoriented(to_las), 'las').as_reoriented(back).dataobj) > 0
    js = np.asarray(nib.load(SEG).dataobj) > 0
    vv = float(abs(np.linalg.det(im.affine[:3, :3]))) / 1000
    d = lambda a, b: round(float(2 * (a & b).sum() / max(1, a.sum() + b.sum())), 5)
    res = {'tta': tta, 'pytorch_cm3': round(float(ref.sum() * vv), 3), 'workbench_cm3': round(float(js.sum() * vv), 3),
           'dice': d(js, ref), 'differing_voxels': int((js != ref).sum()), 'of': int(js.size)}
    tag = '_tta' if tta else ''
    nib.save(nib.Nifti1Image(ref.astype(np.uint8), im.affine), os.path.join(os.path.dirname(SEG), 'pytorch_vs%s.nii.gz' % tag))
    if as_stored:
        st = np.asarray(predict(im, 'stored').dataobj) > 0
        res['as_stored'] = {'pytorch_cm3': round(float(st.sum() * vv), 3), 'dice_vs_workbench': d(js, st), 'dice_vs_reoriented': d(ref, st)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(os.path.dirname(SEG), 'compare_vs%s.json' % tag), 'w'), indent=1)


if __name__ == '__main__':
    main()
