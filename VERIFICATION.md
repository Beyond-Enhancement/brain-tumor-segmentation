# Verification

Tested build: v1.0.0, 2026-11-04. Test machine: Linux container, 2 CPU cores, 7 GB RAM, no GPU,
headless Chrome. Unless stated otherwise the case is TCGA-FG-6692 (the built-in demo: 2D axial
T1, T1c, T2, FLAIR, 0.9 x 0.9 x 5.5 mm).

The executables in `downloads/` were rebuilt from a clean clone of this repository with
`build/build.sh` and came out byte-identical (same SHA-256 as `downloads/SHA256SUMS.txt`). The
Linux binary used for the browser tests is built from the same sources and embedded files in the
same run.

Questions tested:

1. Does the workbench reproduce the original PyTorch HD-GLIO and HD-BET?
2. Is the preprocessing (reorientation, registration, masking) sound?
3. Does it read DICOM correctly, including compressed transfer syntaxes?
4. Does it ever send imaging data to the internet, and does it work with no connection at all?

## 1. Agreement with the original models

### 1a. nnU-Net reimplementation (exact-code test)

`app/js/lib/nnunet.js` reimplements nnU-Net's prediction: crop to the nonzero region (holes
filled), resampling to the target spacing (cubic B-spline, separate-z handling for anisotropic
data), per-channel z-score normalisation, sliding window (step 0.5, Gaussian weighting),
mirroring test-time augmentation, resampling back (linear), argmax and connected-component
postprocessing. It was compared step by step with nnU-Net v1's own functions
(`GenericPreprocessor`, `predict_3D`, `save_segmentation_nifti_from_softmax`,
`load_remove_save`) on the anisotropic demo case, using a random-weight Generic_UNet so that
every class appears (`test/validate/engine_v1_reference.py`, `test/unit/engine_test.mjs`):

| Stage | Result |
|---|---|
| Preprocessed network input | max abs difference 9.5e-7 (float rounding) |
| Class probabilities | max abs difference 7.3e-4, mean 1.4e-5 |
| Segmentation, no augmentation | 56 of 1,490,944 voxels differ (99.996 %), all at near-ties (top-2 margin < 2.5e-4) |
| Segmentation, 8x mirroring | 14 voxels differ (99.999 %) |
| Segmentation, 8x mirroring + connected-component postprocessing | 33 voxels differ, Dice 0.99995 / 1.00000 |

The scikit-image resampling used by nnU-Net is reproduced to 3e-7 (`test/unit/resize_test.mjs`).
One detail matters for exactness: current batchgenerators breaks ties in `resize_segmentation`
by nearest neighbour, while the version HD-GLIO was built with (2019) thresholds labels in
order. The workbench follows the 2019 behaviour; the reference scripts patch it in by default.

### 1b. Real networks, identical inputs

The workbench's network inputs (its *T1/T1c/T2/FLAIR input* exports after HD-BET masking and
registration) were given to the original PyTorch HD-GLIO through nnU-Net v1 `predict_cases`, as
`hd_glio_predict` calls it (fold 0, fp32, CPU; `test/validate/compare_hdglio.py`). HD-BET was
compared with the nnU-Net v2 predictor as the `hd-bet` command runs it, on the same reoriented
T1 (`test/validate/compare_hdbet.py`).

HD-GLIO (no test-time augmentation). Each row compares with PyTorch HD-GLIO run on that row's
own inputs; the int8 build's inputs differ slightly because its HD-BET mask does.

| Weights / runtime | Voxels differing from PyTorch | CE cm3, workbench / PyTorch | NE cm3, workbench / PyTorch | Dice CE / NE / total |
|---|---|---|---|---|
| fp16, browser (CPU) | 7 of 1,490,944 | 7.760 / 7.769 | 13.825 / 13.798 | 0.9994 / 0.9990 / 0.9994 |
| int8, ONNX Runtime native CPU (same inputs as row 1) | 11 | 7.751 / 7.769 | 13.838 / 13.798 | 0.9983 / 0.9982 / 0.9995 |
| **int8, browser (CPU), shipped build, offline** | 18 | 7.734 / 7.751 | 13.918 / 13.838 | 0.9989 / 0.9971 / 0.9986 |

HD-GLIO with test-time augmentation (8 mirrored passes, as published), compared with PyTorch
HD-GLIO also run with augmentation, on the inputs of the shipped build (`compare_hdglio.py --tta
--seg ...`):

| Weights / runtime | Voxels differing from PyTorch | CE cm3, workbench / PyTorch | NE cm3, workbench / PyTorch | Dice CE / NE / total |
|---|---|---|---|---|
| int8, ONNX Runtime native CPU | 13 of 1,490,944 | 7.609 / 7.614 | 13.887 / 13.847 | 0.9985 / 0.9979 / 0.9992 |
| **int8, browser (CPU), shipped build** | 22 | 7.600 / 7.614 | 13.945 / 13.847 | 0.9991 / 0.9965 / 0.9980 |

For scale: switching augmentation on or off changes PyTorch HD-GLIO's own result at 124 voxels
(CE 7.751 vs 7.614 cm3, Dice 0.984), about ten times the difference between the workbench and
PyTorch at either setting.

HD-BET brain mask (PyTorch: 1655.5 cm3, one connected component, about 373,000 voxels):

| Weights / runtime | Voxels differing from PyTorch | Dice |
|---|---|---|
| fp16, ONNX Runtime native CPU | 5 | 0.99999 |
| fp16, ONNX Runtime WebAssembly (as in the browser) | 102 | 0.99986 |
| int8, ONNX Runtime native CPU | 120 | 0.99984 |
| **int8, browser (CPU), shipped build, offline** | 184 | 0.99975 |

The WebAssembly CPU kernels of ONNX Runtime account for about 100 boundary voxels of the brain
mask (0.3 cm3 of 1656 cm3); the native kernels with the same weights match PyTorch to 5 voxels.

**Weight storage.** The fp16 networks (121 MB) would make the executable larger than GitHub's
100 MB file limit, so the shipped build stores weights as int8 per output channel (symmetric
scale) and restores them to fp32 when the network loads; all arithmetic is fp32. The tables
above give the measured effect. `build/export_models.py` without `--q8` makes an fp16 build.

**WebGPU.** The WebGPU backend runs the same graphs. It was checked with the real HD-GLIO and
HD-BET weights on small inputs (software GPU, where full patches would take hours;
`test/validate/webgpu_real_export.py`, `webgpu_real.html`), against ONNX Runtime CPU fp32:

| Network, input | Max abs difference | Argmax agreement |
|---|---|---|
| HD-GLIO, 1 x 4 x 64^3, probabilities | 4.2e-5 | 100.000 % |
| HD-BET, 1 x 1 x 32 x 64 x 64, logits (range +-20) | 6.1e-5 | 99.999 % |

ONNX Runtime reported no operators falling back to the CPU.

### 1c. External plausibility

Not a test of the conversion: HD-GLIO's result on the demo compared with the 5-model BraTS
consensus distributed with the demo data (`fg_refseg`, BraTS Toolkit). That reference sits on
the T1c voxel grid, which in this case is rotated about 8 degrees from the T1 (same header,
different head position), so it was first mapped to T1 space with an independent SimpleITK
registration. Enhancing tumor: consensus 6.12 cm3, HD-GLIO 7.76 cm3, Dice 0.73. Whole
abnormality: 27.96 vs 21.59 cm3, Dice 0.81. For scale, the HD-GLIO model file reports
cross-validation Dice of 0.79 (enhancing) and 0.91 (non-enhancing) on its training data.

## 2. Preprocessing

- **Reorientation** (`fslreorient2std` equivalent) is a lossless permutation/flip of the voxel
  axes; exports undo it exactly, so outputs have the voxel order of the loaded T1.
- **Registration** (6 degrees of freedom, normalised mutual information; coarse rotation and
  translation search at 4 mm, then a 6 / 3 / 1.5 mm pyramid; `test/unit/register_test.mjs`). Each sequence was registered to T1,
  then its header was moved by a known rigid transform (up to 10 degrees and 12 mm) and registered
  again; the recovered transform was compared with the expected one over brain points:

  | Pair | Recovery error, mean (max) over 3 perturbations |
  |---|---|
  | TCGA-FG-6692 T1c to T1 | 0.01-0.04 (0.08) mm |
  | TCGA-FG-6692 T2 to T1 (header offset 14.4 deg, 18 mm) | 0.02-0.03 (0.06) mm |
  | TCGA-FG-6692 FLAIR to T1 (header offset 16 mm) | 0.02-0.03 (0.06) mm |
  | TCGA-DU-7294 T2 to T1 (same 5 mm grid) | 0.00 (0.00) mm |
  | TCGA-DU-7294 FLAIR to T1 | 0.00-0.07 (0.08) mm |
  | TCGA-DU-7294 T1c to T1 (2.5 mm vs 5 mm slices) | 1.0-1.4 (1.8) mm |

  The base solutions' rotation angles agree with an independent SimpleITK Mattes-MI registration
  to within about 1 degree. The demo's T2 and FLAIR headers are offset from the T1 by up to about
  15 degrees and 19 mm; the workbench flags rotations over 10
  degrees or shifts over 20 mm in the *Registration check* panel.

## 3. DICOM

`test/make_dicom.py` writes the demo case as a DICOM study with Siemens-style series names,
shuffled file order, canary patient tags, a 3-slice localiser, and one transfer syntax per
sequence: T1 explicit VR little endian, T1c JPEG 2000 lossless, T2 RLE lossless, FLAIR JPEG-LS
lossless. The workbench ignored the localiser, assigned all four roles correctly, decoded all
four syntaxes, and produced the same volumes as the NIfTI input. None of the exported NIfTI
files or the CSV contains the canary patient name, ID or institution.

`test/make_dicom_variants.py` adds the T1 as JPEG lossless (process 14) and the T1c as a single
enhanced multi-frame MR file with its frames stored in reverse order.
`test/unit/dicom_variants_test.mjs` assembles every series and compares it with the NIfTI it was
written from, after bringing both to the same voxel order:

| Series | Transfer syntax / layout | Voxel difference | Affine difference |
|---|---|---|---|
| T1 | explicit VR little endian | 0 | 0 |
| T1 | JPEG lossless (process 14) | 0 | 0 |
| T1c | JPEG 2000 lossless | 0 | 0 |
| T1c | enhanced multi-frame, frames reversed | 0 | 0 |
| T2 | RLE lossless | 0 | < 1e-17 mm |
| FLAIR | JPEG-LS lossless | 0 | < 1e-17 mm |

## 4. Privacy and offline operation

| Check | Result |
|---|---|
| Code paths that can reach the network | The page and worker fetch only their own files by relative path. ONNX Runtime fetches its `.wasm` from the same origin. No fonts or scripts from elsewhere |
| Content-Security-Policy (set by the launcher) | `default-src 'self'`, `connect-src 'self'`: the browser refuses any request to another origin |
| All requests made during full runs (Playwright, page and worker) | Only `http://127.0.0.1:47821/...`; zero requests elsewhere |
| Launcher | Listens on 127.0.0.1 only; requests with any other Host header get 403 (DNS-rebinding guard); GET/HEAD only |
| Offline (`unshare -rn test/offline_run.sh`: no network interface except loopback, DNS and TCP to the internet fail) | Demo case and DICOM study run end to end with identical results (CE 7.73, NE 13.92, total 21.65, brain 1656 cm3); launcher exits 30 s after the window closes |
| Output files | Fixed names, no DICOM tags; NIfTI description names only the content |

## Run time

Demo case on the 2-core test machine, CPU (WebAssembly), no augmentation: 9 min (HD-BET
5 min 25 s, registration 41 s, HD-GLIO 2 min 56 s). HD-BET runs 8 patches of 96 x 192 x 192 and
HD-GLIO 8 patches of 128^3. Machines with more cores, or with a GPU through WebGPU, should be
considerably faster; that was not measured here. Test-time augmentation multiplies the HD-GLIO
time by 8: the same case with augmentation took 31 min 34 s (HD-GLIO 24 min 51 s).
