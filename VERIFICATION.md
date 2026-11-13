# Verification

Tested build: v1.0.0, 2026-11-04; section 5 covers what v1.1.0 adds (also 2026-11-04). v1.1.1
changes only the page markup (the wrapping of two option labels); the processing code, networks
and add-on file are unchanged, and the demo case gives the same result (checked when taking the
README screenshot). Section 6 covers v1.2.0 (2026-11-09): the Raidionics FLAIR add-on and
runs with fewer than four sequences; HD-GLIO and the BraTS add-on were re-run and are unchanged.
Section 7 covers v1.3.0 (2026-11-09): several models in one run, and their comparison.
Section 8 covers v1.4.0 (2026-11-13): the Raidionics tumor core add-on.
Section 9 covers v1.5.0 (2026-11-13): the AURORA brain metastasis and vestibular schwannoma
add-ons, and runs without brain extraction when only whole-head models are checked.
Test machine: Linux container, 2 CPU cores, 7 GB RAM, no GPU,
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

## 5. Version 1.1.0 additions

**nnU-Net BraTS 2020 add-on.** One network (fold 0) of `nnUNetTrainerV2BraTSRegions_DA4_BN_BD`
from Isensee et al.'s BraTS 2020 entry (Zenodo 4635763, CC BY 4.0), exported with 16-bit weights.
It runs through the same JavaScript nnU-Net as HD-GLIO, plus nnU-Net's region-based output
handling (sigmoid regions WT, TC, ET written in `regions_class_order`) and the conversion to BraTS
labels (`convert_labels_back_to_BraTS`). Compared with the PyTorch network run by nnU-Net v1
`predict_cases` (fold 0, fp32, CPU) on the inputs the workbench gave it, demo case, no
augmentation (`test/validate/compare_brats.py`):

| Compartment | Workbench cm3 | PyTorch cm3 | Dice |
|---|---|---|---|
| ET (4) | 6.997 | 6.997 | 1.00000 |
| NCR (1) | 1.709 | 1.709 | 1.00000 |
| ED (2) | 24.671 | 24.666 | 0.99991 |
| TC (1 + 4) | 8.706 | 8.706 | 1.00000 |
| WT (1 + 2 + 4) | 33.376 | 33.372 | 0.99993 |

One voxel of 1,490,944 differs. The augmentation (mirroring) code is shared with HD-GLIO and was
compared there (section 1b); it was not run separately for this model.

**Add-on files.** The launcher lists `*.btsmodel` files in its own folder and serves only those,
by exact name (a path-traversal request and an unlisted file name both return 404). The page
checks the format and the SHA-256 of the weights before offering a model; a copy with one byte
changed was rejected ("the file is damaged (checksum does not match)").

**Reference sequence.** TCGA-DU-7294, files named as at the requesting institution
(`17_AX_T1_PRE`, `20_SAG_MPR`, `15_AX_T2`, `8_SAG_3D_FLAIR`), T1 and T2 with 5 mm slices, T1c and
FLAIR with 2.5 mm slices. `20_SAG_MPR` (no recognisable sequence name) was assigned to T1c by
elimination and flagged. Registered to T1c and to FLAIR, the segmentation was written on that
sequence's 2.5 mm grid (header identical to the input's):

| Reference | Non-enhancing abnormality cm3 | Enhancing cm3 | Brain mask cm3 |
|---|---|---|---|
| T1c | 21.79 | 0.00 | 1246 |
| FLAIR | 21.83 | 0.00 | 1318 |

DU-7294 is a non-enhancing glioma. The brain-mask volume depends on the sequence HD-BET runs on.
`test/unit/roles_test.mjs` checks the assignment on several naming schemes; with two
unrecognised files nothing is assigned by elimination.

**GPU failure during a run.** `test/gpu_loss_test.mjs` starts a GPU (WebGPU) run of the demo case
and, while HD-BET runs on the GPU, kills the browser's GPU process, as a Windows driver reset
(TDR) would. ONNX Runtime then failed with the error reported from the field (`failed to call
OrtRun() ... webgpu/buffer_manager.cc:669`). HD-BET restarted on the CPU and the run finished
there; the volumes (enhancing 7.73, non-enhancing 13.92, brain 1656 cm3) and the segmentation file
are identical to v1.0.0's CPU result. The viewer, whose WebGL context the GPU reset destroys, was
rebuilt and drew the result; the page noted the fallback and set *Automatic* to use the CPU on
that computer from then on.

**Memory.** A run peaks at about 3 GB in the browser process. Two runs at once on the 7 GB test
machine exceeded its memory limit and one browser tab was stopped by the system. In version 1.2.0, a
FLAIR-only Raidionics run of DU-7294 peaked at 4.1 GB summed over all browser processes
(resident memory sampled every 2 s), and a browser run alongside a raidionicsseg reference run
(about 2 GB) on the same machine was again stopped by the system.

## 6. Version 1.2.0: Raidionics FLAIR hyperintensity add-on

**Model.** `MRI_FLAIRChanges`, folder `t2f`, network of fold 0, from
`Raidionics-MRI_FLAIRChanges-v13.zip` (Raidionics-models release v1.3.0-rc; md5
ec6209b06af8cb226bff50d8890eed24, which matches the release's Slicer model list; the Raidionics
app's list still carries an older checksum for the same URL). BSD 2-Clause. One input (FLAIR),
one sigmoid output (FLAIR hyperintensity). `build/export_models.py raidionics-flair` reads the
model's `pre_processing.ini` and stops unless it is exactly the configuration the JavaScript
implements (1 mm spacing, `minimum` background crop, clipping at the 0 and 99.5th percentiles,
`zeromean_nonzero`, 160^3 patches without offset, sigmoid not in the graph, channels first).

**Reference.** raidionicsseg 1.5.2 (`fit.run_model`) with the runtime settings the Raidionics app
sets through raidionics_rads_lib 1.3.3 (thresholding, `resample_first`, overlapping ratio 0, no
test-time augmentation, first fold only), followed by raidionics_rads_lib's
`perform_brain_overlap_refinement`, copied into `test/validate/raidionics_reference.py`. Both
sides get the same FLAIR file and the same brain mask (the HD-BET mask the workbench exported).
Raidionics itself would supply the mask from its own brain model; the mask only decides which
connected regions are kept (those touching the brain), and none was removed in these cases.

**Implementation, original network (Node, `test/unit/raidionics_test.mjs` with
onnxruntime-node and the original fp32 `model.onnx`).** `test/validate/compare_raidionics.py`
also compares the intermediate arrays:

| Case | FLAIR grid | Resampled grid (both) | Network input max abs diff | Probability max abs diff | Differing voxels | Volume cm3 |
|---|---|---|---|---|---|---|
| TCGA-DU-7294 (axial, 0.94 x 0.94 x 2.5 mm) | 256 x 256 x 68 | 156 x 204 x 156, 2 patches | 4.8e-7 | 1.7e-6 | 0 | 30.270 / 30.270 |
| TCGA-FG-6692 (oblique, 0.90 x 0.90 x 5.5 mm) | 208 x 256 x 26 | 177 x 237 x 200, 8 patches | 4.8e-7 | 2.9e-6 | 0 | 24.089 / 24.089 |

The resampling reproduces scipy's `affine_transform` arithmetic (coordinates accumulated in
scipy's order, which was confirmed on 15,635 boundary cases, and its neighbour order and
weights); the remaining input difference is the mean and standard deviation, which numpy
computes in float32 and the workbench in float64. The brain overlap rule was compared with the
copied raidionics_rads_lib function on random blobs (686 components, 1,743 voxels removed):
identical.

**Weight precision.** With the weights stored as 8-bit integers per channel, 34 voxels changed on
DU-7294 (Dice 0.9988, 30.309 cm3); with 16-bit weights none changed on either case. The add-on
ships with 16-bit weights (44 MB).

**Workbench (browser, WebAssembly, shipped add-on file), `test/e2e.mjs` then
`raidionics_reference.py` and `compare_raidionics.py`:**

| Case | Loaded | Workbench cm3 | raidionicsseg cm3 | Differing voxels | Dice |
|---|---|---|---|---|---|
| TCGA-DU-7294 | FLAIR only | 30.274 | 30.270 | 2 of 4,456,448 | 0.99993 |
| TCGA-FG-6692 | all four, registered to FLAIR | 24.102 | 24.089 | 3 of 1,384,448 | 0.99972 |

The few differing voxels come from ONNX Runtime Web's WebAssembly kernels rounding differently
from the native CPU build (the Node runs above, with the same 16-bit weights, gave none).

**WebGPU.** The test machine has no GPU, so the network was checked on a software WebGPU adapter
(SwiftShader) with a 64^3 block of DU-7294's preprocessed FLAIR around the lesion
(`test/validate/webgpu_raidionics_export.py`): the largest logit difference from ONNX Runtime
CPU was 4.2e-4 (logits range -100 to 10) and all 28,345 voxels above 0.5 agreed. Speed and memory
on real GPUs were not measured.

**Executables.** The 1.2.0 Linux build, run through its own launcher with both add-on files
next to it (both listed under *Segmentation model*), gave the same DU-7294 FLAIR-only
segmentation as the test server, voxel for voxel, with no request leaving the computer.

**FLAIR-only runs and the other models.** The run button needs only the sequences the selected
model uses; others that are chosen are registered to the reference for viewing and exported as
registered images. With HD-GLIO and the BraTS model all four are still required, and the demo
case gives the same results as before: HD-GLIO enhancing 7.73, non-enhancing 13.92, brain 1656 cm3,
and the BraTS model ET 7.00, NCR 1.71, ED 24.67 cm3, with segmentation files identical voxel for
voxel to the version 1.1 runs.

## 7. Version 1.3.0: several models in one run, and their comparison

**What changed.** The checked models run one after the other after a single preparation
(reorientation, HD-BET, registration); each gets the inputs it got before (brain-extracted
for the nnU-Net models, the FLAIR as it is for Raidionics), and each network is released from
memory before the next is loaded. The comparison is computed from the label volumes on the
reference grid: per model the volume of the shared regions (whole abnormality, enhancing tumor,
non-enhancing abnormality, as listed in the README), and per pair and region Dice and the
volumes found by both, by the first only and by the second only.

**Same results as single-model runs.** Two runs of the TCGA-FG-6692 demo, then
`test/validate/check_model_comparison.py`:

| Run | Models | Reference | Result |
|---|---|---|---|
| A | HD-GLIO, nnU-Net BraTS, Raidionics FLAIR | T1 | HD-GLIO and BraTS segmentations identical voxel for voxel to the single-model runs of section 6 (0 of 1,490,944 voxels differ) |
| B | HD-GLIO, Raidionics FLAIR | FLAIR | Raidionics segmentation identical to the single-model run compared with raidionicsseg in section 6 (0 of 1,384,448) |

**Comparison numbers.** The checker recomputes every compartment volume, region volume and
pairwise figure from the exported segmentations with numpy and compares them with the page's
CSV: largest differences 0.0005 cm3 and 0.00004 in Dice (rounding of the CSV). The exported
agreement maps (1 first only, 2 second only, 3 both) match the two segmentations voxel for
voxel. Run A, whole abnormality: HD-GLIO vs BraTS Dice 0.787 (both 21.65 cm3, HD-GLIO only
0.004, BraTS only 11.73), HD-GLIO vs Raidionics 0.908, BraTS vs Raidionics 0.805; enhancing
tumor HD-GLIO vs BraTS 0.900; non-enhancing abnormality 0.673. Run A took 13 min 42 s on the
2-core machine (HD-BET 4 min 52 s, registration 33 s, HD-GLIO 2 min 36 s, BraTS 2 min 52 s,
Raidionics 2 min 50 s).

**Executable and memory.** The same three-model run through the 1.3.0 Linux build and its
launcher (both add-on files next to it) gave segmentations identical to run A for all three
models, with no request leaving the computer, in 13 min 47 s. All browser processes together
peaked at 4.1 GB (resident memory sampled every 2 s), the same as the single-model Raidionics
run in section 6, because each network is released before the next is loaded.

## 8. Version 1.4.0: Raidionics tumor core add-on

**Model.** `MRI_TumorCore`, folder `t1c`, network of fold 0, from
`Raidionics-MRI_TumorCore-v13.zip` (Raidionics-models release v1.3.0-rc; md5
680fb800c9bd2aa35a0177b4b355a86a, as in all three of the release's model lists). BSD 2-Clause.
One input (T1c), one sigmoid output: the preoperative tumor core, defined by Bouget et al.
(Front Neurol 2026) as enhancing tissue, necrosis and cysts. Its `pre_processing.ini` is the
same configuration as the FLAIR model's apart from the class name, so the same JavaScript runs it;
`build/export_models.py raidionics-core` checks this before exporting. After the exporter was
shared between the two models, the FLAIR add-on exports byte for byte as before. Raidionics'
pipeline for it is the same as for the FLAIR model: brain model, tumor core model, brain overlap
refinement.

**Implementation, original network (Node, onnxruntime-node, original fp32 `model.onnx`)**, same
reference runner and comparison as section 6, brain masks from HD-BET:

| Case (T1c) | T1c grid | Resampled grid (both) | Network input max abs diff | Probability max abs diff | Differing voxels | Volume cm3 |
|---|---|---|---|---|---|---|
| TCGA-FG-6692 (axial, 0.90 x 0.90 x 5.5 mm) | 256 x 224 x 26 | 178 x 212 x 164, 8 patches | 2.4e-7 | 3.0e-6 | 0 | 9.945 / 9.945 |
| TCGA-DU-7294 (axial, 0.94 x 0.94 x 2.5 mm) | 256 x 256 x 68 | 158 x 208 x 161, 4 patches | 2.4e-7 | 1.6e-6 | 0 | 9.734 / 9.734 |

On DU-7294 the brain overlap refinement removed 29 voxels (a small finding outside the brain
mask), identically on both sides. With the shipped 16-bit weights, FG-6692 was still identical and
DU-7294 differed at 2 voxels (Dice 0.99977).

**Workbench (browser, WebAssembly, shipped add-on file):**

| Case | Loaded, models | Workbench cm3 | raidionicsseg cm3 | Differing voxels | Dice |
|---|---|---|---|---|---|
| TCGA-DU-7294 | T1c only; tumor core | 9.743 | 9.734 | 4 of 4,456,448 | 0.99955 |
| TCGA-FG-6692 | all four, registered to T1c; BraTS and tumor core | 9.967 | 9.945 | 5 of 1,490,944 | 0.99889 |

The FG-6692 run also checked the new tumor core region of the comparison: BraTS (NCR + ET) vs
Raidionics tumor core Dice 0.827 (both 7.99 cm3, BraTS only 1.36, Raidionics only 1.98), recomputed
from the exported files by `check_model_comparison.py` (largest difference 0.0005 cm3), and the
exported agreement map matched voxel for voxel. A run of HD-GLIO with the tumor core model, which
share no region, showed both volume tables and the note that no overlap is computed.

**What it labels in a non-enhancing glioma.** On DU-7294 the model finds 9.7 cm3 of tumor core,
96% of it inside the Raidionics FLAIR lesion, with a median T1c intensity of 171 against 351 in the
rest of the brain: it labels the T1-hypointense part of the tumor, in keeping with a definition
that includes necrosis and cysts, not enhancement.

**Single-sequence loading.** A lone NIfTI whose name does not reveal its sequence (DU-7294's
T1c, saved as `20_SAG_MPR.nii.gz`) was not assigned to any sequence, so the run could not start.
When the checked models need one sequence and exactly one series is unassigned, the page now
assigns it, marked "only series left - check it". The DU-7294 run above loaded that file alone; it took
4 min 35 s (HD-BET 3 min 20 s, Raidionics tumor core 1 min 15 s for 4 patches).

**Executable.** The 1.4.0 Linux build, run through its own launcher with the three add-on files
next to it (all three listed), gave the same DU-7294 T1c-only segmentation as the test server,
voxel for voxel, with no request leaving the computer. The BraTS and Raidionics FLAIR add-on files
built by 1.4.0 are byte-identical to the published ones.

## 9. Version 1.5.0: AURORA brain metastasis and vestibular schwannoma add-ons

### 9a. AURORA

**Models.** AURORA weights v1.0.0 from Zenodo (record 10557069, CC BY 4.0): `t1c-o_best.tar`
(T1c only, md5 5cbb214b74196c985bde38401e0474df) and `t1c-fla_best.tar` (T1c + FLAIR, md5
5037c42da1a6067f3c1f193924fa7083), as listed in the record. Each is a MONAI BasicUNet (features
32, 32, 64, 128, 256, 32; mish; instance norm) built as brainles_aurora 0.2.7 builds it; the
checkpoints load with `torch.load(weights_only=True)`. For the export the mish activation is
written as x * u / (u + 2) with u = e^x (e^x + 2) (e^x of min(x, 20)), which is the same function
with only elementary operators that ONNX Runtime Web runs on the GPU; on a random 192 x 192 x 32
input the exporter checks that the logits match the original module (largest difference below
1e-3, asserted). UpCat's padding, a no-op for windows divisible by 16, is left out of the graph.

**What brainles_aurora does, and the workbench.**

| Step | brainles_aurora 0.2.7 (MONAI) | Workbench (`app/js/lib/aurora.js`) |
|---|---|---|
| Input | The files as given (expected: brain-extracted, 1 mm, BraTS space) | The HD-BET-masked, registered images on a 1 mm grid in BraTS axis order (i left, j posterior, k superior), 240 x 240 x 155: the reference's own frame when it already is one; else centred on the brain, on the reference's voxels when it is 1 mm and axis-aligned (copied), otherwise linearly resampled |
| Intensities | `ScaleIntensityRangePercentiles(0.5, 99.5, 0, 1, clip, channel_wise)` | Same (numpy's linear percentile, float32 arithmetic) |
| Inference | `SlidingWindowInferer(192 x 192 x 32, overlap 0.5, gaussian, sigma 0.125, replicate padding)` | Same window positions, Gaussian weights (clamped at 0.001) and normalisation |
| Augmentation (default on) | Mean logits of 13 passes: as is; 4 with Gaussian noise (std 0.001); 4 x 2 with the noisy image flipped on axis 1 or 2 | Off by default; with it on, 3 passes weighted 5 : 4 : 4 (the 13 without the noise) |
| Output | sigmoid >= 0.5; whole lesion -> 1, metastasis -> 2 | Same, after linear resampling of the probabilities back to the reference grid |

The BraTS axis order was taken from the AURORA example data: its arrays have the frontal lobes at
low j and the vertex at high k, the layout of BraTS images (whose header orientation is LPS).
The example files themselves carry an identity orientation (RAS) that does not match their
anatomy, so for the workbench tests they were given the BraTS orientation (diag(-1, -1, 1),
arrays unchanged); brainles_aurora, which ignores the orientation, was given the original files.

**Implementation, identical inputs (Node, onnxruntime-node, shipped 16-bit weights, against
brainles_aurora with the fp32 weights).** `test/unit/aurora_test.mjs` runs `aurora.js` on the T1c of
the AURORA example (BraTS-METS sample, 240 x 240 x 155 at 1 mm, brain mask = its nonzero voxels) and
writes the images on its 1 mm grid; `test/validate/compare_aurora.py` gives those files to
`AuroraInferer` (weights read from the local folder, nothing downloaded) and compares the sigmoid
probabilities and labels. Here the 1 mm grid falls on the input's own voxels (240 x 240 x 155, 36
windows), so the input-grid labels correspond voxel for voxel.

| T1c model, input and frame | Max abs probability diff (whole / metastasis) | Differing voxels | Metastasis cm3, workbench / AURORA (Dice) | Edema cm3 (Dice) |
|---|---|---|---|---|
| 1 mm, frame centred on the brain, no augmentation | 0.0027 / 0.0026 | 5 of 8,928,000 | 51.888 / 51.888 (1.0) | 47.346 / 47.349 (0.99995) |
| Same, augmentation: workbench 3 passes, AURORA 13 with its noise set to 0 | 0.0014 / 0.0024 | 2 of 8,928,000 | 51.877 / 51.877 (1.0) | 47.586 / 47.584 (0.99998) |
| 1 mm, the file's own 240 x 240 x 155 frame (AURORA given the file itself), no augmentation | - | 8 of 8,928,000 | 51.796 / 51.796 (1.0) | 47.076 / 47.078 (0.99992) |
| Oblique copy: 0.94 x 0.94 x 3 mm, axes turned 10 degrees, resampled to the 1 mm grid, no augmentation | 0.0039 / 0.0030 | 8 of 8,928,000 (1 mm grid) | 53.174 / 53.173 (0.99999) | 49.071 / 49.073 (0.99992) |

**The frame matters to AURORA itself.** The first two rows used a 240 x 240 x 155 frame centred on
the brain's bounding box, which for this image sits 9 and 4 voxels away from the file's own frame
(j and k). AURORA's own result on the same image in those two frames differs: Dice 0.9956 for the
metastasis (51.80 vs 51.89 cm3), 0.972 for edema (47.08 vs 47.35 cm3), because the sliding
windows then fall on different voxels. The workbench therefore uses a reference that already is
such a box (1 mm, 240 x 240 x 155 along the BraTS axes, as BraTS-preprocessed images are) as it
is, and centres the frame on the brain only for other references; the third row is that case.

**Resampling path.** For the oblique copy (rotated 10 degrees about the superior axis, 3 mm
slices, made with linear interpolation from the example) the 1 mm grid is resampled, so the
labels on the oblique grid come from linearly resampled probabilities; against AURORA's
internal-grid labels taken by nearest voxel they agree with Dice 0.992 (metastasis) and 0.967
(edema), the difference between the two ways of going back. Against AURORA on the original 1 mm
image, the 3 mm result has 53.34 vs 51.80 cm3 of metastasis (Dice 0.961 on the 1 mm grid): what
3 mm slices cost, not a fault of the pipeline.

**Workbench (browser, WebAssembly, shipped add-on files)**, both AURORA models in one run on the
four example files (T1, T1c, T2, FLAIR on one 1 mm grid; registration switched off automatically),
against `test/validate/aurora_end_to_end.py`: brainles_aurora on the T1c (and FLAIR) files as they
are.

| Run | Model | Workbench cm3 (metastasis / edema) | AURORA cm3 | Differing voxels | Dice metastasis / edema |
|---|---|---|---|---|---|
| Brain extraction off, reference T1 | T1c | 51.81 / 47.01 | 51.80 / 47.08 | 274 of 8,928,000 | 0.99975 / 0.9971 |
| Brain extraction off, reference T1 | T1c + FLAIR | 51.99 / 52.16 | 51.97 / 52.46 | 398 | 0.99952 / 0.99626 |
| HD-BET on T1 (default), reference T1 | T1c | 51.82 / 46.96 | 51.80 / 47.08 | 285 | 0.99961 / 0.99697 |
| HD-BET on T1 (default), reference T1 | T1c + FLAIR | 51.98 / 52.26 | 51.97 / 52.46 | 382 | 0.99926 / 0.9965 |

In both runs the 1 mm grid is the files' own frame. With brain extraction off, the inputs are
masked with the reference's nonzero voxels (as for skull-stripped inputs in general); in these
files 7.1 cm3 of the T1c's and 25.2 cm3 of the FLAIR's nonzero voxels are zero in the T1, which
accounts for the differences (the T1c-only run in Node with the T1c unmasked differed at 8 voxels,
row 3 above). With HD-BET, its mask (1,570 cm3, against 1,618 cm3 of nonzero voxels in these
already skull-stripped files) trims the inputs a little more, with the same effect on the result.
For BraTS-preprocessed images, switching brain extraction off and registering to T1c gives
AURORA exactly its own input. The default run also checked the model comparison
(`check_model_comparison.py`: largest difference 0.00003 Dice; agreement map identical): T1c vs
T1c + FLAIR metastasis Dice 0.973, whole lesion 0.890, the FLAIR model finding 13.8 cm3 of
abnormality (edema) on its own.

Run time on the 2-core test machine, CPU: 5.5 min per model without augmentation (36 windows of
192 x 192 x 32), about 3 times that with it.

### 9b. Vestibular schwannoma

**Models.** From Zenodo record 10363647 (CC BY 4.0), `MC-RC+SC-GK-models.zip` (7.2 GB, md5
551f187380b29efca49ddd97f7f9e969 as listed): `Dataset920_VSMCRCSCGKT1` (contrast-enhanced T1,
404 training cases after the authors' balancing of the two datasets) and `Dataset921_VSMCRCSCGKT2`
(T2, 487), nnU-Net v2, the `3d_fullres` configuration that nnU-Net selected for both (cross-validation
Dice 0.913 and 0.870), network of fold 0. Only the files needed were read out of the archive, with
HTTP range requests; Python's zipfile checks each against the archive's CRC-32. The checkpoints
open with `torch.load(weights_only=True)` (plain tensors, numbers and numpy scalars only) before
nnU-Net loads them. These combined-data models were chosen over the multi-centre-only ones
(`Dataset910`/`911`) because the paper found them best on most test sets, clearly so for T1c.

| | T1c (920) | T2 (921) |
|---|---|---|
| Target spacing (z, y, x) | 1.0 x 0.41 x 0.41 mm | 0.70 x 0.47 x 0.47 mm |
| Patch | 64 x 192 x 160 | 40 x 256 x 224 |
| Normalisation | z-score of the whole image (no mask) | same |
| Mirroring at inference | axes 0, 1, 2 | same |

They run through the same JavaScript nnU-Net code as the BraTS add-on and HD-BET (nnU-Net v2
paths), with one change found while adding them: when the probabilities are resampled back to
the input grid, nnU-Net v2 decides on separate-z resampling from the target spacing first and the
input spacing second (v1 the other way round). The order only matters when both spacings are
anisotropic (ratio above 3) along different axes, which the target spacings of these models and of
HD-BET rule out, so no result changes (HD-GLIO and BraTS are v1). The inputs are not brain-extracted (`maskInput: false`), and
when only such models are checked HD-BET does not run. The authors' Zenodo instructions say to
omit nnU-Net's postprocessing, so none is applied (on cross-validation nnU-Net had chosen keeping
the largest region, which changed the mean Dice by less than 0.001).

**Agreement with nnU-Net v2 (browser, WebAssembly, shipped add-on file).** The example image of the
model record (`example_input_images.zip`, a Gamma Knife planning contrast-enhanced T1, 416 x 488 x
81 voxels of 0.41 x 0.41 x 1.5 mm, LPS) was loaded on its own as `vs_gk_ceT1.nii.gz`, a name the
page assigns to T1c (under its own name, `vs_gk_0000.nii.gz`, nnU-Net's channel suffix makes the
page take it for T1, which the drop-down corrects). HD-BET was not run. `test/validate/compare_vs.py` reorients the workbench's export of
the image as the workbench does (LAS), runs `nnUNetPredictor` (PyTorch, CPU, fp32 checkpoint,
fold 0, no mirroring, no postprocessing) and takes the result back:

| Case | Workbench cm3 | nnU-Net v2 cm3 | Differing voxels | Dice |
|---|---|---|---|---|
| Example ceT1, T1c model | 3.761 | 3.761 | 2 of 16,443,648 | 0.99993 |
| Example T2 (on the ceT1 grid), T2 model | 3.329 | 3.328 | 1 of 16,443,648 | 0.99996 |

The workbench reorients every input to the MNI voxel order before the models see it (for this
LPS file, a flip of the second axis), as HD-GLIO's instructions prescribe. Run on the file in its
stored voxel order, as `nnUNetv2_predict` would, nnU-Net gave 3.753 cm3, Dice 0.990 against the
reoriented run: the model is close to, but not exactly, invariant to that flip.

The example T2 is resampled onto the ceT1 grid (1.5 mm slices), not a native thin-slice T2, so it
tests the implementation rather than the T2 model's accuracy. It was loaded on its own with
*Register to* T2 (the T2 model's run grid is then the T2 as given). The page recognised it from
its name.

Run time on the 2-core test machine, CPU, no augmentation: T1c model 29 min 12 s (75 patches of
64 x 192 x 160 on a 122 x 488 x 416 grid at 1.0 x 0.41 x 0.41 mm), T2 model 27 min 57 s (72 patches
of 40 x 256 x 224). The browser peaked at about 3 GB; with other memory-hungry programs running at
the same time the test machine's 7 GB was not enough (the browser tab was stopped once and the run
repeated alone).

**Executable.** The 1.5.0 Linux build, run through its own launcher with the seven add-on files
next to it (all seven listed), ran AURORA T1c on the example T1c alone (brain extraction off, so
the input is the T1c itself) with no request leaving the computer: 51.81 cm3 of metastasis and
47.10 of edema, 62 voxels different from brainles_aurora on the file (Dice 0.99988 metastasis,
0.99935 edema; the Node run of the same input, row 3 of the first table, differed at 8: the
browser's WebAssembly kernels round differently from Node's). The BraTS and both Raidionics add-on
files built by 1.5.0 are byte-identical to the published ones, and the four new add-on files are
the ones tested above.

## Run time

Demo case on the 2-core test machine, CPU (WebAssembly), no augmentation: 9 min (HD-BET
5 min 25 s, registration 41 s, HD-GLIO 2 min 56 s). HD-BET runs 8 patches of 96 x 192 x 192 and
HD-GLIO 8 patches of 128^3. Machines with more cores, or with a GPU through WebGPU, should be
considerably faster; that was not measured here. Test-time augmentation multiplies the HD-GLIO
time by 8: the same case with augmentation took 31 min 34 s (HD-GLIO 24 min 51 s).

Raidionics FLAIR add-on, same machine, CPU: DU-7294 FLAIR alone 7 min 27 s (HD-BET on the FLAIR
6 min 47 s, Raidionics 39 s for 2 patches of 160^3); FG-6692 with all four sequences registered
to FLAIR 5 min 23 s (HD-BET 2 min 19 s, registration 28 s, Raidionics 2 min 36 s for 8 patches).

AURORA add-ons, same machine, CPU: about 5.5 min per model on a 1 mm 240 x 240 x 155 image (36
windows of 192 x 192 x 32), after HD-BET (1 min 20 s there). Vestibular schwannoma add-ons: 28 to
29 min on the 0.41 x 0.41 x 1.5 mm example (72 to 75 patches), with no HD-BET.
