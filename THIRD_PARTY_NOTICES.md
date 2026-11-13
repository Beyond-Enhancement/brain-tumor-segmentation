# Third-party components and data

The executables embed the components below. Full license texts are in the upstream
repositories and records linked here.

## Trained networks

| Model | Source | License | Where it is used |
|---|---|---|---|
| HD-GLIO v2 network parameters | Kickingereder P, Isensee F. Zenodo [10.5281/zenodo.4014850](https://doi.org/10.5281/zenodo.4014850) (`hd_glio_v2_params.zip`, md5 65704898365b11630887389a90c10122) | CC BY 4.0 | `app/models/hdglio/`, converted to ONNX |
| nnU-Net BraTS 2020 parameters (add-on, not in the executables) | Isensee F. Zenodo [10.5281/zenodo.4635763](https://doi.org/10.5281/zenodo.4635763) (`Task082_nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1_bs5_5fold.zip`, md5 b6c771f4921f0a62852e842e4b91d6d0; fold 0 used) | CC BY 4.0 | `downloads/*-BraTS2020-nnUNet.btsmodel`, converted to ONNX (16-bit weights) |
| Raidionics MRI_FLAIRChanges parameters, `t2f` network of fold 0 (add-on, not in the executables) | Bouget D, Faanes MG, et al. [Raidionics-models release v1.3.0-rc](https://github.com/raidionics/Raidionics-models/releases/tag/v1.3.0-rc) (`Raidionics-MRI_FLAIRChanges-v13.zip`, md5 ec6209b06af8cb226bff50d8890eed24, as in the release's model list) | BSD 2-Clause, Copyright (c) 2023, David Bouget (text below and inside the add-on file) | `addons/raidionics-flair/`, converted to ONNX with 16-bit weights, packaged as `BrainTumorSegmentationWorkbench-model-Raidionics-FLAIR.btsmodel` |
| Raidionics MRI_TumorCore parameters, `t1c` network of fold 0 (add-on, not in the executables) | Bouget D, et al. [Raidionics-models release v1.3.0-rc](https://github.com/raidionics/Raidionics-models/releases/tag/v1.3.0-rc) (`Raidionics-MRI_TumorCore-v13.zip`, md5 680fb800c9bd2aa35a0177b4b355a86a, as in the release's model lists) | BSD 2-Clause, Copyright (c) 2023, David Bouget (text below and inside the add-on file) | `addons/raidionics-core/`, converted to ONNX with 16-bit weights, packaged as `BrainTumorSegmentationWorkbench-model-Raidionics-TumorCore.btsmodel` |
| Vestibular schwannoma models trained on the combined multi-centre routine (MC-RC) and single-centre Gamma Knife (SC-GK) data: `Dataset920_VSMCRCSCGKT1` (contrast-enhanced T1) and `Dataset921_VSMCRCSCGKT2` (T2), nnU-Net v2 `3d_fullres`, network of fold 0 (add-ons, not in the executables) | Kujawa A, Dorent R, Connor S, et al. Zenodo [10.5281/zenodo.10363647](https://doi.org/10.5281/zenodo.10363647) (`MC-RC+SC-GK-models.zip`, md5 551f187380b29efca49ddd97f7f9e969 as listed by Zenodo; the fold-0 files, plans and dataset descriptions were read out of the archive with HTTP range requests, each checked against the archive's CRC-32) | CC BY 4.0 (notice and changes inside the add-on files) | `addons/vs-t1c/`, `addons/vs-t2/`, converted to ONNX with 16-bit weights, packaged as `BrainTumorSegmentationWorkbench-model-VS-T1c.btsmodel` and `-VS-T2.btsmodel` |
| AURORA brain metastasis models, weights v1.0.0: `t1c-o_best.tar` (T1c only, md5 5cbb214b74196c985bde38401e0474df) and `t1c-fla_best.tar` (T1c + FLAIR, md5 5037c42da1a6067f3c1f193924fa7083) (add-ons, not in the executables) | Buchner JA, Kofler F, et al. (BrainLesion/AURORA). Zenodo [10.5281/zenodo.10557069](https://doi.org/10.5281/zenodo.10557069) | CC BY 4.0 (notice and changes inside the add-on files) | `addons/aurora-t1c/`, `addons/aurora-t1c-flair/`, converted to ONNX with 16-bit weights, packaged as `BrainTumorSegmentationWorkbench-model-AURORA-T1c.btsmodel` and `-AURORA-T1c-FLAIR.btsmodel` |
| HD-BET v1.5 parameters (HD-BET 2.0 package) | Isensee F, Schell M, Tursunova I, et al. Zenodo [10.5281/zenodo.14445620](https://doi.org/10.5281/zenodo.14445620) (`release_v1.5.0.zip`, md5 f97a36f89f8eb3071ff05c6f9ac66b56) | **CC BY-NC 4.0 (non-commercial)** | `app/models/hdbet/`, converted to ONNX |

The networks were converted with `build/export_models.py`. The weights are stored at reduced
precision; see the README and VERIFICATION.md. Because the HD-BET parameters are licensed for
non-commercial use only, the executables must not be used or redistributed commercially.

Please cite when using results:

- Kickingereder P, Isensee F, Tursunova I, et al. Automated quantitative tumour response
  assessment of MRI in neuro-oncology with artificial neural networks: a multicentre,
  retrospective study. Lancet Oncol. 2019;20(5):728-740. doi:10.1016/S1470-2045(19)30098-1
- Isensee F, Schell M, Pflueger I, et al. Automated brain extraction of multisequence MRI using
  artificial neural networks. Hum Brain Mapp. 2019;40(17):4952-4964. doi:10.1002/hbm.24750
- Isensee F, Jaeger PF, Full PM, Vollmuth P, Maier-Hein KH. nnU-Net for brain tumor
  segmentation. In: Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries
  (BrainLes 2020). LNCS 12659. Springer; 2021:118-132. (BraTS 2020 add-on model)
- Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method
  for deep learning-based biomedical image segmentation. Nat Methods. 2021;18(2):203-211.
- Bouget D, et al. Automatic and standardized reporting of perioperative MRIs in patients with
  central nervous system tumors. Front Neurol. 2026;16. doi:10.3389/fneur.2025.1707481
  (Raidionics tumor core add-on model)
- Faanes MG, Bouget D, Jakola AS, et al. A unified FLAIR hyperintensity segmentation model for
  various CNS tumor types and acquisition time points. Sci Rep. 2026;16:19707.
  doi:10.1038/s41598-026-48496-1 (Raidionics FLAIR add-on model)
- Bouget D, et al. Raidionics: an open software for pre- and postoperative central nervous
  system tumor segmentation and standardized reporting. Sci Rep. 2023;13.
  doi:10.1038/s41598-023-42048-7
- Kujawa A, Dorent R, Connor S, et al. Deep learning for automatic segmentation of vestibular
  schwannoma: a retrospective study from multi-center routine MRI. Front Comput Neurosci.
  2024;18:1365727. doi:10.3389/fncom.2024.1365727 (vestibular schwannoma add-on models)
- Buchner JA, Kofler F, Etzel L, et al. Development and external validation of an MRI-based
  neural network for brain metastasis segmentation in the AURORA multicenter study. Radiother
  Oncol. 2023;178:109425. doi:10.1016/j.radonc.2022.11.014 (AURORA add-on models)
- Buchner JA, Peeken JC, Etzel L, et al. Identifying core MRI sequences for reliable automatic
  brain metastasis segmentation. Radiother Oncol. 2023;188:109901.
  doi:10.1016/j.radonc.2023.109901 (AURORA add-on models)

The inference code in `app/js/lib/nnunet.js` reimplements the prediction steps of nnU-Net
([MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet), Apache-2.0), HD-GLIO
([MIC-DKFZ/HD-GLIO](https://github.com/MIC-DKFZ/HD-GLIO), Apache-2.0) and HD-BET
([MIC-DKFZ/HD-BET](https://github.com/MIC-DKFZ/HD-BET), Apache-2.0). The code in
`app/js/lib/raidionics.js` reimplements the preprocessing, patch-wise inference and
reconstruction of raidionicsseg 1.5.2 ([PyPI `raidionicsseg`](https://pypi.org/project/raidionicsseg/),
BSD 2-Clause, Copyright (c) 2022, SINTEF Digital, Health Research) and the brain overlap
refinement of raidionics_rads_lib 1.3.3
([dbouget/raidionics_rads_lib](https://github.com/dbouget/raidionics_rads_lib), BSD 2-Clause).
The code in `app/js/lib/aurora.js` reimplements the inference of brainles_aurora 0.2.7
([BrainLesion/AURORA](https://github.com/BrainLesion/AURORA), Apache-2.0), including the
intensity scaling (`ScaleIntensityRangePercentiles`) and Gaussian-weighted sliding window
(`SlidingWindowInferer`) of [MONAI](https://github.com/Project-MONAI/MONAI) (Apache-2.0), whose
`BasicUNet` the AURORA networks are.

### License of the Raidionics network parameters

The Raidionics add-on files redistribute network parameters from Raidionics-models under this
license (the same text is in the `README.txt` inside the add-on file):

```
BSD 2-Clause License

Copyright (c) 2023, David Bouget

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Software

| Component | Version | License | Where it is used |
|---|---|---|---|
| [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) | 1.31.1 | MIT | `app/vendor/ort/`, runs the networks (WebGPU or WebAssembly CPU) |
| [NiiVue](https://github.com/niivue/niivue) | 0.69.0 | BSD 2-Clause | `app/vendor/niivue.umd.js`, the WebGL2 image viewer |
| [dicomParser](https://github.com/cornerstonejs/dicomParser) | 1.8.21 | MIT | `app/vendor/dicomParser.mjs` |
| [fflate](https://github.com/101arrowz/fflate) | 0.8.3 | MIT | `app/vendor/fflate.mjs`, gzip and zip |
| [OpenJPEG](https://github.com/uclouvain/openjpeg) via [@cornerstonejs/codec-openjpeg](https://github.com/cornerstonejs/codecs) | 1.3.6 | BSD 2-Clause / MIT | `app/vendor/codecs/`, JPEG 2000 DICOM |
| [CharLS](https://github.com/team-charls/charls) via [@cornerstonejs/codec-charls](https://github.com/cornerstonejs/codecs) | 1.2.7 | BSD 3-Clause / MIT | `app/vendor/codecs/`, JPEG-LS DICOM |
| [jpeg-lossless-decoder-js](https://github.com/rii-mango/JPEGLosslessDecoderJS) | 2.1.2 | MIT | `app/vendor/codecs/`, JPEG lossless DICOM |
| Barlow, Barlow Semi Condensed, IBM Plex Mono via [Fontsource](https://fontsource.org) | - | SIL Open Font License 1.1 | `app/vendor/fonts/` |
| [Go](https://go.dev) standard library | 1.24 | BSD 3-Clause | The launcher (`launcher/`) |

## Demo data

| Case | Source | Terms |
|---|---|---|
| TCGA-FG-6692 (T1, T1c, T2, FLAIR) | [The Cancer Imaging Archive, TCGA-LGG collection](https://www.cancerimagingarchive.net/collection/tcga-lgg/), via the NIfTI files in [BrainLesion tutorials](https://github.com/BrainLesion/tutorials) (as in Beyond-Enhancement/brain-tumor-perfusion) | TCIA data: CC BY 3.0. Cite the collection and TCIA |
| TCGA-DU-7294 (FLAIR; test case and `docs/raidionics-flair.png` only, not shipped) | Same collection and route (as in Beyond-Enhancement/brain-tumor-perfusion) | TCIA data: CC BY 3.0. Cite the collection and TCIA |
| Vestibular schwannoma example (contrast-enhanced T1 and T2; test case only, not shipped) | `example_input_images.zip` (md5 dd77d99b3cf136dd847c4aaa710da76d) in the same Zenodo record as the schwannoma models, [10.5281/zenodo.10363647](https://doi.org/10.5281/zenodo.10363647) | CC BY 4.0 (record license) |
| Brain metastasis example (T1, T1c, T2, FLAIR; test case only, not shipped) | `AURORA/data` in [BrainLesion tutorials](https://github.com/BrainLesion/tutorials), a sample of the ASNR-MICCAI BraTS brain metastasis challenge data | Tutorials repository: Apache-2.0; BraTS-METS data terms apply. Cite the challenge |
