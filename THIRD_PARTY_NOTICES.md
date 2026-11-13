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

The inference code in `app/js/lib/nnunet.js` reimplements the prediction steps of nnU-Net
([MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet), Apache-2.0), HD-GLIO
([MIC-DKFZ/HD-GLIO](https://github.com/MIC-DKFZ/HD-GLIO), Apache-2.0) and HD-BET
([MIC-DKFZ/HD-BET](https://github.com/MIC-DKFZ/HD-BET), Apache-2.0). The code in
`app/js/lib/raidionics.js` reimplements the preprocessing, patch-wise inference and
reconstruction of raidionicsseg 1.5.2 ([PyPI `raidionicsseg`](https://pypi.org/project/raidionicsseg/),
BSD 2-Clause, Copyright (c) 2022, SINTEF Digital, Health Research) and the brain overlap
refinement of raidionics_rads_lib 1.3.3
([dbouget/raidionics_rads_lib](https://github.com/dbouget/raidionics_rads_lib), BSD 2-Clause).

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
| TCGA-FG-6692 (T1, T1c, T2, FLAIR) | [The Cancer Imaging Archive, TCGA-LGG collection](https://www.cancerimagingarchive.net/collection/tcga-lgg/), via the NIfTI files in [BrainLesion tutorials](https://github.com/BrainLesion/tutorials) (as in laetissimus/brain-tumor-perfusion) | TCIA data: CC BY 3.0. Cite the collection and TCIA |
| TCGA-DU-7294 (FLAIR; test case and `docs/raidionics-flair.png` only, not shipped) | Same collection and route (as in laetissimus/brain-tumor-perfusion) | TCIA data: CC BY 3.0. Cite the collection and TCIA |
