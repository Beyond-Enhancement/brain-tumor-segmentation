# Third-party components and data

The executables embed the components below. Full license texts are in the upstream
repositories and records linked here.

## Trained networks

| Model | Source | License | Where it is used |
|---|---|---|---|
| HD-GLIO v2 network parameters | Kickingereder P, Isensee F. Zenodo [10.5281/zenodo.4014850](https://doi.org/10.5281/zenodo.4014850) (`hd_glio_v2_params.zip`, md5 65704898365b11630887389a90c10122) | CC BY 4.0 | `app/models/hdglio/`, converted to ONNX |
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
- Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method
  for deep learning-based biomedical image segmentation. Nat Methods. 2021;18(2):203-211.

The inference code in `app/js/lib/nnunet.js` reimplements the prediction steps of nnU-Net
([MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet), Apache-2.0), HD-GLIO
([MIC-DKFZ/HD-GLIO](https://github.com/MIC-DKFZ/HD-GLIO), Apache-2.0) and HD-BET
([MIC-DKFZ/HD-BET](https://github.com/MIC-DKFZ/HD-BET), Apache-2.0).

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
