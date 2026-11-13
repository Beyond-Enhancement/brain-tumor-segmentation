#!/usr/bin/env python3
"""Export the trained HD-GLIO (nnU-Net v1) and HD-BET (nnU-Net v2) models for the workbench.

For each model this writes app/models/<name>/:
  model.onnx    network graph (weights stored externally)
  weights.pack  weights, fp16 (default) or int8 per output channel (--q8)
  config.json   everything the JavaScript nnU-Net needs, read from the model's own plans

Usage:
  build/export_models.py hdglio <hd_glio_params folder>
  build/export_models.py hdbet  <hd-bet release folder (contains fold_all/)>
  build/export_models.py brats2020 <nnU-Net Task082 trainer folder> --out addons/brats2020
  build/export_models.py raidionics-flair <MRI_FLAIRChanges/t2f folder> --out addons/raidionics-flair
  build/export_models.py raidionics-core <MRI_TumorCore/t1c folder> --out addons/raidionics-core
  build/export_models.py vs-t1c <Dataset910_VSMCRCT1/nnUNetTrainer__nnUNetPlans__3d_fullres> --out addons/vs-t1c
  build/export_models.py vs-t2  <Dataset911_VSMCRCT2/nnUNetTrainer__nnUNetPlans__3d_fullres> --out addons/vs-t2
  build/export_models.py aurora-t1c <t1c-o_best.tar> --out addons/aurora-t1c
  build/export_models.py aurora-t1c-flair <t1c-fla_best.tar> --out addons/aurora-t1c-flair
"""
import argparse, json, os, sys, tempfile, hashlib
import numpy as np
import torch
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'app', 'models')


class SoftmaxWrap(torch.nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
    def forward(self, x):
        return torch.softmax(self.net(x), 1)


class SigmoidWrap(torch.nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
    def forward(self, x):
        return torch.sigmoid(self.net(x))


def export_onnx(net, in_ch, patch, path, softmax, sigmoid=False):
    net.eval()
    model = SigmoidWrap(net) if sigmoid else SoftmaxWrap(net) if softmax else net
    x = torch.zeros(1, in_ch, *patch)
    with torch.no_grad():
        torch.onnx.export(model, x, path, opset_version=17, input_names=['input'], output_names=['output'], dynamo=False,
                          do_constant_folding=True)


def pack_weights(onnx_path, out_dir, q8=False, small_f32=False):
    """Move all initializers to an external fp32 blob, then write a compact pack + index.
    small_f32: keep tensors with fewer than 2 dimensions (norm statistics, biases, PReLU slopes)
    in fp32, so that tiny running variances cannot underflow in fp16."""
    m = onnx.load(onnx_path)
    m.ir_version = 10  # ONNX Runtime Web 1.31 reads IR <= 13; opset 17 needs IR >= 8
    ext_name = 'weights.bin'
    onnx.save_model(m, os.path.join(out_dir, 'model.onnx'), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=ext_name, size_threshold=0, convert_attribute=False)
    m2 = onnx.load(os.path.join(out_dir, 'model.onnx'), load_external_data=False)
    blob = open(os.path.join(out_dir, ext_name), 'rb').read()
    tensors, chunks, pos = [], [], 0
    def add(b):
        nonlocal pos
        pad = (-pos) % 4
        if pad: chunks.append(b'\0' * pad); pos += pad
        start = pos; chunks.append(b); pos += len(b); return start
    for init in m2.graph.initializer:
        info = {e.key: e.value for e in init.external_data}
        off, ln = int(info.get('offset', 0)), int(info['length'])
        raw = blob[off:off + ln]
        ent = {'name': init.name, 'offset': off, 'bytes': ln}
        if init.data_type == onnx.TensorProto.FLOAT and ln >= 64 and not (small_f32 and len(init.dims) < 2):
            a = np.frombuffer(raw, dtype='<f4')
            ent['count'] = int(a.size)
            dims = list(init.dims)
            if q8 and len(dims) >= 4:
                # per output channel (Conv: dim 0, ConvTranspose: dim 1 -> use dim 0 of a transposed view)
                ch = dims[0]
                a2 = a.reshape(ch, -1)
                s = np.maximum(np.abs(a2).max(1), 1e-12) / 127.0
                q = np.clip(np.round(a2 / s[:, None]), -127, 127).astype(np.int8)
                ent.update(fmt='q8', channels=int(ch))
                ent['pack'] = add(q.tobytes())
                ent['scales'] = add(s.astype('<f4').tobytes())
            else:
                ent['fmt'] = 'f16'
                ent['pack'] = add(a.astype('<f2').tobytes())
        else:
            ent['fmt'] = 'raw'
            ent['pack'] = add(raw)
        tensors.append(ent)
    pack = b''.join(chunks)
    open(os.path.join(out_dir, 'weights.pack'), 'wb').write(pack)
    os.remove(os.path.join(out_dir, ext_name))
    return {'pack': 'weights.pack', 'externalName': ext_name, 'externalBytes': len(blob), 'tensors': tensors,
            'format': 'q8' if q8 else 'f16', 'sha256': hashlib.sha256(pack).hexdigest()}


# ------------------------------------------------------------------------------------------------
def allow_numpy_in_checkpoints():
    # The 2019 nnU-Net v1 checkpoint stores numpy scalars next to the weights. Allow exactly those
    # types for torch.load(weights_only=True) instead of turning weight-only loading off.
    import numpy.core.multiarray as ma  # numpy 2 keeps this alias for pickles
    types = [(ma.scalar, 'numpy.core.multiarray.scalar'), np.dtype] + [type(np.dtype(t)) for t in ('float64', 'float32', 'int64', 'int32', 'bool', 'uint8')]
    torch.serialization.add_safe_globals(types)


def export_hdglio(folder, out_dir, q8):
    allow_numpy_in_checkpoints()
    from nnunet.training.model_restore import load_model_and_checkpoint_files
    from nnunet.inference.predict import load_postprocessing
    trainer, params = load_model_and_checkpoint_files(folder, (0,), mixed_precision=False, checkpoint_name='model_final_checkpoint')
    trainer.load_checkpoint_ram(params[0], False)
    net = trainer.network
    net.do_ds = False
    net.cpu().float().eval()
    plans = trainer.plans
    stage = trainer.stage
    stage_plans = plans['plans_per_stage'][stage]
    patch = [int(p) for p in trainer.patch_size]
    n_in = int(trainer.num_input_channels)
    n_cls = int(trainer.num_classes)
    norm = plans['normalization_schemes']
    use_mask = plans['use_mask_for_norm']
    mirror = [int(a) for a in trainer.data_aug_params.get('mirror_axes', (0, 1, 2))]
    pp = None
    pjson = os.path.join(folder, 'postprocessing.json')
    if os.path.isfile(pjson):
        fw, mv = load_postprocessing(pjson)
        pp = {'forWhichClasses': [list(c) if isinstance(c, (list, tuple)) else int(c) for c in fw],
              'minValidObjectSizes': {str(k): v for k, v in (mv or {}).items()} if mv else None}
    export_onnx(net, n_in, patch, os.path.join(out_dir, 'full.onnx'), softmax=True)
    w = pack_weights(os.path.join(out_dir, 'full.onnx'), out_dir, q8)
    os.remove(os.path.join(out_dir, 'full.onnx'))
    sep = trainer.segmentation_export_params if hasattr(trainer, 'segmentation_export_params') else {}
    cfg = {
        'name': 'HD-GLIO', 'version': '2 (Zenodo 4014850)', 'nnunet': 1,
        'trainer': type(trainer).__name__, 'stage': stage,
        'inputChannels': ['T1', 'T1c', 'T2', 'FLAIR'], 'numClasses': n_cls,
        'allClasses': [int(c) for c in plans.get('all_classes', range(1, n_cls))],
        'transposeForward': [int(a) for a in plans['transpose_forward']], 'transposeBackward': [int(a) for a in plans['transpose_backward']],
        'targetSpacing': [float(s) for s in stage_plans['current_spacing']], 'patchSize': patch,
        'normalization': [{'scheme': norm[i], 'useMask': bool(use_mask[i])} for i in range(n_in)],
        'mirrorAxes': mirror, 'stepSize': 0.5, 'aggregate': 'softmax', 'anisotropyThreshold': 3,
        'resampleData': {'order': 3, 'orderZ': 0}, 'resampleProb': {'order': int(sep.get('interpolation_order', 1)), 'orderZ': int(sep.get('interpolation_order_z', 0))},
        'postprocessing': pp, 'onnx': 'model.onnx', 'weights': w,
        # label meaning from the model's own dataset properties (class_dct: 1 'edema', 2 'km' =
        # Kontrastmittel, i.e. contrast-enhancing), named as in Kickingereder et al. 2019
        'classNamesInModel': {str(k): v for k, v in plans['dataset_properties']['class_dct'].items()},
        'labels': [{'value': 1, 'name': 'Non-enhancing T2/FLAIR abnormality', 'short': 'NE', 'color': [255, 210, 63]},
                   {'value': 2, 'name': 'Contrast-enhancing tumor', 'short': 'CE', 'color': [255, 77, 77]}],
        'compartments': [{'key': 'ce', 'name': 'Contrast-enhancing tumor', 'labels': [2]},
                         {'key': 'ne', 'name': 'Non-enhancing T2/FLAIR abnormality', 'labels': [1]},
                         {'key': 'total', 'name': 'Total abnormality (CE + NE)', 'labels': [1, 2]}],
    }
    assert cfg['classNamesInModel'] == {'0': 'background', '1': 'edema', '2': 'km'}, cfg['classNamesInModel']
    return cfg


def export_brats2020(folder, out_dir, q8, fold=0):
    """One network of Isensee et al.'s BraTS 2020 nnU-Net (region-based, batch norm), Zenodo 4635763."""
    allow_numpy_in_checkpoints()
    from nnunet.training.model_restore import load_model_and_checkpoint_files
    from nnunet.inference.predict import load_postprocessing
    trainer, params = load_model_and_checkpoint_files(folder, (fold,), mixed_precision=False, checkpoint_name='model_final_checkpoint')
    trainer.load_checkpoint_ram(params[0], False)
    net = trainer.network
    net.do_ds = False
    net.cpu().float().eval()
    assert isinstance(net.inference_apply_nonlin, torch.nn.Sigmoid), 'expected a region-based (sigmoid) trainer'
    plans, stage = trainer.plans, trainer.stage
    stage_plans = plans['plans_per_stage'][stage]
    patch = [int(p) for p in trainer.patch_size]
    n_in = int(trainer.num_input_channels)
    norm, use_mask = plans['normalization_schemes'], plans['use_mask_for_norm']
    pp = None
    pjson = os.path.join(folder, 'postprocessing.json')
    if os.path.isfile(pjson):
        fw, mv = load_postprocessing(pjson)
        pp = {'forWhichClasses': [list(c) if isinstance(c, (list, tuple)) else int(c) for c in fw],
              'minValidObjectSizes': {str(k): v for k, v in (mv or {}).items()} if mv else None}
    export_onnx(net, n_in, patch, os.path.join(out_dir, 'full.onnx'), softmax=False, sigmoid=True)
    w = pack_weights(os.path.join(out_dir, 'full.onnx'), out_dir, q8)
    os.remove(os.path.join(out_dir, 'full.onnx'))
    sep = trainer.segmentation_export_params if hasattr(trainer, 'segmentation_export_params') else {}
    cfg = {
        'id': 'brats2020', 'addonFormat': 1, 'short': 'nnU-Net BraTS',
        'name': 'nnU-Net BraTS 2020', 'version': 'one network (fold %d) of the BraTS 2020 ensemble, Zenodo 4635763' % fold, 'nnunet': 1,
        'trainer': type(trainer).__name__, 'stage': stage, 'fold': fold,
        'inputChannels': ['T1', 'T1c', 'T2', 'FLAIR'], 'numClasses': len(trainer.regions),
        'regions': {k: list(v) for k, v in trainer.regions.items()}, 'regionClassOrder': [int(c) for c in trainer.regions_class_order],
        'transposeForward': [int(a) for a in plans['transpose_forward']], 'transposeBackward': [int(a) for a in plans['transpose_backward']],
        'targetSpacing': [float(s) for s in stage_plans['current_spacing']], 'patchSize': patch,
        'normalization': [{'scheme': norm[i], 'useMask': bool(use_mask[i])} for i in range(n_in)],
        'mirrorAxes': [int(a) for a in trainer.data_aug_params.get('mirror_axes', (0, 1, 2))], 'stepSize': 0.5,
        'aggregate': 'sigmoid', 'anisotropyThreshold': 3,
        'resampleData': {'order': 3, 'orderZ': 0}, 'resampleProb': {'order': int(sep.get('interpolation_order', 1)), 'orderZ': int(sep.get('interpolation_order_z', 0))},
        'postprocessing': pp, 'onnx': 'model.onnx', 'weights': w,
        # nnU-Net Task082 labels (1 edema, 2 non-enhancing/necrosis, 3 enhancing) back to the BraTS
        # convention, as nnunet.dataset_conversion convert_labels_back_to_BraTS does
        'labelMap': {'1': 2, '2': 1, '3': 4},
        'labels': [{'value': 1, 'name': 'Necrosis / non-enhancing tumor core (NCR)', 'short': 'NCR', 'color': [80, 160, 255]},
                   {'value': 2, 'name': 'Peritumoral edema / infiltrated tissue (ED)', 'short': 'ED', 'color': [255, 210, 63]},
                   {'value': 4, 'name': 'Enhancing tumor (ET)', 'short': 'ET', 'color': [255, 77, 77]}],
        'compartments': [{'key': 'et', 'name': 'Enhancing tumor (ET)', 'labels': [4]},
                         {'key': 'ncr', 'name': 'Necrosis / non-enhancing core (NCR)', 'labels': [1]},
                         {'key': 'ed', 'name': 'Edema / infiltrated tissue (ED)', 'labels': [2]},
                         {'key': 'tc', 'name': 'Tumor core (TC = ET + NCR)', 'labels': [1, 4]},
                         {'key': 'wt', 'name': 'Whole tumor (WT = ET + NCR + ED)', 'labels': [1, 2, 4]}],
        'citation': 'Isensee F, Jaeger PF, Full PM, Vollmuth P, Maier-Hein KH. nnU-Net for brain tumor segmentation. BrainLes 2020, LNCS 12659:118-132.',
        'license': 'CC BY 4.0',
    }
    assert cfg['regions'] == {'whole tumor': [1, 2, 3], 'tumor core': [2, 3], 'enhancing tumor': [3]}, cfg['regions']
    assert cfg['regionClassOrder'] == [1, 2, 3]
    return cfg


RAIDIONICS_LICENSE = """BSD 2-Clause License

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
"""


def _raidionics_network(folder, out_dir, q8, fold, classes):
    """Common part of the Raidionics v1.3 exports: read the model's pre_processing.ini, stop unless it is
    exactly the configuration the JavaScript implements, and pack the weights of one fold.
    folder: the model's sequence folder (e.g. 't2f', 't1c') holding pre_processing.ini and one
    sub-folder per fold. The JavaScript follows raidionicsseg 1.5.2 with the runtime settings the
    Raidionics app and raidionics_rads_lib use (thresholding, resample_first, no overlap, no
    test-time augmentation, first fold only)."""
    import configparser
    ini = configparser.ConfigParser()
    ini.read(os.path.join(folder, 'pre_processing.ini'))
    g = lambda s, k: ini[s][k].split('#')[0].strip()
    floats = lambda s, k: [float(x) for x in g(s, k).split(',')]
    pre = {
        'outputSpacing': floats('PreProcessing', 'output_spacing'), 'crop': g('PreProcessing', 'background_cropping'),
        'clipPercentile': floats('PreProcessing', 'intensity_clipping_range'), 'normalization': g('PreProcessing', 'normalization_method'),
        'patchSize': [int(v) for v in floats('Training', 'patch_size')], 'patchOffset': [int(v) for v in floats('Training', 'patch_offset')],
        'numClasses': int(g('Training', 'nb_classes')), 'thresholds': floats('Training', 'optimal_thresholds'),
        'activation': g('Training', 'activation_layer_type'), 'activationIncluded': g('Training', 'activation_layer_included').lower() == 'true',
        'channelsOrder': g('PreProcessing', 'channels_order'), 'backend': g('Default', 'training_backend'),
        'numberInputs': int(g('PreProcessing', 'number_inputs')), 'swapInput': g('PreProcessing', 'swap_training_input').lower() == 'true',
        'biasCorrection': g('MRI', 'perform_bias_correction').lower() == 'true', 'classes': g('Training', 'classes'),
    }
    # the JavaScript implements exactly this configuration; anything else must be checked first
    expect = {'outputSpacing': [1.0, 1.0, 1.0], 'crop': 'minimum', 'clipPercentile': [0.0, 99.5], 'normalization': 'zeromean_nonzero',
              'patchSize': [160, 160, 160], 'patchOffset': [0, 0, 0], 'numClasses': 1, 'thresholds': [0.5], 'activation': 'sigmoid',
              'activationIncluded': False, 'channelsOrder': 'channels_first', 'backend': 'Torch', 'numberInputs': 1, 'swapInput': False,
              'biasCorrection': False, 'classes': classes}
    assert pre == expect, {k: (pre[k], expect[k]) for k in expect if pre[k] != expect[k]}
    src = os.path.join(folder, str(fold), 'model.onnx')
    m = onnx.load(src, load_external_data=False)
    ins = [i for i in m.graph.input if i.name not in {t.name for t in m.graph.initializer}]
    assert [i.name for i in ins] == ['input'] and [d.dim_value for d in ins[0].type.tensor_type.shape.dim] == [1, 1, 160, 160, 160]
    assert [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim] == [1, 1, 160, 160, 160]
    w = pack_weights(src, out_dir, q8, small_f32=True)
    return pre, w


RAIDIONICS_CITATION = ('Bouget D, et al. Raidionics: an open software for pre- and postoperative central nervous system tumor '
                       'segmentation and standardized reporting. Sci Rep 2023;13. doi:10.1038/s41598-023-42048-7.')


def export_raidionics_flair(folder, out_dir, q8, fold=0):
    """Raidionics MRI_FLAIRChanges (t2f): FLAIR hyperintensity from FLAIR alone, one fold.
    folder: the 't2f' folder from Raidionics-MRI_FLAIRChanges-v13.zip (Raidionics-models v1.3.0-rc)."""
    pre, w = _raidionics_network(folder, out_dir, q8, fold, 'FLAIRChanges')
    return {
        'id': 'raidionics-flair', 'addonFormat': 1, 'kind': 'raidionics', 'short': 'Raidionics FLAIR',
        'name': 'Raidionics FLAIR hyperintensity',
        'version': 'MRI_FLAIRChanges (t2f), network of fold %d, Raidionics-models v1.3.0-rc' % fold, 'fold': fold,
        'trainer': 'AGU-Net (attention U-Net), PyTorch; preprocessing and inference as raidionicsseg 1.5.2',
        'inputChannels': ['FLAIR'], 'tta': False,
        'raidionics': {**pre, 'overlapRatio': 0.0, 'reconstruction': 'thresholding', 'reconstructionOrder': 'resample_first',
                       'refinement': 'brain_overlap'},
        'onnx': 'model.onnx', 'weights': w,
        'labels': [{'value': 1, 'name': 'FLAIR hyperintensity', 'short': 'FLAIR', 'color': [255, 210, 63]}],
        'compartments': [{'key': 'flair', 'name': 'FLAIR hyperintensity (tumor and edema)', 'labels': [1]}],
        'tableNote': 'One class: the whole FLAIR hyperintensity. Enhancing tumor, necrosis and edema are not separated.',
        'citation': ('Faanes MG, Bouget D, Jakola AS, et al. A unified FLAIR hyperintensity segmentation model for various CNS tumor '
                     'types and acquisition time points. Sci Rep 2026;16:19707. doi:10.1038/s41598-026-48496-1. ' + RAIDIONICS_CITATION),
        'license': 'BSD 2-Clause (Copyright (c) 2023, David Bouget)', 'licenseText': RAIDIONICS_LICENSE,
        'sourceUrl': 'https://github.com/raidionics/Raidionics-models/releases/tag/v1.3.0-rc',
    }


def export_raidionics_core(folder, out_dir, q8, fold=0):
    """Raidionics MRI_TumorCore (t1c): preoperative contrast-enhancing tumor core (enhancing tissue,
    necrosis and cysts) of gliomas, meningiomas and metastases from T1c alone, one fold.
    folder: the 't1c' folder from Raidionics-MRI_TumorCore-v13.zip (Raidionics-models v1.3.0-rc)."""
    pre, w = _raidionics_network(folder, out_dir, q8, fold, 'Tumor')
    return {
        'id': 'raidionics-core', 'addonFormat': 1, 'kind': 'raidionics', 'short': 'Raidionics core',
        'name': 'Raidionics tumor core',
        'version': 'MRI_TumorCore (t1c), network of fold %d, Raidionics-models v1.3.0-rc' % fold, 'fold': fold,
        'trainer': 'Attention U-Net, PyTorch; preprocessing and inference as raidionicsseg 1.5.2',
        'inputChannels': ['T1c'], 'tta': False,
        'raidionics': {**pre, 'overlapRatio': 0.0, 'reconstruction': 'thresholding', 'reconstructionOrder': 'resample_first',
                       'refinement': 'brain_overlap'},
        'onnx': 'model.onnx', 'weights': w,
        'labels': [{'value': 1, 'name': 'Tumor core', 'short': 'Core', 'color': [255, 77, 77]}],
        'compartments': [{'key': 'core', 'name': 'Tumor core (enhancing tissue, necrosis, cysts)', 'labels': [1]}],
        'tableNote': ('One class: the contrast-enhancing tumor core, with necrosis and cysts, for gliomas, meningiomas and '
                      'metastases. Edema is not included; the Raidionics FLAIR model gives the FLAIR hyperintensity.'),
        'citation': ('Bouget D, et al. Automatic and standardized reporting of perioperative MRIs in patients with central nervous '
                     'system tumors. Front Neurol 2026;16. doi:10.3389/fneur.2025.1707481. ' + RAIDIONICS_CITATION),
        'license': 'BSD 2-Clause (Copyright (c) 2023, David Bouget)', 'licenseText': RAIDIONICS_LICENSE,
        'sourceUrl': 'https://github.com/raidionics/Raidionics-models/releases/tag/v1.3.0-rc',
    }


def _nnunetv2_network(folder, fold, out_dir, q8):
    """Load one fold of an nnU-Net v2 trained model folder with nnU-Net's own predictor, export the
    network (logits) and pack its weights. The checkpoint is a pickle: it is first opened with
    torch.load(weights_only=True), which only accepts plain tensors, numbers and strings, so that
    nothing else in it can run when nnU-Net then loads it."""
    os.environ['nnUNet_compile'] = 'F'
    allow_numpy_in_checkpoints()
    torch.load(os.path.join(folder, 'fold_%s' % fold, 'checkpoint_final.pth'), map_location='cpu', weights_only=True)
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, perform_everything_on_device=False,
                           device=torch.device('cpu'), verbose=False)
    pred.initialize_from_trained_model_folder(folder, (fold,) if fold != 'all' else 'all', 'checkpoint_final.pth')
    net = pred.network
    if pred.list_of_parameters:
        net.load_state_dict(pred.list_of_parameters[0])
    net.cpu().float().eval()
    if hasattr(net, 'decoder') and hasattr(net.decoder, 'deep_supervision'):
        net.decoder.deep_supervision = False
    cm, pm = pred.configuration_manager, pred.plans_manager
    lm = pm.get_label_manager(pred.dataset_json)
    patch = [int(p) for p in cm.patch_size]
    n_in = len(pred.dataset_json['channel_names'])
    export_onnx(net, n_in, patch, os.path.join(out_dir, 'full.onnx'), softmax=False)
    w = pack_weights(os.path.join(out_dir, 'full.onnx'), out_dir, q8)
    os.remove(os.path.join(out_dir, 'full.onnx'))
    rk, rp = cm.configuration['resampling_fn_data_kwargs'], cm.configuration['resampling_fn_probabilities_kwargs']
    norms, use_mask = cm.normalization_schemes, cm.use_mask_for_norm
    common = {
        'nnunet': 2, 'numClasses': int(lm.num_segmentation_heads), 'regionBased': bool(lm.has_regions),
        'transposeForward': [int(a) for a in pm.transpose_forward], 'transposeBackward': [int(a) for a in pm.transpose_backward],
        'targetSpacing': [float(s) for s in cm.spacing], 'patchSize': patch,
        'normalization': [{'scheme': norms[i], 'useMask': bool(use_mask[i])} for i in range(n_in)],
        'mirrorAxes': [int(a) for a in (pred.allowed_mirroring_axes or [])], 'stepSize': 0.5, 'aggregate': 'logits', 'anisotropyThreshold': 3,
        'resampleData': {'order': int(rk.get('order', 3)), 'orderZ': int(rk.get('order_z', 0))},
        'resampleProb': {'order': int(rp.get('order', 1)), 'orderZ': int(rp.get('order_z', 0))},
        'onnx': 'model.onnx', 'weights': w, 'labelsFromDataset': pred.dataset_json.get('labels'),
    }
    return pred, common


VS_CITATION = ('Kujawa A, Dorent R, Connor S, et al. Deep learning for automatic segmentation of vestibular schwannoma: a '
               'retrospective study from multi-center routine MRI. Front Comput Neurosci 2024;18:1365727. '
               'doi:10.3389/fncom.2024.1365727. Models: Zenodo, doi:10.5281/zenodo.10363647.')
VS_MODELS = {
    # id: (dataset, input, short name, name, table note)
    'vs-t1c': ('Dataset910_VSMCRCT1', 'T1c', 'VS T1c', 'Vestibular schwannoma (T1c)',
               'One class: the vestibular schwannoma (intra- and extrameatal). Trained on contrast-enhanced T1 from routine '
               'scans of 10 UK hospitals (patients with one sporadic tumor; NF2 excluded). The images are not brain-extracted.'),
    'vs-t2': ('Dataset911_VSMCRCT2', 'T2', 'VS T2', 'Vestibular schwannoma (T2)',
              'One class: the vestibular schwannoma (intra- and extrameatal). Trained on T2-weighted images, almost all '
              'high-resolution (under 1 mm in every direction, e.g. CISS/FIESTA), from routine scans of 10 UK hospitals '
              '(patients with one sporadic tumor; NF2 excluded). Only the largest connected region is kept, as in the '
              "model's own postprocessing. The images are not brain-extracted."),
}


def export_vs(folder, out_dir, q8, which, fold=0):
    """One network (3d_fullres, one fold) of the KCL vestibular schwannoma nnU-Net v2 models
    (Kujawa et al. 2024, MC-RC models, Zenodo 10363647). folder: the model's
    nnUNetTrainer__nnUNetPlans__3d_fullres folder (plans.json, dataset.json, fold_<n>/checkpoint_final.pth)."""
    dataset, role, short, name, note = VS_MODELS[which]
    pred, common = _nnunetv2_network(folder, fold, out_dir, q8)
    assert pred.dataset_json['labels'] == {'background': 0, 'Tumor': 1}, pred.dataset_json['labels']
    assert len(pred.dataset_json['channel_names']) == 1
    assert common['normalization'] == [{'scheme': 'ZScoreNormalization', 'useMask': False}], common['normalization']
    assert common['transposeForward'] == [0, 1, 2]
    # nnU-Net's postprocessing found on the cross-validation (crossval_results_folds_0_1_2_3_4/postprocessing.json):
    # only remove_all_but_largest_component_from_segmentation for label 1 is implemented (26-connected, as
    # acvl_utils labels with skimage's full connectivity)
    pp = None
    pj = os.path.join(folder, 'crossval_results_folds_0_1_2_3_4', 'postprocessing.json')
    if os.path.isfile(pj):
        p = json.load(open(pj))
        fns, kw = p['postprocessing_fns'], p['postprocessing_kwargs']
        if fns:
            assert fns == ['remove_all_but_largest_component_from_segmentation'] and kw == [{'labels_or_regions': [1]}], (fns, kw)
            pp = {'forWhichClasses': [1], 'connectivity': 26, 'from': 'nnU-Net v2 remove_all_but_largest_component_from_segmentation'}
    return {
        'id': which, 'addonFormat': 1, 'short': short, 'name': name,
        'version': '%s, nnU-Net v2 3d_fullres, network of fold %d, MC-RC models (Zenodo 10363647)' % (dataset, fold),
        'trainer': 'nnUNetTrainer', 'configuration': '3d_fullres', 'fold': fold,
        'inputChannels': [role], 'maskInput': False, **common, 'postprocessing': pp,
        'labels': [{'value': 1, 'name': 'Vestibular schwannoma', 'short': 'VS', 'color': [255, 77, 77]}],
        'compartments': [{'key': 'vs', 'name': 'Vestibular schwannoma', 'labels': [1]}],
        'compareRegions': {'whole': [1]},
        'tableNote': note, 'citation': VS_CITATION, 'license': 'CC BY 4.0',
        'sourceUrl': 'https://doi.org/10.5281/zenodo.10363647',
    }


class SafeMish(torch.nn.Module):
    """mish(x) = x * tanh(softplus(x)) = x * u / (u + 2) with u = n (n + 2), n = exp(x): only Exp, Add,
    Mul, Div and Clip, which ONNX Runtime Web runs on the GPU. exp is taken of min(x, 20), where the
    factor is already 1 in float32."""
    def forward(self, x):
        n = torch.exp(torch.clamp(x, max=20.0))
        u = n * (n + 2)
        return x * (u / (u + 2))


AURORA_MODELS = {
    # id: (weights file stem, inputs, short name, name, table note)
    'aurora-t1c': ('t1c-o', ['T1c'], 'AURORA T1c', 'AURORA brain metastases (T1c)',
                   'Metastasis = contrast-enhancing tumor with necrosis; edema from T1c alone is less reliable (the T1c + FLAIR '
                   'model is better for edema). Trained on preoperative brain metastases.'),
    'aurora-t1c-flair': ('t1c-fla', ['T1c', 'FLAIR'], 'AURORA T1c+FLAIR', 'AURORA brain metastases (T1c + FLAIR)',
                         'Metastasis = contrast-enhancing tumor with necrosis; edema = FLAIR hyperintensity around it. In the '
                         'published test, models with FLAIR found more false-positive lesions than the T1c-only model. '
                         'Trained on preoperative brain metastases.'),
}
AURORA_CITATION = ('Buchner JA, Kofler F, Etzel L, et al. Development and external validation of an MRI-based neural network for '
                   'brain metastasis segmentation in the AURORA multicenter study. Radiother Oncol 2023;178:109425. '
                   'doi:10.1016/j.radonc.2022.11.014. Buchner JA, Peeken JC, Etzel L, et al. Identifying core MRI sequences for '
                   'reliable automatic brain metastasis segmentation. Radiother Oncol 2023;188:109901. doi:10.1016/j.radonc.2023.109901.')


def export_aurora(tar, out_dir, q8, which):
    """One AURORA model (BrainLesion/AURORA, brainles_aurora 0.2.x; weights v1.0.0, Zenodo 10557069):
    MONAI BasicUNet as brainles_aurora's ModelHandler builds it, the 'best' weights of one input
    combination. tar: <combination>_best.tar from the weights archive. The network gives the two
    logits (whole lesion, metastasis); preprocessing, sliding window, test-time augmentation and
    thresholding are in app/js/lib/aurora.js."""
    from monai.networks.nets import BasicUNet
    stem, inputs, short, name, note = AURORA_MODELS[which]
    assert os.path.basename(tar) == stem + '_best.tar', tar
    net = BasicUNet(spatial_dims=3, in_channels=len(inputs), out_channels=2, features=(32, 32, 64, 128, 256, 32), dropout=0.1, act='mish')
    sd = torch.load(tar, map_location='cpu', weights_only=True)['model_state']
    net.load_state_dict({(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()})
    net.eval()
    roi = [192, 192, 32]
    # the same network with the ONNX-friendly mish; check that it gives the same logits
    x = torch.rand(1, len(inputs), *roi, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = net(x)
    n_mish = 0
    for mod in list(net.modules()):
        for cn, child in list(mod.named_children()):
            if type(child).__name__ == 'Mish':
                setattr(mod, cn, SafeMish()); n_mish += 1
    assert n_mish > 0
    # patch sizes divisible by 16 need no padding after upsampling: leave out UpCat's (zero) replicate
    # padding, which would otherwise be exported as a Pad with computed pads
    for mod in net.modules():
        if type(mod).__name__ == 'UpCat': mod.is_pad = False
    with torch.no_grad():
        d = float((net(x) - ref).abs().max())
    assert d < 1e-3, d
    export_onnx(net, len(inputs), roi, os.path.join(out_dir, 'full.onnx'), softmax=False)
    w = pack_weights(os.path.join(out_dir, 'full.onnx'), out_dir, q8, small_f32=True)
    os.remove(os.path.join(out_dir, 'full.onnx'))
    return {
        'id': which, 'addonFormat': 1, 'kind': 'aurora', 'short': short, 'name': name,
        'version': '%s_best, AURORA weights v1.0.0 (Zenodo 10557069), inference as brainles_aurora 0.2.7' % stem,
        'trainer': 'MONAI BasicUNet (features 32, 32, 64, 128, 256, 32; mish), PyTorch',
        'inputChannels': inputs, 'tta': True,
        'aurora': {
            # brainles_aurora: ScaleIntensityRangePercentiles(0.5, 99.5, 0, 1, clip, channel-wise) on the image as
            # given, MONAI SlidingWindowInferer(roi 192x192x32, overlap 0.5, gaussian, replicate padding),
            # sigmoid >= 0.5; channel 0 whole lesion (label 1), channel 1 metastasis (label 2).
            # The images it was trained on were brain-extracted, 1 mm, in BraTS (SRI-24) space: the
            # workbench gives it the brain-extracted images on a 1 mm grid with BraTS axis order
            # (i to the left, j to posterior, k to superior), 240 x 240 x 155 around the brain.
            'roiSize': roi, 'overlap': 0.5, 'blend': 'gaussian', 'sigmaScale': 0.125, 'padding': 'replicate',
            'percentiles': [0.5, 99.5], 'threshold': 0.5, 'outChannels': ['whole', 'metastasis'],
            'gridSpacing': [1.0, 1.0, 1.0], 'gridAxes': 'LPS', 'gridDims': [240, 240, 155],
            # brainles_aurora's TTA: the plain pass, 4 passes on the image with Gaussian noise (std 0.001)
            # and 4 x 2 passes on it flipped on the first and second axes, logits averaged over 13. Here:
            # the noise is left out, so the 13 passes reduce to 3 weighted 5 : 4 : 4.
            'tta': {'flipAxes': [0, 1], 'weights': [5, 4, 4]},
        },
        'onnx': 'model.onnx', 'weights': w,
        'labels': [{'value': 1, 'name': 'Edema', 'short': 'Edema', 'color': [255, 210, 63]},
                   {'value': 2, 'name': 'Metastasis (enhancing tumor and necrosis)', 'short': 'Met', 'color': [255, 77, 77]}],
        'compartments': [{'key': 'met', 'name': 'Metastasis (enhancing tumor and necrosis)', 'labels': [2]},
                         {'key': 'edema', 'name': 'Edema', 'labels': [1]},
                         {'key': 'total', 'name': 'Total (metastasis + edema)', 'labels': [1, 2]}],
        'compareRegions': {'whole': [1, 2], 'core': [2]},
        'tableNote': note, 'citation': AURORA_CITATION,
        'license': 'CC BY 4.0 (weights); brainles_aurora code Apache-2.0',
        'sourceUrl': 'https://doi.org/10.5281/zenodo.10557069',
    }


def export_hdbet(folder, out_dir, q8):
    os.environ['nnUNet_compile'] = 'F'
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, perform_everything_on_device=False,
                           device=torch.device('cpu'), verbose=False)
    pred.initialize_from_trained_model_folder(folder, 'all')
    net = pred.network
    if pred.list_of_parameters:
        net.load_state_dict(pred.list_of_parameters[0])
    net.cpu().float().eval()
    if hasattr(net, 'decoder') and hasattr(net.decoder, 'deep_supervision'):
        net.decoder.deep_supervision = False
    cm, pm = pred.configuration_manager, pred.plans_manager
    lm = pm.get_label_manager(pred.dataset_json)
    patch = [int(p) for p in cm.patch_size]
    n_in = len(pred.dataset_json['channel_names'])
    n_cls = lm.num_segmentation_heads
    norms = cm.normalization_schemes
    use_mask = cm.use_mask_for_norm
    export_onnx(net, n_in, patch, os.path.join(out_dir, 'full.onnx'), softmax=False)
    w = pack_weights(os.path.join(out_dir, 'full.onnx'), out_dir, q8)
    os.remove(os.path.join(out_dir, 'full.onnx'))
    rk, rp = cm.configuration['resampling_fn_data_kwargs'], cm.configuration['resampling_fn_probabilities_kwargs']
    cfg = {
        'name': 'HD-BET', 'version': '2.0, v1.5 parameters (Zenodo 14445620)', 'nnunet': 2,
        'configuration': torch.load(os.path.join(folder, 'fold_all', 'checkpoint_final.pth'), map_location='cpu', weights_only=False)['init_args']['configuration'],
        'inputChannels': ['image'], 'numClasses': int(n_cls), 'regionBased': bool(lm.has_regions),
        'transposeForward': [int(a) for a in pm.transpose_forward], 'transposeBackward': [int(a) for a in pm.transpose_backward],
        'targetSpacing': [float(s) for s in cm.spacing], 'patchSize': patch,
        'normalization': [{'scheme': norms[i], 'useMask': bool(use_mask[i])} for i in range(n_in)],
        'mirrorAxes': [int(a) for a in (pred.allowed_mirroring_axes or [])], 'stepSize': 0.5, 'aggregate': 'logits', 'anisotropyThreshold': 3,
        'resampleData': {'order': int(rk.get('order', 3)), 'orderZ': int(rk.get('order_z', 0))},
        'resampleProb': {'order': int(rp.get('order', 1)), 'orderZ': int(rp.get('order_z', 0))},
        'postprocessing': None, 'onnx': 'model.onnx', 'weights': w,
        'labelsFromDataset': pred.dataset_json.get('labels'),
    }
    return cfg


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model', choices=['hdglio', 'hdbet', 'brats2020', 'raidionics-flair', 'raidionics-core', *VS_MODELS, *AURORA_MODELS])
    ap.add_argument('folder')
    ap.add_argument('--q8', action='store_true', help='int8 weights (smaller, slightly less exact)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out_dir = a.out or os.path.join(OUT, a.model)
    os.makedirs(out_dir, exist_ok=True)
    fns = {'hdglio': export_hdglio, 'hdbet': export_hdbet, 'brats2020': export_brats2020,
           'raidionics-flair': export_raidionics_flair, 'raidionics-core': export_raidionics_core}
    for k in VS_MODELS: fns[k] = lambda f, o, q, k=k: export_vs(f, o, q, k)
    for k in AURORA_MODELS: fns[k] = lambda f, o, q, k=k: export_aurora(f, o, q, k)
    cfg = fns[a.model](a.folder, out_dir, a.q8)
    json.dump(cfg, open(os.path.join(out_dir, 'config.json'), 'w'), indent=1)
    sz = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    print(a.model, 'exported to', out_dir, '(%.1f MB)' % (sz / 1e6))
    print(json.dumps({k: v for k, v in cfg.items() if k != 'weights'}, indent=1))
