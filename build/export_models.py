#!/usr/bin/env python3
"""Export the trained HD-GLIO (nnU-Net v1) and HD-BET (nnU-Net v2) models for the workbench.

For each model this writes app/models/<name>/:
  model.onnx    network graph (weights stored externally)
  weights.pack  weights, fp16 (default) or int8 per output channel (--q8)
  config.json   everything the JavaScript nnU-Net needs, read from the model's own plans

Usage:
  build/export_models.py hdglio <hd_glio_params folder>
  build/export_models.py hdbet  <hd-bet release folder (contains fold_all/)>
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


def export_onnx(net, in_ch, patch, path, softmax):
    net.eval()
    model = SoftmaxWrap(net) if softmax else net
    x = torch.zeros(1, in_ch, *patch)
    with torch.no_grad():
        torch.onnx.export(model, x, path, opset_version=17, input_names=['input'], output_names=['output'], dynamo=False,
                          do_constant_folding=True)


def pack_weights(onnx_path, out_dir, q8=False):
    """Move all initializers to an external fp32 blob, then write a compact pack + index."""
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
        if init.data_type == onnx.TensorProto.FLOAT and ln >= 64:
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
        'labels': [{'value': 1, 'name': 'Non-enhancing T2/FLAIR abnormality', 'short': 'NE'},
                   {'value': 2, 'name': 'Contrast-enhancing tumor', 'short': 'CE'}],
        'compartments': [{'key': 'ce', 'name': 'Contrast-enhancing tumor', 'labels': [2]},
                         {'key': 'ne', 'name': 'Non-enhancing T2/FLAIR abnormality', 'labels': [1]},
                         {'key': 'total', 'name': 'Total abnormality (CE + NE)', 'labels': [1, 2]}],
    }
    assert cfg['classNamesInModel'] == {'0': 'background', '1': 'edema', '2': 'km'}, cfg['classNamesInModel']
    return cfg


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
    ap.add_argument('model', choices=['hdglio', 'hdbet'])
    ap.add_argument('folder')
    ap.add_argument('--q8', action='store_true', help='int8 weights (smaller, slightly less exact)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out_dir = a.out or os.path.join(OUT, a.model)
    os.makedirs(out_dir, exist_ok=True)
    cfg = (export_hdglio if a.model == 'hdglio' else export_hdbet)(a.folder, out_dir, a.q8)
    json.dump(cfg, open(os.path.join(out_dir, 'config.json'), 'w'), indent=1)
    sz = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    print(a.model, 'exported to', out_dir, '(%.1f MB)' % (sz / 1e6))
    print(json.dumps({k: v for k, v in cfg.items() if k != 'weights'}, indent=1))
