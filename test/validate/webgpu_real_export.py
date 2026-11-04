#!/usr/bin/env python3
"""Export the real HD-GLIO and HD-BET networks with dynamic spatial axes plus a small random
input and the ONNX Runtime (CPU, fp32) output, so the WebGPU backend can be checked with the
real weights in a software-GPU browser where full-size patches would take hours.
Usage: webgpu_real_export.py <hd_glio_params> <hdbet_params> <out dir>"""
import os, sys, numpy as np, torch, onnx, onnxruntime as ort
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build'))
import export_models as E
G, B, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
E.allow_numpy_in_checkpoints()
from nnunet.training.model_restore import load_model_and_checkpoint_files
tr, params = load_model_and_checkpoint_files(G, (0,), mixed_precision=False, checkpoint_name='model_final_checkpoint')
tr.load_checkpoint_ram(params[0], False); g = tr.network; g.do_ds = False; g.cpu().float().eval()
os.environ['nnUNet_compile'] = 'F'
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
p = nnUNetPredictor(device=torch.device('cpu'), perform_everything_on_device=False)
p.initialize_from_trained_model_folder(B, 'all'); b = p.network; b.load_state_dict(p.list_of_parameters[0]); b.cpu().float().eval()
rng = np.random.default_rng(0)
for name, net, shape, sm in (('hdglio', g, (1, 4, 64, 64, 64), True), ('hdbet', b, (1, 1, 32, 64, 64), False)):
    m = E.SoftmaxWrap(net) if sm else net
    path = os.path.join(OUT, name + '_dyn.onnx')
    with torch.no_grad():
        torch.onnx.export(m, torch.zeros(*shape), path, opset_version=17, input_names=['input'], output_names=['output'], dynamo=False,
                          dynamic_axes={'input': {2: 'a', 3: 'b', 4: 'c'}, 'output': {2: 'a', 3: 'b', 4: 'c'}})
    mo = onnx.load(path); mo.ir_version = 10; onnx.save(mo, path)
    x = rng.normal(size=shape).astype(np.float32)
    y = ort.InferenceSession(path, providers=['CPUExecutionProvider']).run(None, {'input': x})[0]
    x.tofile(os.path.join(OUT, name + '_x.raw')); y.astype(np.float32).tofile(os.path.join(OUT, name + '_y.raw'))
    print(name, shape, '->', y.shape, 'output range %.3f..%.3f' % (y.min(), y.max()))
