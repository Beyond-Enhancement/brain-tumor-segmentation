#!/usr/bin/env python3
"""Make the Raidionics FLAIR network (fixed 160^3 input) accept any patch size divisible by 16,
plus a 64^3 test input and the ONNX Runtime (CPU, fp32) output, so the WebGPU backend can be
checked with the real weights in a software-GPU browser (test/out/webgpu_site/index.html).
Usage: webgpu_raidionics_export.py <t2f folder> <out dir> [<js debug dir>]
With a debug dir from test/unit/raidionics_test.mjs, the test input is a 64^3 block of that
case's preprocessed FLAIR around the lesion instead of noise."""
import os, sys, numpy as np, onnx, onnxruntime as ort
src, OUT = sys.argv[1], sys.argv[2]
m = onnx.load(os.path.join(src, '0', 'model.onnx'))
for t in (m.graph.input[0], m.graph.output[0]):
    for k, d in enumerate(t.type.tensor_type.shape.dim[2:]):
        d.ClearField('dim_value'); d.dim_param = 'abc'[k]
del m.graph.value_info[:]
m.ir_version = 10
path = os.path.join(OUT, 'raid_dyn.onnx')
onnx.save(m, path)
shape = (1, 1, 64, 64, 64)
x = np.random.default_rng(0).normal(size=shape).astype(np.float32)
if len(sys.argv) > 3:
    import json
    N = tuple(json.load(open(os.path.join(sys.argv[3], 'js_meta.json')))['N'])
    pre = np.fromfile(os.path.join(sys.argv[3], 'js_pre.raw'), dtype='<f4').reshape(N)
    prob = np.fromfile(os.path.join(sys.argv[3], 'js_prob.raw'), dtype='<f4').reshape(N)
    c = [int(round(v.mean())) for v in np.nonzero(prob > 0.5)]
    s0 = [min(max(ci - 32, 0), n - 64) for ci, n in zip(c, N)]
    x = np.ascontiguousarray(pre[s0[0]:s0[0] + 64, s0[1]:s0[1] + 64, s0[2]:s0[2] + 64])[None, None].astype(np.float32)
y = ort.InferenceSession(path, providers=['CPUExecutionProvider']).run(None, {'input': x})[0]
x.tofile(os.path.join(OUT, 'raid_x.raw')); y.astype(np.float32).tofile(os.path.join(OUT, 'raid_y.raw'))
print('raid', shape, '->', y.shape, 'logit range %.3f..%.3f' % (y.min(), y.max()), 'positive %.3f%%' % (100 * (y > 0).mean()))
