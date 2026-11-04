#!/usr/bin/env python3
"""Stand-in networks for plumbing tests of the page (NOT the real models): 1x1x1 convolutions
with the same input/output layout and config format as HD-BET and HD-GLIO.
Usage: make_standin_models.py <out dir>"""
import json, os, sys
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build'))
from export_models import pack_weights

OUT = sys.argv[1]
LABELS = [{'value': 1, 'name': 'Non-enhancing T2/FLAIR abnormality', 'short': 'NE'}, {'value': 2, 'name': 'Contrast-enhancing tumor', 'short': 'CE'}]
COMPS = [{'key': 'ce', 'name': 'Contrast-enhancing tumor', 'labels': [2]}, {'key': 'ne', 'name': 'Non-enhancing T2/FLAIR abnormality', 'labels': [1]},
         {'key': 'total', 'name': 'Total abnormality (CE + NE)', 'labels': [1, 2]}]

def conv_model(path, cin, cout, W, B, patch, softmax):
    w = numpy_helper.from_array(np.asarray(W, np.float32).reshape(cout, cin, 1, 1, 1), 'w')
    b = numpy_helper.from_array(np.asarray(B, np.float32), 'b')
    # pad the initialisers so they are packed like real weights (>= 64 bytes)
    filler = numpy_helper.from_array(np.zeros(32, np.float32), 'unused')
    nodes = [helper.make_node('Conv', ['input', 'w', 'b'], ['logits' if softmax else 'output'], kernel_shape=[1, 1, 1])]
    if softmax: nodes.append(helper.make_node('Softmax', ['logits'], ['output'], axis=1))
    g = helper.make_graph(nodes, 'standin', [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, cin, *patch])],
                          [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, cout, *patch])], [w, b, filler])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)]), path)

for name in ('hdbet', 'hdglio'):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    patch = [96, 96, 96]
    if name == 'hdbet':
        conv_model(os.path.join(d, 'full.onnx'), 1, 2, [0, 3], [0, -0.6], patch, False)
        cfg = {'name': 'HD-BET (stand-in)', 'version': 'stand-in', 'nnunet': 2, 'inputChannels': ['image'], 'numClasses': 2,
               'transposeForward': [0, 1, 2], 'transposeBackward': [0, 1, 2], 'targetSpacing': [1.0, 1.0, 1.0], 'patchSize': patch,
               'normalization': [{'scheme': 'ZScoreNormalization', 'useMask': False}], 'mirrorAxes': [0, 1, 2], 'stepSize': 0.5,
               'aggregate': 'logits', 'anisotropyThreshold': 3, 'resampleData': {'order': 3, 'orderZ': 0}, 'resampleProb': {'order': 1, 'orderZ': 0},
               'postprocessing': None, 'onnx': 'model.onnx'}
    else:
        conv_model(os.path.join(d, 'full.onnx'), 4, 3, [0, 0, 0, 0, 0, 0, 0, 2.2, -2.5, 2.5, 0, 0], [0, -3.2, -3.2], patch, True)
        cfg = {'name': 'HD-GLIO (stand-in)', 'version': 'stand-in', 'nnunet': 1, 'inputChannels': ['T1', 'T1c', 'T2', 'FLAIR'], 'numClasses': 3,
               'transposeForward': [0, 1, 2], 'transposeBackward': [0, 1, 2], 'targetSpacing': [1.0, 1.0, 1.0], 'patchSize': patch,
               'normalization': [{'scheme': 'nonCT', 'useMask': True}] * 4, 'mirrorAxes': [0, 1, 2], 'stepSize': 0.5, 'aggregate': 'softmax',
               'anisotropyThreshold': 3, 'resampleData': {'order': 3, 'orderZ': 0}, 'resampleProb': {'order': 1, 'orderZ': 0},
               'postprocessing': {'forWhichClasses': [[1, 2], 2]}, 'onnx': 'model.onnx', 'labels': LABELS, 'compartments': COMPS}
    cfg['weights'] = pack_weights(os.path.join(d, 'full.onnx'), d)
    os.remove(os.path.join(d, 'full.onnx'))
    json.dump(cfg, open(os.path.join(d, 'config.json'), 'w'), indent=1)
    print(name, sorted(os.listdir(d)))
