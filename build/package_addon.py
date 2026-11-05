#!/usr/bin/env python3
"""Package an exported add-on model (config.json, model.onnx, weights.pack) as one .btsmodel file
(a zip with fixed timestamps, so the same export gives a byte-identical file).
Usage: package_addon.py <exported folder> <out.btsmodel>"""
import json, os, sys, zipfile
src, out = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(src, 'config.json')))
notice = (f"{cfg['name']} - add-on model for the Brain Tumor Segmentation Workbench\n\n"
          f"{cfg.get('version', '')}\nTrainer: {cfg.get('trainer', '')}\n\n"
          f"License of the network parameters: {cfg.get('license', 'see source')}\n"
          f"Please cite: {cfg.get('citation', '')}\n\n"
          "Place this file in the same folder as the workbench program; it then appears under\n"
          "Processing options > Segmentation model. Or open it with 'Add model file'.\n")
files = [('config.json', open(os.path.join(src, 'config.json'), 'rb').read()), ('README.txt', notice.encode('ascii', 'replace')),
         (cfg['onnx'], open(os.path.join(src, cfg['onnx']), 'rb').read()), (cfg['weights']['pack'], open(os.path.join(src, cfg['weights']['pack']), 'rb').read())]
with zipfile.ZipFile(out, 'w') as z:
    for name, data in files:
        zi = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = 0o644 << 16
        z.writestr(zi, data, compresslevel=9)
print(out, '%.1f MB' % (os.path.getsize(out) / 1e6))
