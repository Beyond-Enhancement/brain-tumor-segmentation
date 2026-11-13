#!/usr/bin/env python3
"""Recompute the workbench's model comparison from its exported files and check its CSV.

Usage: check_model_comparison.py <e2e out dir> [--same ID=FILE ...]
Reads <dir>/model_comparison_volumes.csv and <dir>/<id>_segmentation.nii.gz for each model in it,
recomputes every compartment volume, shared-region volume, and for each pair and region Dice and
the volumes found by both, by the first model only and by the second only, and reports the largest
difference from the CSV. Every agreement map exported (agreement_<a>_vs_<b>_<region>.nii.gz) is
checked against the two segmentations. --same compares a model's segmentation with another file
(for example the same model run on its own) voxel by voxel."""
import csv, glob, os, re, sys
import numpy as np, nibabel as nib

D = sys.argv[1]
same = dict(a.split('=', 1) for a in sys.argv[sys.argv.index('--same') + 1:]) if '--same' in sys.argv else {}
REGIONS = {  # app/js/worker.js MODEL_REGIONS
    'hdglio': {'whole': [1, 2], 'enh': [2], 'nonenh': [1]},
    'brats2020': {'whole': [1, 2, 4], 'core': [1, 4], 'enh': [4], 'nonenh': [1, 2]},
    'raidionics-flair': {'whole': [1]},
    'raidionics-core': {'core': [1]},
}
NAMES = {'Whole abnormality': 'whole', 'Tumor core': 'core', 'Enhancing tumor': 'enh', 'Non-enhancing abnormality': 'nonenh'}

rows = [r for r in csv.reader(open(os.path.join(D, 'model_comparison_volumes.csv'))) if r and not r[0].startswith('#')]
head, rows = rows[0], [dict(zip(rows[0], r)) for r in rows[1:]]
ids = list(dict.fromkeys(r['model'] for r in rows if r['table'] == 'volume' and r['model'] != 'hdbet'))
segs, vv = {}, None
for i in ids:
    img = nib.load(os.path.join(D, i + '_segmentation.nii.gz'))
    segs[i] = np.asarray(img.dataobj).astype(np.int32)
    v = abs(np.linalg.det(img.affine[:3, :3])) / 1000
    assert vv is None or abs(v - vv) < 1e-9
    vv = v
worst = {'volume': 0.0, 'region': 0.0, 'dice': 0.0, 'overlap': 0.0}
for r in rows:
    if r['table'] == 'volume' and r['model'] in segs:
        n = int(np.isin(segs[r['model']], [int(x) for x in r['labels'].split('+')]).sum())
        assert n == int(r['voxels']), (r, n)
        worst['volume'] = max(worst['volume'], abs(n * vv - float(r['volume_cm3'])))
    elif r['table'] == 'region':
        k = NAMES[r['compartment_or_region']]
        n = int(np.isin(segs[r['model']], REGIONS[r['model']][k]).sum())
        assert n == int(r['voxels']), (r, n)
        worst['region'] = max(worst['region'], abs(n * vv - float(r['volume_cm3'])))
    elif r['table'] == 'agreement':
        k = NAMES[r['compartment_or_region']]
        a = np.isin(segs[r['model']], REGIONS[r['model']][k]); b = np.isin(segs[r['model_b']], REGIONS[r['model_b']][k])
        both, ao, bo = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum())
        dice = 2 * both / (2 * both + ao + bo) if (2 * both + ao + bo) else None
        if dice is not None: worst['dice'] = max(worst['dice'], abs(dice - float(r['dice'])))
        for x, c in ((both, 'both_cm3'), (ao, 'model_only_cm3'), (bo, 'model_b_only_cm3')):
            worst['overlap'] = max(worst['overlap'], abs(x * vv - float(r[c])))
        print('%-16s vs %-16s %-9s Dice %.4f  both %.3f  first only %.3f  second only %.3f cm3' % (r['model'], r['model_b'], k, dice or 0, both * vv, ao * vv, bo * vv))
for f in glob.glob(os.path.join(D, 'agreement_*_vs_*_*.nii.gz')):
    m = re.match(r'agreement_(.+)_vs_(.+)_(whole|core|enh|nonenh)\.nii\.gz$', os.path.basename(f))
    a, b, k = m.groups()
    want = np.isin(segs[a], REGIONS[a][k]).astype(np.int32) + 2 * np.isin(segs[b], REGIONS[b][k]).astype(np.int32)
    got = np.asarray(nib.load(f).dataobj).astype(np.int32)
    print('agreement map', os.path.basename(f), 'differing voxels', int((got != want).sum()))
for i, f in same.items():
    o = np.asarray(nib.load(f).dataobj).astype(np.int32)
    print('%s vs %s: differing voxels %d of %d' % (i, f, int((o != segs[i]).sum()), o.size))
print('largest differences from the CSV (cm3, Dice):', {k: float('%.2g' % v) for k, v in worst.items()})
