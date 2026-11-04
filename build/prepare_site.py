#!/usr/bin/env python3
"""Assemble launcher/site/ (the files embedded in the executable) from app/.

Files that shrink by more than 10 % are stored gzip-compressed and listed in gzip.json; the
launcher serves them with Content-Encoding: gzip. Fails if a model is missing.
"""
import gzip, json, os, shutil, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(REPO, 'app')
SITE = os.path.join(REPO, 'launcher', 'site')
SKIP_DIRS = {'__pycache__'}


def main():
    # BTSW_TEST_MODELS=<dir> embeds other networks (test builds only; never for downloads/)
    test_models = os.environ.get('BTSW_TEST_MODELS')
    mdir = test_models or os.path.join(APP, 'models')
    for m in ('hdbet', 'hdglio'):
        for f in ('config.json', 'model.onnx', 'weights.pack'):
            if not os.path.isfile(os.path.join(mdir, m, f)):
                sys.exit(f'prepare_site: {mdir}/{m}/{f} is missing (run build/export_models.py)')
    if os.path.isdir(SITE):
        shutil.rmtree(SITE)
    gz, raw_total, stored_total = [], 0, 0
    sources = []
    for root, dirs, files in os.walk(APP):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not (test_models and root == APP and d == 'models'))
        for f in sorted(files):
            sources.append((os.path.join(root, f), os.path.relpath(os.path.join(root, f), APP).replace(os.sep, '/')))
    if test_models:
        print('prepare_site: TEST BUILD with networks from', test_models)
        for root, dirs, files in os.walk(test_models):
            for f in sorted(files):
                sources.append((os.path.join(root, f), 'models/' + os.path.relpath(os.path.join(root, f), test_models).replace(os.sep, '/')))
    for src, rel in sources:
            f = os.path.basename(src)
            dst = os.path.join(SITE, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            data = open(src, 'rb').read()
            raw_total += len(data)
            if len(data) > 4096 and not f.endswith(('.gz', '.woff2', '.png')):
                comp = gzip.compress(data, compresslevel=9, mtime=0)
                if len(comp) < 0.9 * len(data):
                    open(dst + '.gz', 'wb').write(comp)
                    gz.append(rel)
                    stored_total += len(comp)
                    continue
            shutil.copyfile(src, dst)
            stored_total += len(data)
    json.dump(gz, open(os.path.join(SITE, 'gzip.json'), 'w'))
    print('site ready: %.1f MB of files stored as %.1f MB (%d gzip-compressed)' % (raw_total / 1e6, stored_total / 1e6, len(gz)))


if __name__ == '__main__':
    main()
