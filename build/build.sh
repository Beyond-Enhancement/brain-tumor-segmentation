#!/usr/bin/env bash
# Build the standalone executables into dist/.
#   dist/BrainTumorSegmentationWorkbench.exe                 Windows 10/11, x64
#   dist/BrainTumorSegmentationWorkbench-macOS-AppleSilicon.zip  macOS 11+, Apple M-series
#   dist/BrainTumorSegmentationWorkbench-macOS-Intel.zip     macOS 11+, Intel
#   dist/btsw-linux-amd64                                    Linux, x64 (used by the tests)
#   dist/*.btsmodel                                          add-on models, if exported into addons/
# Needs Go >= 1.22 and Python 3 with Pillow. Cross-compiles from any OS; no cgo.
# The models must have been exported first (build/export_models.py).
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION="1.5.0"  # keep in sync with appVer in launcher/main.go
NAME="Brain Tumor Segmentation Workbench"

python3 build/prepare_site.py
python3 build/make_syso.py launcher/rsrc_windows_amd64.syso "$VERSION"

mkdir -p dist build/tmp
export CGO_ENABLED=0
LDF="-s -w"
( cd launcher
  GOOS=windows GOARCH=amd64 go build -trimpath -buildvcs=false -ldflags "$LDF -H windowsgui" -o ../dist/BrainTumorSegmentationWorkbench.exe .
  GOOS=darwin  GOARCH=arm64 go build -trimpath -buildvcs=false -ldflags "$LDF" -o ../build/tmp/btsw-darwin-arm64 .
  GOOS=darwin  GOARCH=amd64 go build -trimpath -buildvcs=false -ldflags "$LDF" -o ../build/tmp/btsw-darwin-amd64 .
  GOOS=linux   GOARCH=amd64 go build -trimpath -buildvcs=false -ldflags "$LDF" -o ../dist/btsw-linux-amd64 .
)
mac_zip() {  # mac_zip <binary> <zip name>
  rm -rf build/tmp/zip && mkdir -p build/tmp/zip
  cp "$1" "build/tmp/zip/$NAME"
  chmod 755 "build/tmp/zip/$NAME"
  touch -t 202601010000 "build/tmp/zip/$NAME"
  rm -f "dist/$2"
  ( cd build/tmp/zip && zip -X -9 -q "../../../dist/$2" "$NAME" )
}
mac_zip build/tmp/btsw-darwin-arm64 BrainTumorSegmentationWorkbench-macOS-AppleSilicon.zip
mac_zip build/tmp/btsw-darwin-amd64 BrainTumorSegmentationWorkbench-macOS-Intel.zip
# add-on model files (exported with build/export_models.py into addons/<id>/)
ADDONS=()
if [ -f addons/brats2020/config.json ]; then
  python3 build/package_addon.py addons/brats2020 dist/BrainTumorSegmentationWorkbench-model-BraTS2020-nnUNet.btsmodel
  ADDONS+=(BrainTumorSegmentationWorkbench-model-BraTS2020-nnUNet.btsmodel)
fi
if [ -f addons/raidionics-flair/config.json ]; then
  python3 build/package_addon.py addons/raidionics-flair dist/BrainTumorSegmentationWorkbench-model-Raidionics-FLAIR.btsmodel
  ADDONS+=(BrainTumorSegmentationWorkbench-model-Raidionics-FLAIR.btsmodel)
fi
if [ -f addons/raidionics-core/config.json ]; then
  python3 build/package_addon.py addons/raidionics-core dist/BrainTumorSegmentationWorkbench-model-Raidionics-TumorCore.btsmodel
  ADDONS+=(BrainTumorSegmentationWorkbench-model-Raidionics-TumorCore.btsmodel)
fi
for pair in aurora-t1c:AURORA-T1c aurora-t1c-flair:AURORA-T1c-FLAIR vs-t1c:VS-T1c vs-t2:VS-T2; do
  if [ -f "addons/${pair%%:*}/config.json" ]; then
    python3 build/package_addon.py "addons/${pair%%:*}" "dist/BrainTumorSegmentationWorkbench-model-${pair#*:}.btsmodel"
    ADDONS+=("BrainTumorSegmentationWorkbench-model-${pair#*:}.btsmodel")
  fi
done
( cd dist && sha256sum BrainTumorSegmentationWorkbench.exe BrainTumorSegmentationWorkbench-macOS-*.zip "${ADDONS[@]}" btsw-linux-amd64 > SHA256SUMS.txt && cat SHA256SUMS.txt )
ls -la dist
