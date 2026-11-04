#!/bin/bash
# Offline test: runs the launcher and a headless browser inside a fresh network namespace that
# has no interfaces except loopback, then works through the demo case and a DICOM study.
# Usage (Linux, from the repo root, after build/build.sh):
#   python3 test/make_dicom.py app/demo/tcga-fg-6692 test/out/dicom_canary
#   unshare -rn test/offline_run.sh
set -u
cd "$(dirname "$0")/.."
REPO=$(pwd)
mkdir -p test/out
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
python3 -c "
import fcntl, socket, struct
s = socket.socket()
fcntl.ioctl(s, 0x8914, struct.pack('16sH', b'lo', 0x1 | 0x8 | 0x40))  # SIOCSIFFLAGS: lo up
"
echo "== network check inside sandbox"
awk 'NR>2{print "  interface:", $1}' /proc/net/dev
python3 - <<'PY'
import socket
for host in ("example.com", "zenodo.org", "cdn.jsdelivr.net"):
    try:
        socket.getaddrinfo(host, 443); print("  DNS", host, "RESOLVED (unexpected)")
    except Exception as e:
        print("  DNS", host, "-> fails:", type(e).__name__)
for ip in ("1.1.1.1", "8.8.8.8"):
    s = socket.socket(); s.settimeout(3)
    try:
        s.connect((ip, 443)); print("  TCP", ip, "CONNECTED (unexpected)")
    except Exception as e:
        print("  TCP", ip, "-> fails:", e)
PY
echo "== starting launcher"
"$REPO/dist/btsw-linux-amd64" --no-browser > "$REPO/test/out/offline-server.log" 2>&1 &
sleep 1
echo "== demo case"
node "$REPO/test/e2e.mjs" http://127.0.0.1:47821/ "$REPO/test/out/offline-demo" "$@" | grep -v '^\[steps\]'
echo "== DICOM study"
node "$REPO/test/e2e.mjs" http://127.0.0.1:47821/ "$REPO/test/out/offline-dicom" --dicom "$REPO/test/out/dicom_canary" "$@" | grep -v '^\[steps\]'
echo "== waiting for the launcher to notice the window closed"
sleep 35
cat "$REPO/test/out/offline-server.log"
