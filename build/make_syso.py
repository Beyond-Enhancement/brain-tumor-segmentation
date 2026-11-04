#!/usr/bin/env python3
"""Write a COFF .syso holding the Windows icon, version info and manifest for the Go linker.

Usage: make_syso.py <out.syso> <version e.g. 1.0.0>
"""
import io, os, struct, sys
from PIL import Image

OUT, VER = sys.argv[1], sys.argv[2]
ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon")
LANG = 0x0409
RT_ICON, RT_GROUP_ICON, RT_VERSION, RT_MANIFEST = 3, 14, 16, 24


# ---------------------------------------------------------------- icon images
def dib(img):
    w, h = img.size
    px = img.convert("RGBA").tobytes()
    rows = []
    for y in range(h - 1, -1, -1):  # bottom-up
        row = bytearray()
        for x in range(w):
            r, g, b, a = px[(y * w + x) * 4:(y * w + x) * 4 + 4]
            row += bytes((b, g, r, a))
        rows.append(bytes(row))
    mask_stride = ((w + 31) // 32) * 4
    hdr = struct.pack("<IiiHHIIiiII", 40, w, h * 2, 1, 32, 0, 0, 0, 0, 0, 0)
    return hdr + b"".join(rows) + b"\0" * (mask_stride * h)


icons = []  # (size, bytes)
for s in (16, 32, 48):
    icons.append((s, dib(Image.open(f"{ICON_DIR}/icon-{s}.png"))))
buf = io.BytesIO(); Image.open(f"{ICON_DIR}/icon-256.png").save(buf, "PNG", optimize=True)
icons.append((256, buf.getvalue()))

grp = struct.pack("<HHH", 0, 1, len(icons))
for i, (s, data) in enumerate(icons, start=1):
    wb = 0 if s >= 256 else s
    grp += struct.pack("<BBBBHHIH", wb, wb, 0, 0, 1, 32, len(data), i)


# ---------------------------------------------------------------- version info
def u16z(s):
    return (s + "\0").encode("utf-16-le")


def pad4(b):
    return b + b"\0" * (-len(b) % 4)


def block(key, value=b"", children=(), wtype=1, value_len=None):
    body = struct.pack("<HHH", 0, 0, 0) + u16z(key)
    body = pad4(body) + value
    if children:
        body = pad4(body)
        for c in children:
            body = pad4(body) + c
    vl = value_len if value_len is not None else len(value)
    return struct.pack("<HHH", len(body), vl, wtype) + body[6:]


def string(key, val):
    v = u16z(val)
    return block(key, v, wtype=1, value_len=len(v) // 2)


parts = [int(x) for x in (VER.split(".") + ["0", "0", "0"])[:4]]
ms, ls = (parts[0] << 16) | parts[1], (parts[2] << 16) | parts[3]
fixed = struct.pack("<13I", 0xFEEF04BD, 0x00010000, ms, ls, ms, ls, 0x3F, 0, 0x40004, 1, 0, 0, 0)
strings = [
    ("CompanyName", ""),
    ("FileDescription", "Brain Tumor Segmentation Workbench"),
    ("FileVersion", VER),
    ("InternalName", "BrainTumorSegmentationWorkbench"),
    ("LegalCopyright", ""),
    ("OriginalFilename", "BrainTumorSegmentationWorkbench.exe"),
    ("ProductName", "Brain Tumor Segmentation Workbench"),
    ("ProductVersion", VER),
]
st = block("040904B0", children=[string(k, v) for k, v in strings if v])
sfi = block("StringFileInfo", children=[st])
vfi = block("VarFileInfo", children=[block("Translation", struct.pack("<HH", LANG, 0x04B0), wtype=0)])
version = block("VS_VERSION_INFO", fixed, children=[sfi, vfi], wtype=0)

# ---------------------------------------------------------------- manifest
manifest = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <assemblyIdentity type="win32" name="BrainTumorSegmentationWorkbench" version="%s.0"/>
  <description>Brain Tumor Segmentation Workbench</description>
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3"><security><requestedPrivileges>
    <requestedExecutionLevel level="asInvoker" uiAccess="false"/>
  </requestedPrivileges></security></trustInfo>
  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1"><application>
    <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"/>
  </application></compatibility>
  <application xmlns="urn:schemas-microsoft-com:asm.v3"><windowsSettings>
    <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true/pm</dpiAware>
    <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2</dpiAwareness>
  </windowsSettings></application>
  <dependency><dependentAssembly>
    <assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls" version="6.0.0.0"
      processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*"/>
  </dependentAssembly></dependency>
</assembly>
""" % VER.encode()

# ---------------------------------------------------------------- resource tree
# {type: {id: bytes}}, all ids numeric, single language
tree = {
    RT_ICON: {i: data for i, (_, data) in enumerate(icons, start=1)},
    RT_GROUP_ICON: {1: grp},
    RT_VERSION: {1: version},
    RT_MANIFEST: {1: manifest},
}

DIR, ENT, DATA = 16, 8, 16
types = sorted(tree)
# sizes of each directory level
off = DIR + ENT * len(types)                      # level-1 directory
l2 = {}
for t in types:
    l2[t] = off; off += DIR + ENT * len(tree[t])  # level-2 directories
l3 = {}
for t in types:
    for i in sorted(tree[t]):
        l3[(t, i)] = off; off += DIR + ENT * 1      # level-3 (language) directories
de = {}
for t in types:
    for i in sorted(tree[t]):
        de[(t, i)] = off; off += DATA               # data entries
blob = {}
for t in types:
    for i in sorted(tree[t]):
        off += -off % 8
        blob[(t, i)] = off; off += len(tree[t][i])
size = off + (-off % 8)

sec = bytearray(size)
relocs = []


def put_dir(at, entries):
    struct.pack_into("<IIHHHH", sec, at, 0, 0, 0, 0, 0, len(entries))
    for k, (ident, target, is_dir) in enumerate(entries):
        struct.pack_into("<II", sec, at + DIR + ENT * k, ident, target | (0x80000000 if is_dir else 0))


put_dir(0, [(t, l2[t], True) for t in types])
for t in types:
    put_dir(l2[t], [(i, l3[(t, i)], True) for i in sorted(tree[t])])
    for i in sorted(tree[t]):
        put_dir(l3[(t, i)], [(LANG, de[(t, i)], False)])
        struct.pack_into("<IIII", sec, de[(t, i)], blob[(t, i)], len(tree[t][i]), 0, 0)
        relocs.append(de[(t, i)])  # OffsetToData is an RVA: relocate against .rsrc
        sec[blob[(t, i)]:blob[(t, i)] + len(tree[t][i])] = tree[t][i]

# ---------------------------------------------------------------- COFF object
IMAGE_REL_AMD64_ADDR32NB = 3
raw_ptr = 20 + 40
rel_ptr = raw_ptr + len(sec)
sym_ptr = rel_ptr + 10 * len(relocs)
out = bytearray()
out += struct.pack("<HHIIIHH", 0x8664, 1, 0, sym_ptr, 1, 0, 0)
out += struct.pack("<8sIIIIIIHHI", b".rsrc", 0, 0, len(sec), raw_ptr, rel_ptr, 0, len(relocs), 0, 0x40000040)
out += sec
for r in relocs:
    out += struct.pack("<IIH", r, 0, IMAGE_REL_AMD64_ADDR32NB)
out += struct.pack("<8sIhHBB", b".rsrc", 0, 1, 0, 3, 0)
out += struct.pack("<I", 4)
open(OUT, "wb").write(out)
print(f"wrote {OUT}: {len(out)} bytes, {len(relocs)} resources")
