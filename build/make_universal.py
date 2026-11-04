#!/usr/bin/env python3
"""Combine thin Mach-O executables into one universal (fat) binary, like `lipo -create`.

Usage: make_universal.py <out> <thin1> <thin2> ...
"""
import os, struct, sys

ALIGN = 14  # 16 KiB slice alignment, as lipo uses for arm64
out_path, inputs = sys.argv[1], sys.argv[2:]
datas = [open(p, "rb").read() for p in inputs]
layout, cur = [], 8 + 20 * len(datas)
for d in datas:
    magic, cpu, sub = struct.unpack_from("<Iii", d, 0)
    if magic != 0xFEEDFACF:
        sys.exit(f"not a 64-bit Mach-O: {inputs[len(layout)]}")
    cur = (cur + (1 << ALIGN) - 1) & ~((1 << ALIGN) - 1)
    layout.append((cpu, sub & 0x00FFFFFF, cur, len(d)))
    cur += len(d)
out = bytearray(struct.pack(">II", 0xCAFEBABE, len(datas)))
for cpu, sub, off, size in layout:
    out += struct.pack(">iiIII", cpu, sub, off, size, ALIGN)
for (cpu, sub, off, size), d in zip(layout, datas):
    out += b"\0" * (off - len(out))
    out += d
with open(out_path, "wb") as f:
    f.write(out)
os.chmod(out_path, 0o755)
print(f"universal binary: {out_path} ({len(out):,} bytes, {len(datas)} slices)")
