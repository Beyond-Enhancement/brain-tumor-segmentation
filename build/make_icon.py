#!/usr/bin/env python3
"""Draw the app icon (an axial brain with a segmented tumour) into build/icon/."""
import math, os
from PIL import Image, ImageDraw
S = 2048
img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
pad = int(S * 0.06); r = int(S * 0.2)
d.rounded_rectangle([pad, pad, S - pad, S - pad], radius=r, fill=(18, 23, 28, 255), outline=(36, 48, 58, 255), width=int(S * 0.012))
cx, cy = S / 2, S * 0.51
# brain outline: slightly egg-shaped ellipse, wider posteriorly
def brain_pt(t, sx=0.31, sy=0.36):
    x = math.cos(t); y = math.sin(t)
    w = sx * (1 + 0.06 * y)
    return (cx + S * w * x, cy + S * sy * y)
outline = [brain_pt(2 * math.pi * i / 720) for i in range(720)]
d.polygon(outline, fill=(44, 52, 62, 255))
d.line(outline + [outline[0]], fill=(205, 212, 222, 255), width=int(S * 0.028), joint='curve')
# interhemispheric fissure
d.line([(cx, cy - S * 0.36), (cx, cy - S * 0.20)], fill=(205, 212, 222, 255), width=int(S * 0.02))
d.line([(cx, cy + S * 0.24), (cx, cy + S * 0.36)], fill=(205, 212, 222, 255), width=int(S * 0.02))
# tumour: non-enhancing abnormality (yellow) with an enhancing rim (red) around a dark core
tx, ty = cx - S * 0.10, cy - S * 0.07
def blob(rad, wob):
    return [(tx + S * rad * (1 + wob * math.sin(3 * a + 0.7) + 0.5 * wob * math.cos(5 * a)) * math.cos(a),
             ty + S * rad * (1 + wob * math.sin(3 * a + 0.7) + 0.5 * wob * math.cos(5 * a)) * math.sin(a)) for a in [2 * math.pi * i / 360 for i in range(360)]]
d.polygon(blob(0.16, 0.10), fill=(255, 210, 63, 255))
d.polygon(blob(0.085, 0.08), fill=(255, 77, 77, 255))
d.polygon(blob(0.045, 0.05), fill=(44, 52, 62, 255))
img = img.resize((1024, 1024), Image.LANCZOS)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon')
os.makedirs(OUT, exist_ok=True)
img.save(f'{OUT}/icon-1024.png')
for s in (256, 48, 32, 16):
    img.resize((s, s), Image.LANCZOS).save(f'{OUT}/icon-{s}.png')
print('ok')
