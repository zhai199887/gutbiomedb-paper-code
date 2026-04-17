"""Stitch fig1 panels a–e into a publication-ready composite fig1.pdf.

Layout (3 rows, no whitespace gutters):
  Row 1: a — full-width thin pipeline banner
  Row 2: b + d + e — three panels across, fills row
  Row 3: c — alone, full canvas width (hero panel, scaled up)
Panel internals are NOT modified — each sub-image is scaled as-is.
"""
from PIL import Image, ImageDraw, ImageFont
import os

BASE = r"E:\tasks\screenshots\fig1"
OUT_PDF = os.path.join(BASE, "fig1.pdf")
OUT_PNG = os.path.join(BASE, "fig1.png")

paths = {
    "a": os.path.join(BASE, "fig1a.png"),
    "b": os.path.join(BASE, "fig1b.png"),
    "c": os.path.join(BASE, "fig1c.png"),
    "d": os.path.join(BASE, "fig1d.png"),
    "e": os.path.join(BASE, "fig1e", "fig1e.png"),
}
ims = {k: Image.open(v).convert("RGB") for k, v in paths.items()}

W = 3600           # Nature double-column 183 mm @ 500 dpi
H_A4 = 4724        # Nature full-page height 240 mm @ 500 dpi
GAP = 50
GAP_AC = 10        # smaller gap between a and c (pull c up)
HGAP = 60
LABEL_PAD = 95     # row vertical slot — keeps label position stable across rows
IMG_PAD = 50       # image starts this far below label baseline (smaller = closer)
BG = (255, 255, 255)

def scale_w(im, w):
    return im.resize((w, int(round(im.height * w / im.width))), Image.LANCZOS)

# Row 1: a — mask source's small baked-in "a", will redraw at unified size
a_src = ims["a"].copy()
ImageDraw.Draw(a_src).rectangle((40, 5, 140, 80), fill=BG)
a = scale_w(a_src, W)

# Mask "a" baked into c source top-left
c_src = ims["c"].copy()
ImageDraw.Draw(c_src).rectangle((130, 20, 300, 200), fill=BG)
ims["c"] = c_src

# Row 3: e right column, height = canvas bottom - bde row top, so e's longest genus name touches bottom
# calculate bde row start y (aligned with a/c/gap)
a_h = int(round(ims["a"].height * W / ims["a"].width))
c_h = int(round(ims["c"].height * W / ims["c"].width))
y_offset_calc = 30
gap_ac_calc = 10
gap_after_c = 30
bde_top = y_offset_calc + a_h + gap_ac_calc + c_h + gap_after_c
e_h_target = H_A4 - bde_top
e_shift = 80                                  # shift e rightward
e_w = int(round(e_h_target * 4200 / 3300)) - e_shift
e = scale_w(ims["e"], e_w)
left_w = W - HGAP - e_w - e_shift
left_margin = 130                             # left margin for b/d
b_w = left_w - left_margin
b_left = scale_w(ims["b"], b_w)
d_left = scale_w(ims["d"], b_w)
row_h = max(e.height, b_left.height + 50 + d_left.height)
bde_wrap = Image.new("RGB", (W, row_h), BG)
# Left column: b at top, d at bottom (aligned to e bottom)
bde_wrap.paste(b_left, (left_margin, 0))
d_top = e.height - d_left.height
bde_wrap.paste(d_left, (left_margin, d_top))
# Right column: e (shifted right)
e_x = left_w + HGAP + e_shift
e_y = 0
bde_wrap.paste(e, (e_x, e_y))

# Row 3: c spans full width (scaled proportionally)
c = scale_w(ims["c"], W)

rows = [
    [("a",), a, [(0, 0)]],
    [("c",), c, [(0, 0)]],
    [("b", "d", "e"), bde_wrap, [(left_margin, 0), (left_margin, d_top), (e_x, e_y)]],
]
content_h = sum(r[1].height for r in rows) + GAP * (len(rows) - 1)
canvas = Image.new("RGB", (W, H_A4), BG)
y_offset = 30

try:
    font = ImageFont.truetype("arialbd.ttf", 55)
except Exception:
    font = ImageFont.load_default()

draw = ImageDraw.Draw(canvas)
y = y_offset
for i, (labels, img, positions) in enumerate(rows):
    canvas.paste(img, (0, y))
    for lbl, (lx, ly) in zip(labels, positions):
        draw.text((lx + 20, y + ly + 5), lbl, fill=(0, 0, 0), font=font)
    gap = GAP_AC if i == 0 else (gap_after_c if i == 1 else GAP)
    y += img.height + gap

canvas.save(OUT_PNG, "PNG", optimize=True)
canvas.save(OUT_PDF, "PDF", resolution=500.0)
aspect = canvas.height / canvas.width
print(f"wrote {OUT_PDF} size={canvas.size} aspect={aspect:.3f} (Nature max 1.35)")