"""Composite Fig 2 (a-h).

Layout (7.2 x 10.4 inch, double-column):
  Row 1: a                  (global map, full width)
  Row 2: b | c | d          (PCoA | Volcano | LEfSe bar)
  Row 3: e | f              (cross-study forest | Faecalibacterium forest)
  Row 4: g | h              (SparCC circular network | lifecycle heatmap)

Reads each sub-panel PNG via PIL (with decompression guard),
places via matplotlib gridspec, exports fig2.pdf + fig2.png + LaTeX copy.
"""
import os
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["font.family"]  = "Arial"
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def trim_to_xaxis(im, bottom_frac=0.30, dark_frac=0.30, tick_pad_frac=0.028):
    """Crop image bottom to just below its main x-axis line (removes trailing annotation).
    Searches the bottom `bottom_frac` of the image for the bottom-most row whose dark
    pixels span > `dark_frac` of width (the axis line), keeps `tick_pad_frac` below it."""
    arr = np.asarray(im.convert("L"))
    H, W = arr.shape
    dark = arr < 100
    row_dark = dark.sum(axis=1)
    search_start = int(H * (1 - bottom_frac))
    broad_rows = np.where(row_dark[search_start:] > W * dark_frac)[0]
    if len(broad_rows) == 0:
        return im
    axis_y = search_start + broad_rows[-1]
    cut_y = min(H, axis_y + int(H * tick_pad_frac))
    return im.crop((0, 0, W, cut_y))

ROOT = r"E:\tasks\screenshots\fig2"
LATEX_DIR = r"E:\microbiomap_clone\compendium_website\docs\NatureMethods_LaTeX"

PANELS = "abcdefgh"

def load(letter, max_dim=3200):
    p = os.path.join(ROOT, f"fig2{letter}", f"fig2{letter}.png")
    im = Image.open(p)
    if letter == "f":
        w0, h0 = im.size
        pad = int(h0 * 0.16)
        new_im = Image.new("RGB", (w0, h0 + pad), "white")
        new_im.paste(im, (0, pad))
        im = new_im
    w, h = im.size
    m = max(w, h)
    if m > max_dim:
        s = max_dim / m
        im = im.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return im

imgs = {L: load(L) for L in PANELS}

FW, FH = 7.2, 9.45
fig = plt.figure(figsize=(FW, FH))
gs = fig.add_gridspec(
    nrows=4, ncols=240,
    height_ratios=[1.87, 2.20, 2.77, 2.15],
    hspace=0.05, wspace=0.04,
    left=0.010, right=0.995, top=0.996, bottom=0.004,
)

axes = {}
axes["a"] = fig.add_subplot(gs[0, 0:177])
axes["b"] = fig.add_subplot(gs[0, 177:240])
axes["c"] = fig.add_subplot(gs[1, 0:100])
axes["d"] = fig.add_subplot(gs[1, 100:215])
axes["e"] = fig.add_subplot(gs[2, 0:80])
axes["g"] = fig.add_subplot(gs[2, 80:240])
axes["f"] = fig.add_subplot(gs[3, 0:80])
axes["h"] = fig.add_subplot(gs[3, 80:240])

for L in PANELS:
    ax = axes[L]
    im_L = imgs[L]
    W_img, H_img = im_L.size
    if L in ("e", "f"):
        dx = int(W_img * 0.06)  # shift image left within cell
        ax.imshow(im_L, extent=(-dx, W_img - dx, H_img, 0))
    else:
        ax.imshow(im_L)
    ax.set_anchor("SW" if L == "e" else "NW")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-0.010, 1.00, L, transform=ax.transAxes,
            fontsize=8, fontweight="bold",
            ha="right", va="bottom", family="Arial")

out_pdf = os.path.join(ROOT, "fig2a", "fig2.pdf")
out_png = os.path.join(ROOT, "fig2a", "fig2.png")
fig.savefig(out_pdf, dpi=600, bbox_inches="tight", pad_inches=0.0)
fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.0)
print(f"[fig2] saved {out_pdf}")
print(f"[fig2] saved {out_png}")

# sync to LaTeX
import shutil
for name in ["fig2.pdf"]:
    src = os.path.join(ROOT, "fig2a", name)
    dst = os.path.join(LATEX_DIR, name)
    shutil.copy2(src, dst)
    print(f"[fig2] copied -> {dst}")
