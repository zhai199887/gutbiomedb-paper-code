"""
Fig 1g1 — Universal GBHI leave-one-cohort-out (LOCO) validation.
Render 9 per-disease panels at fig1d scale, then stitch into a 3x3 PNG.
Same proportions/fonts as the v6 stitcher; single navy square per cohort.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

C_NAVY    = "#3C5488"
C_DIAMOND = "#1a1a1a"
C_TXT     = "#1a1a1a"
C_NULL    = "#2c3e50"
C_BG      = "#f7f9fc"
C_RULE    = "#d5dbdb"
FONT      = "Arial"

IN_JSON  = r"E:\tasks\screenshots\fig1g\gbhi_universal_loco_per_cohort.json"
TILE_DIR = r"E:\tasks\screenshots\fig1g\_tiles_univ"
OUT_PNG  = r"E:\tasks\screenshots\fig3\figS2_gbhi_loco.png"
OUT_PDF  = r"E:\tasks\screenshots\fig3\figS2_gbhi_loco.pdf"
os.makedirs(TILE_DIR, exist_ok=True)

NICE = {"c_difficile_infection":"CDI","UC":"UC","CD":"CD",
        "obesity":"Obesity","adenoma":"Adenoma","colorectal_cancer":"CRC",
        "rheumatoid arthritis":"RA","HIV":"HIV","IBS":"IBS"}
fmt = lambda v: "NA" if v is None else f"{v:.3f}"

d = json.load(open(IN_JSON, "r", encoding="utf-8"))
valid = [(k, v) for k, v in d["per_disease"].items()
         if not v.get("skipped") and v.get("auc_mean") is not None]
valid.sort(key=lambda kv: -kv[1]["auc_mean"])

DPI = 70

def render_tile(i, dis, info, path):
    cohorts = [co for co in info["per_cohort"] if co.get("auc_universal") is not None]
    cohorts.sort(key=lambda co: co["auc_universal"])
    n = len(cohorts); yy = list(range(n - 1, -1, -1)); y_pool = -1.8
    fig_h = max(22, n * 2.0 + 7)
    fig, ax = plt.subplots(figsize=(46, fig_h), facecolor="white")
    ax.set_facecolor("white")
    ax.set_ylim(y_pool - 1.5, n - 1 + 1.6)

    for j in range(n):
        if j % 2 == 0:
            ax.axhspan(yy[j] - 0.46, yy[j] + 0.46, color=C_BG, zorder=0)
    ax.axvline(0.5, color=C_NULL, lw=8, ls="--", zorder=1)

    for co, y in zip(cohorts, yy):
        ax.plot(co["auc_universal"], y, "s", color=C_NAVY, ms=44,
                mec=C_DIAMOND, mew=3.0, zorder=6)

    ax.axhline(-0.72, color=C_RULE, lw=4, zorder=1)
    pm = info["auc_mean"]
    dw, dh = 0.014, 0.42
    def diamond(cx, cy, color):
        dx = [cx, cx + dw, cx, cx - dw, cx]
        dy = [cy + dh, cy, cy - dh, cy, cy + dh]
        ax.fill(dx, dy, color=color, zorder=8)
        ax.plot(dx, dy, color="white", lw=3.0, zorder=9)
    if pm is not None:
        diamond(pm, y_pool, C_NAVY)

    ax.set_xlim(0.30, 1.0)
    ax.set_yticks([])
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(6); ax.spines["bottom"].set_color(C_TXT)
    ax.tick_params(axis="x", width=8, length=22, labelsize=64, colors=C_TXT)
    plt.setp(ax.get_xticklabels(), fontweight="bold", fontfamily=FONT)
    ax.set_xlabel("AUROC  (universal softmax, LOCO)", fontsize=66, fontfamily=FONT,
                  color=C_TXT, labelpad=26, fontweight="bold")

    for co, y in zip(cohorts, yy):
        lab = f"{co['cohort']}  (n={co['n_nc']+co['n_dis']})"
        ax.text(-0.005, y, lab, transform=ax.get_yaxis_transform(),
                fontsize=62, fontfamily=FONT, color=C_TXT,
                va="center", ha="right", fontweight="bold")
    ax.text(-0.005, y_pool, "Pooled  (mean)",
            transform=ax.get_yaxis_transform(),
            fontsize=62, fontfamily=FONT, color=C_DIAMOND,
            va="center", ha="right", fontweight="bold")

    header_y = n - 1 + 1.10
    col_x = 1.08
    ax.text(col_x, header_y, "AUROC", transform=ax.get_yaxis_transform(),
            fontsize=62, fontfamily=FONT, color=C_NAVY,
            va="center", ha="center", fontweight="bold")
    for co, y in zip(cohorts, yy):
        ax.text(col_x, y, fmt(co["auc_universal"]),
                transform=ax.get_yaxis_transform(),
                fontsize=60, fontfamily=FONT, color=C_TXT,
                va="center", ha="center", fontweight="bold")
    ax.text(col_x, y_pool, fmt(pm), transform=ax.get_yaxis_transform(),
            fontsize=60, fontfamily=FONT, color=C_NAVY,
            va="center", ha="center", fontweight="bold")

    nice = NICE.get(dis, dis[:22])
    ax.set_title(
        f"{chr(97+i)}   {nice}   \u2014   Leave-One-Cohort-Out validation   ({n} cohorts)",
        fontsize=72, fontfamily=FONT, color=C_TXT,
        fontweight="bold", loc="left", pad=32)

    plt.subplots_adjust(left=0.11, right=0.55, top=0.93, bottom=0.12)
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)

tiles = []
for i, (dis, info) in enumerate(valid):
    p = os.path.join(TILE_DIR, f"tile_{i}.png")
    print(f"[{i+1}/{len(valid)}] {dis}")
    render_tile(i, dis, info, p)
    tiles.append(p)

while len(tiles) < 9:
    # pad blank tile to keep 3x3 grid
    blank = os.path.join(TILE_DIR, f"blank_{len(tiles)}.png")
    fig, ax = plt.subplots(figsize=(46, 22), facecolor="white")
    ax.axis("off")
    fig.savefig(blank, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    tiles.append(blank)

imgs = [Image.open(p) for p in tiles]
rows = [imgs[r*3:(r+1)*3] for r in range(3)]
row_heights = [max(im.height for im in r) for r in rows]
col_widths = [max(rows[r][c].width for r in range(3)) for c in range(3)]
W = sum(col_widths); H = sum(row_heights)
canvas = Image.new("RGB", (W, H), "white")
y = 0
for r in range(3):
    x = 0
    for c in range(3):
        im = rows[r][c]
        canvas.paste(im, (x + (col_widths[c]-im.width)//2,
                          y + (row_heights[r]-im.height)//2))
        x += col_widths[c]
    y += row_heights[r]

canvas.save(OUT_PNG, optimize=True)
canvas.save(OUT_PDF, "PDF", resolution=150)
print(f"saved {OUT_PNG}  {canvas.size}")
print(f"saved {OUT_PDF}")


