#!/usr/bin/env python3
"""
Final volcano plot v3:
  - 2× font sizes throughout
  - Thicker axes and lines
  - Top 5 labels per box
  - Strict: adj.p >= 0.05 → grey (no color) regardless of cache inconsistency
"""
import sys, io, os, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42   # TrueType → fully editable in Illustrator
matplotlib.rcParams['ps.fonttype']  = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

OUTPUT_DIR = r"E:\tasks\screenshots\fig2c"
CACHE_FILE = os.path.join(OUTPUT_DIR, "_cd_analysis_cache.pkl")
FONT   = "Arial"
C_TXT  = "#2c3e50"
C_SUB  = "#5d6d7e"
C_RULE = "#d5dbdb"

VCOL = {
    "crc_hi": "#c0392b",
    "crc":    "#e67e22",
    "nc_hi":  "#16a085",
    "nc":     "#2aa198",
    "normal": "#888888",
}

print("Loading cache …")
with open(CACHE_FILE, "rb") as f:
    c = pickle.load(f)
diff_results = c["diff_results"]
n_crc, n_nc  = c["n_crc"], c["n_nc"]
print(f"  CD n={n_crc}, NC n={n_nc}, taxa={len(diff_results)}")

x_vals = np.array([r["log2fc"]                for r in diff_results])
y_vals = np.array([r["neg_log10_adjusted_p"]   for r in diff_results])
adj_ps = np.array([r["adjusted_p"]             for r in diff_results])
tnames = [r["taxon"] for r in diff_results]

POWER = 1/3
def ytrans(v):  return np.sign(v) * np.abs(v) ** POWER
def ytrans1(v): return math.copysign(abs(v) ** POWER, v)

y_plot    = ytrans(y_vals)
y_max     = float(y_plot.max()) * 1.12
sig_001   = -math.log10(0.01)
sig_001_p = ytrans1(sig_001)
sig_005   = -math.log10(0.05)
sig_005_p = ytrans1(sig_005)
x_max     = float(np.percentile(np.abs(x_vals), 99.8)) * 1.12

_yt_actual = [0, 5, 10, 20, 50, 100, 200, 300, 400]
_yt_plot   = [ytrans1(t) for t in _yt_actual if ytrans1(t) <= y_max * 1.02]
_yt_labels = [str(t) for t in _yt_actual[:len(_yt_plot)]]

def vol_cat(r):
    ap, fc = r["adjusted_p"], r["log2fc"]
    if ap >= 0.05:          return "normal"
    if -1 <= fc <= 1:       return "normal"   # |logFC| < 1 → grey regardless of p
    if fc > 1:              return "crc_hi" if ap < 0.001 else "crc"
    return                   "nc_hi"  if ap < 0.001 else "nc"

cats    = [vol_cat(r) for r in diff_results]
cat_arr = np.array(cats)

# ── Figure — extra left margin for outside legend ─────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(17, 16), facecolor="white")
# bottom enlarged to make room for legend below the axes
fig.subplots_adjust(left=0.12, right=0.96, top=0.91, bottom=0.20)

# ── Background color panels ───────────────────────────────────────────────────
ax.add_patch(mpatches.FancyBboxPatch((-x_max, 0), x_max, y_max,
    boxstyle="square,pad=0", facecolor="#2aa198", alpha=0.07,
    edgecolor="none", zorder=0))
ax.add_patch(mpatches.FancyBboxPatch((0, 0), x_max, y_max,
    boxstyle="square,pad=0", facecolor="#e67e22", alpha=0.07,
    edgecolor="none", zorder=0))

# ── Threshold lines — thicker ─────────────────────────────────────────────────
ax.axhline(sig_005_p, color="#e74c3c", lw=5.0, ls=(0,(5,4)), zorder=3, alpha=0.85)
ax.axhline(sig_001_p, color="#c0392b", lw=4.0, ls=(0,(3,3)), zorder=3, alpha=0.65)
ax.axvline(-1, color="#474747", lw=4.0, ls="--", zorder=1, alpha=0.75)
ax.axvline(+1, color="#474747", lw=4.0, ls="--", zorder=1, alpha=0.75)
ax.axvline( 0, color="#cccccc", lw=3.0, zorder=1)
ax.text(x_max * 0.99, sig_005_p - 0.18,
        "adj.p = 0.05", ha="right", va="top", fontsize=30, color="#e74c3c", fontfamily=FONT)

# ── Scatter — STRICT: colored only if y_plot > sig_005_p ─────────────────────
m_bg = cat_arr == "normal"
ax.scatter(x_vals[m_bg], y_plot[m_bg],
           s=28, c=VCOL["normal"], edgecolors="none",
           alpha=0.38, zorder=2)

for cat in ["nc", "nc_hi", "crc", "crc_hi"]:
    # Extra guard: only color points that are actually above p=0.05 in plot space
    m = (cat_arr == cat) & (y_plot > sig_005_p)
    if m.any():
        ax.scatter(x_vals[m], y_plot[m],
                   s=56, c=VCOL[cat],
                   edgecolors="black", linewidths=1.2,
                   alpha=0.88, zorder=3)
    # Points with "colored" category but below p=0.05 → force grey
    m_below = (cat_arr == cat) & (y_plot <= sig_005_p)
    if m_below.any():
        ax.scatter(x_vals[m_below], y_plot[m_below],
                   s=28, c=VCOL["normal"], edgecolors="none",
                   alpha=0.38, zorder=2)

# ── Direct label → point annotations (no boxes, no "Top N" header) ────────────
def label_top_points(direction, n_top=5):
    if direction == "right":
        mask = (adj_ps < 0.01) & (x_vals > 1) & (y_plot >= sig_001_p)
    else:
        mask = (adj_ps < 0.01) & (x_vals < -1) & (y_plot >= sig_001_p)

    idx = np.where(mask)[0]
    if len(idx) == 0:
        return

    sorted_idx = idx[np.argsort(adj_ps[idx])][:n_top]
    # Sort by point y descending (most significant on top)
    sorted_idx = sorted(sorted_idx, key=lambda si: -float(y_plot[si]))

    # Stack label positions evenly — lower than region text to avoid overlap
    y_top_lbl = y_max * 0.72   # taxon names start here (moved down from 0.82)
    y_bot_lbl = y_max * 0.44   # taxon names end here   (moved down from 0.56)
    n = len(sorted_idx)
    step = (y_top_lbl - y_bot_lbl) / max(n - 1, 1)

    # "Top 5" header — fixed at original height, independent of label column
    if direction == "right":
        tx   = x_max * 0.55
        ha_l = "left"
    else:
        tx   = -x_max * 0.30
        ha_l = "right"

    ax.text(tx, y_max * 0.845, f"Top {n_top}",
            fontsize=30, color=C_TXT, fontfamily=FONT, fontweight="bold",
            ha=ha_l, va="bottom",
            bbox=dict(facecolor="white", edgecolor="#888888",
                      linewidth=2.0, pad=3.0, boxstyle="round,pad=0.3"),
            zorder=9)

    for rank, si in enumerate(sorted_idx):
        xi = float(x_vals[si])
        yi = float(y_plot[si])
        name = tnames[si]
        ty = y_top_lbl - rank * step

        ax.annotate(
            f" {name} ",
            xy=(xi, yi),
            xytext=(tx, ty),
            fontsize=28, fontstyle="italic", fontfamily=FONT,
            color=C_TXT, fontweight="bold",
            ha=ha_l, va="center",
            bbox=dict(facecolor="white", edgecolor="#bbbbbb",
                      linewidth=1.8, boxstyle="round,pad=0.3", alpha=0.92),
            arrowprops=dict(arrowstyle="-", color="#999999", lw=2.0,
                            connectionstyle="arc3,rad=0.08"),
            zorder=8)

label_top_points("right", n_top=5)
label_top_points("left",  n_top=5)

# ── Region text ───────────────────────────────────────────────────────────────
ax.text(-x_max * 0.97, y_max * 0.96,
        "NC-enriched\n(CD-depleted)",
        color="#16a085", fontsize=30, fontfamily=FONT,
        fontweight="bold", ha="left", va="top", zorder=5)
ax.text(x_max * 0.97, y_max * 0.96,
        "CD-enriched\n(NC-depleted)",
        color="#c0392b", fontsize=30, fontfamily=FONT,
        fontweight="bold", ha="right", va="top", zorder=5)

# ── Axes — thicker spines & ticks ────────────────────────────────────────────
ax.set_xlim(-x_max, x_max)
ax.set_ylim(0, y_max)
ax.set_yticks(_yt_plot)
ax.set_yticklabels(_yt_labels, fontsize=30)

auto_ticks  = [t for t in np.arange(-15, 16, 5) if -x_max <= t <= x_max]
x_tick_vals = sorted(set(auto_ticks) | {-1, 1})
x_tick_lbls = [str(int(t)) if t != 0 else "0" for t in x_tick_vals]
ax.set_xticks(x_tick_vals)
ax.set_xticklabels(x_tick_lbls, fontsize=30)

for lbl in ax.get_xticklabels():
    if lbl.get_text() in ("-1", "1"):
        lbl.set_color("#474747")
        lbl.set_fontweight("bold")
        lbl.set_fontsize(30)
    else:
        lbl.set_color(C_SUB)

for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax.spines[sp].set_linewidth(6.0)
    ax.spines[sp].set_color("#444444")
ax.tick_params(axis="both", width=5.0, length=14, colors=C_SUB)

ax.set_title(
    f"CD vs Healthy Control: Differential Microbiome (n={n_crc:,} CD, {n_nc:,} NC)",
    fontsize=28, fontfamily=FONT, fontweight="bold", color=C_TXT, pad=14)
ax.set_xlabel(r"CD vs NC  (log$_2$FC)", fontsize=30, fontfamily=FONT,
              color=C_TXT, labelpad=12)
ax.set_ylabel(r"Significance  ($-\log_{10}$ adj.$p$)", fontsize=30,
              fontfamily=FONT, color=C_TXT, labelpad=12)

# ── Legend — OUTSIDE plot, below axes (2 columns) ────────────────────────────
leg_handles = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=VCOL["crc_hi"],
           ms=28, mec="black", mew=2.0, label=r"CD-enriched (adj.$p$<0.001)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=VCOL["crc"],
           ms=28, mec="black", mew=2.0, label=r"CD-enriched (adj.$p$<0.05)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=VCOL["nc_hi"],
           ms=28, mec="black", mew=2.0, label=r"NC-enriched (adj.$p$<0.001)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=VCOL["nc"],
           ms=28, mec="black", mew=2.0, label=r"NC-enriched (adj.$p$<0.05)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=VCOL["normal"],
           ms=22, mec="none", label="Non-significant"),
]
ax.legend(handles=leg_handles,
          fontsize=30, framealpha=0.95, edgecolor=C_RULE,
          loc="upper center",
          bbox_to_anchor=(0.5, -0.14),   # closer to plot
          ncol=3,
          borderpad=1.0, labelspacing=0.7, columnspacing=1.5,
          handlelength=1.5)

for ext in ("png", "pdf"):
    out = os.path.join(OUTPUT_DIR, f"fig2c.{ext}")
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"  → {out}")
plt.close()
print("Done.")
