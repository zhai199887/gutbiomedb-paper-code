#!/usr/bin/env python3
"""
Fig 1b LEfSe LDA Bar Chart — landscape, Top 12 per group
Data: platform LEfSe results stored in _cd_analysis_cache.pkl
CD-enriched → right (+), NC-enriched → left (−)
Color: CD=#2b8cbe, NC=#7bccc4
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = r"E:\tasks\screenshots\fig2d"
CACHE_FILE = os.path.join(OUTPUT_DIR, "_cd_analysis_cache.pkl")
C_CD  = "#2b8cbe"
C_NC   = "#7bccc4"
FONT   = "Arial"
TOP_N  = 12
FS     = 18   # base font size (tick labels, bar labels, legend)

# ── Load data ──────────────────────────────────────────────────────────────────
with open(CACHE_FILE, "rb") as f:
    c = pickle.load(f)
lefse = c["lefse"]

def valid_taxon(name):
    """Reject NA, 'group', empty, or purely numeric names."""
    if not name or name.strip().lower() in ("na", "group", ""):
        return False
    if name.strip().replace(".", "").isdigit():
        return False
    return True

crc_all = sorted([r for r in lefse if r["enriched_group"] == "A" and valid_taxon(r["taxon"])],
                 key=lambda x: x["lda_score"], reverse=True)
nc_all  = sorted([r for r in lefse if r["enriched_group"] == "B" and valid_taxon(r["taxon"])],
                 key=lambda x: x["lda_score"], reverse=True)

crc_entries = crc_all[:TOP_N]
nc_entries  = nc_all[:TOP_N]
n_crc = len(crc_entries)
n_nc  = len(nc_entries)

# NC reversed so lowest LDA nearest separator (mirrored layout)
all_entries = crc_entries + nc_entries[::-1]
n_total = len(all_entries)

y_pos  = np.arange(n_total, 0, -1, dtype=float)
labels = [r["taxon"] for r in all_entries]
values = [r["lda_score"] if r["enriched_group"] == "A"
          else -r["lda_score"] for r in all_entries]
colors = [C_CD if r["enriched_group"] == "A" else C_NC for r in all_entries]

x_max = max(abs(v) for v in values)
xlim  = x_max * 1.30

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 10), facecolor="white")
ax.set_facecolor("white")

# Subtle horizontal reference lines at each bar row (behind bars)
for yp in y_pos:
    ax.axhline(yp, color="#e8e8e8", lw=0.7, zorder=1)

# Bars
ax.barh(y_pos, values, height=0.72,
        color=colors, edgecolor="none", zorder=3)

# Zero line
ax.axvline(0, color="#2c3e50", lw=2.5, zorder=5)

# Separator between CD and NC groups — midpoint between last CD row and first NC row
sep_y = (y_pos[n_crc - 1] + y_pos[n_crc]) / 2
ax.axhline(sep_y, color="#aaaaaa", lw=1.4, ls="--", alpha=0.7, zorder=4)

# ── LDA score labels at bar tips ──────────────────────────────────────────────
for yp, val, entry in zip(y_pos, values, all_entries):
    offset = 0.08
    if val >= 0:
        ax.text(val + offset, yp, f"{entry['lda_score']:.2f}",
                ha="left", va="center", fontsize=FS - 2,
                fontfamily=FONT, color="#333333")
    else:
        ax.text(val - offset, yp, f"{entry['lda_score']:.2f}",
                ha="right", va="center", fontsize=FS - 2,
                fontfamily=FONT, color="#333333")

# ── Y-axis: italic genus names colored by group ───────────────────────────────
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=FS, fontfamily=FONT, fontstyle="italic")
for tick, entry in zip(ax.get_yticklabels(), all_entries):
    tick.set_color(C_CD if entry["enriched_group"] == "A" else C_NC)

# ── X-axis ────────────────────────────────────────────────────────────────────
ax.set_xlim(-xlim, xlim)
ax.set_xlabel("LDA Score (log$_{10}$)", fontsize=FS,
              fontfamily=FONT, color="#2c3e50", labelpad=10)

x_ticks = np.arange(0, int(xlim) + 1, 1.0)
all_ticks = sorted(set(list(-x_ticks[1:]) + list(x_ticks)))
ax.set_xticks(all_ticks)
ax.set_xticklabels([f"{int(abs(t))}" for t in all_ticks],
                   fontsize=FS, fontfamily=FONT)
ax.tick_params(axis="x", length=7, width=2.5, color="#555555")
ax.tick_params(axis="y", length=0, pad=6)

# ── Spines ────────────────────────────────────────────────────────────────────
for sp in ["top", "right", "left"]:
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_linewidth(2.5)
ax.spines["bottom"].set_color("#444444")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title("LEfSe Analysis: CD vs NC",
             fontsize=FS + 6, fontfamily=FONT, fontweight="bold",
             color="#2c3e50", pad=16)

# ── Legend (lower right — NC bars extend left so this corner is empty) ────────
legend_handles = [
    mpatches.Patch(facecolor=C_CD, label=f"CD-enriched  (n={len(crc_all)})"),
    mpatches.Patch(facecolor=C_NC,  label=f"NC-enriched  (n={len(nc_all)})"),
]
ax.legend(handles=legend_handles, fontsize=FS - 2, framealpha=0.95,
          edgecolor="#cccccc", loc="lower right",
          handlelength=1.6, borderpad=1.0,
          prop={"family": FONT, "size": FS - 2})

# ── Footer note (figure-level, below x-label, no overlap) ────────────────────
plt.tight_layout(rect=[0, 0.08, 1, 1])

fig.text(0.5, 0.025,
         "LDA threshold >= 2.0  |  Kruskal-Wallis p < 0.05  |  CD n=2,173  |  NC n=82,106",
         ha="center", va="bottom",
         fontsize=FS, fontfamily=FONT, color="#888888")

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    out = os.path.join(OUTPUT_DIR, f"fig2d.{ext}")
    fig.savefig(out, dpi=600 if ext == "png" else 300,
                bbox_inches="tight", facecolor="white")
    print(f"  → {out}")
plt.close(fig)
print("Done.")
