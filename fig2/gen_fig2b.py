#!/usr/bin/env python3
"""
Fig 1b PCoA — half-violin + boxplot + jitter margin style (R reference)
  - Top  margin: PC1 distribution per group (horizontal)
  - Right margin: PC2 distribution per group (vertical)
  - Main: scatter + 95% ellipses
Color: CD=#2b8cbe, NC=#7bccc4  |  RANDOM_SEED = 42
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
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde

OUTPUT_DIR  = r"E:\tasks\screenshots\fig2b"
CACHE_FILE  = os.path.join(OUTPUT_DIR, "_cd_analysis_cache.pkl")
RANDOM_SEED = 42
C_CD = "#2b8cbe"
C_NC  = "#7bccc4"
FONT  = "Arial"
FS    = 22

print("Loading cache …")
with open(CACHE_FILE, "rb") as f:
    c = pickle.load(f)

coords      = np.array(c["pcoa_coords"])
groups      = np.array(c["pcoa_groups"])
pcoa_var    = c["pcoa_var"]
perm_result = c.get("perm_result", {})

mask_crc = groups == "A"
mask_nc  = groups == "B"
pc1_crc, pc2_crc = coords[mask_crc, 0], coords[mask_crc, 1]
pc1_nc,  pc2_nc  = coords[mask_nc,  0], coords[mask_nc,  1]
print(f"  CD n={mask_crc.sum()}, NC n={mask_nc.sum()}")

rng = np.random.default_rng(RANDOM_SEED)

# ── Shared axis limits ─────────────────────────────────────────────────────────
pad  = 0.07
x_lo = coords[:, 0].min() - pad;  x_hi = coords[:, 0].max() + pad
y_lo = coords[:, 1].min() - pad;  y_hi = coords[:, 1].max() + pad

# ── Layout ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 11), facecolor="white")
gs  = gridspec.GridSpec(
    2, 2,
    height_ratios=[1.4, 4], width_ratios=[4, 1.4],
    hspace=0.02, wspace=0.02,
    left=0.12, right=0.97, top=0.94, bottom=0.10
)
ax_top    = fig.add_subplot(gs[0, 0])
ax_main   = fig.add_subplot(gs[1, 0])
ax_right  = fig.add_subplot(gs[1, 1])
ax_corner = fig.add_subplot(gs[0, 1])
ax_corner.set_visible(False)

# ── Helpers ────────────────────────────────────────────────────────────────────
def half_violin_h(ax, data, y_pos, color, nudge=0.18, max_h=0.58, alpha=0.35):
    kde = gaussian_kde(data, bw_method="scott")
    xr  = np.linspace(data.min(), data.max(), 300)
    den = kde(xr); den = den / den.max() * max_h
    ax.fill_between(xr, y_pos + nudge, y_pos + nudge + den,
                    color=color, alpha=alpha, linewidth=0, zorder=2)
    ax.plot(xr, y_pos + nudge + den, color=color, lw=1.2, alpha=0.65, zorder=2)

def half_violin_v(ax, data, x_pos, color, nudge=0.18, max_w=0.58, alpha=0.35):
    kde = gaussian_kde(data, bw_method="scott")
    yr  = np.linspace(data.min(), data.max(), 300)
    den = kde(yr); den = den / den.max() * max_w
    ax.fill_betweenx(yr, x_pos + nudge, x_pos + nudge + den,
                     color=color, alpha=alpha, linewidth=0, zorder=2)
    ax.plot(x_pos + nudge + den, yr, color=color, lw=1.2, alpha=0.65, zorder=2)

def box_h(ax, data, y_pos, color, nudge=0.20, w=0.13):
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    wlo = max(data.min(), q1 - 1.5*iqr)
    whi = min(data.max(), q3 + 1.5*iqr)
    yc  = y_pos + nudge
    ax.add_patch(mpatches.Rectangle((q1, yc - w/2), q3-q1, w,
                 facecolor="white", edgecolor=color, lw=2.2, zorder=5))
    ax.plot([med, med], [yc-w/2, yc+w/2], color=color, lw=2.5, zorder=6)
    ax.plot([wlo, q1], [yc, yc], color=color, lw=1.6, zorder=4)
    ax.plot([q3, whi], [yc, yc], color=color, lw=1.6, zorder=4)
    cap = w * 0.35
    for xv in [wlo, whi]:
        ax.plot([xv, xv], [yc-cap, yc+cap], color=color, lw=1.6, zorder=4)

def box_v(ax, data, x_pos, color, nudge=0.20, w=0.13):
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    wlo = max(data.min(), q1 - 1.5*iqr)
    whi = min(data.max(), q3 + 1.5*iqr)
    xc  = x_pos + nudge
    ax.add_patch(mpatches.Rectangle((xc-w/2, q1), w, q3-q1,
                 facecolor="white", edgecolor=color, lw=2.2, zorder=5))
    ax.plot([xc-w/2, xc+w/2], [med, med], color=color, lw=2.5, zorder=6)
    ax.plot([xc, xc], [wlo, q1], color=color, lw=1.6, zorder=4)
    ax.plot([xc, xc], [q3, whi], color=color, lw=1.6, zorder=4)
    cap = w * 0.35
    for yv in [wlo, whi]:
        ax.plot([xc-cap, xc+cap], [yv, yv], color=color, lw=1.6, zorder=4)

# ── TOP panel (PC1) ───────────────────────────────────────────────────────────
ax_top.set_facecolor("white")
for data, color, ypos, label in [
        (pc1_crc, C_CD, 2, "CD"),
        (pc1_nc,  C_NC,  1, "NC")]:
    jit = rng.uniform(-0.09, 0.09, len(data))
    ax_top.scatter(data, ypos + jit, s=28, c=color, alpha=0.38,
                   edgecolors="none", zorder=3)
    half_violin_h(ax_top, data, ypos, color)
    box_h(ax_top, data, ypos, color)
    ax_top.text(x_lo - 0.06, ypos + 0.25, label, fontsize=FS-1,
                fontfamily=FONT, color=color, fontweight="bold", va="center")

ax_top.set_xlim(x_lo, x_hi)
ax_top.set_ylim(0.5, 3.10)
ax_top.axis("off")
_pos = ax_top.get_position()
ax_top.set_position([_pos.x0, _pos.y0 + 0.02, _pos.width, _pos.height])

# ── RIGHT panel (PC2) ─────────────────────────────────────────────────────────
ax_right.set_facecolor("white")
for data, color, xpos, label in [
        (pc2_crc, C_CD, 1, "CD"),
        (pc2_nc,  C_NC,  2, "NC")]:
    jit = rng.uniform(-0.09, 0.09, len(data))
    ax_right.scatter(xpos + jit, data, s=28, c=color, alpha=0.38,
                     edgecolors="none", zorder=3)
    half_violin_v(ax_right, data, xpos, color)
    box_v(ax_right, data, xpos, color)
    ax_right.text(xpos + 0.25, y_hi - 0.01, label, fontsize=FS-1,
                  fontfamily=FONT, color=color, fontweight="bold",
                  va="top", ha="center")

ax_right.set_xlim(0.5, 3.10)
ax_right.set_ylim(y_lo, y_hi)
ax_right.axis("off")

# ── MAIN scatter ───────────────────────────────────────────────────────────────
ax_main.set_facecolor("#fafafa")

def conf_ellipse(ax, x, y, color, n_std=2.0, alpha_f=0.14):
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(xy=(x.mean(), y.mean()), width=w, height=h,
                         angle=theta, facecolor=color, alpha=alpha_f, zorder=1))
    ax.add_patch(Ellipse(xy=(x.mean(), y.mean()), width=w, height=h,
                         angle=theta, facecolor="none",
                         edgecolor=color, lw=2.0, alpha=0.65, zorder=2))

conf_ellipse(ax_main, pc1_nc,  pc2_nc,  C_NC)
conf_ellipse(ax_main, pc1_crc, pc2_crc, C_CD)

ax_main.scatter(pc1_nc,  pc2_nc,  s=116, c=C_NC,  alpha=0.72,
                edgecolors="white", lw=0.5, zorder=3,
                label=f"NC  (n={mask_nc.sum()})")
ax_main.scatter(pc1_crc, pc2_crc, s=116, c=C_CD, alpha=0.72,
                edgecolors="white", lw=0.5, zorder=4,
                label=f"CD  (n={mask_crc.sum()})")

ax_main.axhline(0, color="#aaaaaa", lw=1.2, ls="--", alpha=0.45)
ax_main.axvline(0, color="#aaaaaa", lw=1.2, ls="--", alpha=0.45)

ax_main.set_xlim(x_lo, x_hi)
ax_main.set_ylim(y_lo, y_hi)
ax_main.set_xlabel(f"PCo1 ({pcoa_var[0]:.1f}%)", fontsize=FS,
                   fontfamily=FONT, color="#2c3e50", labelpad=8)
ax_main.set_ylabel(f"PCo2 ({pcoa_var[1]:.1f}%)", fontsize=FS,
                   fontfamily=FONT, color="#2c3e50", labelpad=8)
ax_main.tick_params(axis="both", labelsize=FS-2, width=2.5, length=7)
for sp in ["top", "right"]:
    ax_main.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax_main.spines[sp].set_linewidth(2.5)
    ax_main.spines[sp].set_color("#444444")

leg = ax_main.legend(fontsize=FS-2, framealpha=0.95, edgecolor="#cccccc",
                     loc="upper right", bbox_to_anchor=(1.0, 1.08),
                     markerscale=1.3,
                     handletextpad=0.5, borderpad=0.8)
for t in leg.get_texts():
    t.set_fontfamily(FONT)


# PERMANOVA annotation (top-left of main panel)
r2   = perm_result.get("r_squared", float("nan"))
pv   = perm_result.get("p_value",   float("nan"))
fst  = perm_result.get("f_statistic", float("nan"))
perm_txt = (f"PERMANOVA\nR² = {r2:.4f},  F = {fst:.2f}\n"
            f"p = {pv:.3f}  (999 permutations)")
ax_main.text(0.03, 1.05, perm_txt,
             transform=ax_main.transAxes,
             fontsize=FS-4, fontfamily=FONT, color="#2c3e50",
             va="top", linespacing=1.7,
             bbox=dict(facecolor="white", edgecolor="#cccccc",
                       alpha=0.90, pad=4, boxstyle="round,pad=0.4"))

fig.suptitle(
    "Beta Diversity: PCoA of Bray–Curtis Distances",
    fontsize=FS, fontfamily=FONT, fontweight="bold",
    color="#2c3e50", y=0.98)

# ── Save ─────────────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    out = os.path.join(OUTPUT_DIR, f"fig2b.{ext}")
    fig.savefig(out, dpi=600 if ext == "png" else 300,
                bbox_inches="tight", facecolor="white")
    print(f"  → {out}")
plt.close(fig)
print("Done.")
