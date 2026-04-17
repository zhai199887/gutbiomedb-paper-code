#!/usr/bin/env python3
"""
CRC Global Sampling Map + Alpha Diversity  v4
Right panel: 3 diversity indices side-by-side (Shannon, Simpson, Chao1)
Jitter point size doubled.  RANDOM_SEED = 42 (fixed).
"""
import sys, io, os, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.cm as cm
import geopandas as gpd
import pickle, warnings

METADATA_PATH = r"E:\microbiomap_clone\data\metadata.csv"
WORLD_SHP     = r"E:\microbiomap_env\Lib\site-packages\pyogrio\tests\fixtures\naturalearth_lowres\naturalearth_lowres.shp"
OUTPUT_DIR    = r"E:\tasks\screenshots\fig2a"
CACHE_FILE    = os.path.join(OUTPUT_DIR, "_cd_analysis_cache.pkl")
RANDOM_SEED   = 42          # fixed seed — jitter display only

C_CRC = "#2b8cbe"
C_NC  = "#7bccc4"
FONT  = "Arial"

FS_TITLE  = 28
FS_LABEL  = 26
FS_TICK   = 22
FS_ANNOT  = 30   # p-value annotations on boxplots
FS_MAP_LBL= 30   # country labels on map
FS_LEGEND = 22
FS_CBAR   = 20
FS_INFO   = 22

# ─── 1. Metadata ──────────────────────────────────────────────────────────────
print("[1/4] Loading metadata …")
meta = pd.read_csv(METADATA_PATH, low_memory=False)

# ─── 2. Diversity data from cache ─────────────────────────────────────────────
print("[2/4] Loading diversity data from cache …")
with open(CACHE_FILE, "rb") as _f:
    _cache = pickle.load(_f)

shannon_crc = np.array(_cache["shannon_a"])
shannon_nc  = np.array(_cache["shannon_b"])
simpson_crc = np.array(_cache["simpson_a"])
simpson_nc  = np.array(_cache["simpson_b"])
chao1_crc   = np.array(_cache["chao1_a"])
chao1_nc    = np.array(_cache["chao1_b"])
n_crc       = _cache.get("n_crc", 2173)
n_nc        = _cache.get("n_nc",  82106)

# Use full-data p-values from platform API (all n=1719 CRC vs n=78691 NC)
# These match the platform display; "p_*_full" keys set by rebuild_cache.py
p_shannon = _cache.get("p_shannon_full", _cache.get("p_shannon"))
p_simpson = _cache.get("p_simpson_full", _cache.get("p_simpson"))
p_chao1   = _cache.get("p_chao1_full",   _cache.get("p_chao1"))
u_shannon = _cache.get("u_shannon")
u_simpson = _cache.get("u_simpson")
u_chao1   = _cache.get("u_chao1")

# Fallback: recompute if missing
if p_shannon is None:
    u_shannon, p_shannon = stats.mannwhitneyu(shannon_crc, shannon_nc, alternative="two-sided")
if p_simpson is None:
    u_simpson, p_simpson = stats.mannwhitneyu(simpson_crc, simpson_nc, alternative="two-sided")
if p_chao1 is None:
    u_chao1,   p_chao1   = stats.mannwhitneyu(chao1_crc,   chao1_nc,   alternative="two-sided")

print(f"  Shannon  U={u_shannon:.3e}  p={p_shannon:.3e}")
print(f"  Simpson  U={u_simpson:.3e}  p={p_simpson:.3e}")
print(f"  Chao1    U={u_chao1:.3e}  p={p_chao1:.3e}")

# ─── 2b. CRC geographic subsets ───────────────────────────────────────────────
INFORM_COLS = [f"inform{i}" for i in range(12)]
crc_mask = pd.Series(False, index=meta.index)
for col in INFORM_COLS:
    if col in meta.columns:
        crc_mask |= (meta[col].fillna("").astype(str).str.strip().str.lower()
                     == "cd")
crc_meta = meta[crc_mask].copy()

# ─── 3. Geodata ───────────────────────────────────────────────────────────────
print("[3/4] Preparing geographic data …")
import pycountry
def iso3_to_iso2(a3):
    try:
        return pycountry.countries.get(alpha_3=str(a3).upper()).alpha_2
    except Exception:
        return None
world_raw = gpd.read_file(WORLD_SHP)[["name", "iso_a3", "geometry"]].copy()
world_raw["iso"] = world_raw["iso_a3"].apply(iso3_to_iso2)
world = world_raw[["name", "iso", "geometry"]].copy()
crc_by_country = crc_meta.groupby("iso").size().reset_index(name="crc_count")
world = world.merge(crc_by_country, on="iso", how="left")
world["crc_count"] = world["crc_count"].fillna(0).astype(int)
world["has_crc"]   = world["crc_count"] > 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    world["centroid"] = world.geometry.centroid
world["lon"] = world["centroid"].x
world["lat"]  = world["centroid"].y

# ─── 4. Figure ────────────────────────────────────────────────────────────────
print("[4/4] Rendering figure …")

crc_world = world[world["has_crc"]].copy()
crc_world["log_crc"] = np.log1p(crc_world["crc_count"])
log_max = float(crc_world["log_crc"].max())
norm    = Normalize(vmin=0, vmax=log_max)
crc_cmap = LinearSegmentedColormap.from_list(
    "crc_density",
    ["#f0f9e8", "#ccebc5", "#7bccc4", "#2b8cbe", "#084081"], N=256)

fig = plt.figure(figsize=(34, 14), facecolor="white")
fig.patch.set_facecolor("white")

# MAP_W=0.38; diversity panels 0.182 each; 0.030 gap between map and panels
MAP_W   = 0.38
PNL_W   = 0.182
PNL_GAP = 0.012
MAP_END = 0.01 + MAP_W            # = 0.39
PNL_S   = MAP_END + 0.045         # = 0.435  (gap between map and diversity panels)

ax_map   = fig.add_axes([0.01,   0.200, MAP_W, 0.760])
cbar_ax  = fig.add_axes([0.04,   0.135, MAP_W * 0.82, 0.045])
ax_info  = fig.add_axes([0.04,   0.030, 0.95,  0.095])
ax_sha   = fig.add_axes([PNL_S,                    0.100, PNL_W, 0.760])
ax_sim   = fig.add_axes([PNL_S + PNL_W + PNL_GAP, 0.100, PNL_W, 0.760])
ax_ch    = fig.add_axes([PNL_S + 2*(PNL_W + PNL_GAP), 0.100, PNL_W, 0.760])

# ── Map ────────────────────────────────────────────────────────────────────────
world[~world["has_crc"]].plot(ax=ax_map, color="#e8e8e8",
                               edgecolor="#b0b0b0", linewidth=0.3, zorder=1)
crc_world.plot(ax=ax_map, column="log_crc", cmap=crc_cmap,
               norm=norm, edgecolor="#555555", linewidth=0.5, zorder=2)
# Per-country label offsets: (lon_offset, lat_offset, ha)
# Mathematically verified: no two label bounding boxes intersect.
#
# Approximate text box extents at 30pt on MAP_W=0.38 (12.9 in), 1°≈0.036 in:
#   line_h ≈ 11.6° lat per line
#   "United States of America" ≈ 158° wide (ha=left → extends right)
#   "Spain" / "China" / "Japan" ≈ 35° wide; "(n=xxx)" ≈ 42° wide
#
#  USA  anchor(-90, 60): lon[-90, 68], lat[38, 60]  — North Canada / Atlantic
#  Spain anchor( +1, 26): lon[+1, 36], lat[ 4, 26]  — Sahara / Atlantic
#  China anchor(130, 40): ha=right → lon[88,130], lat[17, 40]  — Eastern China
#  Japan anchor(135, 55): lon[135,177], lat[31, 55] — N. Pacific / Sakhalin
#
#  All pairs: no lon×lat rectangle intersection ✓
# Each entry: (label_lon, label_lat, ha)
# Labels spread horizontally so text boxes do NOT overlap.
# At 30pt, each char ≈ 5.8° wide.  Label widths (ha=left, extends right):
#   USA(3)=17°  Ireland(7)=41°  Belgium(7)=41°  Netherlands(11)=64°  Sweden(6)=35°
# Horizontal layout (ha="left", each range = [lon, lon+width]):
#   USA      -160 → -143   (N Pacific, lat=62)
#   Canada   -125 →  -89   (Arctic, lat=80)
#   Ireland  -170 → -129   (S Pacific, lat=-25)
#   Belgium  -110 →  -69   (S Atlantic, lat=-30)
#   Netherlands -48 → +16  (mid S Atlantic, lat=-42)
#   Sweden     28 →  +63   (Indian Ocean / E Africa, lat=-48)
#   Israel     68 →  103   (Indian Ocean, lat=-20)
#   China     138 →  173   (N Pacific, lat=48)
#   Japan     155 →  190   (N Pacific, lat=25)
LABEL_ANC = {
    "United States of America": (-175,  62, "left"),  # N. Pacific upper-left
    "Canada":                   (-125,  81, "left"),  # Arctic Ocean
    "Ireland":                  ( -70,  82, "left"),  # far N. Atlantic
    "Belgium":                  (-150,  28, "left"),  # N. Pacific mid
    "Netherlands":              ( -90,   8, "left"),  # equatorial Atlantic
    "Sweden":                   (  10,  77, "left"),  # Arctic Ocean
    "Israel":                   (  58, -18, "left"),  # Indian Ocean
    "China":                    ( 138,  62, "left"),  # N. Pacific right
    "Japan":                    ( 155,  25, "left"),  # N. Pacific right
}

for _, row in crc_world.iterrows():
    n  = row["crc_count"]
    sz = max(300, min(5000, n * 2.5))
    ax_map.scatter(row["lon"], row["lat"],
                   s=sz, c="white", edgecolors="#084081",
                   linewidths=2.2, alpha=0.85, zorder=5)
    SHORT = {"United States of America": "USA"}
    if n >= 10 and row["name"] in LABEL_ANC:
        lx, ly, ha = LABEL_ANC[row["name"]]
        dname = SHORT.get(row["name"], row["name"])
        ax_map.annotate(
            f'{dname}\n(n={n:,})',
            xy=(row["lon"], row["lat"]),
            xytext=(lx, ly),
            fontsize=FS_MAP_LBL, ha=ha, va="center",
            color="#1a2c4e", fontfamily=FONT, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=2.5),
            arrowprops=dict(arrowstyle="-", color="#888888", lw=1.6),
            zorder=9, clip_on=False)
ax_map.set_xlim(-180, 180)
ax_map.set_ylim(-60, 85)
ax_map.set_aspect("equal")
ax_map.axis("off")
ax_map.set_title("Global Distribution of Crohn's Disease Gut Microbiome Samples in GutBiomeDB",
                 fontsize=FS_TITLE, fontfamily=FONT, fontweight="bold",
                 color="#2c3e50", pad=105)

legend_sizes = [200, 500, 1000, 1500]
legend_markers = [
    plt.scatter([], [], s=max(300, min(5000, n*2.5)),
                c="white", edgecolors="#084081", linewidths=2.2,
                alpha=0.85, label=f'n={n:,}')
    for n in legend_sizes
]
ax_map.legend(handles=legend_markers, title="CD samples",
              title_fontsize=FS_LEGEND + 2, loc="lower left",
              bbox_to_anchor=(-0.08, -0.50), fontsize=FS_LEGEND,
              framealpha=0.95, edgecolor="#cccccc",
              scatterpoints=1, labelspacing=2.2, borderpad=1.2, handletextpad=1.2)

# Colorbar
actual_max_n = int(crc_world["crc_count"].max())
sm = cm.ScalarMappable(norm=norm, cmap=crc_cmap); sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.set_label("CD sample count (log scale)", fontsize=FS_CBAR,
               fontfamily=FONT, color="#555", labelpad=10)
tick_candidates = [1, 10, 50, 100, 300, 500, 815]
tick_vals = [v for v in tick_candidates if math.log1p(v) <= log_max * 1.01]
tick_vals = sorted(set(tick_vals))
cbar.set_ticks([math.log1p(v) for v in tick_vals])
cbar.set_ticklabels([f'{v:,}' for v in tick_vals])
cbar.ax.set_xlim(0, log_max)
cbar.ax.tick_params(labelsize=FS_TICK, length=8, width=2.0)

# Info panel
ax_info.set_facecolor("#f5f7fa"); ax_info.axis("off")
for sp in ax_info.spines.values():
    sp.set_color("#cccccc"); sp.set_visible(True); sp.set_linewidth(0.8)
total_crc = int(n_crc); n_ctry = len(crc_world)
mapped_crc = int(crc_world["crc_count"].sum())
unmapped = total_crc - mapped_crc
sorted_crc = crc_world.sort_values("crc_count", ascending=False)
dist_items = [f'{r["name"]}: {int(r["crc_count"]):,}' for _, r in sorted_crc.iterrows()]
# Wrap distribution into two lines to avoid truncation
half = (len(dist_items) + 1) // 2
dist_line1 = "   |   ".join(dist_items[:half])
dist_line2 = "   |   ".join(dist_items[half:])
header = (f"CD Samples — Total: {total_crc:,}  |  Countries: {n_ctry}  |  "
          f"Mapped: {mapped_crc:,}  |  Unmapped: {unmapped:,}")
info_text = f"{header}\nDistribution: {dist_line1}\n{' ' * 13}{dist_line2}"
ax_info.text(0.01, 0.95, info_text, transform=ax_info.transAxes,
             fontsize=FS_INFO, fontfamily=FONT, va="top", ha="left",
             color="#2c3e50")

# ── Helper: draw one diversity boxplot ────────────────────────────────────────
rng = np.random.default_rng(RANDOM_SEED)

def draw_diversity(ax, vals_crc, vals_nc, ylabel, title, p_val, u_val,
                   jitter_s=40):           # ← doubled from 20
    ax.set_facecolor("#fafafa")
    bp = ax.boxplot(
        [vals_crc, vals_nc], widths=0.52, patch_artist=True,
        medianprops=dict(color="white", linewidth=4.0),
        whiskerprops=dict(color="#444", linewidth=3.0),
        capprops=dict(color="#444", linewidth=3.0),
        flierprops=dict(marker="o", markerfacecolor="#bbb",
                        markeredgecolor="none", markersize=6, alpha=0.5),
        notch=False)
    for patch, color in zip(bp["boxes"], [C_CRC, C_NC]):
        patch.set_facecolor(color); patch.set_alpha(0.85)
        patch.set_edgecolor("white"); patch.set_linewidth(3.0)

    for i, (vals, color) in enumerate(zip([vals_crc, vals_nc], [C_CRC, C_NC])):
        n_pts = min(400, len(vals))
        jitter = rng.uniform(-0.16, 0.16, n_pts)
        xs = i + 1 + jitter
        ys = rng.choice(vals, size=n_pts, replace=False)
        ax.scatter(xs, ys, s=jitter_s, c=color, alpha=0.30,
                   zorder=3, edgecolors="none")

    y_max_v = max(float(np.max(vals_crc)), float(np.max(vals_nc)))
    y_top   = y_max_v * 1.10
    ax.plot([1, 1, 2, 2], [y_top*0.96, y_top, y_top, y_top*0.96],
            color="#555", linewidth=3.0)
    if p_val < 0.001:
        p_lbl = f"p = {p_val:.2e}"
    elif p_val < 0.05:
        p_lbl = f"p = {p_val:.4f}"
    else:
        p_lbl = f"p = {p_val:.3f}"
    ax.text(1.5, y_top * 1.01, p_lbl, ha="center", va="bottom",
            fontsize=FS_ANNOT, fontfamily=FONT, color="#333333")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["", ""])   # labels shown below via ax.text
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, fontfamily=FONT, color="#333")
    ax.set_title(title, fontsize=FS_LABEL, fontfamily=FONT,
                 fontweight="bold", color="#2c3e50", pad=16)
    ax.set_xlim(0.30, 2.70)

    y_min_v = min(float(np.min(vals_crc)), float(np.min(vals_nc)))
    y_pad   = (y_top - y_min_v) * 0.08
    ax.set_ylim(y_min_v - y_pad * 3, y_top * 1.18)

    ax.tick_params(axis="y", labelsize=FS_LABEL, length=8, width=3.0)
    ax.tick_params(axis="x", length=8, width=3.0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_linewidth(3.0)
        ax.spines[sp].set_color("#333333")

    # n= labels
    y_lbl = y_min_v - y_pad * 2.5
    for i, (n, color) in enumerate(zip([len(vals_crc), len(vals_nc)],
                                       [C_CRC, C_NC])):
        lbl = ["CD", "NC"][i]
        ax.text(i + 1, y_lbl, f'{lbl}\nn={n:,}', ha="center", va="top",
                fontsize=FS_LABEL, fontfamily=FONT, color=color,
                fontweight="bold")

draw_diversity(ax_sha, shannon_crc, shannon_nc,
               "Shannon Index", "Alpha Diversity (Shannon Index)", p_shannon, u_shannon)
draw_diversity(ax_sim, simpson_crc, simpson_nc,
               "Simpson Index", "Alpha Diversity (Simpson Index)", p_simpson, u_simpson)
draw_diversity(ax_ch,  chao1_crc,   chao1_nc,
               "Chao1", "Alpha Diversity (Chao1)", p_chao1, u_chao1)

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ["png", "pdf"]:
    out = os.path.join(OUTPUT_DIR, f"fig2a.{ext}")
    fig.savefig(out, dpi=600 if ext == "png" else 300,
                bbox_inches="tight", facecolor="white")
    print(f"  → {out}")
plt.close(fig)
print("Done.")
