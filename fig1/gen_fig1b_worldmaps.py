#!/usr/bin/env python3
"""
Fig 1a — Global Database World Map (choropleth style)
Fixes:
  - Use ISO_A2_EH as fallback key (covers FR, NO, etc.)
  - Manual point markers for small countries not in 110m shapefile (SG, MT)
  - All 72 countries labeled
  - Wider colorbar, no overlapping labels
  - Full-width colorbar
"""

import sys, io, os, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import geopandas as gpd

# ─── Paths ───────────────────────────────────────────────────────────────────
METADATA_PATH = r"E:\microbiomap_clone\data\metadata.csv"
WORLD_SHP     = r"E:\tasks\screenshots\ne_110m_countries.zip"
OUTPUT_DIR    = r"E:\tasks\screenshots\fig1"
FONT = "Arial"

# Manual coordinates for countries too small for 110m shapefile
MANUAL_COORDS = {
    "SG": (103.82, 1.35,  "Singapore"),
    "MT": (14.37,  35.90, "Malta"),
}

# ─── 1. Load metadata ─────────────────────────────────────────────────────────
print("[1/3] Loading metadata …")
meta = pd.read_csv(METADATA_PATH, low_memory=False)
meta = meta[meta["iso"].notna() & (meta["iso"] != "unknown")]

by_country = meta.groupby("iso").agg(
    total=("srr", "count"),
    region=("region", lambda x: x.value_counts().index[0]),
).reset_index()
by_country["log_total"] = np.log1p(by_country["total"])
total_samples = int(meta.shape[0])

print(f"    {total_samples:,} samples, {len(by_country)} countries")
print(f"    ISO codes: {sorted(by_country['iso'].tolist())}")

# ─── 2. World shapefile — use ISO_A2_EH as fallback ──────────────────────────
print("[2/3] Loading world shapefile …")
world_raw = gpd.read_file(WORLD_SHP)

# Build a unified ISO key: prefer ISO_A2, fall back to ISO_A2_EH
world_raw["iso_key"] = world_raw["ISO_A2"].where(
    world_raw["ISO_A2"] != "-99", world_raw["ISO_A2_EH"]
)
world = world_raw[["NAME", "iso_key", "geometry"]].copy()
world.columns = ["name", "iso", "geometry"]

# Merge
world = world.merge(by_country, on="iso", how="left")
world["total"]     = world["total"].fillna(0).astype(int)
world["log_total"] = world["log_total"].fillna(0)
world["region"]    = world["region"].fillna("unknown")
world["has_data"]  = world["total"] > 0

# Compute centroids (suppress CRS warning — this is for label placement only)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    world["centroid"] = world.geometry.centroid
world["lon"] = world["centroid"].x
world["lat"]  = world["centroid"].y

# Check mapping
mapped_isos = set(world.loc[world["has_data"], "iso"])
our_isos = set(by_country["iso"])
unmapped = our_isos - mapped_isos
print(f"    Mapped: {len(mapped_isos)}, unmapped (too small for 110m): {sorted(unmapped)}")

# ─── 3. FIGURE ────────────────────────────────────────────────────────────────
print("[3/3] Rendering figure …")

# Custom teal-blue colormap
cmap_a = LinearSegmentedColormap.from_list(
    "gutbiomedb",
    ["#f0f9e8", "#ccebc5", "#7bccc4", "#2b8cbe", "#084081"],
    N=256
)

world_data = world[world["has_data"]].copy()
log_max = float(world_data["log_total"].max())
norm_a = Normalize(vmin=0, vmax=log_max)

fig, ax = plt.subplots(1, 1, figsize=(20, 10), facecolor="white")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── Base map ───────────────────────────────────────────────────────────────────
world[~world["has_data"]].plot(
    ax=ax, color="#f0f0f0", edgecolor="#d0d0d0", linewidth=0.3, zorder=1
)
world_data.plot(
    ax=ax, column="log_total", cmap=cmap_a,
    norm=norm_a, edgecolor="#888888", linewidth=0.4, zorder=2
)

# ── Large circle markers at country centroids ─────────────────
for _, row in world_data.iterrows():
    n = row["total"]
    sz = max(80, min(8000, (n ** 0.52) * 18))
    ax.scatter(row["lon"], row["lat"],
               s=sz, c="white", edgecolors="#2c3e50",
               linewidths=1.2, alpha=0.50, zorder=4)

# ── Manual markers for small countries ────────────────────────────────────────
for iso, (lon, lat, cname) in MANUAL_COORDS.items():
    if iso in by_country["iso"].values:
        n = int(by_country.loc[by_country["iso"]==iso, "total"].values[0])
        log_v = math.log1p(n)
        rgba = cmap_a(norm_a(log_v))
        sz = max(80, min(8000, (n**0.52)*18))
        ax.scatter(lon, lat, s=sz+200, c=[rgba], edgecolors="#888888",
                   linewidths=1.0, alpha=0.92, zorder=6, marker="D")

# ── Build label data ───────────────────────────────────────────────────────────
label_rows = []
for _, row in world_data.iterrows():
    label_rows.append({
        "name": row["name"], "iso": row["iso"],
        "total": row["total"], "lon": row["lon"], "lat": row["lat"],
        "manual": False
    })
for iso, (lon, lat, cname) in MANUAL_COORDS.items():
    if iso in by_country["iso"].values:
        n = int(by_country.loc[by_country["iso"]==iso, "total"].values[0])
        label_rows.append({
            "name": cname, "iso": iso,
            "total": n, "lon": lon, "lat": lat,
            "manual": True
        })

label_df = pd.DataFrame(label_rows).sort_values("total", ascending=False)

# Format numbers: ≥1000 → "X,XXX" or "XK"
def fmt_n(n):
    if n >= 10000: return f"{n//1000}K"
    if n >= 1000:  return f"{n/1000:.1f}K"
    return str(n)

# ── Label TOP 20 countries only — large font (5× original) ────────────────────
TOP_N = 20
top20 = label_df.head(TOP_N)

# Manual overrides for overlapping clusters (Europe + East/South Asia)
# Keys match name_short (after the replace() chain below)
LABEL_OVERRIDES = {
    # Europe cluster — spread vertically & horizontally
    "Sweden":      (-8,  10, "right"),
    "Finland":     ( 22,  4, "left"),
    "Denmark":     ( 12, 10, "left"),
    "Germany":     ( 18,  0, "left"),
    "Netherlands": (-12, -4, "right"),
    "Italy":       ( 12, -10, "left"),
    "Austria":     ( 28, -16, "left"),
    "United Kingdom": (-14,  4, "right"),
    "Spain":       (-10, -6, "right"),
    # East/South Asia cluster
    "China":       ( -6,  8, "right"),
    "South Korea": ( 18, 10, "left"),
    "Japan":       (  6, -14, "center"),
    "India":       (-14, -6, "right"),
    "Bangladesh":  ( 10,  6, "left"),
}

for _, row in top20.iterrows():
    n = row["total"]; lon = row["lon"]; lat = row["lat"]
    name = row["name"]
    name_short = name  # Unified: always use full country names

    if name_short in LABEL_OVERRIDES:
        dx, dy, ha = LABEL_OVERRIDES[name_short]
    elif lon < -60:   dx, dy, ha = -2.5,  2.5, "right"
    elif lon > 140: dx, dy, ha = 2.5,   2.5, "left"
    elif lat > 60:  dx, dy, ha = 0,     3.5, "center"
    elif lat < -30: dx, dy, ha = 0,    -4.0, "center"
    else:           dx, dy, ha = 1.5,   2.0, "left"

    ax.text(lon + dx, lat + dy,
            f"{name_short}\n({fmt_n(n)})",
            ha=ha, va="bottom", fontsize=18, fontweight="bold",
            fontfamily=FONT, color="#1a252f",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.80, pad=0.8),
            zorder=7)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(-180, 180)
ax.set_ylim(-58, 84)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(
    "Global Geographic Distribution of GutBiomeDB Samples",
    fontsize=40, fontfamily=FONT, fontweight="bold", color="#2c3e50", pad=16
)
# Subtitle omitted

# ── Colorbar — full-width, tall ───────────────────────────
cbar_ax = fig.add_axes([0.02, 0.03, 0.96, 0.048])
sm = cm.ScalarMappable(norm=norm_a, cmap=cmap_a)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Sample count (log scale)", fontsize=25, fontfamily=FONT,
               color="#444444", labelpad=10)

# Custom ticks — drop 5,000 to avoid overlap with 10,000
tick_actual = [1, 10, 100, 1000, 10000, 50000]
tick_actual = [v for v in tick_actual if math.log1p(v) <= log_max * 1.02]
tick_pos    = [math.log1p(v) for v in tick_actual]
tick_labels = ["1", "10", "100", "1,000", "10,000", "50,000"][:len(tick_actual)]
cbar.set_ticks(tick_pos)
cbar.set_ticklabels(tick_labels)
cbar.ax.tick_params(labelsize=22, length=8, width=1.5)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.10, 1, 1])
for ext in ["png", "pdf"]:
    out = os.path.join(OUTPUT_DIR, f"fig1b.{ext}")
    fig.savefig(out, dpi=600 if ext=="png" else 300,
                bbox_inches="tight", facecolor="white")
    print(f"  → {out}")
plt.close(fig)
print("Done.")
