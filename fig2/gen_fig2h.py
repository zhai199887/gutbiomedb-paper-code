#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig 2h -- CD vs NC Full Lifecycle Stacked Area Chart (Top 30 Genera)
+ Statistics: PERMANOVA (Bray-Curtis, 999 perms), Kruskal-Wallis+BH-FDR, Spearman rho

Data logic: directly imports platform backend main.py, calls _lifecycle_internal()
Color scheme: identical to fig1e phylumColors.ts (build_genus_colors)
Layout: left (CD) + right (NC) stacked area charts + right-side legend
Output:
  E:/tasks/screenshots/fig2h/fig2h.{png,pdf}
  E:/tasks/screenshots/fig2h/fig2h_stats_NC.txt
  E:/tasks/screenshots/fig2h/fig2h_stats_CD.txt
  E:/tasks/screenshots/fig2h/fig2h_kruskal_NC.csv
  E:/tasks/screenshots/fig2h/fig2h_spearman_NC.csv
"""

import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── Add platform backend api directory to PATH and import platform code ──────
API_DIR = r"E:\microbiomap_clone\compendium_website\api"
sys.path.insert(0, API_DIR)
os.chdir(API_DIR)   # main.py uses relative paths that depend on this directory

print("[1/5] Importing platform backend (main.py) ...")
from main import (
    _lifecycle_internal,
    AGE_GROUP_ORDER as BACKEND_AGE_ORDER,
    get_metadata,
    get_abundance,
    relative_abundance_matrix,
    aggregate_by_level,
    is_valid_genus,
    _lifecycle_filter_meta,
)
print("      Import OK")

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb
from PIL import Image
from scipy.interpolate import PchipInterpolator   # equivalent to d3.curveMonotoneX
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
import pandas as pd
from collections import defaultdict, OrderedDict
import warnings
warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = r"E:\tasks\screenshots\fig2h"
FONT       = "Arial"
TOP_N      = 30
PERMANOVA_N_PERMS    = 999      # 999 permutations
PERMANOVA_MAX_SAMPLE = 500      # max 500 samples per age group (avoid memory overflow)
SPEARMAN_ALPHA       = 0.05     # FDR threshold

C_CD = "#2b8cbe"
C_NC = "#22863a"

AGE_GROUP_ORDER = ["Infant", "Child", "Adolescent", "Adult",
                   "Older_Adult", "Oldest_Old", "Centenarian"]
# Age group ordinal values (for Spearman continuous encoding)
AGE_ORDINAL = {ag: i+1 for i, ag in enumerate(AGE_GROUP_ORDER)}

AGE_LABEL = {
    "Infant":      "Infant",
    "Child":       "Child",
    "Adolescent":  "Adolescent",
    "Adult":       "Adult",
    "Older_Adult": "Older\nAdult",
    "Oldest_Old":  "Oldest\nOld",
    "Centenarian": "Centenarian",
}

# ── Color system identical to fig1e (gen_fig1e_circular.py) ─────────────────
import matplotlib.colors as mcolors

_PHYLUM_COLOR_RANGES = {
    "Bacillota":               ("#1A6B82", "#B2E8EF"),
    "Firmicutes":              ("#1A6B82", "#B2E8EF"),
    "Bacteroidota":            ("#253D6E", "#8EB3D5"),
    "Bacteroidetes":           ("#253D6E", "#8EB3D5"),
    "Pseudomonadota":          ("#00836F", "#91D1C2"),
    "Proteobacteria":          ("#00836F", "#91D1C2"),
    "Actinomycetota":          ("#D4735A", "#F7B89A"),
    "Actinobacteria":          ("#D4735A", "#F7B89A"),
    "Fusobacteriota":          ("#5271A4", "#B3BDDA"),
    "Fusobacteria":            ("#5271A4", "#B3BDDA"),
    "Thermodesulfobacteriota": ("#4A148C", "#CE93D8"),
    "Verrucomicrobiota":       ("#C41435", "#F9C0B5"),
    "Verrucomicrobia":         ("#C41435", "#F9C0B5"),
    "Other":                   ("#424242", "#BDBDBD"),
}

LARGE_PHY_THRESH = 8
_QUAL_PALETTE = [
    "#4DBBD5", "#3C5488", "#00A087", "#F39B7F", "#8491B4",
    "#1DBD9E", "#E8836A", "#2B8FA3", "#6484B8", "#56D0B8",
    "#D4735A", "#6ECCD8", "#5271A4", "#91D1C2", "#F7B89A",
    "#0D5C70", "#7A9DC8", "#3CCAB0", "#B3BDDA", "#37A8BC",
    "#4A6BA8", "#72D8C2", "#C47C60", "#5EC3D2", "#9CA8C8",
    "#A8DDD4", "#253D6E", "#EBA080", "#5BA4A4", "#8EB3D5",
]

PHYLUM_DISP = {
    "Bacteroidota": "Bacteroidota",     "Bacteroidetes": "Bacteroidota",
    "Pseudomonadota": "Pseudomonadota", "Proteobacteria": "Pseudomonadota",
    "Bacillota": "Bacillota (Firmicutes)", "Firmicutes": "Bacillota (Firmicutes)",
    "Actinomycetota": "Actinomycetota", "Actinobacteria": "Actinomycetota",
    "Fusobacteriota": "Fusobacteriota", "Fusobacteria": "Fusobacteriota",
    "Verrucomicrobiota": "Verrucomicrobiota", "Verrucomicrobia": "Verrucomicrobiota",
    "Spirochaetota": "Spirochaetota",   "Spirochaetes": "Spirochaetota",
    "Desulfobacterota": "Desulfobacterota",
    "Thermodesulfobacteriota": "Desulfobacterota",
}

PHYLUM_TITLE_COLORS = {
    ph: rng[0] for ph, rng in _PHYLUM_COLOR_RANGES.items()
}
PHYLUM_TITLE_COLORS["Other/Unknown"] = "#424242"


def _gradient_colors(color_dark, color_light, n):
    if n <= 0:
        return []
    d = np.array(mcolors.to_rgb(color_dark))
    l = np.array(mcolors.to_rgb(color_light))
    if n == 1:
        rgb = ((d + l) / 2 * 255).clip(0, 255).astype(int)
        return [f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"]
    colors = []
    for i in range(n):
        t = i / (n - 1)
        rgb = ((d * (1 - t) + l * t) * 255).clip(0, 255).astype(int)
        colors.append(f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}")
    return colors


def build_genus_colors(all_genera, genus_to_phylum=None):
    if genus_to_phylum is None:
        genus_to_phylum = {}
    phylum_genera: dict = {}
    for g in sorted(set(all_genera)):
        if g == "Other":
            continue
        phy = genus_to_phylum.get(g, "Other")
        phylum_genera.setdefault(phy, []).append(g)
    result = {"Other": "#BDBDBD"}
    for phy, genera in phylum_genera.items():
        genera_sorted = sorted(genera)
        if len(genera_sorted) >= LARGE_PHY_THRESH:
            for i, g in enumerate(genera_sorted):
                result[g] = _QUAL_PALETTE[i % len(_QUAL_PALETTE)]
        else:
            dark, light = _PHYLUM_COLOR_RANGES.get(phy, _PHYLUM_COLOR_RANGES["Other"])
            shades = _gradient_colors(dark, light, len(genera_sorted))
            for g, c in zip(genera_sorted, shades):
                result[g] = c
    result_rgb = {g: to_rgb(c) for g, c in result.items()}
    return result_rgb, result


# ─────────────────────────────────────────────────────────────────────────────
#  STATISTICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction (BH 1995)."""
    pvals = np.array(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return np.array([])
    order = np.argsort(pvals)
    ranked_p = pvals[order]
    adjusted = np.minimum(1.0, ranked_p * m / (np.arange(1, m + 1)))
    # monotone non-decreasing from the tail
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    result = np.empty(m)
    result[order] = adjusted
    return result


def bray_curtis_matrix(X):
    """Bray-Curtis dissimilarity matrix (n_samples x n_features).
    BC(i,j) = sum|xi-xj| / (sum(xi)+sum(xj))
    Uses float64 internally for numerical stability.
    """
    X = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X_norm = X / row_sum          # normalize each row to sum=1
    dist = cdist(X_norm, X_norm, metric="braycurtis")
    # cdist returns NaN when both rows are all-zero (0/0); treat as 0 (identical)
    dist = np.nan_to_num(dist, nan=0.0, posinf=1.0, neginf=0.0)
    np.fill_diagonal(dist, 0.0)   # ensure exact zero diagonal
    return dist.astype(np.float64)


def permanova(dist_matrix, group_labels, n_perms=999, seed=42):
    """
    PERMANOVA (Anderson 2001) on a precomputed square distance matrix.

    SS formulae (Anderson 2001):
      SStotal   = sum_{i} sum_{j} d²_{ij} / (2N)
      SSwithin  = sum_k [ sum_{i in k} sum_{j in k} d²_{ij} / (2*n_k) ]
      SSbetween = SStotal - SSwithin
      pseudo-F  = (SSbetween/df_a) / (SSwithin/df_w)
      R²        = SSbetween / SStotal

    Returns: R2, F_obs, p_val, df_a, df_w
    """
    rng = np.random.default_rng(seed)
    labels = np.array(group_labels, dtype=object)
    n = len(labels)
    unique_groups = np.unique(labels)
    a = len(unique_groups)
    D2 = dist_matrix.astype(np.float64) ** 2

    def ss_total(D2, n):
        return float(np.sum(D2)) / (2.0 * n)

    def ss_within(D2, labels):
        sw = 0.0
        for g in np.unique(labels):
            idx = np.where(labels == g)[0]
            ni  = len(idx)
            if ni < 2:
                continue
            sub = D2[np.ix_(idx, idx)]
            sw += float(np.sum(sub)) / (2.0 * ni)
        return sw

    sst = ss_total(D2, n)
    if sst <= 0.0 or np.isnan(sst):
        return np.nan, np.nan, 1.0, a - 1, n - a

    ssw_obs = ss_within(D2, labels)
    ssa_obs = sst - ssw_obs
    df_a = a - 1
    df_w = n - a

    if df_w <= 0:
        return np.nan, np.nan, 1.0, df_a, df_w
    if ssw_obs <= 0.0 or ssa_obs <= 0.0:
        return 0.0, 0.0, 1.0, df_a, df_w

    F_obs = (ssa_obs / df_a) / (ssw_obs / df_w)
    R2    = ssa_obs / sst

    # Permutation test (999 permutations)
    count_ge = 0
    for _ in range(n_perms):
        perm_labels = rng.permutation(labels)
        ssw_perm = ss_within(D2, perm_labels)
        ssa_perm = sst - ssw_perm
        if ssw_perm <= 0:
            continue
        F_perm = (ssa_perm / df_a) / (ssw_perm / df_w)
        if np.isfinite(F_perm) and F_perm >= F_obs:
            count_ge += 1
    p_val = (count_ge + 1) / (n_perms + 1)
    return float(R2), float(F_obs), float(p_val), int(df_a), int(df_w)


def run_spearman_by_age(genus_rel_matrix, genus_names, sample_ages, age_ordinal):
    """
    Spearman rank correlation: each genus RA vs. ordinal age code.
    Returns DataFrame with columns: genus, rho, pval, adj_pval, significant
    """
    age_codes = np.array([age_ordinal.get(a, np.nan) for a in sample_ages])
    valid_mask = ~np.isnan(age_codes)
    age_codes_v = age_codes[valid_mask]
    rows = []
    pvals = []
    for gi, genus in enumerate(genus_names):
        ra = genus_rel_matrix[valid_mask, gi]
        rho, p = spearmanr(age_codes_v, ra)
        rows.append({"genus": genus, "rho": round(float(rho), 4),
                     "pval": float(p)})
        pvals.append(float(p))
    adj = bh_fdr(pvals)
    for i, row in enumerate(rows):
        row["adj_pval"] = round(float(adj[i]), 6)
        row["significant"] = bool(adj[i] < SPEARMAN_ALPHA)
    return pd.DataFrame(rows)


# ─── 2. Call platform backend function to fetch aggregated data (for figure) ─
print("[2/5] Calling _lifecycle_internal() for aggregated data ...")
print("      (First call loads ~1.5GB data, takes ~2-3 minutes...)")
cd_data = _lifecycle_internal(disease="CD", top_genera=TOP_N, use_cache=True)
print(f"      CD done: total_samples={cd_data['total_samples']:,}, genera={len(cd_data['genera'])}")

nc_data = _lifecycle_internal(disease="NC", top_genera=TOP_N, use_cache=True)
print(f"      NC done: total_samples={nc_data['total_samples']:,}, genera={len(nc_data['genera'])}")

# ─── Align genera (take union) ──────────────────────────────────────────────
cd_genera_raw = [g for g in cd_data["genera"] if g != "Other"]
nc_genera_raw = [g for g in nc_data["genera"] if g != "Other"]
union_genera  = list(cd_genera_raw)
for g in nc_genera_raw:
    if g not in union_genera:
        union_genera.append(g)
final_gen = union_genera[:TOP_N]

phylum_map = {**nc_data.get("phylum_map", {}), **cd_data.get("phylum_map", {})}
print(f"      Final genera ({len(final_gen)}): {final_gen[:5]} ...")

# Kruskal-Wallis + BH-FDR (already computed by backend)
kw_nc = nc_data.get("kruskal_results", [])
kw_cd = cd_data.get("kruskal_results", [])
print(f"      Kruskal-Wallis (NC): {len(kw_nc)} genera tested, "
      f"{sum(1 for r in kw_nc if r.get('significant'))} significant (FDR<0.05)")
print(f"      Kruskal-Wallis (CD): {len(kw_cd)} genera tested, "
      f"{sum(1 for r in kw_cd if r.get('significant'))} significant (FDR<0.05)")

# ─── 3. Load sample-level data for PERMANOVA & Spearman ─────────────────
print("[3/5] Loading sample-level data for PERMANOVA & Spearman ...")
meta  = get_metadata()
abund = get_abundance()

def get_sample_level(disease_label):
    """Return (genus_rel_matrix, sample_ages, genus_names) for a disease group."""
    filtered = _lifecycle_filter_meta(meta, disease=disease_label, country="")
    valid_keys = abund.index.intersection(filtered["sample_key"].unique()).tolist()
    if len(valid_keys) < 10:
        return None, None, None

    raw = abund.loc[valid_keys].to_numpy(dtype=np.float32, copy=True)
    rel = relative_abundance_matrix(raw).astype(np.float32, copy=False)
    col_names = abund.columns.tolist()
    genus_rel, genus_labels, _ = aggregate_by_level(rel, col_names, "genus")
    valid_pos = [idx for idx, g in enumerate(genus_labels) if is_valid_genus(g)]
    genus_rel = genus_rel[:, valid_pos]
    genus_names = [genus_labels[i] for i in valid_pos]

    # Restrict to final_gen columns only
    col_map = {g: i for i, g in enumerate(genus_names)}
    target_cols = [col_map[g] for g in final_gen if g in col_map]
    target_names = [final_gen[i] for i, g in enumerate(final_gen) if g in col_map]
    genus_rel_top = genus_rel[:, target_cols]

    age_lookup = (
        filtered[["sample_key", "age_group"]]
        .drop_duplicates("sample_key")
        .set_index("sample_key")["age_group"]
        .fillna("Unknown").astype(str).str.strip()
        .to_dict()
    )
    sample_ages = np.array([age_lookup.get(k, "Unknown") for k in valid_keys], dtype=object)
    return genus_rel_top, sample_ages, target_names

genus_rel_nc, ages_nc_s, gen_names_nc = get_sample_level("NC")
genus_rel_cd, ages_cd_s, gen_names_cd = get_sample_level("CD")
print(f"      NC sample-level: {genus_rel_nc.shape[0]:,} samples, {genus_rel_nc.shape[1]} genera")
print(f"      CD sample-level: {genus_rel_cd.shape[0]:,} samples, {genus_rel_cd.shape[1]} genera")


def run_permanova_subsampled(genus_rel, sample_ages, disease_label):
    """
    PERMANOVA with subsampling (max PERMANOVA_MAX_SAMPLE per age group).
    Returns dict with R2, F, p_val.
    """
    present_ages = [a for a in AGE_GROUP_ORDER if a in sample_ages]
    if len(present_ages) < 2:
        return None

    rng = np.random.default_rng(42)
    sub_idx, sub_labels = [], []
    for ag in present_ages:
        idx = np.where(sample_ages == ag)[0]
        if len(idx) > PERMANOVA_MAX_SAMPLE:
            idx = rng.choice(idx, PERMANOVA_MAX_SAMPLE, replace=False)
        sub_idx.extend(idx.tolist())
        sub_labels.extend([ag] * len(idx))

    sub_idx = np.array(sub_idx)
    sub_mat = genus_rel[sub_idx]
    n_sub = len(sub_idx)
    print(f"      PERMANOVA ({disease_label}): n_sub={n_sub}, "
          f"{len(present_ages)} groups, computing Bray-Curtis ...")
    dist_mat = bray_curtis_matrix(sub_mat)
    print(f"      Running {PERMANOVA_N_PERMS} permutations ...")
    R2, F_stat, p_val, df_a, df_w = permanova(
        dist_mat, sub_labels, n_perms=PERMANOVA_N_PERMS
    )
    print(f"      {disease_label} PERMANOVA: R²={R2:.4f}, F={F_stat:.2f}, P={p_val:.4f}")
    return {"R2": R2, "F": F_stat, "p_val": p_val, "df_a": df_a, "df_w": df_w,
            "n_subsampled": n_sub, "n_groups": len(present_ages)}

perm_nc = run_permanova_subsampled(genus_rel_nc, ages_nc_s, "NC")
perm_cd = run_permanova_subsampled(genus_rel_cd, ages_cd_s, "CD")

# Spearman rank correlation
print("      Computing Spearman correlations ...")
spear_nc = run_spearman_by_age(genus_rel_nc, gen_names_nc, ages_nc_s, AGE_ORDINAL)
spear_cd = run_spearman_by_age(genus_rel_cd, gen_names_cd, ages_cd_s, AGE_ORDINAL)
print(f"      NC Spearman: {spear_nc['significant'].sum()} / {len(spear_nc)} significant (FDR<0.05)")
print(f"      CD Spearman: {spear_cd['significant'].sum()} / {len(spear_cd)} significant (FDR<0.05)")


# ─── 4. Parse aggregated data (for figure) ──────────────────────────────
def parse_rows(lc_data, target_genera):
    all_gen = [g for g in lc_data["genera"] if g != "Other"]
    rows = {}
    for row in lc_data["data"]:
        age = row["age_group"]
        n   = row["sample_count"]
        means = []
        for g in target_genera:
            means.append(float(row.get(g, 0.0)) if g in all_gen else 0.0)
        other = float(row.get("Other", max(0.0, 100.0 - sum(means))))
        rows[age] = {"n": n, "means": means, "other": other}
    present = [a for a in AGE_GROUP_ORDER if a in rows]
    return rows, present

rows_cd, ages_cd_p = parse_rows(cd_data, final_gen)
rows_nc, ages_nc_p = parse_rows(nc_data, final_gen)

# ─── 5. Colors ───────────────────────────────────────────────────────────────
genus_colors, genus_colors_hex = build_genus_colors(
    final_gen + ["Other"], phylum_map
)

# ─── 6. Draw figure ──────────────────────────────────────────────────────────
print("[4/5] Plotting ...")

UF = 32    # unified font size (all text consistent)
fig = plt.figure(figsize=(56, 18), facecolor="white")

# ── Main plot uses independent gridspec ──
gs_plots = fig.add_gridspec(1, 2, width_ratios=[1, 1],
                            left=0.05, right=0.64, top=0.86, bottom=0.14,
                            wspace=0.12)
ax_cd  = fig.add_subplot(gs_plots[0])
ax_nc  = fig.add_subplot(gs_plots[1])

# ── Legend uses independent axes: slightly right-shifted to avoid overlap with right panel labels ──
ax_leg = fig.add_axes([0.663, 0.00, 0.18, 1.00])
ax_leg.axis("off")

def plot_panel(ax, rows, present_ages, title, title_color, show_ylabel=True,
               permanova_res=None, stat_y=-0.18):
    n_x  = len(present_ages)
    x    = np.arange(n_x, dtype=float)
    bands = final_gen + ["Other"]
    mat   = np.zeros((len(bands), n_x))

    for j, age in enumerate(present_ages):
        for i, g in enumerate(final_gen):
            mat[i, j] = rows[age]["means"][i]
        mat[-1, j] = rows[age]["other"]

    # normalize columns to 100%
    cs = mat.sum(axis=0); cs[cs == 0] = 1.0
    mat = mat / cs * 100.0

    # ── PchipInterpolator = d3.curveMonotoneX (Steffen 1990) ────────────────
    x_fine = np.linspace(0, n_x - 1, 600)
    fine   = np.zeros((len(bands), 600))
    for i in range(len(bands)):
        if n_x >= 2:
            pchip = PchipInterpolator(x, mat[i])
            fine[i] = np.clip(pchip(x_fine), 0, None)
        else:
            fine[i] = np.full(600, mat[i, 0])

    fs = fine.sum(axis=0); fs[fs == 0] = 1.0
    fine = fine / fs * 100.0

    cumul = np.zeros(600)
    for i, g in enumerate(bands):
        yb = cumul.copy()
        yt = cumul + fine[i]
        c  = genus_colors.get(g, to_rgb("#94a3b8"))
        ax.fill_between(x_fine, yb, yt, color=c, alpha=0.92, linewidth=0, zorder=2)
        ax.plot(x_fine, yt, color="white", linewidth=0.6, alpha=0.4, zorder=3)
        cumul = yt

    ax.set_xlim(0, n_x - 1)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(n_x))
    ax.set_xticklabels([AGE_LABEL[a] for a in present_ages],
                       fontsize=UF, fontfamily=FONT, color="#333333")
    ax.set_xlabel("Age group", fontsize=UF, fontfamily=FONT, color="#333333", labelpad=10)
    if show_ylabel:
        ax.set_ylabel("Relative abundance (%)", fontsize=UF,
                      fontfamily=FONT, color="#333333", labelpad=16)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.tick_params(axis="y", labelsize=UF)
    ax.tick_params(axis="x", length=0, pad=12)
    ax.set_facecolor("#f5f6f7")
    for sp in ["top", "right", "bottom", "left"]:
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(6)
        ax.spines[sp].set_color("#000000")

    for j, age in enumerate(present_ages):
        ax.text(j, 102, f"n={rows[age]['n']:,}",
                ha="center", va="bottom",
                fontsize=UF, fontfamily=FONT, color="#555555")

    # ── PERMANOVA statistics annotation (bottom-right) ───────────────────
    if permanova_res is not None:
        R2  = permanova_res["R2"]
        pv  = permanova_res["p_val"]
        p_str = "P < 0.001" if pv < 0.001 else f"P = {pv:.3f}"
        stat_txt = f"PERMANOVA: R² = {R2:.3f}, {p_str}\n(Bray-Curtis, {PERMANOVA_N_PERMS} perms)"
        ax.text(0.50, stat_y, stat_txt, transform=ax.transAxes,
                ha="center", va="top", fontsize=UF, fontfamily=FONT,
                color="#444444",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                          edgecolor="#cccccc", alpha=0.85))

    ax.set_title(title, fontsize=UF, fontfamily=FONT,
                 color=title_color, fontweight="bold", pad=80)

plot_panel(ax_cd, rows_cd, ages_cd_p,
           title="Crohn's Disease (CD)", title_color=C_CD, show_ylabel=True,
           permanova_res=perm_cd, stat_y=-0.152)
plot_panel(ax_nc, rows_nc, ages_nc_p,
           title="Healthy Control (NC)",  title_color=C_NC, show_ylabel=False,
           permanova_res=perm_nc, stat_y=-0.152)

# ─── Legend ──────────────────────────────────────────────────────────────────
ph_groups_leg = OrderedDict()
for g in final_gen:
    ph_raw  = phylum_map.get(g, "Unknown")
    ph_disp = PHYLUM_DISP.get(ph_raw, "Other/Unknown")
    ph_groups_leg.setdefault(ph_disp, []).append((g, genus_colors[g]))
ph_groups_leg.setdefault("Other/Unknown", []).append(("Other", genus_colors["Other"]))

ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)
n_headers = len(ph_groups_leg)
n_genera  = len(final_gen) + 1
legend_top = 0.998
legend_bottom = 0.002
avail_height = legend_top - legend_bottom
# legend spacing only between groups, no extra whitespace after last group
header_step = 1.4 * 1.3
gap_step = 1.4 * 0.8
total_units = sum(header_step + len(gens) for gens in ph_groups_leg.values())
if n_headers > 1:
    total_units += (n_headers - 1) * gap_step
line_h = avail_height / total_units
hdr_h  = line_h * 1.4
gap    = line_h * gap_step
# font size scales proportionally with line height
leg_inches = 1.00 * 18
fsize_gen = line_h * leg_inches * 72 * 1.00
fsize_hdr = fsize_gen * 1.10

y_cur = legend_top

legend_groups = list(ph_groups_leg.items())
for idx, (ph_disp, gens) in enumerate(legend_groups):
    if y_cur < legend_bottom:
        break
    ph_raw_for_color = next((k for k, v in PHYLUM_DISP.items() if v == ph_disp), ph_disp)
    tc = PHYLUM_TITLE_COLORS.get(ph_raw_for_color,
         PHYLUM_TITLE_COLORS.get(ph_disp, "#424242"))
    ax_leg.text(0.0, y_cur, ph_disp, fontsize=fsize_hdr, fontfamily=FONT,
                color=tc, fontweight="bold", va="top",
                transform=ax_leg.transAxes)
    y_cur -= hdr_h * 1.3
    for g, c in gens:
        if y_cur < legend_bottom:
            break
        y_cur -= line_h
        sh = line_h * 0.65
        rect = mpatches.FancyBboxPatch(
            (0.0, y_cur + line_h * 0.15), 0.09, sh,
            boxstyle="round,pad=0.002",
            facecolor=c, edgecolor="none",
            transform=ax_leg.transAxes, clip_on=False)
        ax_leg.add_patch(rect)
        gname = g if len(g) <= 22 else g[:21] + "."
        ax_leg.text(0.11, y_cur + line_h * 0.50, gname, fontsize=fsize_gen,
                    fontfamily=FONT, color="#333333", va="center",
                    transform=ax_leg.transAxes)
    if idx < len(legend_groups) - 1:
        y_cur -= gap

fig.text(0.35, 0.96,
         f"Microbiome Composition Across the Human Lifespan  (Top {TOP_N} genera)",
         ha="center", va="bottom",
         fontsize=UF, fontfamily=FONT, fontweight="bold", color="#2c3e50")

# ─── 7. Save figure ──────────────────────────────────────────────────────────
print("[5/5] Saving outputs ...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
for ext, dpi in [("png", 300), ("pdf", 300)]:
    out = os.path.join(OUTPUT_DIR, f"fig2h.{ext}")
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    if ext == "png":
        img = Image.open(out).convert("RGB")
        mask = img.point(lambda p: 255 if p < 249 else 0)
        bbox = mask.getbbox()
        if bbox is not None:
            left, top, right, bottom = bbox
            pad_left = 0
            pad_top = 4
            pad_right = 4
            pad_bottom = 10
            crop_box = (
                max(0, left - pad_left),
                max(0, top - pad_top),
                min(img.size[0], right + pad_right),
                min(img.size[1], bottom + pad_bottom),
            )
            cropped = img.crop(crop_box)
            bottom_safe_pad = 18
            canvas = Image.new("RGB", (cropped.size[0], cropped.size[1] + bottom_safe_pad), "white")
            canvas.paste(cropped, (0, 0))
            canvas.save(out)
    print(f"  -> {out}")
plt.close(fig)

# ─── 8. Save statistics results ──────────────────────────────────────────────
def save_stats_report(disease, perm_res, kw_rows, spear_df, n_samples):
    lines = []
    lines.append(f"=== fig1f Statistical Report: {disease} ===")
    lines.append(f"Total samples: {n_samples:,}")
    lines.append(f"Age groups: {', '.join(AGE_GROUP_ORDER)}")
    lines.append("")
    lines.append("── PERMANOVA (Bray-Curtis dissimilarity, 999 permutations) ──")
    if perm_res:
        pv = perm_res['p_val']
        p_str = "P < 0.001" if pv < 0.001 else f"P = {pv:.4f}"
        lines.append(f"  R² = {perm_res['R2']:.4f}")
        lines.append(f"  Pseudo-F({perm_res['df_a']},{perm_res['df_w']}) = {perm_res['F']:.4f}")
        lines.append(f"  {p_str}")
        lines.append(f"  (subsampled n = {perm_res['n_subsampled']}, "
                     f"{perm_res['n_groups']} age groups)")
        lines.append(f"  Interpretation: Life-stage group explains {perm_res['R2']*100:.1f}% "
                     f"of total gut microbiota compositional variance.")
    else:
        lines.append("  PERMANOVA could not be computed (too few samples).")
    lines.append("")
    lines.append("── Kruskal-Wallis H test (per genus, BH-FDR corrected) ──")
    lines.append("  Format: genus | H statistic | raw P | FDR adj-P | significant")
    sig_kw = [r for r in kw_rows if r.get("significant")]
    lines.append(f"  Significant genera (FDR<0.05): {len(sig_kw)} / {len(kw_rows)}")
    for r in sorted(kw_rows, key=lambda x: x.get("adjusted_p", 1.0)):
        sig = "***" if r.get("significant") else "   "
        lines.append(f"  {sig} {r['genus']:30s} H={r['kruskal_h']:8.2f} "
                     f"P={r['kruskal_p']:.2e}  adj-P={r['adjusted_p']:.2e}")
    lines.append("")
    lines.append("── Spearman rank correlation (genus RA vs. ordinal age) ──")
    lines.append("  Age coding: Infant=1, Child=2, Adolescent=3, Adult=4, "
                 "Older_Adult=5, Oldest_Old=6, Centenarian=7")
    lines.append("  BH-FDR corrected; threshold FDR < 0.05")
    sig_sp = spear_df[spear_df["significant"]]
    lines.append(f"  Significant genera: {len(sig_sp)} / {len(spear_df)}")
    for _, row in spear_df.sort_values("adj_pval").iterrows():
        sig = "***" if row["significant"] else "   "
        direction = "↑ with age" if row["rho"] > 0 else "↓ with age"
        lines.append(f"  {sig} {row['genus']:30s} rho={row['rho']:+.4f} "
                     f"P={row['pval']:.2e}  adj-P={row['adj_pval']:.2e}  {direction}")
    lines.append("")
    lines.append("── References ──")
    lines.append("  PERMANOVA: Anderson MJ. Austral Ecology 2001;26:32-46.")
    lines.append("  Kruskal-Wallis: Kruskal WH, Wallis WA. J Am Stat Assoc 1952;47:583-621.")
    lines.append("  BH-FDR: Benjamini Y, Hochberg Y. J R Stat Soc B 1995;57:289-300.")
    lines.append("  Spearman: Spearman C. Am J Psychol 1904;15:72-101.")
    lines.append("  PCHIP interpolation: Steffen M. Astron Astrophys 1990;239:443-450.")
    return "\n".join(lines)

for dis, perm_res, kw_rows, spear_df, n_total in [
    ("NC", perm_nc, kw_nc, spear_nc, nc_data["total_samples"]),
    ("CD", perm_cd, kw_cd, spear_cd, cd_data["total_samples"]),
]:
    report = save_stats_report(dis, perm_res, kw_rows, spear_df, n_total)
    txt_path = os.path.join(OUTPUT_DIR, f"fig1f_stats_{dis}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  -> {txt_path}")

    # save CSV
    kw_df = pd.DataFrame(kw_rows)
    kw_csv = os.path.join(OUTPUT_DIR, f"fig1f_kruskal_{dis}.csv")
    kw_df.to_csv(kw_csv, index=False)
    print(f"  -> {kw_csv}")

    sp_csv = os.path.join(OUTPUT_DIR, f"fig1f_spearman_{dis}.csv")
    spear_df.to_csv(sp_csv, index=False)
    print(f"  -> {sp_csv}")

print("\nAll done.")
print(f"Output directory: {OUTPUT_DIR}")
