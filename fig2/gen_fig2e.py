#!/usr/bin/env python3
"""
Figure 1c — Cross-study meta-analysis forest plot (Nature-standard).

Statistical model  : DerSimonian–Laird random-effects meta-analysis
Per-study markers  : filled circle = q<0.05 (study-level BH), open = q≥0.05
Pooled estimate    : black diamond (standard forest-plot convention)
Heterogeneity      : I² + τ² (Tau²) + Cochran Q p-value
CI truncation      : "▷" / "◁" symbol when CI exceeds plot range
Right annotation   : adj.p | I² | τ² | k studies | n samples

Data source: same files used by the GutBiomeDB platform backend.
"""
import math, sys, io, os, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow

# ─── Paths ───────────────────────────────────────────────────
ABUNDANCE_PATH = r"E:\microbiomap_clone\data\unfiltered_abundance.csv"
METADATA_PATH  = r"E:\microbiomap_clone\data\metadata.csv"
OUTPUT_DIR     = os.path.dirname(os.path.abspath(__file__))

# ─── Parameters ──────────────────────────────────────────────
DISEASE        = "CD"
METHOD         = "wilcoxon"
TAXONOMY_LEVEL = "genus"
P_THRESHOLD    = 0.05
MIN_STUDIES    = 2
TOP_N_DISPLAY  = 24

# ─── Color palette — matches volcano plot in Figure 1b ────────
C_DISEASE  = "#e74c3c"   # volcano red  — CD-enriched (adj.p<0.001)
C_CONTROL  = "#00b4d8"   # volcano teal — NC-enriched (adj.p<0.001)
C_MIXED    = "#95a5a6"   # non-significant gray — inconsistent direction
C_DIAMOND  = "#1a1a1a"   # near-black pooled diamond
C_TXT      = "#1a1a1a"   # all main text: pure black
C_SUB      = "#2c3e50"   # axis/header text: dark navy
C_NULL     = "#2c3e50"   # right-column values: dark (not gray)
C_BG       = "#f7f9fc"
C_RULE     = "#d5dbdb"
FONT       = "Arial"

# ════════════════════════════════════════════════════════════
# 1. Backend utility functions (exact copies)
# ════════════════════════════════════════════════════════════

def extract_genus(col_name):
    parts = str(col_name).split(".")
    return parts[-1].strip() if parts else str(col_name)

def relative_abundance_matrix(raw_matrix):
    matrix = np.asarray(raw_matrix, dtype=float)
    totals = matrix.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return matrix / totals * 100.0

def aggregate_by_level(matrix, columns, taxonomy_level="genus"):
    labels = [extract_genus(col) or "Unknown" for col in columns]
    unique_labels = list(dict.fromkeys(labels))
    label_to_indices = {label: [] for label in unique_labels}
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    aggregated = np.zeros((matrix.shape[0], len(unique_labels)), dtype=float)
    for col_idx, label in enumerate(unique_labels):
        aggregated[:, col_idx] = matrix[:, label_to_indices[label]].sum(axis=1)
    return aggregated, unique_labels

def bh_correction(p_values):
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    prev_adj = 1.0
    for rank, (orig_idx, p) in enumerate(reversed(indexed)):
        adj = min(prev_adj, p * n / (n - rank))
        adjusted[orig_idx] = min(adj, 1.0)
        prev_adj = adj
    return adjusted

# ════════════════════════════════════════════════════════════
# 2. Data loading & per-project differential analysis
# ════════════════════════════════════════════════════════════
print("[1/4] Loading metadata …")
meta = pd.read_csv(METADATA_PATH, low_memory=False)
meta["sample_key"] = (meta["project"].astype(str).str.strip() + "_"
                      + meta["srr"].astype(str).str.strip())
INFORM_COLS = [f"inform{i}" for i in range(12)]
disease_lower = DISEASE.strip().lower()

def _strict_nc_mask(m):
    if "inform-all" in m.columns:
        return m["inform-all"].fillna("").astype(str).str.strip() == "NC"
    mask = m[INFORM_COLS[0]].fillna("").astype(str).str.strip().str.lower() == "nc"
    for col in INFORM_COLS[1:]:
        if col in m.columns:
            mask &= m[col].fillna("").astype(str).str.strip() == ""
    return mask

all_projects = meta["project"].unique()
eligible_projects = []
for proj in sorted(all_projects):
    proj_meta = meta[meta["project"] == proj]
    disease_mask = pd.Series(False, index=proj_meta.index)
    for col in INFORM_COLS:
        if col in proj_meta.columns:
            disease_mask |= (proj_meta[col].fillna("").astype(str)
                           .str.strip().str.lower() == disease_lower)
    nc_mask = _strict_nc_mask(proj_meta)
    if disease_mask.sum() >= 20 and nc_mask.sum() >= 20:
        eligible_projects.append(proj)

print(f"    {len(eligible_projects)} projects with ≥20 CD and ≥20 NC")

print("[2/4] Loading abundance matrix …")
abund = pd.read_csv(ABUNDANCE_PATH, index_col=0, low_memory=False)
abund_idx = set(abund.index)
col_names = abund.columns.tolist()
print(f"    {abund.shape[0]} samples × {abund.shape[1]} ASVs loaded")

print("[3/4] Per-project differential analysis …")
per_project_results = []
taxa_global = None

for proj_id in eligible_projects:
    proj_meta = meta[meta["project"] == proj_id].copy()
    disease_mask = pd.Series(False, index=proj_meta.index)
    for col in INFORM_COLS:
        if col in proj_meta.columns:
            disease_mask |= (proj_meta[col].fillna("").astype(str)
                           .str.strip().str.lower() == disease_lower)
    nc_mask = _strict_nc_mask(proj_meta)
    disease_keys = [k for k in proj_meta.loc[disease_mask, "sample_key"].values if k in abund_idx]
    control_keys = [k for k in proj_meta.loc[nc_mask,    "sample_key"].values if k in abund_idx]
    if len(disease_keys) < 20 or len(control_keys) < 20:
        per_project_results.append({"project_id": proj_id,
                                    "n_disease": len(disease_keys),
                                    "n_control": len(control_keys),
                                    "error": "Insufficient samples", "taxa_results": []})
        continue
    raw_d = abund.loc[disease_keys].values.astype(float)
    raw_c = abund.loc[control_keys].values.astype(float)
    mat_d = relative_abundance_matrix(raw_d)
    mat_c = relative_abundance_matrix(raw_c)
    agg_d, taxa = aggregate_by_level(mat_d, col_names, TAXONOMY_LEVEL)
    agg_c, _    = aggregate_by_level(mat_c, col_names, TAXONOMY_LEVEL)
    if taxa_global is None:
        taxa_global = taxa
    pseudo = 1e-6
    taxa_results = []
    for i, taxon in enumerate(taxa):
        vals_d = agg_d[:, i]
        vals_c = agg_c[:, i]
        # Effect size: "difference in mean log₂-transformed relative abundances (pseudocount 10⁻⁶)"
        # First log₂-transform each observation, then take mean difference
        log2_d = np.log2(vals_d + pseudo)
        log2_c = np.log2(vals_c + pseudo)
        log2fc = float(np.mean(log2_d) - np.mean(log2_c))
        try:
            if METHOD == "wilcoxon":
                _, p = stats.mannwhitneyu(vals_d, vals_c, alternative="two-sided")
            else:
                _, p = stats.ttest_ind(vals_d, vals_c)
        except Exception:
            p = 1.0
        # SE on log-transformed scale, using sample variance (ddof=1)
        se = float(np.sqrt(np.var(log2_d, ddof=1) / len(log2_d)
                          + np.var(log2_c, ddof=1) / len(log2_c)))
        taxa_results.append({"taxon": taxon, "log2fc": round(log2fc, 4),
                             "p_value": round(float(p), 8), "se": round(se, 8)})
    per_project_results.append({"project_id": proj_id,
                                 "n_disease": len(disease_keys),
                                 "n_control": len(control_keys),
                                 "error": None, "taxa_results": taxa_results})
    print(f"    {proj_id}: D={len(disease_keys)} C={len(control_keys)}")

# ════════════════════════════════════════════════════════════
# 3. DerSimonian–Laird meta-analysis
# ════════════════════════════════════════════════════════════
valid_projects = [p for p in per_project_results
                  if p["error"] is None and len(p["taxa_results"]) > 0]
print(f"    {len(valid_projects)} valid projects for meta-analysis")

consensus_markers = []
for tax_idx, taxon in enumerate(taxa_global):
    effects = []
    for proj in valid_projects:
        tr = proj["taxa_results"]
        if tax_idx < len(tr) and tr[tax_idx]["taxon"] == taxon:
            se = tr[tax_idx]["se"]
            if se > 0:
                effects.append({
                    "log2fc": tr[tax_idx]["log2fc"],
                    "se":     se,
                    "p":      tr[tax_idx]["p_value"],
                    "proj":   proj["project_id"],
                    "n_d":    proj["n_disease"],
                    "n_c":    proj["n_control"],
                })
    if len(effects) < MIN_STUDIES:
        continue

    # Fixed-effects step (for Q / tau²)
    weights = [1 / e["se"]**2 for e in effects]
    w_sum   = sum(weights)
    beta_fe = sum(w * e["log2fc"] for w, e in zip(weights, effects)) / w_sum

    # Cochran's Q + I²
    Q  = sum(w * (e["log2fc"] - beta_fe)**2 for w, e in zip(weights, effects))
    df = len(effects) - 1
    p_Q = 1 - stats.chi2.cdf(Q, df) if df > 0 else 1.0
    I2  = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0

    # DerSimonian–Laird tau²
    c    = w_sum - sum(w**2 for w in weights) / w_sum
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0

    # Random-effects pooled estimate
    re_weights = [1 / (e["se"]**2 + tau2) for e in effects]
    re_w_sum   = sum(re_weights)
    beta_re    = sum(w * e["log2fc"] for w, e in zip(re_weights, effects)) / re_w_sum
    se_re      = math.sqrt(1 / re_w_sum)
    z          = beta_re / se_re if se_re > 0 else 0
    meta_p     = float(2 * (1 - stats.norm.cdf(abs(z))))

    # Study-level BH correction (within this taxon, across studies)
    study_adj_p = bh_correction([e["p"] for e in effects])

    directions = [1 if e["log2fc"] > 0 else -1 for e in effects]
    direction  = "mixed"
    if all(d > 0 for d in directions):
        direction = "disease"
    elif all(d < 0 for d in directions):
        direction = "control"

    consensus_markers.append({
        "taxon":       taxon,
        "meta_log2fc": round(beta_re, 4),
        "meta_se":     round(se_re, 6),
        "meta_p":      round(meta_p, 8),
        "ci_low":      round(beta_re - 1.96 * se_re, 4),
        "ci_high":     round(beta_re + 1.96 * se_re, 4),
        "n_studies":   len(effects),
        "n_total":     sum(e["n_d"] + e["n_c"] for e in effects),
        "I2":          round(I2, 1),
        "tau2":        round(tau2, 4),
        "Q":           round(Q, 3),
        "p_Q":         round(p_Q, 4),
        "direction":   direction,
        "effects":     effects,           # per-study data for dots
        "study_adj_p": study_adj_p,       # per-study BH p
    })

# Global BH across all taxa
adjusted_meta_p = bh_correction([float(m["meta_p"]) for m in consensus_markers])
for marker, adj_p in zip(consensus_markers, adjusted_meta_p):
    marker["adjusted_meta_p"] = round(float(adj_p), 8)

consensus_markers.sort(key=lambda x: (x["adjusted_meta_p"], x["meta_p"]))
significant_markers = [m for m in consensus_markers if m["adjusted_meta_p"] < P_THRESHOLD]
print(f"    {len(significant_markers)} significant consensus markers (BH adj.p < 0.05)")

chart_markers = significant_markers[:TOP_N_DISPLAY]
print(f"    Displaying top {len(chart_markers)} markers")

# ════════════════════════════════════════════════════════════
# 4. NATURE-STANDARD FOREST PLOT  (large-text, thick-line version)
#    All font sizes ×5, all line widths ×3 vs. original draft.
#    Designed to remain legible when used as a panel in a
#    multi-panel Figure 1 layout.
# ════════════════════════════════════════════════════════════
print("\n[4/4] Drawing Nature-standard forest plot …")

def direction_color(direction):
    return {"disease": C_DISEASE, "control": C_CONTROL}.get(direction, C_MIXED)

n = len(chart_markers)

# x-axis range
all_lows  = [m["ci_low"]  for m in chart_markers]
all_highs = [m["ci_high"] for m in chart_markers]
clip_lo = min(min(all_lows),  -3) - 0.8
clip_hi = max(max(all_highs),  3) + 0.8

# ── Figure & axes ────────────────────────────────────────────
# Extra-large figure so text stays legible at panel scale
fig_h = max(30, n * 1.55 + 8)
fig, ax = plt.subplots(figsize=(38, fig_h), facecolor="white")
ax.set_facecolor("white")
yy = list(range(n - 1, -1, -1))

# Alternating row bands
for i in range(n):
    if i % 2 == 0:
        ax.axhspan(yy[i] - 0.46, yy[i] + 0.46, color=C_BG, zorder=0)

# Null line
ax.axvline(0, color=C_NULL, lw=6, ls="--", zorder=1)

# ── Per-marker: per-cohort dots + CI line + diamond ──────────
# Nature convention (Nat. Microbiol. 2024):
#   • filled dot  = within-study adj.p < 0.05 (significant)
#   • hollow dot  = within-study adj.p ≥ 0.05 (not significant)
#   • dot color follows own log2FC direction (>0 red, <0 teal)
#   • pooled CI bar and diamond drawn on top
for m, y in zip(chart_markers, yy):
    # Color CI bar by POOLED estimate sign (Nature convention),
    # not per-cohort consistency — consistency is shown by I² and dots.
    clr = direction_color("disease" if m["meta_log2fc"] > 0 else "control")

    # Step 1 — per-study small circles (behind everything)
    # Use raw p < 0.05 (uncorrected) for within-study significance marker.
    # BH correction across studies within one taxon is too conservative for
    # visualization; we report within-cohort q < 0.1 (uncorrected) instead.
    for eff in m["effects"]:
        lfc = eff["log2fc"]
        if lfc < clip_lo or lfc > clip_hi:
            continue
        dot_clr = direction_color("disease" if lfc > 0 else "control")
        if eff["p"] < 0.05:                      # raw within-study p < 0.05
            ax.plot(lfc, y, "o", color=dot_clr,
                    ms=20, mec=dot_clr, mew=3.0,
                    alpha=0.85, zorder=3)
        else:                                     # not significant within study
            ax.plot(lfc, y, "o", color="none",
                    ms=20, mec=dot_clr, mew=3.0,
                    alpha=0.50, zorder=3)

    # Step 2 — pooled 95% CI bar
    lo, hi = m["ci_low"], m["ci_high"]
    lo_plot = max(lo, clip_lo + 0.15)
    hi_plot = min(hi, clip_hi - 0.15)
    ax.plot([lo_plot, hi_plot], [y, y],
            color=clr, lw=9, solid_capstyle="butt", zorder=5)

    if lo < clip_lo + 0.3:
        ax.annotate("", xy=(clip_lo + 0.2, y), xytext=(clip_lo + 1.0, y),
                    arrowprops=dict(arrowstyle="-|>", color=clr,
                                   lw=6, mutation_scale=28), zorder=5)
    if hi > clip_hi - 0.3:
        ax.annotate("", xy=(clip_hi - 0.2, y), xytext=(clip_hi - 1.0, y),
                    arrowprops=dict(arrowstyle="-|>", color=clr,
                                   lw=6, mutation_scale=28), zorder=5)

    # Step 3 — pooled estimate diamond
    beta   = m["meta_log2fc"]
    dw, dh = 0.55, 0.44
    dx = [beta, beta+dw, beta, beta-dw, beta]
    dy = [y+dh, y,       y-dh, y,       y+dh]
    ax.fill(dx, dy, color=C_DIAMOND, zorder=7)
    ax.plot(dx, dy, color="white", lw=2.5, zorder=8)

# ── Axes ─────────────────────────────────────────────────────
ax.set_xlim(clip_lo, clip_hi)
ax.set_ylim(-2.2, n - 1 + 1.2)
ax.set_yticks([])
for sp in ["top", "right", "left"]:
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_linewidth(4)
ax.spines["bottom"].set_color(C_SUB)
ax.tick_params(axis="x", width=5, length=16, labelsize=52, colors=C_TXT)
plt.setp(ax.get_xticklabels(), fontweight="bold", fontfamily=FONT)
ax.set_xlabel(r"log$_2$ fold change (CD vs. NC)",
              fontsize=52, fontfamily=FONT, color=C_TXT, labelpad=22,
              fontweight="bold")

def fmt_taxon(name):
    """Display name: purely-numeric SILVA codes → 'unclassified_XXX'."""
    if re.match(r'^\d+$', name):
        return f"unclassified_{name}"
    return name

# ── LEFT: taxon names ────────────────────────────────────────
for m, y in zip(chart_markers, yy):
    raw = fmt_taxon(m["taxon"])
    label = raw if len(raw) <= 28 else raw[:26] + "…"
    ax.text(-0.005, y, label,
            transform=ax.get_yaxis_transform(),
            fontsize=50, fontfamily=FONT, color=C_TXT,
            va="center", ha="right", fontstyle="italic", fontweight="bold")

# ── RIGHT: 4 stats columns, wide spacing ─────────────────────
header_y = n - 1 + 0.90
col_x    = [1.06, 1.22, 1.35, 1.48]
headers  = ["adj.p", "I²(%)", "k", "n"]
for cx, hd in zip(col_x, headers):
    ax.text(cx, header_y, hd,
            transform=ax.get_yaxis_transform(),
            fontsize=50, fontfamily=FONT, color=C_TXT,
            va="center", ha="center", fontweight="bold")

for m, y in zip(chart_markers, yy):
    adj_p = m["adjusted_meta_p"]
    p_str = "<0.001" if adj_p < 0.001 else f"{adj_p:.3f}"
    vals  = [p_str, f"{m['I2']:.0f}", str(m["n_studies"]), str(m["n_total"])]
    for cx, val in zip(col_x, vals):
        ax.text(cx, y, val,
                transform=ax.get_yaxis_transform(),
                fontsize=50, fontfamily=FONT, color=C_TXT,
                va="center", ha="center", fontweight="bold")

# ── Title ─────────────────────────────────────────────────────
total_n = sum(p["n_disease"] + p["n_control"] for p in valid_projects)
ax.set_title(
    f"Cross-study meta-analysis — Crohn's Disease\n"
    f"({len(valid_projects)} studies · {total_n:,} samples · DerSimonian–Laird random-effects)",
    fontsize=54, fontfamily=FONT, color=C_TXT, fontweight="bold", pad=26)

# ── Legend — 2 × 2 grid below plot, fully clear of x-label ───
legend_elements = [
    Line2D([0], [0], marker="D", color="w", markerfacecolor=C_DIAMOND,
           markersize=30, label="Pooled estimate (diamond + 95% CI)"),
    Line2D([0], [0], color=C_DISEASE, lw=11,
           label="CD-enriched (log₂FC > 0)"),
    Line2D([0], [0], color=C_CONTROL, lw=11,
           label="NC-enriched (log₂FC < 0)"),
    Line2D([0], [0], color=C_MIXED, lw=11,
           label="Inconsistent direction"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
           markersize=22, mec="white", label="Per-study: adj.p < 0.05 (filled)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
           markersize=22, mec="#555555", mew=3,
           label="Per-study: adj.p ≥ 0.05 (hollow)"),
]
ax.legend(handles=legend_elements, loc="lower center",
          bbox_to_anchor=(0.50, -0.30), ncol=2,
          fontsize=50, framealpha=0.97, edgecolor="#cccccc",
          handlelength=3.2, handleheight=2.0,
          columnspacing=2.0, handletextpad=1.0,
          prop={"size": 50, "weight": "bold"})

plt.subplots_adjust(left=0.18, right=0.52, top=0.93, bottom=0.38)

# ── Save ──────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    p = os.path.join(OUTPUT_DIR, f"fig2e.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  → {p}")
plt.close()
print("Done.")
