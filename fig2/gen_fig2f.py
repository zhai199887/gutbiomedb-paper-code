#!/usr/bin/env python3
"""
Figure 1d — Faecalibacterium depletion in Crohn's disease (per-genus drill-down).

Visual style: identical to Figure 1c (same figsize, fonts, colours, line weights).
Statistics: DerSimonian–Laird random-effects meta-analysis.
  log2FC  = mean(log2(x+1e-6))_CD  −  mean(log2(x+1e-6))_NC
  SE      = sqrt(var_d/n_d + var_c/n_c),  ddof=1 (Bessel correction)

python gen_fig1d_forest.py
"""
import csv, math, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ABUNDANCE_PATH = r"E:\microbiomap_clone\data\unfiltered_abundance.csv"
METADATA_PATH  = r"E:\microbiomap_clone\data\metadata.csv"
OUTPUT_DIR     = os.path.dirname(os.path.abspath(__file__))
MIN_SAMPLES    = 20
PSEUDO         = 1e-6
GENUS          = "Faecalibacterium"

# ── Colour palette: IDENTICAL to fig1c ──────────────────────
C_DISEASE  = "#e74c3c"   # CD-enriched red  (volcano red)
C_CONTROL  = "#00b4d8"   # NC-enriched teal (volcano teal)
C_DIAMOND  = "#1a1a1a"   # near-black  — pooled diamond
C_TXT      = "#1a1a1a"   # all main text
C_SUB      = "#2c3e50"   # axis / header text
C_NULL     = "#2c3e50"   # null-line + right-column values
C_BG       = "#f7f9fc"   # alternating row bands
C_RULE     = "#d5dbdb"   # separator rule
FONT       = "Arial"

# Faecalibacterium is DEPLETED in CD → NC-enriched colour
C_GENUS    = C_CONTROL

# ════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════
print("[1/4] Loading abundance data …")
with open(ABUNDANCE_PATH) as f:
    hdr = f.readline().strip().split(",")

fcol = None
for i, c in enumerate(hdr):
    if GENUS.lower() in c.lower():
        fcol = i
        print(f"    Column {i}: {c}")
        break
if fcol is None:
    raise ValueError(f"{GENUS} not found in column headers")

sv = {}   # sample_key → relative abundance (0–1)
with open(ABUNDANCE_PATH) as f:
    reader = csv.reader(f); next(reader)
    for row in reader:
        s = row[0]
        try:    fval = float(row[fcol])
        except: fval = 0.0
        try:    total = sum(float(x) for x in row[1:] if x)
        except: total = 0.0
        sv[s] = fval / total if total > 0 else 0.0
print(f"    {len(sv):,} samples loaded.")

print("[2/4] Loading metadata …")
meta = pd.read_csv(METADATA_PATH, low_memory=False)
meta["sk"] = (meta["project"].astype(str).str.strip()
              + "_" + meta["srr"].astype(str).str.strip())

ic = ["inform-all"] + [f"inform{i}" for i in range(12)]
av = [c for c in ic if c in meta.columns]
cd_m = pd.Series(False, index=meta.index)
for c in av:
    cd_m |= meta[c].fillna("").astype(str).str.strip().str.lower() == "cd"
nc_m = meta["inform-all"].fillna("").astype(str).str.strip() == "NC"
for c in av[1:]:
    if c in meta.columns:
        nc_m &= meta[c].fillna("").astype(str).str.strip().isin(["NC", ""])

# ════════════════════════════════════════════════════════════
# 2. PER-STUDY EFFECTS  (same method as fig1c / API backend)
# ════════════════════════════════════════════════════════════
print("[3/4] Per-study differential analysis …")
results = []
for proj in sorted(meta["project"].unique()):
    cdk = meta[(meta["project"] == proj) & cd_m]["sk"].tolist()
    nck = meta[(meta["project"] == proj) & nc_m]["sk"].tolist()
    cdv = [k for k in cdk if k in sv]
    ncv = [k for k in nck if k in sv]
    if len(cdv) < MIN_SAMPLES or len(ncv) < MIN_SAMPLES:
        continue

    # Corrected log₂FC: mean on log₂ scale (NOT log₂ of means)
    vd = np.array([math.log2(sv[k] + PSEUDO) for k in cdv])
    vc = np.array([math.log2(sv[k] + PSEUDO) for k in ncv])
    lfc = float(np.mean(vd) - np.mean(vc))

    # SE with Bessel correction on log₂ scale
    se = math.sqrt(np.var(vd, ddof=1) / len(vd)
                 + np.var(vc, ddof=1) / len(vc))
    if se == 0 or math.isnan(se):
        continue

    # Wilcoxon rank-sum on raw relative abundances
    try:
        _, p = stats.mannwhitneyu(
            np.array([sv[k] for k in cdv]),
            np.array([sv[k] for k in ncv]),
            alternative="two-sided")
        p = float(p)
    except Exception:
        p = 1.0

    results.append(dict(project=proj, n_cd=len(cdv), n_nc=len(ncv),
                        log2fc=lfc, se=se, p=p))
    print(f"    {proj:30s}  n_CD={len(cdv):4d}  n_NC={len(ncv):4d}"
          f"  log2FC={lfc:+.3f}  SE={se:.4f}  P={p:.3e}")

k = len(results)
print(f"    k = {k} qualifying studies")

# ════════════════════════════════════════════════════════════
# 3. DerSimonian–Laird RANDOM-EFFECTS META-ANALYSIS
# ════════════════════════════════════════════════════════════
eff  = [(r["log2fc"], r["se"]) for r in results]
wf   = [1 / se**2  for _, se in eff]
ws   = sum(wf)
bfe  = sum(w * e for w, (e, _) in zip(wf, eff)) / ws   # fixed-effects

Q    = sum(w * (e - bfe)**2 for w, (e, _) in zip(wf, eff))
df_  = k - 1
I2   = max(0.0, (Q - df_) / Q * 100) if Q > 0 else 0.0

c_   = ws - sum(w**2 for w in wf) / ws
tau2 = max(0.0, (Q - df_) / c_) if (c_ > 0 and Q > df_) else 0.0

rw   = [1 / (se**2 + tau2) for _, se in eff]
rws  = sum(rw)
bre  = sum(w * e for w, (e, _) in zip(rw, eff)) / rws
sre  = math.sqrt(1 / rws)
mp   = float(2 * (1 - stats.norm.cdf(abs(bre / sre))))
cil  = bre - 1.96 * sre
cih  = bre + 1.96 * sre
p_het = float(1 - stats.chi2.cdf(Q, df=df_))

print(f"\n    Pooled = {bre:.3f}  [{cil:.3f}, {cih:.3f}]  P = {mp:.3e}")
print(f"    I² = {I2:.1f}%  τ² = {tau2:.3f}  Q = {Q:.2f}  P(het) = {p_het:.3e}")

# ════════════════════════════════════════════════════════════
# 4. BUILD SORTED ROW DATA
# ════════════════════════════════════════════════════════════
results.sort(key=lambda x: x["log2fc"])
n = len(results)

rw_sorted  = [1 / (r["se"]**2 + tau2) for r in results]
rws_sorted = sum(rw_sorted)
rw_max     = max(rw_sorted)

rows = []
for i, r in enumerate(results):
    w_pct = rw_sorted[i] / rws_sorted * 100
    rows.append({
        **r,
        "cl":      r["log2fc"] - 1.96 * r["se"],   # study-level 95% CI
        "ch":      r["log2fc"] + 1.96 * r["se"],
        "wpct":    w_pct,
        "rw_norm": rw_sorted[i] / rw_max,           # for square size scaling
    })

# ════════════════════════════════════════════════════════════
# 5. FIGURE  — identical visual style to Figure 1c
#    figsize=(38, …), fontsize 50/52/54, lw 6/9/4, bold
# ════════════════════════════════════════════════════════════
print("\n[4/4] Drawing figure …")

yy     = list(range(n - 1, -1, -1))
y_pool = -1.8

fig_h  = max(22, n * 2.0 + 7)
fig, ax = plt.subplots(figsize=(46, fig_h), facecolor="white")
ax.set_facecolor("white")
ax.set_ylim(y_pool - 1.5, n - 1 + 1.6)

# ── Alternating row bands ──
for i in range(n):
    if i % 2 == 0:
        ax.axhspan(yy[i] - 0.46, yy[i] + 0.46, color=C_BG, zorder=0)

# ── Null line: lw=8, ls="--" ──
ax.axvline(0, color=C_NULL, lw=8, ls="--", zorder=1)

# ── Per-study squares + CI bars ──
all_x   = [r["cl"] for r in rows] + [r["ch"] for r in rows] + [cil, cih]
clip_lo = min(all_x) - 0.5
clip_hi = max(all_x) + 0.5

for row, y in zip(rows, yy):
    lo_plot = max(row["cl"], clip_lo + 0.1)
    hi_plot = min(row["ch"], clip_hi - 0.1)

    # CI whisker: lw=11
    ax.plot([lo_plot, hi_plot], [y, y],
            color=C_GENUS, lw=11, solid_capstyle="butt", zorder=4)

    # Square size ∝ RE weight (22–44 pt)
    sq_sz = 22 + row["rw_norm"] * 22
    ax.plot(row["log2fc"], y, "s",
            color=C_GENUS, ms=sq_sz,
            mec=C_DIAMOND, mew=3.0, zorder=5)

# ── Separator ──
ax.axhline(-0.72, color=C_RULE, lw=4, zorder=1)

# ── Pooled diamond: dw=0.60, dh=0.48 ──
dw, dh = 0.60, 0.48
dx = [bre, bre + dw, bre, bre - dw, bre]
dy = [y_pool + dh, y_pool, y_pool - dh, y_pool, y_pool + dh]
ax.fill(dx, dy, color=C_DIAMOND, zorder=7)
ax.plot(dx, dy, color="white", lw=3.0, zorder=8)

# ── Axes: spine lw=6, tick labelsize=64, ALL BOLD ──
ax.set_xlim(clip_lo, clip_hi)
ax.set_ylim(y_pool - 1.5, n - 1 + 1.6)
ax.set_yticks([])
for sp in ["top", "right", "left"]:
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_linewidth(6)
ax.spines["bottom"].set_color(C_TXT)
ax.tick_params(axis="x", width=8, length=22, labelsize=64,
               colors=C_TXT, which="both")
plt.setp(ax.get_xticklabels(), fontweight="bold", fontfamily=FONT)
ax.set_xlabel(r"$\log_2$ fold change  (CD vs. healthy control)",
              fontsize=66, fontfamily=FONT, color=C_TXT,
              labelpad=26, fontweight="bold")

# ── LEFT: study labels — fontsize=64, bold ──
for row, y in zip(rows, yy):
    ax.text(-0.005, y, row["project"],
            transform=ax.get_yaxis_transform(),
            fontsize=62, fontfamily=FONT, color=C_TXT,
            va="center", ha="right", fontweight="bold")

ax.text(-0.005, y_pool, "Pooled  (RE model)",
        transform=ax.get_yaxis_transform(),
        fontsize=62, fontfamily=FONT, color=C_DIAMOND,
        va="center", ha="right", fontweight="bold")

# ── RIGHT: 4 annotation columns — wide spacing, no overlap ──
# With figsize=(46,...), left=0.11, right=0.55: axes_width = 0.44*46 = 20.2 in
# Each 0.01 axes unit = 0.20 in.  fontsize=62 → ~0.47 in/char
# n_CD gap: 0.14 → 2.83 in; P gap: 0.20 → 4.04 in; Wt gap: 0.22 → 4.44 in
header_y = n - 1 + 1.10
col_x    = [1.04, 1.18, 1.36, 1.58]
headers  = ["n_CD", "n_NC", "P", "Weight%"]

for cx, hd in zip(col_x, headers):
    ax.text(cx, header_y, hd,
            transform=ax.get_yaxis_transform(),
            fontsize=62, fontfamily=FONT, color=C_TXT,
            va="center", ha="center", fontweight="bold")

def fmt_p(p: float) -> str:
    if p < 0.001:  return "<0.001"
    if p < 0.01:   return f"{p:.3f}"
    return f"{p:.2f}"

for row, y in zip(rows, yy):
    vals = [
        str(row["n_cd"]),
        str(row["n_nc"]),
        fmt_p(row["p"]),
        f"{row['wpct']:.1f}%",
    ]
    for cx, val in zip(col_x, vals):
        ax.text(cx, y, val,
                transform=ax.get_yaxis_transform(),
                fontsize=60, fontfamily=FONT, color=C_TXT,
                va="center", ha="center", fontweight="bold")

# Pooled row
pool_vals = ["", "", fmt_p(mp), "100%"]
for cx, val in zip(col_x, pool_vals):
    if val:
        ax.text(cx, y_pool, val,
                transform=ax.get_yaxis_transform(),
                fontsize=60, fontfamily=FONT, color=C_DIAMOND,
                va="center", ha="center", fontweight="bold")

# ── Title: fontsize=68, bold ──
total_n = sum(r["n_cd"] + r["n_nc"] for r in results)
ax.set_title(
    f"Faecalibacterium  \u2014  Crohn's disease vs. healthy control\n"
    f"({k} studies \u00b7 {total_n:,} samples \u00b7 DerSimonian\u2013Laird random-effects)",
    fontsize=68, fontfamily=FONT, color=C_TXT,
    fontweight="bold", pad=30)

# ── Footer: fontsize=48, bold ──
p_het_str  = "<0.001" if p_het < 0.001 else f"{p_het:.3f}"
p_meta_str = f"{mp:.2e}" if mp < 0.0001 else f"{mp:.4f}"
# Use only Arial-safe characters: no subscript/superscript Unicode
footer = (
    f"k = {k},  I2 = {I2:.1f}%,  "
    f"tau2 = {tau2:.3f},  "
    f"Q = {Q:.2f} (df = {df_}),  "
    f"P(het) = {p_het_str}  |  "
    f"Pooled log2FC = {bre:.2f} [{cil:.2f}, {cih:.2f}],  "
    f"P = {p_meta_str}"
)
ax.text(0.50, -0.13, footer,
        transform=ax.transAxes, clip_on=False,
        fontsize=48, fontfamily=FONT, color=C_NULL,
        ha="center", va="top", fontweight="bold")

# ── Layout: left smaller (shift plot left), right larger (room for columns) ──
plt.subplots_adjust(left=0.11, right=0.55, top=0.91, bottom=0.20)

# ════════════════════════════════════════════════════════════
# 6. SAVE
# ════════════════════════════════════════════════════════════
for ext in ("png", "pdf"):
    out = os.path.join(OUTPUT_DIR, f"fig2f.{ext}")
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"  -> {out}")
plt.close()
print("Done.")
