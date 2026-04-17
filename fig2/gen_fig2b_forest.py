#!/usr/bin/env python3
"""
Figure 2b — Veillonella enrichment in Crohn's disease.
Ultra-compact visual forest plot for multi-panel figure.
All numeric details go in figure legend / Extended Data.

python gen_fig2b_forest.py
"""
import csv, math, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ABUNDANCE_PATH = r"E:\microbiomap_clone\data\unfiltered_abundance.csv"
METADATA_PATH  = r"E:\microbiomap_clone\data\metadata.csv"
OUTPUT_DIR     = os.path.dirname(os.path.abspath(__file__))
MIN_SAMPLES    = 20

# ═══════════════ DATA ═══════════════
print("[1] Loading …")
with open(ABUNDANCE_PATH) as f:
    hdr = f.readline().strip().split(",")
vcol = next(i for i, c in enumerate(hdr) if c.endswith(".Veillonella"))
sv, st = {}, {}
with open(ABUNDANCE_PATH) as f:
    r = csv.reader(f); next(r)
    for row in r:
        s = row[0]
        try: v = float(row[vcol])
        except: v = 0.0
        sv[s] = v
        st[s] = sum((float(x) for x in row[1:] if x), start=0.0)
srel = {s: (sv[s]/st[s] if st[s]>0 else 0) for s in sv}

meta = pd.read_csv(METADATA_PATH, low_memory=False)
meta["sk"] = meta["project"].astype(str).str.strip()+"_"+meta["srr"].astype(str).str.strip()
ic = ["inform-all"]+[f"inform{i}" for i in range(12)]
av = [c for c in ic if c in meta.columns]
cd_m = pd.Series(False, index=meta.index)
for c in av: cd_m |= meta[c].fillna("").astype(str).str.strip().str.lower()=="cd"
nc_m = meta["inform-all"].fillna("").astype(str).str.strip()=="NC"
for c in av[1:]:
    if c in meta.columns: nc_m &= meta[c].fillna("").astype(str).str.strip().isin(["NC",""])

pseudo = 1e-6
results = []
for proj in sorted(meta[cd_m].groupby("project")["sk"].apply(list).to_dict()):
    cdk = meta[(meta["project"]==proj)&cd_m]["sk"].tolist()
    nck = meta[(meta["project"]==proj)&nc_m]["sk"].tolist()
    cdv = [k for k in cdk if k in srel]; ncv = [k for k in nck if k in srel]
    if len(cdv)<MIN_SAMPLES or len(ncv)<MIN_SAMPLES: continue
    vd, vc = np.array([srel[k] for k in cdv]), np.array([srel[k] for k in ncv])
    md, mc = float(np.mean(vd)), float(np.mean(vc))
    lfc = math.log2((md+pseudo)/(mc+pseudo))
    se = float(np.sqrt(np.var(vd)/len(vd)+np.var(vc)/len(vc)))
    try: _, p = stats.mannwhitneyu(vd, vc, alternative="two-sided"); p=float(p)
    except: p=1.0
    results.append(dict(project=proj, n_cd=len(cdv), n_nc=len(ncv),
                        log2fc=lfc, se=se, p=p))

# ═══════════════ META-ANALYSIS ═══════════════
eff = [(r["log2fc"],r["se"]) for r in results if r["se"]>0]
wf = [1/se**2 for _,se in eff]; ws = sum(wf)
bfe = sum(w*e for w,(e,_) in zip(wf,eff))/ws
Q = sum(w*(e-bfe)**2 for w,(e,_) in zip(wf,eff))
dfv = len(eff)-1
I2 = max(0,(Q-dfv)/Q*100) if Q>0 else 0
c_ = ws-sum(w**2 for w in wf)/ws
tau2 = max(0,(Q-dfv)/c_) if c_>0 else 0
rw = [1/(se**2+tau2) for _,se in eff]; rws = sum(rw)
bre = sum(w*e for w,(e,_) in zip(rw,eff))/rws
sre = math.sqrt(1/rws); zz = bre/sre
mp = float(2*(1-stats.norm.cdf(abs(zz))))
cil, cih = bre-1.96*sre, bre+1.96*sre
print(f"    k={len(results)}, pooled={bre:.2f} [{cil:.2f},{cih:.2f}] P={mp:.1e}")

results.sort(key=lambda x: x["log2fc"])
n = len(results)
twfe = sum(1/r["se"]**2 for r in results if r["se"]>0)
rows = []
for r in results:
    re_se = math.sqrt(r["se"]**2+tau2)
    rows.append({**r, "cl":r["log2fc"]-1.96*re_se, "ch":r["log2fc"]+1.96*re_se,
                 "wpct":(1/r["se"]**2)/twfe*100 if r["se"]>0 else 0})

# ═══════════════ FIGURE ═══════════════
print("[2] Drawing …")

# Lifecycle stacked-area colour scheme (warm orange/teal/blue)
C_SQ   = "#e67e22"    # warm orange squares
C_SQ_E = "#d35400"    # darker orange edge
C_DM   = "#c0392b"    # red diamond
C_DM_E = "#922b21"
C_TXT  = "#2c3e50"    # dark navy text
C_SUB  = "#5d6d7e"    # steel-blue grey
C_BG   = "#f7f9fc"    # very light blue-grey
C_RULE = "#d5dbdb"
C_NULL = "#aeb6bf"
FONT   = "Arial"

yy = list(range(n-1, -1, -1))
y_pool = -1.8

fig, ax = plt.subplots(figsize=(8, 9), facecolor="white")
ax.set_facecolor("white")
ax.set_ylim(y_pool-1.5, n-1+0.8)

# Alternating rows
for i in range(n):
    if i % 2 == 0:
        ax.axhspan(yy[i]-0.42, yy[i]+0.42, color=C_BG, zorder=0)

# Null line
ax.axvline(0, color=C_NULL, lw=1.2, ls="--", zorder=1)

# Studies
for row, y in zip(rows, yy):
    ax.plot([row["cl"], row["ch"]], [y, y],
            color=C_SUB, lw=2.0, solid_capstyle="butt", zorder=3)
    sz = max(8, min(20, 7+row["wpct"]*0.45))
    ax.plot(row["log2fc"], y, "s", color=C_SQ, ms=sz,
            mec=C_SQ_E, mew=1.0, zorder=5)

# Separator
ax.axhline(-0.7, color=C_RULE, lw=1.0, zorder=1)

# Diamond
dh = 0.35
ax.add_patch(Polygon(
    [(cil,y_pool),(bre,y_pool+dh),(cih,y_pool),(bre,y_pool-dh)],
    closed=True, fc=C_DM, ec=C_DM_E, lw=1.2, zorder=5, alpha=0.92))

# X axis
all_x = [r["cl"] for r in rows]+[r["ch"] for r in rows]+[cil,cih]
ax.set_xlim(min(all_x)-0.8, max(all_x)+0.8)
ax.set_yticks([])
for sp in ["top","right","left"]:
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_linewidth(0.8)
ax.spines["bottom"].set_color(C_SUB)
ax.tick_params(axis="x", width=0.8, length=5, labelsize=18, colors=C_SUB)
ax.tick_params(axis="y", length=0)
ax.set_xlabel(r"log$_2$ fold change (CD vs. control)",
              fontsize=20, fontfamily=FONT, color=C_TXT, labelpad=12)

# LEFT: study names only — HUGE font
for row, y in zip(rows, yy):
    ax.text(-0.03, y, row["project"],
            transform=ax.get_yaxis_transform(),
            fontsize=19, fontfamily=FONT, color=C_TXT,
            va="center", ha="right", fontweight="medium")

# Pooled label
ax.text(-0.03, y_pool,
        f'Pooled  {bre:.2f} [{cil:.2f}, {cih:.2f}]',
        transform=ax.get_yaxis_transform(),
        fontsize=19, fontfamily=FONT, color=C_DM,
        va="center", ha="right", fontweight="bold")

# Footer — below xlabel with enough gap
ax.text(0.5, -0.14,
        f"k = {len(results)},  I\u00B2 = {I2:.1f}%,  P = {mp:.1e}",
        transform=ax.transAxes,
        fontsize=14, fontfamily=FONT, color=C_NULL, ha="center")

plt.subplots_adjust(left=0.42, right=0.95, top=0.97, bottom=0.15)

for ext in ("png", "pdf"):
    p = os.path.join(OUTPUT_DIR, f"fig2b_forest_CD_Veillonella_real.{ext}")
    fig.savefig(p, dpi=300, facecolor="white")
    print(f"  -> {p}")
plt.close()
print("Done.")
