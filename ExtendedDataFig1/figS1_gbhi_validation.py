"""
Supplementary figure — In-sample refit vs cohort-LOCO OOF for universal GBHI.
Left bar = training-fit AUROC (482-cohort refit, optimistic ceiling).
Right bar = cohort-LOCO pooled OOF AUROC (1-P(NC) vs disease, honest external).
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

sys.path.insert(0, r"E:\microbiomap_clone\compendium_website\api")
from main import _inform_label_mask

CACHE    = r"E:\tasks\screenshots\fig1g\v6_cache.npz"
META_CSV = r"E:\microbiomap_clone\data\metadata.csv"
OOF_NPZ  = r"E:\tasks\screenshots\fig1g\gbhi_universal_oof.npz"
DIAG     = r"E:\tasks\screenshots\fig1g\gbhi_universal_loco.json"
OUT_PNG  = r"E:\tasks\screenshots\fig3\figS1_gbhi_validation.png"
OUT_PDF  = r"E:\tasks\screenshots\fig3\figS1_gbhi_validation.pdf"

NICE = {"c_difficile_infection":"CDI","CD":"CD","UC":"UC",
        "rheumatoid arthritis":"RA","HIV":"HIV",
        "adenoma":"Adenoma","obesity":"Obesity","IBS":"IBS",
        "colorectal_cancer":"CRC"}
C_IN = "#8491B4"   # lavender = in-sample refit (optimistic)
C_LO = "#3C5488"   # navy     = cohort-LOCO OOF (honest)

plt.rcParams.update({"font.family":"Arial","font.size":9,
                     "axes.labelsize":10,"axes.titlesize":11,
                     "xtick.labelsize":8,"ytick.labelsize":8,
                     "legend.fontsize":8,"axes.linewidth":0.9})

# ---- in-sample AUROCs from diagnostics json (full-data refit) -----
diag = json.load(open(DIAG, encoding="utf-8"))["diagnostics"]
in_auc = {k: diag[k]["in_sample_auc_1mPNC"] for k in NICE}

# ---- cohort-LOCO pooled OOF AUROCs (same formula as fig1g1 panel a) ----
c = np.load(CACHE, allow_pickle=True)
nc_mask = c["nc_mask"]
oof = np.load(OOF_NPZ, allow_pickle=True)
CLASSES = [str(x) for x in oof["classes"]]
P = oof["oof_probs"].astype(np.float32)
NC_CI = CLASSES.index("NC")
P_nc = P[:, NC_CI]
has_oof = (P.sum(1) > 0.5)
dys = 1.0 - P_nc

meta = pd.read_csv(META_CSV, low_memory=False)

def auroc(score, y):
    y = np.asarray(y).astype(bool)
    pos = score[y]; neg = score[~y]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    u, _ = mannwhitneyu(pos, neg, alternative="two-sided")
    a = u / (len(pos) * len(neg))
    return float(max(a, 1 - a))

oof_auc = {}
for raw in NICE:
    dm = (_inform_label_mask(meta, raw).values if hasattr(_inform_label_mask(meta, raw),"values")
          else np.asarray(_inform_label_mask(meta, raw))) & has_oof
    nm = nc_mask & has_oof
    sel = dm | nm
    oof_auc[raw] = auroc(dys[sel], dm[sel])

rows = [(NICE[k], in_auc[k], oof_auc[k]) for k in NICE]
rows.sort(key=lambda r: -r[2])
labels = [r[0] for r in rows]
isv    = [r[1] for r in rows]
lov    = [r[2] for r in rows]
print("[figS1] in-sample:", {l:round(v,3) for l,v in zip(labels,isv)})
print("[figS1] cohort-LOCO OOF:", {l:round(v,3) for l,v in zip(labels,lov)})

fig = plt.figure(figsize=(7.2, 9.45))
ax = fig.add_subplot(111)
y = np.arange(len(labels))[::-1]
h = 0.36
ax.barh(y + h/2, isv, height=h, color=C_IN, edgecolor="#1a1a1a", linewidth=0.6,
        label="In-sample refit (training fit, 482-cohort pool)")
ax.barh(y - h/2, lov, height=h, color=C_LO, edgecolor="#1a1a1a", linewidth=0.6,
        label="cohort-LOCO OOF (1-P(NC), honest external)")
for yi, v in zip(y, isv):
    ax.text(v + 0.005, yi + h/2, f"{v:.3f}", va="center", fontsize=7, fontweight="bold")
for yi, v in zip(y, lov):
    ax.text(v + 0.005, yi - h/2, f"{v:.3f}", va="center", fontsize=7, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9, fontweight="bold")
ax.axvline(0.50, color="#2c3e50", ls="--", lw=1.0)
ax.axvline(0.70, color="#F39B7F", ls=":",  lw=1.0)
ax.set_xlim(0.40, 1.02)
ax.set_xlabel("AUROC  (disease vs NC, one-vs-rest)", fontweight="bold")
ax.set_title("Supplementary — Universal GBHI: in-sample refit vs cohort-LOCO OOF",
             loc="left", fontweight="bold", pad=10)
ax.legend(frameon=False, loc="upper right", fontsize=8,
          bbox_to_anchor=(1.0, -0.12), ncol=2)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=400, bbox_inches="tight", transparent=True)
fig.savefig(OUT_PDF, bbox_inches="tight", transparent=True)
print(f"saved {OUT_PNG}")
print(f"saved {OUT_PDF}")
