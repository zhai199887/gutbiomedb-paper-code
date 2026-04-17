"""Fig 2a — Per-disease cohort-LOCO OOF AUROC bar chart."""
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
sys.path.insert(0, r"E:\microbiomap_clone\compendium_website\api")
from main import _inform_label_mask

CACHE=r"E:\tasks\screenshots\fig3\v6_cache.npz"
META =r"E:\microbiomap_clone\data\metadata.csv"
OOF  =r"E:\tasks\screenshots\fig3\gbhi_universal_oof.npz"
OUT_PNG=r"E:\tasks\screenshots\fig3\fig3a.png"
OUT_PDF=r"E:\tasks\screenshots\fig3\fig3a.pdf"

NICE={"c_difficile_infection":"CDI","CD":"CD","UC":"UC",
      "rheumatoid arthritis":"RA","HIV":"HIV","adenoma":"Adenoma",
      "obesity":"Obesity","IBS":"IBS","colorectal_cancer":"CRC"}
BAR_HI="#3C5488"; BAR_LO="#8491B4"

plt.rcParams.update({"font.family":"Arial","font.size":9,
    "axes.labelsize":10,"axes.titlesize":11,
    "xtick.labelsize":8,"ytick.labelsize":8,"axes.linewidth":0.9})

c=np.load(CACHE,allow_pickle=True); nc_mask=c["nc_mask"]
oof=np.load(OOF,allow_pickle=True)
CLASSES=[str(x) for x in oof["classes"]]
P=oof["oof_probs"].astype(np.float32)
P_nc=P[:,CLASSES.index("NC")]
has_oof=(P.sum(1)>0.5)
dys=1.0-P_nc
meta=pd.read_csv(META,low_memory=False)

def auroc(score,y):
    y=np.asarray(y).astype(bool)
    pos=score[y]; neg=score[~y]
    if len(pos)==0 or len(neg)==0: return float("nan")
    u,_=mannwhitneyu(pos,neg,alternative="two-sided")
    a=u/(len(pos)*len(neg))
    return float(max(a,1-a))

aucs=[]
for dis,nice in NICE.items():
    m=_inform_label_mask(meta,dis)
    dm=(m.values if hasattr(m,"values") else np.asarray(m)) & has_oof
    nm=nc_mask & has_oof
    aucs.append((nice, auroc(dys[dm|nm], dm[dm|nm])))
aucs.sort(key=lambda t:-t[1])
labels=[t[0] for t in aucs]; vals=[t[1] for t in aucs]
colors=[BAR_HI if v>=0.70 else BAR_LO for v in vals]
print("[fig3a]", {l:round(v,3) for l,v in aucs})

fig,ax=plt.subplots(figsize=(5.0,4.6))
y=np.arange(len(labels))[::-1]
ax.barh(y,vals,color=colors,edgecolor="#1a1a1a",linewidth=0.8)
for yi,v in zip(y,vals):
    ax.text(v+0.005,yi,f"{v:.3f}",va="center",fontsize=7.5,fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=8)
ax.axvline(0.50,color="#2c3e50",ls="--",lw=1.0)
ax.axvline(0.70,color="#F39B7F",ls=":",lw=1.0)
ax.set_xlim(0.40,0.85)
ax.set_xlabel("AUROC  (1$-$P(NC), one-vs-rest)",fontweight="bold")
ax.set_title("a  Per-disease cohort-LOCO OOF AUROC",loc="left",fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.patch.set_alpha(0.0)
fig.savefig(OUT_PNG,dpi=400,bbox_inches="tight",transparent=True)
fig.savefig(OUT_PDF,bbox_inches="tight",transparent=True)
print(f"[fig3a] saved {OUT_PNG}")
