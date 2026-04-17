"""Supp Fig 3 — 3x3 grid of NC vs disease health-score densities for all 9
training-pool diseases, scored with the cohort-LOCO OOF probability matrix
from gbhi_universal_oof.npz (same source as Fig 2a/b)."""
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
sys.path.insert(0, r"E:\microbiomap_clone\compendium_website\api")
from main import _inform_label_mask

CACHE=r"E:\tasks\screenshots\fig3\v6_cache.npz"
META =r"E:\microbiomap_clone\data\metadata.csv"
OOF  =r"E:\tasks\screenshots\fig3\gbhi_universal_oof.npz"
OUT_PNG=r"E:\tasks\screenshots\fig3\figS3_gbhi_all9.png"
OUT_PDF=r"E:\tasks\screenshots\fig3\figS3_gbhi_all9.pdf"

NICE={"c_difficile_infection":"CDI","CD":"CD","UC":"UC",
      "rheumatoid arthritis":"RA","HIV":"HIV","adenoma":"Adenoma",
      "obesity":"Obesity","IBS":"IBS","colorectal_cancer":"CRC"}
NC_COLOR="#3CB4E5"; DIS_COLOR="#E05C5C"

plt.rcParams.update({"font.family":"Arial","font.size":9,
    "axes.labelsize":9,"axes.titlesize":8,
    "xtick.labelsize":7.5,"ytick.labelsize":7.5,
    "legend.fontsize":7,"axes.linewidth":0.9})

c=np.load(CACHE,allow_pickle=True); nc_mask=c["nc_mask"]
oof=np.load(OOF,allow_pickle=True)
CLASSES=[str(x) for x in oof["classes"]]
P=oof["oof_probs"].astype(np.float32)
P_nc=P[:,CLASSES.index("NC")]
has_oof=(P.sum(1)>0.5)
meta=pd.read_csv(META,low_memory=False)

nm_all=nc_mask & has_oof
nc_scores=P_nc[nm_all]*100.0

def logit(p):
    p=np.clip(p,1e-4,1-1e-4)
    return np.log(p/(1-p))
nc_logit=logit(P_nc[nm_all])

def auroc(pos,neg):
    u,_=mannwhitneyu(pos,neg,alternative="two-sided")
    a=u/(len(pos)*len(neg))
    return float(max(a,1-a))

# compute per-disease and sort by AUROC desc so layout matches fig2a ordering
rows=[]
for dis,nice in NICE.items():
    m=_inform_label_mask(meta,dis)
    dm=(m.values if hasattr(m,"values") else np.asarray(m)) & has_oof
    dscores=P_nc[dm]*100.0
    dlogit=logit(P_nc[dm])
    auc=auroc(dscores,nc_scores)  # dysbiotic side lower
    rows.append((nice,dscores,dlogit,auc))
rows.sort(key=lambda r:-r[3])
print("[figS3]",[(r[0],round(r[3],3),len(r[1])) for r in rows])

fig,axes=plt.subplots(3,3,figsize=(9.6,8.4),sharex=True)
LLO,LHI=-9.2,9.2  # logit clipped range matches p in [1e-4, 1-1e-4]
bins=np.linspace(LLO,LHI,55)
panel_letters="abcdefghi"
for i,(nice,dscores,dlogit,auc) in enumerate(rows):
    ax=axes[i//3,i%3]
    ax.hist(nc_logit,bins=bins,color=NC_COLOR,alpha=0.6,density=True,
            label=f"NC (n={len(nc_scores):,})",edgecolor="white",linewidth=0.3)
    ax.hist(dlogit,bins=bins,color=DIS_COLOR,alpha=0.6,density=True,
            label=f"{nice} (n={len(dscores):,})",edgecolor="white",linewidth=0.3)
    ax.axvline(np.median(nc_logit),color=NC_COLOR,ls="--",lw=1.1)
    ax.axvline(np.median(dlogit),color=DIS_COLOR,ls="--",lw=1.1)
    ax.axvline(0,color="#888",ls=":",lw=0.7)
    ax.set_xlim(LLO,LHI)
    ax.set_title(f"{panel_letters[i]}  {nice} vs NC   AUROC = {auc:.3f}",
                 loc="left",fontweight="bold")
    ax.legend(frameon=False,loc="upper right",fontsize=6.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if i%3==0: ax.set_ylabel("Density",fontweight="bold")
    if i//3==2: ax.set_xlabel("logit P(NC)  =  log[P(NC)/(1$-$P(NC))]",
                              fontweight="bold")

fig.suptitle("Extended Data Fig. 3 $|$ Cohort-LOCO OOF health-score densities for 9 training-pool diseases",
             fontweight="bold",fontsize=11,y=0.995)
fig.tight_layout(rect=[0,0,1,0.975])
fig.patch.set_alpha(0.0)
fig.savefig(OUT_PNG,dpi=400,bbox_inches="tight",transparent=True)
fig.savefig(OUT_PDF,bbox_inches="tight",transparent=True)
print(f"[figS3] saved {OUT_PNG}")
