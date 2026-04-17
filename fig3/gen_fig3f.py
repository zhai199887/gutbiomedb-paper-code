"""Fig 2f — Health-score distribution: NC training pool vs unseen-disease pool."""
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, r"E:\microbiomap_clone\compendium_website\api")
from main import _inform_label_mask

CACHE=r"E:\tasks\screenshots\fig3\v6_cache.npz"
META =r"E:\microbiomap_clone\data\metadata.csv"
OOF  =r"E:\tasks\screenshots\fig3\gbhi_universal_oof.npz"
OUT_PNG=r"E:\tasks\screenshots\fig3\fig3f.png"
OUT_PDF=r"E:\tasks\screenshots\fig3\fig3f.pdf"

NICE9=["c_difficile_infection","CD","UC","rheumatoid arthritis","HIV",
       "adenoma","obesity","IBS","colorectal_cancer"]
plt.rcParams.update({"font.family":"Arial","font.size":9,
    "axes.labelsize":10,"axes.titlesize":11,
    "legend.fontsize":8,"axes.linewidth":0.9})

c=np.load(CACHE,allow_pickle=True); nc_mask=c["nc_mask"]; n=nc_mask.shape[0]
oof=np.load(OOF,allow_pickle=True)
CLASSES=[str(x) for x in oof["classes"]]
P=oof["oof_probs"].astype(np.float32)
P_nc=P[:,CLASSES.index("NC")]
has_oof=(P.sum(1)>0.5)
health_oof=P_nc*100.0
meta=pd.read_csv(META,low_memory=False)

y=np.full(n,-1,dtype=np.int8); y[nc_mask]=0
for i,dis in enumerate(NICE9,start=1):
    m=_inform_label_mask(meta,dis)
    dm=m.values if hasattr(m,"values") else np.asarray(m)
    y[dm & (y==-1)]=i

nc_sel=(y==0) & has_oof
ext_sel=(y==-1)
nc_health=health_oof[nc_sel]
ext_health=health_oof[ext_sel]
print(f"[fig3f] NC n={len(nc_health):,}  unseen pool n={len(ext_health):,}")

def logit(p):
    p=np.clip(p,1e-4,1-1e-4); return np.log(p/(1-p))
nc_lg=logit(P_nc[nc_sel]); ext_lg=logit(P_nc[ext_sel])

fig,ax=plt.subplots(figsize=(5.0,4.6))
LLO,LHI=-9.2,9.2
bins=np.linspace(LLO,LHI,55)
ax.hist(nc_lg,bins=bins,color="#3CB4E5",alpha=0.6,density=True,
        label=f"NC (n={len(nc_health):,})",edgecolor="white",lw=0.3)
ax.hist(ext_lg,bins=bins,color="#E05C5C",alpha=0.6,density=True,
        label=f"Unseen (n={len(ext_health):,})",
        edgecolor="white",lw=0.3)
ax.axvline(np.median(nc_lg),color="#3CB4E5",ls="--",lw=1.2)
ax.axvline(np.median(ext_lg),color="#E05C5C",ls="--",lw=1.2)
ax.axvline(0,color="#888",ls=":",lw=0.7)
ax.set_xlim(LLO,LHI)
ax.set_xlabel("logit P(NC)  =  log[P(NC)/(1$-$P(NC))]",fontweight="bold")
ax.set_ylabel("Density",fontweight="bold")
ax.legend(frameon=False,loc="upper right",fontsize=7,
          bbox_to_anchor=(0.98,0.98),handlelength=1.2,handletextpad=0.5,borderaxespad=0.2)
ax.set_title("f  NC training vs unseen-disease pool",loc="left",fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.patch.set_alpha(0.0)
fig.savefig(OUT_PNG,dpi=400,bbox_inches="tight",transparent=True)
fig.savefig(OUT_PDF,bbox_inches="tight",transparent=True)
print(f"[fig3f] saved {OUT_PNG}")
