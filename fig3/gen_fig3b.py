"""Fig 2b — CD vs NC health-score density (cohort-LOCO OOF)."""
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
sys.path.insert(0, r"E:\microbiomap_clone\compendium_website\api")
from main import _inform_label_mask

CACHE=r"E:\tasks\screenshots\fig3\v6_cache.npz"
META =r"E:\microbiomap_clone\data\metadata.csv"
OOF  =r"E:\tasks\screenshots\fig3\gbhi_universal_oof.npz"
OUT_PNG=r"E:\tasks\screenshots\fig3\fig3b.png"
OUT_PDF=r"E:\tasks\screenshots\fig3\fig3b.pdf"

NC_COLOR="#3CB4E5"; DIS_COLOR="#E05C5C"
plt.rcParams.update({"font.family":"Arial","font.size":9,
    "axes.labelsize":10,"axes.titlesize":11,
    "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
    "axes.linewidth":0.9})

c=np.load(CACHE,allow_pickle=True); nc_mask=c["nc_mask"]
oof=np.load(OOF,allow_pickle=True)
CLASSES=[str(x) for x in oof["classes"]]
P=oof["oof_probs"].astype(np.float32)
P_nc=P[:,CLASSES.index("NC")]
has_oof=(P.sum(1)>0.5)
meta=pd.read_csv(META,low_memory=False)

cd_m_raw=_inform_label_mask(meta,"CD")
cd_m=(cd_m_raw.values if hasattr(cd_m_raw,"values") else np.asarray(cd_m_raw)) & has_oof
nm_all=nc_mask & has_oof
nc_scores=P_nc[nm_all]*100.0
cd_scores=P_nc[cd_m]*100.0
u,_=mannwhitneyu(nc_scores,cd_scores,alternative="two-sided")
a=u/(len(nc_scores)*len(cd_scores)); cd_auc=max(a,1-a)
print(f"[fig3b] CD vs NC OOF AUROC = {cd_auc:.3f}")

def logit(p):
    p=np.clip(p,1e-4,1-1e-4); return np.log(p/(1-p))
nc_lg=logit(P_nc[nm_all]); cd_lg=logit(P_nc[cd_m])

fig,ax=plt.subplots(figsize=(4.6,4.6))
LLO,LHI=-9.2,9.2
bins=np.linspace(LLO,LHI,55)
ax.hist(nc_lg,bins=bins,color=NC_COLOR,alpha=0.6,density=True,
        label=f"NC (n={len(nc_scores):,})",edgecolor="white",linewidth=0.3)
ax.hist(cd_lg,bins=bins,color=DIS_COLOR,alpha=0.6,density=True,
        label=f"CD (n={len(cd_scores):,})",edgecolor="white",linewidth=0.3)
ax.axvline(np.median(nc_lg),color=NC_COLOR,ls="--",lw=1.2)
ax.axvline(np.median(cd_lg),color=DIS_COLOR,ls="--",lw=1.2)
ax.axvline(0,color="#888",ls=":",lw=0.7)
ax.set_xlim(LLO,LHI)
ax.set_xlabel("logit P(NC)  =  log[P(NC)/(1$-$P(NC))]",fontweight="bold")
ax.set_ylabel("Density",fontweight="bold")
ax.legend(frameon=False,loc="upper right",fontsize=7,
          bbox_to_anchor=(0.98,0.98),handlelength=1.2,handletextpad=0.5,borderaxespad=0.2)
ax.set_title(f"b  CD vs NC   OOF AUROC = {cd_auc:.3f}",loc="left",fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.patch.set_alpha(0.0)
fig.savefig(OUT_PNG,dpi=400,bbox_inches="tight",transparent=True)
fig.savefig(OUT_PDF,bbox_inches="tight",transparent=True)
print(f"[fig3b] saved {OUT_PNG}")
