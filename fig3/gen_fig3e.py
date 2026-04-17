"""Fig 2e — ROC: NC training pool vs unseen-disease pool (cohort-LOCO OOF)."""
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
sys.path.insert(0, r"E:\microbiomap_clone\compendium_website\api")
from main import _inform_label_mask

CACHE=r"E:\tasks\screenshots\fig3\v6_cache.npz"
META =r"E:\microbiomap_clone\data\metadata.csv"
OOF  =r"E:\tasks\screenshots\fig3\gbhi_universal_oof.npz"
OUT_PNG=r"E:\tasks\screenshots\fig3\fig3e.png"
OUT_PDF=r"E:\tasks\screenshots\fig3\fig3e.pdf"

NICE9=["c_difficile_infection","CD","UC","rheumatoid arthritis","HIV",
       "adenoma","obesity","IBS","colorectal_cancer"]
plt.rcParams.update({"font.family":"Arial","font.size":9,
    "axes.labelsize":10,"axes.titlesize":11,"axes.linewidth":0.9})

c=np.load(CACHE,allow_pickle=True); nc_mask=c["nc_mask"]; n=nc_mask.shape[0]
oof=np.load(OOF,allow_pickle=True)
CLASSES=[str(x) for x in oof["classes"]]
P=oof["oof_probs"].astype(np.float32)
P_nc=P[:,CLASSES.index("NC")]
has_oof=(P.sum(1)>0.5)
score_oof=1.0-P_nc
meta=pd.read_csv(META,low_memory=False)

y=np.full(n,-1,dtype=np.int8); y[nc_mask]=0
for i,dis in enumerate(NICE9,start=1):
    m=_inform_label_mask(meta,dis)
    dm=m.values if hasattr(m,"values") else np.asarray(m)
    y[dm & (y==-1)]=i

nc_sel=(y==0) & has_oof
ext_sel=(y==-1)
nc_score=score_oof[nc_sel]; ext_score=score_oof[ext_sel]
y_true=np.concatenate([np.zeros(len(nc_score)),np.ones(len(ext_score))])
y_score=np.concatenate([nc_score,ext_score])
fpr,tpr,_=roc_curve(y_true,y_score)
auc=roc_auc_score(y_true,y_score)
print(f"[fig3e] NC vs unseen-disease pool AUROC = {auc:.3f}")

fig,ax=plt.subplots(figsize=(4.6,4.6))
ax.plot(fpr,tpr,color="#3C5488",lw=2.0)
ax.plot([0,1],[0,1],color="#8491B4",ls="--",lw=1.0)
ax.set_xlabel("False positive rate",fontweight="bold")
ax.set_ylabel("True positive rate",fontweight="bold")
ax.set_title(f"e  NC vs unseen-disease pool\n(AUROC = {auc:.3f})",
             loc="left",fontweight="bold")
ax.set_xlim(0,1); ax.set_ylim(0,1.005)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.patch.set_alpha(0.0)
fig.savefig(OUT_PNG,dpi=400,bbox_inches="tight",transparent=True)
fig.savefig(OUT_PDF,bbox_inches="tight",transparent=True)
print(f"[fig3e] saved {OUT_PNG}")
