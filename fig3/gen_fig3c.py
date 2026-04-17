"""Fig 2c — Single-sample GBHI gauge readout (CD patient)."""
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
sys.path.insert(0, r"E:\microbiomap_clone\compendium_website\api")
from main import _inform_label_mask

CACHE=r"E:\tasks\screenshots\fig3\v6_cache.npz"
META =r"E:\microbiomap_clone\data\metadata.csv"
OOF  =r"E:\tasks\screenshots\fig3\gbhi_universal_oof.npz"
OUT_PNG=r"E:\tasks\screenshots\fig3\fig3c.png"
OUT_PDF=r"E:\tasks\screenshots\fig3\fig3c.pdf"

GAUGE_POOR="#E05C5C"; GAUGE_FAIR="#F39B7F"; GAUGE_GOOD="#00A087"
plt.rcParams.update({"font.family":"Arial","font.size":9,
    "axes.labelsize":10,"axes.titlesize":11,"axes.linewidth":0.9})

c=np.load(CACHE,allow_pickle=True); nc_mask=c["nc_mask"]
proj_all=np.array([str(x) for x in c["project"]])
_idkey="sample_key" if "sample_key" in c.files else ("sample_id" if "sample_id" in c.files else None)
samp_ids=np.array([str(x) for x in c[_idkey]]) if _idkey else None
oof=np.load(OOF,allow_pickle=True)
CLASSES=[str(x) for x in oof["classes"]]
P=oof["oof_probs"].astype(np.float32)
P_nc=P[:,CLASSES.index("NC")]; P_cd=P[:,CLASSES.index("CD")]
has_oof=(P.sum(1)>0.5)
meta=pd.read_csv(META,low_memory=False)

cd_m_raw=_inform_label_mask(meta,"CD")
cd_m=(cd_m_raw.values if hasattr(cd_m_raw,"values") else np.asarray(cd_m_raw)) & has_oof
cd_rows=np.where(cd_m)[0]
rng=np.random.default_rng(42)
pick=int(rng.choice(cd_rows))
p_cd_pick=float(P_cd[pick]); p_nc_pick=float(P_nc[pick])
health=p_nc_pick*100.0
cohort=proj_all[pick]
sid=str(samp_ids[pick]) if samp_ids is not None else f"row#{pick}"
print(f"[fig3c] pick {sid}  P(NC)={p_nc_pick:.3f}  health={health:.1f}")

fig,ax=plt.subplots(figsize=(5.0,4.6))
ax.set_aspect("equal"); ax.patch.set_alpha(0.0)
def ang(s): return 180.0-1.80*s
R_out,R_in=1.00,0.74
ax.add_patch(Wedge((0,0),R_out,ang(33),ang(0),width=R_out-R_in,facecolor=GAUGE_POOR,edgecolor="none"))
ax.add_patch(Wedge((0,0),R_out,ang(66),ang(33),width=R_out-R_in,facecolor=GAUGE_FAIR,edgecolor="none"))
ax.add_patch(Wedge((0,0),R_out,ang(100),ang(66),width=R_out-R_in,facecolor=GAUGE_GOOD,edgecolor="none"))
for s,lab in [(0,"0"),(50,"50"),(100,"100")]:
    a_=np.deg2rad(ang(s))
    ax.text(1.12*np.cos(a_),1.12*np.sin(a_),lab,ha="center",va="center",
            fontsize=10,color="#1a1a1a",fontweight="bold")
a_p=np.deg2rad(ang(health))
ax.plot([0,0.86*np.cos(a_p)],[0,0.86*np.sin(a_p)],
        color="#1a1a1a",lw=3.2,solid_capstyle="round")
ax.add_patch(Circle((0,0),0.055,color="#1a1a1a",zorder=6))
if health>=66:   tier,tc="Good",GAUGE_GOOD
elif health>=33: tier,tc="Fair",GAUGE_FAIR
else:            tier,tc="Poor",GAUGE_POOR
ax.text(0,0.38,f"{health:.0f}",ha="center",va="center",
        fontsize=40,fontweight="bold",color="#1a1a1a")
ax.text(0,0.12,tier,ha="center",va="center",fontsize=13,fontweight="bold",color=tc)
ax.text(0,-0.22,"Universal GBHI health score",ha="center",va="center",
        fontsize=9,color="#4a5568",fontweight="bold")
ax.text(0,-0.36,f"CD patient  {sid}",ha="center",va="center",
        fontsize=8,color="#1a1a1a",style="italic")
ax.text(0,-0.48,f"cohort {cohort}   P(CD)={p_cd_pick:.3f}",
        ha="center",va="center",fontsize=7.5,color="#4a5568")
ax.set_xlim(-1.25,1.25); ax.set_ylim(-0.62,1.25)
ax.set_xticks([]); ax.set_yticks([])
for sp in ["top","right","bottom","left"]: ax.spines[sp].set_visible(False)
ax.set_title("c  Single-sample readout (CD patient)",loc="left",
             fontweight="bold",color="#1a1a1a")

fig.patch.set_alpha(0.0)
fig.savefig(OUT_PNG,dpi=400,bbox_inches="tight",transparent=True)
fig.savefig(OUT_PDF,bbox_inches="tight",transparent=True)
print(f"[fig3c] saved {OUT_PNG}")
