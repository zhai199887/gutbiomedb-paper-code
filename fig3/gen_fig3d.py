"""Fig 2d — Health-score distribution by label stratum (NC / Disease / Unknown)."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt

CACHE=r"E:\tasks\screenshots\fig3\v6_cache.npz"
META =r"E:\microbiomap_clone\data\metadata.csv"
OOF  =r"E:\tasks\screenshots\fig3\gbhi_universal_oof.npz"
OUT_PNG=r"E:\tasks\screenshots\fig3\fig3d.png"
OUT_PDF=r"E:\tasks\screenshots\fig3\fig3d.pdf"

GAUGE_POOR="#E05C5C"; GAUGE_FAIR="#F39B7F"; GAUGE_GOOD="#00A087"
plt.rcParams.update({"font.family":"Arial","font.size":9,
    "axes.labelsize":10,"axes.titlesize":11,
    "xtick.labelsize":8,"ytick.labelsize":8,"axes.linewidth":0.9})

c=np.load(CACHE,allow_pickle=True); nc_mask=c["nc_mask"]
oof=np.load(OOF,allow_pickle=True)
CLASSES=[str(x) for x in oof["classes"]]
P=oof["oof_probs"].astype(np.float32)
P_nc=P[:,CLASSES.index("NC")]
has_oof=(P.sum(1)>0.5)
H_all=P_nc*100.0
meta=pd.read_csv(META,low_memory=False)

ia_all=meta["inform-all"].fillna("").astype(str).str.strip()
str_nc =ia_all.str.upper().eq("NC").values & has_oof
str_unk=((ia_all.eq("")|ia_all.str.lower().isin(["nan","unknown","none","na"])).values) & has_oof
str_dis=(~(ia_all.str.upper().eq("NC").values)) & (~((ia_all.eq("")|ia_all.str.lower().isin(["nan","unknown","none","na"])).values)) & has_oof
STRATA=[("NC",str_nc,"#3CB4E5"),("Disease",str_dis,"#E05C5C"),("Unknown",str_unk,"#8491B4")]
print(f"[fig3d] NC={str_nc.sum():,} Disease={str_dis.sum():,} Unknown={str_unk.sum():,}")

fig,ax=plt.subplots(figsize=(6.0,4.6))
ax.set_facecolor("white")
ax.axhspan(0,40,color=GAUGE_POOR,alpha=0.055,zorder=0)
ax.axhspan(40,70,color=GAUGE_FAIR,alpha=0.065,zorder=0)
ax.axhspan(70,100,color=GAUGE_GOOD,alpha=0.065,zorder=0)
for yy in (40,70):
    ax.axhline(yy,color="#b8c0c6",lw=0.6,ls=(0,(2,3)),zorder=1)

xpos=np.array([0,1,2],dtype=float)
parts=ax.violinplot([H_all[m] for _,m,_ in STRATA],positions=xpos,widths=0.82,
    showmeans=False,showmedians=False,showextrema=False,bw_method=0.22)
for pc,(_,_,col) in zip(parts["bodies"],STRATA):
    pc.set_facecolor(col); pc.set_edgecolor("#2c3e50")
    pc.set_linewidth(1.0); pc.set_alpha(0.78)

for x,(_,m,col) in zip(xpos,STRATA):
    s=H_all[m]
    q1,med,q3=np.percentile(s,[25,50,75])
    lo_w,hi_w=np.percentile(s,[5,95])
    ax.vlines(x,lo_w,hi_w,color="#1a1a1a",lw=1.1,zorder=5)
    ax.add_patch(plt.Rectangle((x-0.075,q1),0.15,q3-q1,facecolor="white",
        edgecolor="#1a1a1a",linewidth=1.3,zorder=6))
    ax.hlines(med,x-0.095,x+0.095,color="#1a1a1a",lw=2.8,zorder=7)
    ax.hlines(med,x-0.095,x+0.095,color="white",lw=1.0,zorder=8)

for x,(name,m,col) in zip(xpos,STRATA):
    s=H_all[m]; med=np.median(s)
    ax.text(x,108,f"{med:.1f}",ha="center",va="bottom",
            fontsize=13,fontweight="bold",color=col)
    ax.text(x,104,"median",ha="center",va="bottom",
            fontsize=6.8,color="#4a5568",fontweight="bold",fontstyle="italic")

bar_y=-11; bar_h=5.5
for x,(name,m,col) in zip(xpos,STRATA):
    s=H_all[m]
    hi=(s>=70).mean(); mo=((s>=40)&(s<70)).mean(); lo=(s<40).mean()
    bw=0.68; x0=x-bw/2
    for frac,cc in [(lo,GAUGE_POOR),(mo,GAUGE_FAIR),(hi,GAUGE_GOOD)]:
        ax.add_patch(plt.Rectangle((x0,bar_y),bw*frac,bar_h,facecolor=cc,
            edgecolor="none",alpha=0.92,zorder=4))
        if frac>=0.12:
            ax.text(x0+bw*frac/2,bar_y+bar_h/2,f"{frac*100:.0f}%",
                    ha="center",va="center",fontsize=6.5,color="white",
                    fontweight="bold",zorder=5)
        x0+=bw*frac
    ax.add_patch(plt.Rectangle((x-bw/2,bar_y),bw,bar_h,facecolor="none",
        edgecolor="#1a1a1a",linewidth=0.8,zorder=6))

ax.text(1.015,85,"high\n$\\geq$70",transform=ax.get_yaxis_transform(),
        ha="left",va="center",fontsize=7.2,color=GAUGE_GOOD,fontweight="bold")
ax.text(1.015,55,"moderate\n40–70",transform=ax.get_yaxis_transform(),
        ha="left",va="center",fontsize=7.2,color=GAUGE_FAIR,fontweight="bold")
ax.text(1.015,20,"low\n$<$40",transform=ax.get_yaxis_transform(),
        ha="left",va="center",fontsize=7.2,color=GAUGE_POOR,fontweight="bold")

ax.set_xticks(xpos)
ax.set_xticklabels(
    [f"$\\bf{{NC}}$\nn = {str_nc.sum():,}",
     f"$\\bf{{Disease}}$\nn = {str_dis.sum():,}",
     f"$\\bf{{Unknown}}$\nn = {str_unk.sum():,}"],fontsize=8.2)
ax.set_ylabel("Universal GBHI health score",fontweight="bold",labelpad=6)
ax.set_ylim(-18,122); ax.set_xlim(-0.70,2.90)
ax.set_yticks([0,20,40,60,80,100])
ax.tick_params(axis="y",length=3,width=0.8,colors="#2c3e50")
ax.tick_params(axis="x",length=0,pad=6)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#2c3e50"); ax.spines["left"].set_linewidth(0.9)
ax.spines["bottom"].set_color("#2c3e50"); ax.spines["bottom"].set_linewidth(0.9)
ax.set_title("d  Score distribution by label stratum  (n = 168{,}464)",
             loc="left",fontweight="bold",color="#1a1a1a",pad=14)

fig.patch.set_alpha(0.0)
fig.savefig(OUT_PNG,dpi=400,bbox_inches="tight",transparent=True)
fig.savefig(OUT_PDF,bbox_inches="tight",transparent=True)
print(f"[fig3d] saved {OUT_PNG}")
