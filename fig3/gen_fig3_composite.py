"""Fig 3 — 3x2 composite from fig3a-f PNGs via matplotlib."""
import matplotlib.pyplot as plt, matplotlib.image as mpimg, os

SRC = r"E:\tasks\screenshots\fig3"
PANELS = ["fig3a","fig3b","fig3c","fig3d","fig3e","fig3f"]
LABELS = "abcdef"
OUT_PDF = os.path.join(SRC,"fig3.pdf")
OUT_PNG = os.path.join(SRC,"fig3.png")

fig, axes = plt.subplots(3, 2, figsize=(7.2, 9.45))
for ax, name, lbl in zip(axes.ravel(), PANELS, LABELS):
    img = mpimg.imread(os.path.join(SRC, name+".png"))
    ax.imshow(img); ax.set_axis_off()
    ax.text(-0.010, 1.00, lbl, transform=ax.transAxes,
            fontsize=8, fontweight="bold",
            ha="right", va="bottom", family="Arial")

fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01,wspace=0.02,hspace=0.02)
fig.patch.set_alpha(0.0)
fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight", transparent=True)
fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", transparent=True)
print(f"[fig3] saved {OUT_PDF}")
print(f"[fig3] saved {OUT_PNG}")
