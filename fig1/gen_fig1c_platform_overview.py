#!/usr/bin/env python3
"""
Fig 1c - GutBiomeDB Platform Functional Overview
v4: aligned with fig1a4 palette + 5 real page screenshots embedded

Layout:
  - Top bar: hero + tagline + gold line
  - Browsing  (6 cards, navy band) — first 5 use real screenshots (Playwright)
  - Analytics (5 cards, coral band) — all matplotlib v4 palette
  - Export    (2 cards, teal band)  — all matplotlib v4 palette
  - Double soft-glow shadows, band gradient fills, funnel triangle flow

Run: python gen_fig1c_platform_overview.py
"""
import os
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, Circle, Arc, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

# ── Paths ────────────────────────────────────────────────────────────────────
THUMB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thumbs")
OUTPUT_DIR = r"E:\tasks\screenshots\fig1"

# ── Nature rcParams ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "Arial",
    "font.sans-serif":  ["Arial", "DejaVu Sans"],
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "svg.fonttype":     "none",
    "axes.linewidth":   0.8,
    "patch.linewidth":  0.8,
    "figure.facecolor": "white",
    "savefig.facecolor":"white",
    "savefig.dpi":      600,
})

# ── Fig 1a4 base color palette ───────────────────────────────────────────────
NAVY="#253D6E"; MID="#5271A4"; PALE="#8EB3D5"
PEACH="#F7B89A"; CORAL="#D4735A"; CRIMSON="#C41435"
TEAL="#1A6B82"; FOREST="#00836F"

# Color palette definition
V4 = {
    "browse": dict(
        band="#EAF0F8", band2="#DBE5F1",
        border=NAVY, ink="#1B2D52",
        hero=NAVY, tag_bg=NAVY, tag_fg="#FFFFFF",
    ),
    "analytics": dict(
        band="#E6EEF3", band2="#D1DFE8",
        border=TEAL, ink="#13425B",
        hero=NAVY, tag_bg=TEAL, tag_fg="#FFFFFF",
    ),
    "export": dict(
        band="#E7EEEC", band2="#D4E0DC",
        border=FOREST, ink="#0E4A5A",
        hero=FOREST, tag_bg=FOREST, tag_fg="#FFFFFF",
    ),
}

GOLD = "#C9A227"
MUTE = "#5A5A5A"

# ── Modules: (row, title, hero, detail, thumb_key, crop_hint) ─────────────────
# crop_hint: (left_frac, top_frac, right_frac, bottom_frac) for real screenshots
MODULES = [
    # row, title, hero, detail, thumb, crop(None=full image), badge
    ("browse", "Phenotype\nExplorer",    "3-way",      "disease × age × sex",      "browse_phenotype_real",  None, "Filter"),
    ("browse", "Disease\nBrowser",       "224",        "diseases  ·  MeSH",        "browse_disease_real",    None, "MeSH"),
    ("browse", "Genus\nSearch",          "3,143",      "genera · cross-disease",   "browse_search_real",     None, "Trie"),
    ("browse", "Study\nDatasets",        "482",        "projects · 72 countries",  "browse_studies_real",    None, "SRA"),
    ("browse", "Metabolic\nFunction",    "15",         "categories · KEGG",        "browse_metabolism_real", None, "KEGG"),
    # Analytics
    ("analytics", "Similarity\nSearch",           "k-NN",      "Bray-Curtis · Jaccard",     "browse_similarity_v7",        None, "k-NN"),
    ("analytics", "Differential\nAnalysis",       "Wilcoxon",  "CD vs NC · 3,143 genera",  "analytics_differential_v11", None, "Wilcoxon"),
    ("analytics", "Cross-study\nMeta-analysis",   "Wilcoxon + REM", "7 cohorts · mean I²=59.0%", "analytics_meta_v11",          None, "REM"),
    ("analytics", "Network\nAnalysis",            "SparCC",    "62 nodes · 180 edges",       "analytics_network_v11",      None, "SparCC"),
    ("analytics", "Lifecycle\nAtlas",             "Kruskal-Wallis", "7 stages · 130,499 samples", "analytics_lifecycle_v11", None, "KW"),
    ("analytics", "GBHI\nHealth Score",           "75.4",      "NC median · 0–100 scale",   "analytics_gbhi_v11",          None, "LOCO"),
    ("export", "Download\n+ REST API",            "46",        "endpoints · CSV·SVG·PNG",  "export_download_v7",         None, "OpenAPI"),
    ("export", "About\n& Cite",                   "DOI",       "authors · license · cite", "export_about_v7",            None, "CC-BY"),
]

# ── Canvas layout ────────────────────────────────────────────────────────────
FIG_W, FIG_H = 17.6, 11.95
LEFT_PAD  = 0.70
RIGHT_PAD = 0.55
TOP_BAR_H = 1.45

# Layout dimensions for card grid
CARD_W_ROW = {"browse": 3.00, "analytics": 2.50, "export": 2.80}
CARD_H_ROW = {"browse": 3.10, "analytics": 2.55, "export": 2.35}
ROW_HEADER_H = 0.48
BAND_H_ROW = {
    "browse":    CARD_H_ROW["browse"]    + ROW_HEADER_H + 0.10,
    "analytics": CARD_H_ROW["analytics"] + ROW_HEADER_H + 0.10,
    "export":    CARD_H_ROW["export"]    + ROW_HEADER_H + 0.10,
}
GAP_X = 0.26
BAND_GAP = 0.24

_band_top_of_browse = FIG_H - TOP_BAR_H - 0.25
BAND_Y = {
    "browse":    _band_top_of_browse - BAND_H_ROW["browse"],
    "analytics": _band_top_of_browse - BAND_H_ROW["browse"] - BAND_GAP - BAND_H_ROW["analytics"],
    "export":    _band_top_of_browse - BAND_H_ROW["browse"] - BAND_GAP - BAND_H_ROW["analytics"] - BAND_GAP - BAND_H_ROW["export"],
}
# card coordinates: each band top reserves ROW_HEADER_H for row header, cards below
ROW_Y = {k: BAND_Y[k] + 0.05 for k in BAND_Y}
HEADER_Y_TOP = {k: BAND_Y[k] + BAND_H_ROW[k] for k in BAND_Y}

def row_usable_w():
    return FIG_W - LEFT_PAD - RIGHT_PAD

# ── Inline thumbnail drawing functions ───────────────────────────────────────
rng = np.random.default_rng(42)

def _ui_bg(ax, color="#F4F7FB"):
    ax.set_facecolor(color)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

def _thumb_phenotype(ax):
    """Phenotype Explorer — 1:1 replica matching the real platform screenshot."""
    _ui_bg(ax, "white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    T = ax.transAxes
    NAVY = "#1F3864"; RED = "#C41435"; INK = "#1A1A1A"
    MGRAY = "#BCC6D6"; LGRAY = "#EEF1F6"

    # ── title ────────────────────────────────────────────────────────────────
    ax.text(0.025, 0.965, "Phenotype  ·  age × sex × disease",
            ha="left", va="center", fontsize=5.6, color=INK,
            fontweight="bold", transform=T, zorder=2)

    # ── filter bar ───────────────────────────────────────────────────────────
    btn_y = 0.870; bh = 0.068
    filter_items = [
        # (x, w, label, facecolor, textcolor, edgecolor, fontsize)
        (0.025, 0.075, "age",        "white", INK,    MGRAY, 4.4),
        (0.108, 0.075, "sex",        RED,     "white", RED,   4.4),
        (0.191, 0.098, "disease",    "white", INK,    MGRAY, 4.4),
        (0.302, 0.125, "female  v",  "white", INK,    MGRAY, 4.0),
        (0.435, 0.036, "VS",         "white", "#666", "none", 4.0),
        (0.480, 0.115, "male  v",    "white", INK,    MGRAY, 4.0),
        (0.608, 0.085, "Genus",      RED,     "white", RED,   4.4),
        (0.703, 0.088, "Phylum",     "white", INK,    MGRAY, 4.4),
        (0.820, 0.160, "run analysis", NAVY,  "white", NAVY,  4.4),
    ]
    for bx, bw, lbl, fc, tc, ec, fs in filter_items:
        if ec != "none":
            ax.add_patch(FancyBboxPatch((bx, btn_y), bw, bh,
                boxstyle="round,pad=0.002,rounding_size=0.008",
                facecolor=fc, edgecolor=ec, lw=0.7,
                transform=T, zorder=3))
        ax.text(bx + bw/2, btn_y + bh/2, lbl, ha="center", va="center",
                fontsize=fs, color=tc, transform=T, zorder=4)

    # ── 4 stat boxes ─────────────────────────────────────────────────────────
    stat_y = 0.742; sh = 0.108; sw = 0.232; gap = 0.011
    stats = [
        ("female N", "35,860", RED),
        ("male N",   "38,778", NAVY),
        ("p<0.05",   "128",    NAVY),
        ("effect",   "118",    NAVY),
    ]
    for i, (lbl, num, col) in enumerate(stats):
        sx = 0.025 + i * (sw + gap)
        ax.add_patch(FancyBboxPatch((sx, stat_y), sw, sh,
            boxstyle="round,pad=0.003,rounding_size=0.006",
            facecolor="white", edgecolor=MGRAY, lw=0.6,
            transform=T, zorder=3))
        # thin colored top strip
        ax.add_patch(Rectangle((sx + 0.004, stat_y + sh - 0.007),
            sw - 0.008, 0.006, facecolor=col, edgecolor="none",
            transform=T, zorder=4))
        ax.text(sx + sw/2, stat_y + sh * 0.67, lbl,
                ha="center", va="center", fontsize=3.8, color="#666", transform=T)
        ax.text(sx + sw/2, stat_y + sh * 0.28, num,
                ha="center", va="center", fontsize=7.2, color=INK,
                fontweight="bold", transform=T)

    # ── chart-type toggle bar (Butterfly | Effect | Prev-Abund + Export) ─────
    tg_y = 0.665; tg_h = 0.044
    toggles = [("Butterfly", True), ("Effect", False), ("Prev-Abund", False)]
    tx = 0.025
    for lbl, sel in toggles:
        tw = 0.115
        ax.add_patch(FancyBboxPatch((tx, tg_y), tw, tg_h,
            boxstyle="round,pad=0.001,rounding_size=0.005",
            facecolor=RED if sel else "white",
            edgecolor=RED if sel else MGRAY, lw=0.5,
            transform=T, zorder=3))
        ax.text(tx + tw/2, tg_y + tg_h/2, lbl, ha="center", va="center",
                fontsize=3.4, color="white" if sel else "#555",
                fontweight="bold" if sel else "normal", transform=T, zorder=4)
        tx += tw + 0.012
    # export buttons (right side)
    ex_x = 0.975
    for lbl in ["PNG", "SVG", "CSV"]:
        ew = 0.046
        ex_x -= ew + 0.005
        ax.add_patch(FancyBboxPatch((ex_x, tg_y + 0.005), ew, tg_h - 0.010,
            boxstyle="round,pad=0.001,rounding_size=0.004",
            facecolor="white", edgecolor=MGRAY, lw=0.4,
            transform=T, zorder=3))
        ax.text(ex_x + ew/2, tg_y + tg_h/2, lbl, ha="center", va="center",
                fontsize=3.0, color="#555", transform=T, zorder=4)

    # ── 2x2 chart grid ───────────────────────────────────────────────────────
    area_x0, area_x1 = 0.025, 0.975
    area_y0, area_y1 = 0.035, 0.650
    cgap = 0.010
    col_split = area_x0 + (area_x1 - area_x0) * 0.50
    row_split = area_y0 + (area_y1 - area_y0) * 0.42

    # ========================================================================
    # Panel A (top-left): Butterfly chart (lollipop, 6 genera)
    # ========================================================================
    ax_x0, ax_y0 = area_x0, row_split + cgap
    ax_x1, ax_y1 = col_split - cgap, area_y1
    ax.add_patch(Rectangle((ax_x0, ax_y0), ax_x1 - ax_x0, ax_y1 - ax_y0,
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
    ax.text(ax_x0 + 0.006, ax_y1 - 0.018, "Butterfly",
            ha="left", va="center", fontsize=3.0, color=INK,
            fontweight="bold", transform=T, zorder=4)
    # direction indicators (below title, above plot)
    ax.text(ax_x0 + 0.095, ax_y1 - 0.032, "← female",
            ha="center", va="center", fontsize=2.0, color=NAVY, transform=T, zorder=4)
    ax.text(ax_x0 + 0.170, ax_y1 - 0.032, "male →",
            ha="center", va="center", fontsize=2.0, color=RED, transform=T, zorder=4)

    bgenera = ["Bacteroides", "Prevotella", "Faecalibact", "Bifidobact",
               "Akkermansia", "Parabacter"]
    bvals   = [ 0.85, -0.80, -0.55, 0.65, 0.50, -0.20]
    bsig    = ["***", "***", "***", "***", "***", "*"]
    blfc    = ["+0.23", "-0.31", "-0.20", "+0.28", "+0.33", "-0.09"]
    blabel_x = ax_x0 + 0.005
    bmid     = ax_x0 + 0.170
    brt      = ax_x1 - 0.040
    bbot     = ax_y0 + 0.036
    btop     = ax_y1 - 0.038
    n = len(bgenera)
    rh = (btop - bbot) / n
    bhalf_l = bmid - blabel_x - 0.112
    bhalf_r = brt - bmid
    # x-axis ticks (below chart)
    for frac, lbl in [(-1.0, "10%"), (-0.5, "5%"), (0.0, "0"),
                       (0.5, "5%"), (1.0, "10%")]:
        if frac < 0:
            xp = bmid + frac * bhalf_l
        else:
            xp = bmid + frac * bhalf_r
        ax.plot([xp, xp], [bbot - 0.005, bbot],
                color="#999", lw=0.3, transform=T, zorder=4)
        ax.text(xp, bbot - 0.007, lbl, ha="center", va="top",
                fontsize=1.7, color="#888", transform=T, zorder=4)
    ax.plot([bmid, bmid], [bbot, btop], color="#888", lw=0.4,
            transform=T, zorder=4)
    # log2FC column header (at top right, aligned with title row area)
    ax.text(brt + 0.008, ax_y1 - 0.018, "log2FC", ha="center", va="center",
            fontsize=1.8, color="#666", fontweight="bold",
            transform=T, zorder=4)
    for i, (g, v, sig, lfc) in enumerate(zip(bgenera, bvals, bsig, blfc)):
        cy = bbot + (n - 1 - i) * rh + rh * 0.5
        col = RED if v > 0 else NAVY
        end = bmid + v * (bhalf_r if v > 0 else bhalf_l)
        # thick bar
        ax.plot([bmid, end], [cy, cy], color=col, lw=3.4,
                solid_capstyle="round", transform=T, zorder=5)
        # big lollipop end dot
        ax.scatter([end], [cy], s=22, c=col, zorder=6, transform=T,
                   edgecolors="white", linewidths=0.4)
        ax.text(blabel_x + 0.085, cy, g, ha="right", va="center",
                fontsize=2.2, color=INK, style="italic", transform=T, zorder=6)
        # significance asterisks at bar end
        ax.text(end + (0.005 if v > 0 else -0.005), cy, sig,
                ha="left" if v > 0 else "right", va="center",
                fontsize=1.9, color=col, fontweight="bold",
                transform=T, zorder=7)
        # log2FC numeric column on right
        ax.text(brt + 0.008, cy, lfc, ha="center", va="center",
                fontsize=1.9, color="#444", transform=T, zorder=6)

    # ========================================================================
    # Panel B (top-right): Effect Plot — volcano (-log10(adj.p) vs log2FC)
    # ========================================================================
    bx0, by0 = col_split, row_split + cgap
    bx1, by1 = area_x1, area_y1
    ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
    ax.text(bx0 + 0.006, by1 - 0.018, "Effect Plot",
            ha="left", va="center", fontsize=3.0, color=INK,
            fontweight="bold", transform=T, zorder=4)
    # legend (top-right)
    ax.text(bx1 - 0.008, by1 - 0.018,
            "♀74 | ♂26", ha="right", va="center",
            fontsize=1.9, color="#666", transform=T, zorder=4)
    # plot region
    p_x0 = bx0 + 0.044
    p_x1 = bx1 - 0.010
    p_y0 = by0 + 0.030
    p_y1 = by1 - 0.040
    ax.plot([p_x0, p_x0, p_x1], [p_y1, p_y0, p_y0],
            color="#999", lw=0.4, transform=T, zorder=4)
    # y-axis ticks + labels
    for frac, lbl in [(0.0, "0"), (0.33, "40"), (0.67, "80"), (1.0, "120")]:
        ty = p_y0 + frac * (p_y1 - p_y0)
        ax.plot([p_x0 - 0.003, p_x0], [ty, ty],
                color="#999", lw=0.3, transform=T, zorder=4)
        ax.text(p_x0 - 0.005, ty, lbl, ha="right", va="center",
                fontsize=1.8, color="#888", transform=T, zorder=4)
    # x-axis ticks + labels
    for frac, lbl in [(0.0, "-0.6"), (0.25, "-0.2"), (0.5, "0"),
                       (0.75, "+0.4"), (1.0, "+0.8")]:
        tx2 = p_x0 + frac * (p_x1 - p_x0)
        ax.plot([tx2, tx2], [p_y0, p_y0 - 0.003],
                color="#999", lw=0.3, transform=T, zorder=4)
        ax.text(tx2, p_y0 - 0.005, lbl, ha="center", va="top",
                fontsize=1.7, color="#888", transform=T, zorder=4)
    ax.text((p_x0 + p_x1)/2, p_y0 - 0.018, "log2FC",
            ha="center", va="top", fontsize=2.0, color="#666",
            transform=T, zorder=4)
    # y axis title (vertical)
    ax.text(p_x0 - 0.022, (p_y0 + p_y1)/2, "−log10(adj.p)",
            ha="center", va="center", fontsize=1.8, color="#666",
            rotation=90, transform=T, zorder=4)
    # zero line
    zx = p_x0 + (p_x1 - p_x0) * 0.5
    ax.plot([zx, zx], [p_y0, p_y1], color="#BBB", lw=0.35,
            linestyle=(0, (1.5, 1.5)), transform=T, zorder=4)
    # significance line
    sigy = p_y0 + (p_y1 - p_y0) * 0.05
    ax.plot([p_x0, p_x1], [sigy, sigy], color="#BBB", lw=0.35,
            linestyle=(0, (1.5, 1.5)), transform=T, zorder=4)
    ax.text(p_x1 - 0.003, sigy + 0.003, "adj.p=0.05",
            ha="right", va="bottom", fontsize=1.6, color="#888",
            transform=T, zorder=4)
    # lollipop volcano: vertical stem from y=0 to point, ♀=filled / ♂=open
    GREEN = "#2BB573"; SKY = "#3CA0E0"; YELLOW = "#E8B83A"
    PURPLE = "#B574D9"; GRAY = "#8C9BAA"
    # (px, py, color, sex 'F'/'M', label)
    pts = [
        (0.55, 0.92, GREEN,  "F", "Bacteroides"),
        (0.72, 0.80, GREEN,  "F", "Alistipes"),
        (0.38, 0.68, SKY,    "F", "Blautia"),
        (0.80, 0.62, PURPLE, "F", "Akkermansia"),
        (0.55, 0.69, GRAY,   "F", None),
        (0.62, 0.66, SKY,    "F", None),
        (0.50, 0.55, RED,    "F", None),
        (0.46, 0.42, SKY,    "M", None),
        (0.48, 0.48, SKY,    "F", None),
        (0.58, 0.55, SKY,    "F", None),
        (0.42, 0.34, GREEN,  "M", None),
        (0.45, 0.30, SKY,    "M", None),
        (0.65, 0.45, SKY,    "F", None),
        (0.70, 0.32, SKY,    "F", None),
        (0.74, 0.45, SKY,    "F", None),
        (0.30, 0.32, YELLOW, "M", None),
        (0.22, 0.28, GREEN,  "M", None),
        (0.85, 0.45, YELLOW, "F", None),
        (0.92, 0.48, YELLOW, "F", None),
        (0.18, 0.18, YELLOW, "M", None),
        (0.10, 0.10, SKY,    "M", None),
        (0.78, 0.20, SKY,    "F", None),
        (0.40, 0.55, RED,    "F", None),
        (0.34, 0.18, GREEN,  "M", None),
        (0.52, 0.28, SKY,    "F", None),
        (0.36, 0.40, SKY,    "M", None),
    ]
    for px, py, col, sex, lbl in pts:
        x = p_x0 + px * (p_x1 - p_x0)
        y = p_y0 + py * (p_y1 - p_y0)
        # stem from x-axis to point
        ax.plot([x, x], [p_y0, y], color=col, lw=0.8, alpha=0.7,
                transform=T, zorder=4)
        if sex == "F":
            ax.scatter([x], [y], s=24, c=col, edgecolors="white",
                       linewidths=0.4, transform=T, zorder=6)
        else:
            ax.scatter([x], [y], s=24, facecolors="white",
                       edgecolors=col, linewidths=0.9,
                       transform=T, zorder=6)
        if lbl:
            ax.text(x + 0.004, y + 0.008, lbl, ha="left", va="bottom",
                    fontsize=2.0, color=INK, style="italic",
                    transform=T, zorder=7)

    # ========================================================================
    # Panel C (bottom-left): Prevalence-Abundance scatter
    # ========================================================================
    cx0_, cy0_ = area_x0, area_y0
    cx1_, cy1_ = col_split - cgap, row_split - cgap
    ax.add_patch(Rectangle((cx0_, cy0_), cx1_ - cx0_, cy1_ - cy0_,
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
    ax.text(cx0_ + 0.006, cy1_ - 0.016, "Prev-Abund",
            ha="left", va="center", fontsize=2.8, color=INK,
            fontweight="bold", transform=T, zorder=4)
    # legend
    ax.text(cx1_ - 0.008, cy1_ - 0.016, "● female  ○ male",
            ha="right", va="center", fontsize=1.7, color="#666",
            transform=T, zorder=4)
    pp_x0 = cx0_ + 0.050
    pp_x1 = cx1_ - 0.008
    pp_y0 = cy0_ + 0.028
    pp_y1 = cy1_ - 0.030
    ax.plot([pp_x0, pp_x0, pp_x1], [pp_y1, pp_y0, pp_y0],
            color="#999", lw=0.4, transform=T, zorder=4)
    # gridlines
    for frac in (0.25, 0.5, 0.75):
        gx = pp_x0 + frac * (pp_x1 - pp_x0)
        gy = pp_y0 + frac * (pp_y1 - pp_y0)
        ax.plot([gx, gx], [pp_y0, pp_y1], color="#EEE", lw=0.25,
                transform=T, zorder=4)
        ax.plot([pp_x0, pp_x1], [gy, gy], color="#EEE", lw=0.25,
                transform=T, zorder=4)
    # y ticks
    for frac, lbl in [(1.0, "20%"), (0.66, "2%"), (0.33, "0.2%"), (0.0, "0.05%")]:
        ty_ = pp_y0 + frac * (pp_y1 - pp_y0)
        ax.text(pp_x0 - 0.004, ty_, lbl, ha="right", va="center",
                fontsize=1.7, color="#888", transform=T, zorder=4)
    # x ticks
    for frac, lbl in [(0.0, "0%"), (0.33, "33%"), (0.66, "66%"), (1.0, "100%")]:
        tx_ = pp_x0 + frac * (pp_x1 - pp_x0)
        ax.text(tx_, pp_y0 - 0.005, lbl, ha="center", va="top",
                fontsize=1.7, color="#888", transform=T, zorder=4)
    ax.text((pp_x0 + pp_x1)/2, pp_y0 - 0.008, "Prevalence",
            ha="center", va="top", fontsize=1.8, color="#666",
            transform=T, zorder=4)
    ax.text(pp_x0 - 0.027, (pp_y0 + pp_y1)/2, "Abundance (log)",
            ha="center", va="center", fontsize=1.7, color="#666",
            rotation=90, transform=T, zorder=4)
    # paired points: (female_px, female_py, male_px, male_py, color)
    GREEN = "#2BB573"; SKY = "#3CA0E0"; YELLOW = "#E8B83A"
    pa_pairs = [
        (0.95, 0.95, 0.92, 0.91, GREEN),
        (0.85, 0.78, 0.80, 0.74, SKY),
        (0.78, 0.72, 0.74, 0.68, GREEN),
        (0.70, 0.65, 0.66, 0.60, RED),
        (0.65, 0.55, 0.62, 0.51, SKY),
        (0.60, 0.62, 0.55, 0.58, GREEN),
        (0.55, 0.45, 0.51, 0.41, SKY),
        (0.48, 0.48, 0.43, 0.43, SKY),
        (0.45, 0.40, 0.40, 0.36, YELLOW),
        (0.40, 0.35, 0.36, 0.31, SKY),
        (0.35, 0.30, 0.30, 0.26, GREEN),
        (0.30, 0.32, 0.26, 0.28, SKY),
        (0.25, 0.22, 0.21, 0.18, YELLOW),
        (0.20, 0.18, 0.16, 0.14, GREEN),
        (0.15, 0.12, 0.11, 0.09, SKY),
        (0.10, 0.08, 0.07, 0.05, SKY),
    ]
    for fx, fy, mx, my, col in pa_pairs:
        # line connecting the pair (same taxon)
        x_f = pp_x0 + fx * (pp_x1 - pp_x0)
        y_f = pp_y0 + fy * (pp_y1 - pp_y0)
        x_m = pp_x0 + mx * (pp_x1 - pp_x0)
        y_m = pp_y0 + my * (pp_y1 - pp_y0)
        ax.plot([x_f, x_m], [y_f, y_m], color=col, lw=0.45,
                alpha=0.6, transform=T, zorder=4)
        # female = filled
        ax.scatter([x_f], [y_f], s=14, c=col, edgecolors="white",
                   linewidths=0.3, transform=T, zorder=5)
        # male = open
        ax.scatter([x_m], [y_m], s=14, facecolors="white",
                   edgecolors=col, linewidths=0.7,
                   transform=T, zorder=5)

    # ========================================================================
    # Panel D (bottom-right): mini Full Results Table
    # ========================================================================
    tx0, ty0 = col_split, area_y0
    tx1, ty1 = area_x1, row_split - cgap
    ax.add_patch(Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0,
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
    ax.text(tx0 + 0.006, ty1 - 0.016, "Full Results Table",
            ha="left", va="center", fontsize=2.8, color=INK,
            fontweight="bold", transform=T, zorder=4)
    ax.text(tx1 - 0.008, ty1 - 0.016, "100 results",
            ha="right", va="center", fontsize=1.8, color="#888",
            transform=T, zorder=4)
    cols = ["Genus", "Phylum", "F%", "M%", "log2FC", "adj.p", "Sig", "Enr."]
    col_fracs = [0.20, 0.18, 0.10, 0.10, 0.13, 0.11, 0.08, 0.10]
    table_x0 = tx0 + 0.006
    table_x1 = tx1 - 0.006
    col_x = [table_x0]
    for cf in col_fracs[:-1]:
        col_x.append(col_x[-1] + cf * (table_x1 - table_x0))
    hdr_y = ty1 - 0.034
    ax.plot([table_x0, table_x1], [hdr_y - 0.006, hdr_y - 0.006],
            color=MGRAY, lw=0.4, transform=T, zorder=4)
    for cxi, c in zip(col_x, cols):
        ax.text(cxi, hdr_y, c, ha="left", va="center",
                fontsize=1.9, color="#666", fontweight="bold",
                transform=T, zorder=5)
    rows = [
        ("Bacteroides",  "Bacterio.", "12.96", "11.02", "+0.23", "<.001", "***", "♀"),
        ("Alistipes",    "Bacterio.", "1.48",  "1.20",  "+0.31", "<.001", "***", "♀"),
        ("Blautia",      "Bacillota", "3.53",  "3.08",  "+0.20", "<.001", "***", "♀"),
        ("Akkermansia",  "Verruco.",  "1.36",  "1.11",  "+0.28", "<.001", "***", "♀"),
        ("Bilophila",    "Thermode.", "0.12",  "0.10",  "+0.23", "<.001", "***", "♀"),
        ("Anaerobut.",   "Bacillota", "0.54",  "0.42",  "+0.37", "<.001", "***", "♀"),
        ("Ruminococ.",   "Bacillota", "1.11",  "0.88",  "+0.35", "<.001", "***", "♀"),
        ("Acutalibact.", "Bacillota", "0.26",  "0.20",  "+0.33", "<.001", "***", "♀"),
        ("Monoglobus",   "Bacillota", "0.17",  "0.14",  "+0.26", "<.001", "***", "♀"),
    ]
    row_y0 = hdr_y - 0.018
    row_step = (row_y0 - ty0 - 0.008) / len(rows)
    for ri, row in enumerate(rows):
        ry = row_y0 - (ri + 0.5) * row_step
        for cxi, val, ci in zip(col_x, row, range(8)):
            if ci == 0:
                col = INK; sty = "italic"
            elif ci == 1:
                col = "#888"; sty = "normal"
            elif ci == 4:
                col = SKY; sty = "normal"
            elif ci == 6:
                col = RED; sty = "normal"
            elif ci == 7:
                col = RED; sty = "normal"
            else:
                col = "#444"; sty = "normal"
            ax.text(cxi, ry, val, ha="left", va="center",
                    fontsize=1.8, color=col, style=sty,
                    transform=T, zorder=5)

def _thumb_disease(ax):
    """Disease Browser — left sidebar + right main panel with Crohn's Disease selected."""
    _ui_bg(ax, "white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    T = ax.transAxes
    NAVY = "#1F3864"; RED = "#C41435"; INK = "#1A1A1A"
    MGRAY = "#BCC6D6"; LGRAY = "#EEF1F6"
    TEAL = "#1A6B82"

    # ── title ────────────────────────────────────────────────────────────────
    ax.text(0.025, 0.965, "Disease Browser",
            ha="left", va="center", fontsize=5.6, color=INK,
            fontweight="bold", transform=T, zorder=2)
    ax.text(0.025, 0.925, "Select a disease to view microbiome signature",
            ha="left", va="center", fontsize=3.4, color="#666", transform=T, zorder=2)

    # ── LEFT SIDEBAR ─────────────────────────────────────────────────────────
    lp_x, lp_w = 0.008, 0.285
    lp_y0, lp_y1 = 0.035, 0.900
    ax.add_patch(Rectangle((lp_x, lp_y0), lp_w, lp_y1 - lp_y0,
        facecolor="white", edgecolor=MGRAY, lw=0.6, transform=T, zorder=2))

    # search box
    ax.add_patch(FancyBboxPatch((lp_x + 0.008, 0.855), lp_w - 0.016, 0.036,
        boxstyle="round,pad=0.002,rounding_size=0.004",
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
    ax.text(lp_x + 0.014, 0.873, "Search diseases...", ha="left", va="center",
            fontsize=3.1, color="#999", transform=T)

    # category dropdown
    ax.add_patch(FancyBboxPatch((lp_x + 0.008, 0.810), lp_w - 0.016, 0.036,
        boxstyle="round,pad=0.002,rounding_size=0.004",
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
    ax.text(lp_x + 0.014, 0.828, "Disease Category  v", ha="left", va="center",
            fontsize=3.1, color="#555", transform=T)

    # sort-by label + toggle
    ax.text(lp_x + 0.010, 0.786, "SORT BY", ha="left", va="center",
            fontsize=2.7, color="#888", fontweight="bold", transform=T)
    sort_btns = [("Sample Count", RED, "white", True),
                 ("Alphabetical", "white", "#666", False)]
    bx = lp_x + 0.008
    for lbl, fc, tc, sel in sort_btns:
        bw = 0.135
        ax.add_patch(FancyBboxPatch((bx, 0.745), bw, 0.032,
            boxstyle="round,pad=0.001,rounding_size=0.003",
            facecolor=fc, edgecolor=RED if sel else MGRAY,
            lw=0.5, transform=T, zorder=3))
        ax.text(bx + bw/2, 0.761, lbl, ha="center", va="center",
                fontsize=2.9, color=tc, transform=T)
        bx += bw + 0.003

    # disease list
    dis_items = [
        ("Non-disease Control",       "82,106", False),
        ("Hematopoietic SCT",         "8,341",  False),
        ("Hematologic malignancy",    "4,711",  False),
        ("Preterm infant",            "3,111",  False),
        ("Chickenpox",                "3,051",  False),
        ("HFR of Crohn's Pts",        "2,671",  False),
        ("Ulcerative Colitis",        "2,398",  False),
        ("Crohn's Disease",           "2,173",  True),   # ← SELECTED
        ("C. difficile Infection",    "1,912",  False),
        ("Obesity",                   "1,680",  False),
        ("HIV",                       "1,498",  False),
        ("Colorectal Cancer",         "1,245",  False),
        ("Type 2 Diabetes",           "1,102",  False),
        ("Parkinson's Disease",       "987",    False),
        ("Autism Spectrum",           "912",    False),
        ("Liver Cirrhosis",           "864",    False),
        ("NAFLD",                     "798",    False),
        ("IBS",                       "742",    False),
        ("Rheumatoid Arthritis",      "689",    False),
        ("Asthma",                    "631",    False),
        ("Atopic Dermatitis",         "578",    False),
        ("Multiple Sclerosis",        "524",    False),
        ("Alzheimer's Disease",       "487",    False),
        ("Type 1 Diabetes",           "452",    False),
        ("Gastric Cancer",            "418",    False),
        ("Pancreatic Cancer",         "385",    False),
        ("Depression",                "342",    False),
        ("Psoriasis",                 "298",    False),
        ("Celiac Disease",            "271",    False),
        ("Lupus",                     "246",    False),
        ("Ankylosing Spondylitis",    "218",    False),
        ("Bipolar Disorder",          "189",    False),
        ("Schizophrenia",             "164",    False),
        ("Eczema",                    "142",    False),
    ]
    list_y0 = 0.720
    row_h = 0.0198
    for di, (dname, dcount, sel) in enumerate(dis_items):
        iy = list_y0 - (di + 1) * row_h
        if sel:
            ax.add_patch(Rectangle((lp_x + 0.004, iy), lp_w - 0.008, row_h - 0.003,
                facecolor="#FCE8EC", edgecolor=RED, lw=0.6,
                transform=T, zorder=3))
        ax.text(lp_x + 0.012, iy + row_h * 0.55, dname, ha="left", va="center",
                fontsize=2.5, color=RED if sel else INK,
                fontweight="bold" if sel else "normal", transform=T, zorder=4)
        ax.text(lp_x + lp_w - 0.010, iy + row_h * 0.55, dcount, ha="right", va="center",
                fontsize=2.3, color="#777", transform=T, zorder=4)

    # ── RIGHT MAIN PANEL ─────────────────────────────────────────────────────
    rp_x = lp_x + lp_w + 0.012
    rp_w = 0.978 - rp_x
    rp_y0, rp_y1 = 0.035, 0.900
    ax.add_patch(Rectangle((rp_x, rp_y0), rp_w, rp_y1 - rp_y0,
        facecolor="white", edgecolor=MGRAY, lw=0.6, transform=T, zorder=2))

    # tabs (wider buttons + bigger gaps)
    tabs = [("Profile", True), ("Biomarkers", False),
            ("Diff Abund.", False), ("Studies", False)]
    tab_x = rp_x + 0.012
    tw = 0.105
    tgap = 0.022
    for lbl, sel in tabs:
        ax.add_patch(FancyBboxPatch((tab_x, 0.850), tw, 0.042,
            boxstyle="round,pad=0.001,rounding_size=0.016",
            facecolor=RED if sel else "white",
            edgecolor=RED if sel else MGRAY,
            lw=0.5, transform=T, zorder=3))
        ax.text(tab_x + tw/2, 0.871, lbl, ha="center", va="center",
                fontsize=3.4, color="white" if sel else "#555",
                fontweight="bold" if sel else "normal", transform=T, zorder=4)
        tab_x += tw + tgap

    # 4 stat boxes
    stat_y = 0.755; sh = 0.080
    stats = [
        ("samples",   "2,173",  RED),
        ("controls",  "82,341", NAVY),
        ("studies",   "15",     NAVY),
        ("top genera","40",     NAVY),
    ]
    stat_gap = 0.018
    sw = (rp_w - 0.008 - 3 * stat_gap) / 4
    for i, (lbl, num, col) in enumerate(stats):
        sx = rp_x + 0.010 + i * (sw + stat_gap)
        ax.add_patch(FancyBboxPatch((sx, stat_y), sw, sh,
            boxstyle="round,pad=0.002,rounding_size=0.005",
            facecolor="white", edgecolor=MGRAY, lw=0.5,
            transform=T, zorder=3))
        ax.add_patch(Rectangle((sx + 0.004, stat_y + sh - 0.006),
            sw - 0.008, 0.005, facecolor=col, edgecolor="none",
            transform=T, zorder=4))
        ax.text(sx + sw/2, stat_y + sh * 0.70, lbl,
                ha="center", va="center", fontsize=3.4, color="#666", transform=T)
        ax.text(sx + sw/2, stat_y + sh * 0.24, num,
                ha="center", va="center", fontsize=6.8, color=INK,
                fontweight="bold", transform=T)

    # metadata tag row
    tag_y = 0.712
    tags = [("Gastrointestinal", LGRAY, INK),
            ("Std: Crohn's Disease", LGRAY, INK),
            ("MeSH: D003424", LGRAY, INK),
            ("ICD-10: K50", LGRAY, INK)]
    tx = rp_x + 0.010
    for lbl, fc, tc in tags:
        tw = 0.006 + len(lbl) * 0.0105
        ax.add_patch(FancyBboxPatch((tx, tag_y), tw, 0.028,
            boxstyle="round,pad=0.001,rounding_size=0.003",
            facecolor=fc, edgecolor=MGRAY, lw=0.4, transform=T, zorder=3))
        ax.text(tx + tw/2, tag_y + 0.014, lbl, ha="center", va="center",
                fontsize=2.7, color=tc, transform=T, zorder=4)
        tx += tw + 0.008

    # ── chart area split into 3 subpanels ────────────────────────────────────
    area_x0 = rp_x + 0.010
    area_x1 = rp_x + rp_w - 0.010
    area_y0 = rp_y0 + 0.012
    area_y1 = 0.685
    gap = 0.008
    split_x = area_x0 + (area_x1 - area_x0) * 0.46   # left 46% / right 54%
    split_y = area_y0 + (area_y1 - area_y0) * 0.48   # right panel vsplit

    # ========================================================================
    # Panel A (left): Abundance vs Prevalence scatter
    # ========================================================================
    ax_x0, ax_y0 = area_x0, area_y0
    ax_x1, ax_y1 = split_x - gap, area_y1
    ax.add_patch(Rectangle((ax_x0, ax_y0), ax_x1 - ax_x0, ax_y1 - ax_y0,
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
    ax.text((ax_x0 + ax_x1) / 2, ax_y1 - 0.018, "Abundance vs Prevalence",
            ha="center", va="center", fontsize=3.2, color=INK,
            fontweight="bold", transform=T, zorder=4)
    # plot region
    p_x0 = ax_x0 + 0.028
    p_x1 = ax_x1 - 0.010
    p_y0 = ax_y0 + 0.025
    p_y1 = ax_y1 - 0.035
    # axis frame
    ax.plot([p_x0, p_x0, p_x1], [p_y1, p_y0, p_y0],
            color="#999", lw=0.4, transform=T, zorder=4)
    # gridlines
    for frac in (0.25, 0.5, 0.75):
        gx = p_x0 + (p_x1 - p_x0) * frac
        gy = p_y0 + (p_y1 - p_y0) * frac
        ax.plot([gx, gx], [p_y0, p_y1], color="#EEE", lw=0.3,
                transform=T, zorder=3)
        ax.plot([p_x0, p_x1], [gy, gy], color="#EEE", lw=0.3,
                transform=T, zorder=3)
    # axis labels
    ax.text(p_x0 - 0.006, p_y1, "23.07%", ha="right", va="center",
            fontsize=2.3, color="#666", transform=T, zorder=4)
    ax.text(p_x0 - 0.006, p_y0 + (p_y1 - p_y0) * 0.55, "2.77%",
            ha="right", va="center", fontsize=2.3, color="#666", transform=T, zorder=4)
    ax.text(p_x0 - 0.006, p_y0, "0.33%", ha="right", va="center",
            fontsize=2.3, color="#666", transform=T, zorder=4)
    ax.text(p_x0, p_y0 - 0.012, "0%", ha="center", va="top",
            fontsize=2.3, color="#666", transform=T, zorder=4)
    ax.text(p_x1, p_y0 - 0.012, "100%", ha="center", va="top",
            fontsize=2.3, color="#666", transform=T, zorder=4)
    # scatter points with fixed positions matching the reference
    pts = [
        # (prev_frac, abund_frac, color, label)
        (0.95, 0.93, NAVY,  "Bacteroides"),
        (0.62, 0.82, NAVY,  "Shigella"),
        (0.85, 0.58, CORAL, "Faecalibact."),
        (0.45, 0.72, CORAL, "Blautia"),
        (0.28, 0.62, TEAL,  "Akkermansia"),
        (0.12, 0.42, NAVY,  "Segatella"),
        (0.72, 0.40, CORAL, "Bifidobact."),
        (0.38, 0.50, RED,   None),
        (0.48, 0.40, NAVY,  None), (0.52, 0.45, NAVY, None),
        (0.55, 0.38, CORAL, None), (0.60, 0.35, CORAL, None),
        (0.30, 0.30, TEAL,  None), (0.42, 0.28, CORAL, None),
        (0.20, 0.22, CORAL, None), (0.15, 0.12, CORAL, None),
        (0.05, 0.08, NAVY,  None), (0.18, 0.10, CORAL, None),
        (0.65, 0.42, CORAL, None), (0.58, 0.30, CORAL, None),
    ]
    left_labels = {"Akkermansia", "Segatella"}
    for pv, ab, col, lbl in pts:
        px = p_x0 + pv * (p_x1 - p_x0)
        py = p_y0 + ab * (p_y1 - p_y0)
        ax.scatter([px], [py], s=12, c=col, edgecolors="none",
                   transform=T, zorder=5)
        if lbl:
            if lbl in left_labels:
                ax.text(px + 0.004, py + 0.010, lbl, ha="left", va="bottom",
                        fontsize=2.2, color=INK, transform=T, zorder=6)
            else:
                ax.text(px - 0.004, py + 0.010, lbl, ha="right", va="bottom",
                        fontsize=2.2, color=INK, transform=T, zorder=6)

    # ========================================================================
    # Panel B (right-top): Top Genera bar chart (6 genera, reduced)
    # ========================================================================
    bx0, by0 = split_x, split_y + gap
    bx1, by1 = area_x1, area_y1
    ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
    ax.text(bx0 + 0.006, by1 - 0.018, "Top Genera",
            ha="left", va="center", fontsize=3.0, color=INK,
            fontweight="bold", transform=T, zorder=4)
    # export buttons (mini)
    ex_x = bx1 - 0.005
    for lbl in ["PNG", "SVG", "CSV"]:
        ew = 0.028
        ex_x -= ew + 0.003
        ax.add_patch(FancyBboxPatch((ex_x, by1 - 0.028), ew, 0.008,
            boxstyle="round,pad=0.001,rounding_size=0.002",
            facecolor="white", edgecolor=MGRAY, lw=0.4, transform=T, zorder=4))
        ax.text(ex_x + ew/2, by1 - 0.018, lbl, ha="center", va="center",
                fontsize=2.2, color="#555", transform=T, zorder=5)

    # bars — 6 genera only
    genera6 = ["Bacteroides", "Faecalibact.", "Escherichia",
               "Prevotella",  "Ruminococcus", "Blautia"]
    dis_vals = [0.92, 0.35, 0.88, 0.72, 0.40, 0.30]
    ctl_vals = [0.78, 0.85, 0.20, 0.30, 0.70, 0.58]
    br_x0 = bx0 + 0.108
    br_x1 = bx1 - 0.008
    br_y0 = by0 + 0.008
    br_y1 = by1 - 0.032
    n6 = 6
    g_h = (br_y1 - br_y0) / n6
    b_h = g_h * 0.36
    for i, (g, dv, cv) in enumerate(zip(genera6, dis_vals, ctl_vals)):
        gy = br_y0 + (n6 - 1 - i) * g_h + g_h * 0.5
        dy = gy + b_h * 0.55
        cy = gy - b_h * 0.55
        ax.add_patch(Rectangle((br_x0, dy - b_h / 2), dv * (br_x1 - br_x0),
            b_h, facecolor=RED, edgecolor="none", transform=T, zorder=4))
        ax.add_patch(Rectangle((br_x0, cy - b_h / 2), cv * (br_x1 - br_x0),
            b_h, facecolor=TEAL, edgecolor="none", transform=T, zorder=4))
        ax.text(br_x0 - 0.004, gy, g, ha="right", va="center",
                fontsize=2.4, color=INK, style="italic", transform=T, zorder=5)

    # ========================================================================
    # Panel C (right-bottom): Demographics — By Sex | By Age | By Country
    # ========================================================================
    tx0, ty0 = split_x, area_y0
    tx1, ty1 = area_x1, split_y - gap
    ax.add_patch(Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0,
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))

    ORANGE = "#F59A2E"   # demographics accent
    # three sub-sections split horizontally
    sub_w = (tx1 - tx0 - 0.012) / 3.0
    sub_gap = 0.004

    # -- By Sex -----------------------------------------------------------
    sx0 = tx0 + 0.006
    sx1 = sx0 + sub_w - sub_gap
    ax.text((sx0 + sx1) / 2, ty1 - 0.012, "By Sex",
            ha="center", va="center", fontsize=2.6, color=INK,
            fontweight="bold", transform=T, zorder=4)
    sex_rows = [("unknown", 1060, 1060),
                ("female",   622, 1060),
                ("male",     491, 1060)]
    row_top = ty1 - 0.025
    row_bot = ty0 + 0.008
    row_h = (row_top - row_bot) / 3
    for i, (lbl, val, mx) in enumerate(sex_rows):
        y = row_top - (i + 0.5) * row_h
        lab_x = sx0 + 0.014
        bar_x0 = sx0 + 0.036
        bar_x1 = sx1 - 0.018
        ax.text(sx0 + 0.002, y, lbl, ha="left", va="center",
                fontsize=2.1, color="#666", transform=T, zorder=5)
        # background track
        ax.add_patch(FancyBboxPatch((bar_x0, y - 0.006),
            bar_x1 - bar_x0, 0.012,
            boxstyle="round,pad=0,rounding_size=0.006",
            facecolor="#EEF1F6", edgecolor="none",
            transform=T, zorder=4))
        # filled portion
        fill_w = (val / mx) * (bar_x1 - bar_x0)
        ax.add_patch(FancyBboxPatch((bar_x0, y - 0.006), fill_w, 0.012,
            boxstyle="round,pad=0,rounding_size=0.006",
            facecolor=ORANGE, edgecolor="none",
            transform=T, zorder=5))
        ax.text(sx1 - 0.002, y, f"{val:,}", ha="right", va="center",
                fontsize=2.1, color="#888", transform=T, zorder=6)

    # -- By Age Group -----------------------------------------------------
    ax0 = sx1 + sub_gap
    ax1 = ax0 + sub_w - sub_gap
    ax.text((ax0 + ax1) / 2, ty1 - 0.012, "By Age",
            ha="center", va="center", fontsize=2.6, color=INK,
            fontweight="bold", transform=T, zorder=4)
    # all 6 age groups preserved, abbreviated to fit
    ages = [("Adt", 1322), ("Chd", 369), ("Unk", 307),
            ("Old", 109), ("Ado", 57), ("Oldest", 9)]
    amax = 1322
    col_w = (ax1 - ax0 - 0.010) / len(ages)
    base_y = ty0 + 0.008
    top_y  = ty1 - 0.032
    for i, (lbl, v) in enumerate(ages):
        cx = ax0 + 0.005 + (i + 0.5) * col_w
        h = (v / amax) * (top_y - base_y)
        ax.add_patch(Rectangle((cx - col_w * 0.32, base_y),
            col_w * 0.64, h, facecolor=ORANGE, edgecolor="none",
            transform=T, zorder=5))
        ax.text(cx, base_y + h + 0.005, str(v), ha="center", va="bottom",
                fontsize=2.0, color="#888", transform=T, zorder=6)
        ax.text(cx - 0.002, base_y - 0.004, lbl, ha="right", va="top",
                fontsize=2.0, color="#666", rotation=40,
                rotation_mode="anchor", transform=T, zorder=6)

    # -- By Country -------------------------------------------------------
    cx0 = ax1 + sub_gap
    cx1 = tx1 - 0.006
    ax.text((cx0 + cx1) / 2, ty1 - 0.012, "By Country",
            ha="center", va="center", fontsize=2.6, color=INK,
            fontweight="bold", transform=T, zorder=4)
    countries = [("USA",     815, 815),
                 ("Unknown", 700, 815),
                 ("Israel",  234, 815),
                 ("China",   186, 815),
                 ("Ireland", 109, 815),
                 ("Japan",    16, 815)]
    row_top_c = ty1 - 0.025
    row_bot_c = ty0 + 0.008
    row_h_c = (row_top_c - row_bot_c) / len(countries)
    for i, (lbl, val, mx) in enumerate(countries):
        y = row_top_c - (i + 0.5) * row_h_c
        bar_x0 = cx0 + 0.028
        bar_x1 = cx1 - 0.018
        ax.text(cx0 - 0.004, y, lbl, ha="left", va="center",
                fontsize=1.9, color="#666", transform=T, zorder=5)
        ax.add_patch(FancyBboxPatch((bar_x0, y - 0.005),
            bar_x1 - bar_x0, 0.010,
            boxstyle="round,pad=0,rounding_size=0.005",
            facecolor="#EEF1F6", edgecolor="none",
            transform=T, zorder=4))
        fill_w = (val / mx) * (bar_x1 - bar_x0)
        ax.add_patch(FancyBboxPatch((bar_x0, y - 0.005), fill_w, 0.010,
            boxstyle="round,pad=0,rounding_size=0.005",
            facecolor=ORANGE, edgecolor="none",
            transform=T, zorder=5))
        ax.text(cx1 - 0.002, y, str(val), ha="right", va="center",
                fontsize=1.9, color="#888", transform=T, zorder=6)

def _thumb_search(ax):
    """Genus Search — search + 4 stats + Disease/Country/Age/Sex 2x2 grid."""
    _ui_bg(ax, "white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    T = ax.transAxes
    NAVY = "#1F3864"; RED = "#C41435"; INK = "#1A1A1A"
    MGRAY = "#BCC6D6"; LGRAY = "#EEF1F6"
    TEAL = "#1A6B82"; MAGENTA = "#C735C7"

    # ── title ───────────────────────────────────────────────────────────────
    ax.text(0.025, 0.965, "Genus Search", ha="left", va="center",
            fontsize=5.6, color=INK, fontweight="bold", transform=T)

    # ── search bar (selected genus chip + Search CTA) ───────────────────────
    bh = 0.068
    btn_y = 0.870
    # search input box
    ax.add_patch(FancyBboxPatch((0.025, btn_y), 0.720, bh,
        boxstyle="round,pad=0.002,rounding_size=0.008",
        facecolor="white", edgecolor=MGRAY, lw=0.6, transform=T, zorder=3))
    ax.text(0.040, btn_y + bh/2, "Bacteroides", ha="left", va="center",
            fontsize=4.4, color=INK, transform=T, zorder=4)
    # CTA "Search"
    ax.add_patch(FancyBboxPatch((0.760, btn_y), 0.215, bh,
        boxstyle="round,pad=0.002,rounding_size=0.008",
        facecolor=NAVY, edgecolor=NAVY, lw=0.6, transform=T, zorder=3))
    ax.text(0.868, btn_y + bh/2, "Search", ha="center", va="center",
            fontsize=4.4, color="white", fontweight="bold",
            transform=T, zorder=4)

    # ── quick select row ────────────────────────────────────────────────────
    ax.text(0.025, 0.795, "Quick select:", ha="left", va="center",
            fontsize=3.8, color="#666", transform=T)
    ax.add_patch(FancyBboxPatch((0.180, 0.780), 0.300, 0.050,
        boxstyle="round,pad=0.002,rounding_size=0.006",
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
    ax.text(0.330, 0.805, "-- Common genera --  v", ha="center", va="center",
            fontsize=3.6, color="#555", transform=T, zorder=4)

    # ── genus header "Bacteroides →" italic magenta ─────────────────────────
    ax.text(0.500, 0.730, "Bacteroides →", ha="center", va="center",
            fontsize=5.0, color=MAGENTA, style="italic",
            fontweight="bold", transform=T)

    # ── 4 stat boxes ────────────────────────────────────────────────────────
    stat_y = 0.608
    sh = 0.100
    stats = [
        ("TOTAL SAMPLES", "168,464", NAVY),
        ("PRESENT IN",    "124,933", NAVY),
        ("PREVALENCE",    "74.2%",   NAVY),
        ("MEAN ABUNDANCE","12.8%",   NAVY),
    ]
    stat_gap = 0.012
    sw = (0.950 - 3 * stat_gap) / 4
    for i, (lbl, num, col) in enumerate(stats):
        sx = 0.025 + i * (sw + stat_gap)
        ax.add_patch(FancyBboxPatch((sx, stat_y), sw, sh,
            boxstyle="round,pad=0.002,rounding_size=0.005",
            facecolor="white", edgecolor=MGRAY, lw=0.6, transform=T, zorder=3))
        ax.add_patch(Rectangle((sx + 0.004, stat_y + sh - 0.006),
            sw - 0.008, 0.005, facecolor=col, edgecolor="none",
            transform=T, zorder=4))
        ax.text(sx + sw/2, stat_y + sh * 0.70, lbl,
                ha="center", va="center", fontsize=3.2, color="#666", transform=T)
        ax.text(sx + sw/2, stat_y + sh * 0.24, num,
                ha="center", va="center", fontsize=6.6, color=INK,
                fontweight="bold", transform=T)

    # ── export buttons row ──────────────────────────────────────────────────
    ex_y = 0.540
    ex_labels = ["Export CSV", "Export SVG", "Export PNG"]
    ex_x = 0.025
    for lbl in ex_labels:
        ew = 0.145
        ax.add_patch(FancyBboxPatch((ex_x, ex_y), ew, 0.040,
            boxstyle="round,pad=0.002,rounding_size=0.006",
            facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
        ax.text(ex_x + ew/2, ex_y + 0.008, lbl, ha="center", va="center",
                fontsize=3.0, color="#555", fontweight="bold", transform=T, zorder=4)
        ex_x += ew + 0.008

    # ── 2x2 panel grid ──────────────────────────────────────────────────────
    area_x0, area_x1 = 0.025, 0.975
    area_y0, area_y1 = 0.035, 0.505
    col_split = 0.500
    row_split = 0.265
    cgap = 0.008

    def _panel(x0, y0, x1, y1, title_text, title_color=INK):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
            facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
        ax.text(x0 + 0.008, y1 - 0.016, title_text,
                ha="left", va="center", fontsize=3.0, color=title_color,
                fontweight="bold", transform=T, zorder=4)

    # ------ Panel A: Disease Landscape (top-left) ------
    ax0, ay0 = area_x0, row_split + cgap
    ax1, ay1 = col_split - cgap, area_y1
    _panel(ax0, ay0, ax1, ay1, "DISEASE LANDSCAPE", "#666")
    dis_items = [
        ("Non-CDI diarrhea w/ amp", 0.711),
        ("IgA glomerulonephritis",  0.678),
        ("Aplastic anemia",         0.658),
        ("Localized scleroderma",   0.570),
        ("Cerebral infarction",     0.538),
        ("Thyroid carcinoma",       0.462),
        ("Primary biliary cirr.",   0.445),
        ("Chronic constipation",    0.420),
        ("Secondary hypertension",  0.416),
        ("Autoimmune thyroiditis",  0.399),
        ("Schizophrenia",           0.369),
        ("Kidney stone",            0.354),
    ]
    db_x0 = ax0 + 0.148
    db_x1 = ax1 - 0.042
    db_top = ay1 - 0.033
    db_bot = ay0 + 0.010
    n_d = len(dis_items)
    rh = (db_top - db_bot) / n_d
    for i, (lbl, v) in enumerate(dis_items):
        cy = db_top - (i + 0.5) * rh
        bw = v * (db_x1 - db_x0)
        ax.add_patch(Rectangle((db_x0, cy - rh * 0.30), bw, rh * 0.60,
            facecolor=MAGENTA, edgecolor="none", transform=T, zorder=4))
        ax.text(db_x0 - 0.003, cy, lbl, ha="right", va="center",
                fontsize=1.9, color="#888", transform=T, zorder=5)
        ax.text(db_x0 + bw + 0.003, cy, f"{v*100:.1f}%",
                ha="left", va="center", fontsize=1.9,
                color="#888", transform=T, zorder=5)

    # ------ Panel B: Country Landscape (top-right) ------
    bx0, by0 = col_split, row_split + cgap
    bx1, by1 = area_x1, area_y1
    _panel(bx0, by0, bx1, by1, "COUNTRY LANDSCAPE", "#666")
    countries = [
        ("Indonesia",   0.351),
        ("Armenia",     0.286),
        ("Finland",     0.268),
        ("Malaysia",    0.243),
        ("Kazakhstan",  0.227),
        ("Estonia",     0.220),
        ("Brazil",      0.191),
        ("South Korea", 0.189),
        ("Denmark",     0.186),
        ("France",      0.185),
        ("Japan",       0.180),
        ("Germany",     0.161),
    ]
    cb_x0 = bx0 + 0.120
    cb_x1 = bx1 - 0.042
    cb_top = by1 - 0.033
    cb_bot = by0 + 0.010
    n_c = len(countries)
    rh = (cb_top - cb_bot) / n_c
    BLUE = "#5B6FE5"
    for i, (lbl, v) in enumerate(countries):
        cy = cb_top - (i + 0.5) * rh
        bw = (v / 0.40) * (cb_x1 - cb_x0)
        ax.add_patch(Rectangle((cb_x0, cy - rh * 0.30), bw, rh * 0.60,
            facecolor=BLUE, edgecolor="none", transform=T, zorder=4))
        ax.text(cb_x0 - 0.003, cy, lbl, ha="right", va="center",
                fontsize=1.9, color="#888", transform=T, zorder=5)
        ax.text(cb_x0 + bw + 0.003, cy, f"{v*100:.1f}%",
                ha="left", va="center", fontsize=1.9,
                color="#888", transform=T, zorder=5)

    # ------ Panel C: Abundance by Age Group (bottom-left) ------
    cx0_, cy0_ = area_x0, area_y0
    cx1_, cy1_ = col_split - cgap, row_split - cgap
    _panel(cx0_, cy0_, cx1_, cy1_, "ABUNDANCE BY AGE GROUP", "#666")
    ages = [
        ("Infant",       0.0748),
        ("Child",        0.1190),
        ("Adolescent",   0.2060),
        ("Adult",        0.1390),
        ("Older Adult",  0.1900),
        ("Oldest Old",   0.2200),
        ("Centenarian",  0.1870),
        ("Unknown",      0.1420),
    ]
    ag_x0 = cx0_ + 0.112
    ag_x1 = cx1_ - 0.038
    ag_top = cy1_ - 0.028
    ag_bot = cy0_ + 0.012
    n_a = len(ages)
    rh = (ag_top - ag_bot) / n_a
    for i, (lbl, v) in enumerate(ages):
        cy = ag_top - (i + 0.5) * rh
        bw = (v / 0.25) * (ag_x1 - ag_x0)
        ax.add_patch(Rectangle((ag_x0, cy - rh * 0.30), bw, rh * 0.60,
            facecolor=TEAL, edgecolor="none", transform=T, zorder=4))
        ax.text(ag_x0 - 0.003, cy, lbl, ha="right", va="center",
                fontsize=1.9, color="#888", transform=T, zorder=5)
        ax.text(ag_x0 + bw + 0.003, cy, f"{v*100:.1f}%",
                ha="left", va="center", fontsize=1.9,
                color="#888", transform=T, zorder=5)

    # ------ Panel D: Abundance by Sex (bottom-right, table) ------
    dx0, dy0 = col_split, area_y0
    dx1, dy1 = area_x1, row_split - cgap
    _panel(dx0, dy0, dx1, dy1, "ABUNDANCE BY SEX", "#666")
    # column headers
    col_xs = [dx0 + 0.030, dx0 + 0.110, dx0 + 0.200, dx0 + 0.295, dx0 + 0.390]
    headers = ["SEX", "MEAN ABUND.", "PREVALENCE", "N"]
    # 4-col table: SEX | MEAN ABUND | PREVALENCE | N
    t_x0 = dx0 + 0.015
    t_x1 = dx1 - 0.015
    col_ws = [0.22, 0.30, 0.28, 0.20]  # fractions
    col_lefts = []
    _cx = t_x0
    for w in col_ws:
        col_lefts.append(_cx)
        _cx += w * (t_x1 - t_x0)
    hdr_y = dy1 - 0.040
    for lbl, lx in zip(headers, col_lefts):
        ax.text(lx, hdr_y, lbl, ha="left", va="center",
                fontsize=2.3, color="#888", fontweight="bold",
                transform=T, zorder=4)
    # divider line
    div_y = hdr_y - 0.014
    ax.plot([t_x0, t_x1], [div_y, div_y], color=MGRAY,
            lw=0.5, transform=T, zorder=4)
    # rows
    rows = [("female", "13.0%", "76.0%", "35,860"),
            ("male",   "11.0%", "68.6%", "38,778")]
    row_h_t = 0.046
    for ri, row in enumerate(rows):
        ry = div_y - 0.014 - ri * row_h_t
        for val, lx in zip(row, col_lefts):
            ax.text(lx, ry, val, ha="left", va="center",
                    fontsize=2.6, color=INK, transform=T, zorder=4)
        if ri == 0:
            ax.plot([t_x0, t_x1], [ry - row_h_t * 0.5, ry - row_h_t * 0.5],
                    color=LGRAY, lw=0.4, transform=T, zorder=3)

def _thumb_studies(ax):
    """Studies & Projects — 6 stats + map + timeline + filters + table."""
    _ui_bg(ax, "white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    T = ax.transAxes
    NAVY = "#1F3864"; RED = "#C41435"; INK = "#1A1A1A"
    MGRAY = "#BCC6D6"; LGRAY = "#EEF1F6"
    GREEN = "#2BB573"; TEAL = "#1A6B82"

    # ── title + subtitle ────────────────────────────────────────────────────
    ax.text(0.025, 0.965, "Studies & Projects", ha="left", va="center",
            fontsize=5.6, color=INK, fontweight="bold", transform=T)
    ax.text(0.025, 0.930,
            "A project browser for cohort provenance, study metadata, "
            "and cross-study launch.",
            ha="left", va="center", fontsize=3.0, color="#888", transform=T)

    # ── 6 stat boxes ────────────────────────────────────────────────────────
    stat_y = 0.830
    sh = 0.072
    stats = [
        ("PROJECTS",  "482",     NAVY),
        ("SAMPLES",   "168,464", NAVY),
        ("NC",        "82,106",  GREEN),
        ("DISEASE",   "72,679",  RED),
        ("COUNTRIES", "72",      NAVY),
        ("DISEASES",  "224",     NAVY),
    ]
    stat_gap = 0.010
    sw = (0.950 - 5 * stat_gap) / 6
    for i, (lbl, num, col) in enumerate(stats):
        sx = 0.025 + i * (sw + stat_gap)
        ax.add_patch(FancyBboxPatch((sx, stat_y), sw, sh,
            boxstyle="round,pad=0.002,rounding_size=0.004",
            facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
        ax.add_patch(Rectangle((sx + 0.004, stat_y + sh - 0.005),
            sw - 0.008, 0.004, facecolor=col, edgecolor="none",
            transform=T, zorder=4))
        ax.text(sx + sw/2, stat_y + sh * 0.72, lbl,
                ha="center", va="center", fontsize=2.8, color="#666", transform=T)
        ax.text(sx + sw/2, stat_y + sh * 0.26, num,
                ha="center", va="center", fontsize=5.4, color=INK,
                fontweight="bold", transform=T)

    # ── map panel + timeline panel (side-by-side) ───────────────────────────
    mp_x0, mp_x1 = 0.025, 0.540
    tl_x0, tl_x1 = 0.555, 0.975
    pn_y0, pn_y1 = 0.540, 0.800

    # Map panel frame
    ax.add_patch(Rectangle((mp_x0, pn_y0), mp_x1 - mp_x0, pn_y1 - pn_y0,
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
    ax.text(mp_x0 + 0.010, pn_y1 - 0.018, "Projects Map",
            ha="left", va="center", fontsize=3.2, color=INK,
            fontweight="bold", transform=T, zorder=4)
    ax.text(mp_x1 - 0.010, pn_y1 - 0.018, "Clear country filter",
            ha="right", va="center", fontsize=2.4, color="#888",
            transform=T, zorder=4)

    # map inset (light bg, green continents, bigger bubbles)
    map_ax = ax.inset_axes([mp_x0 + 0.010, pn_y0 + 0.014,
                            (mp_x1 - mp_x0) - 0.008,
                            (pn_y1 - pn_y0) - 0.084])
    _ui_bg(map_ax, "#F4F7FB")
    map_ax.set_xlim(0, 1); map_ax.set_ylim(0, 1)
    CONT_FACE = "#BFE3C9"
    CONT_EDGE = "#8FCFA3"
    na = np.array([[0.05,0.82],[0.22,0.84],[0.24,0.52],[0.12,0.44],[0.04,0.58]])
    sa = np.array([[0.18,0.44],[0.26,0.42],[0.28,0.18],[0.20,0.15],[0.14,0.30]])
    eu = np.array([[0.42,0.84],[0.55,0.84],[0.56,0.68],[0.48,0.64],[0.42,0.72]])
    af = np.array([[0.44,0.64],[0.56,0.64],[0.58,0.28],[0.50,0.18],[0.42,0.32]])
    as_ = np.array([[0.56,0.86],[0.92,0.86],[0.90,0.55],[0.72,0.50],[0.58,0.62]])
    au = np.array([[0.78,0.32],[0.92,0.32],[0.92,0.18],[0.78,0.18]])
    for poly in (na, sa, eu, af, as_, au):
        map_ax.add_patch(plt.Polygon(poly, closed=True,
            facecolor=CONT_FACE, edgecolor=CONT_EDGE, lw=0.5, zorder=3))
    # bubbles
    bubbles = [
        (0.16, 0.66, 63, "US"), (0.78, 0.70, 58, "CN"),
        (0.52, 0.74, 28, "DE"), (0.43, 0.80, 22, "GB"),
        (0.57, 0.82, 12, "DK"), (0.86, 0.74, 14, "JP"),
        (0.12, 0.72, 10, "CA"), (0.72, 0.62,  8, "BD"),
        (0.85, 0.25,  5, "AU"), (0.44, 0.68,  6, "FR"),
    ]
    for bx_, by_, cnt, cc in bubbles:
        sz = 22 + cnt * 2.2
        map_ax.scatter([bx_], [by_], s=sz, c=GREEN, alpha=0.85,
                       edgecolors="white", lw=0.6, zorder=5)
        map_ax.text(bx_, by_ + 0.09, cc, fontsize=2.4, ha="center",
                    color="#1A6B35", zorder=6)
    for sp in map_ax.spines.values():
        sp.set_visible(False)
    map_ax.set_xticks([]); map_ax.set_yticks([])

    # Timeline panel frame
    ax.add_patch(Rectangle((tl_x0, pn_y0), tl_x1 - tl_x0, pn_y1 - pn_y0,
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
    ax.text(tl_x0 + 0.010, pn_y1 - 0.018, "Timeline",
            ha="left", va="center", fontsize=3.2, color=INK,
            fontweight="bold", transform=T, zorder=4)
    ax.text(tl_x1 - 0.010, pn_y1 - 0.018, "2012-2021",
            ha="right", va="center", fontsize=2.4, color="#888",
            transform=T, zorder=4)

    # timeline area chart inset
    tl_ax = ax.inset_axes([tl_x0 + 0.008, pn_y0 + 0.014,
                           (tl_x1 - tl_x0) - 0.034,
                           (pn_y1 - pn_y0) - 0.084])
    _ui_bg(tl_ax, "white")
    years = np.arange(2012, 2022)
    samples = np.array([5, 9, 16, 22, 30, 44, 58, 76, 92, 108]) * 1.0
    projects = np.array([3, 5, 9, 13, 18, 26, 33, 42, 50, 58]) * 1.0
    tl_ax.fill_between(years, 0, samples, color=GREEN, alpha=0.25, zorder=3)
    tl_ax.plot(years, samples, color=GREEN, lw=1.2, zorder=4)
    tl_ax.plot(years, projects, color=TEAL, lw=1.0, zorder=5)
    tl_ax.fill_between(years, 0, projects, color=TEAL, alpha=0.18, zorder=3)
    tl_ax.set_xlim(2012, 2021)
    tl_ax.set_ylim(0, 120)
    tl_ax.set_xticks([2012, 2015, 2018, 2021])
    tl_ax.set_xticklabels(["2012", "2015", "2018", "2021"], fontsize=2.0)
    tl_ax.set_yticks([0, 30, 60, 90, 120])
    tl_ax.tick_params(axis='y', labelsize=2.0, pad=1)
    tl_ax.tick_params(axis='x', labelsize=2.0, pad=1)
    for sp in tl_ax.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color("#BBB")
    tl_ax.spines["top"].set_visible(False)
    tl_ax.spines["right"].set_visible(False)

    # ── filter row ──────────────────────────────────────────────────────────
    fr_y = 0.475
    fr_h = 0.046
    filters = [
        ("Search by project ID or disease…", 0.255),
        ("All countries  v",                 0.130),
        ("Filter by disease  v",             0.165),
        ("NC filter: all  v",                0.140),
        ("Sort by samples  v",               0.150),
    ]
    fx = 0.025
    for lbl, fw in filters:
        ax.add_patch(FancyBboxPatch((fx, fr_y), fw, fr_h,
            boxstyle="round,pad=0.002,rounding_size=0.006",
            facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
        ax.text(fx + 0.006, fr_y + fr_h/2, lbl, ha="left", va="center",
                fontsize=2.6, color="#666", transform=T, zorder=4)
        fx += fw + 0.008
    # 492 / 492 projects · Reset filters
    ax.text(0.025, fr_y - 0.018, "482 / 482 projects",
            ha="left", va="center", fontsize=2.4, color="#888", transform=T)
    ax.text(0.975, fr_y - 0.018, "Reset filters",
            ha="right", va="center", fontsize=2.4, color="#888", transform=T)

    # ── projects table ──────────────────────────────────────────────────────
    tb_x0, tb_x1 = 0.025, 0.975
    tb_y0, tb_y1 = 0.035, 0.440
    ax.add_patch(Rectangle((tb_x0, tb_y0), tb_x1 - tb_x0, tb_y1 - tb_y0,
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))

    # header row
    col_specs = [
        ("PROJECT ID",    0.130),
        ("YEAR",          0.050),
        ("COUNTRY",       0.140),
        ("SAMPLES",       0.070),
        ("NC",            0.060),
        ("DISEASE",       0.070),
        ("REGION (EST.)", 0.120),
        ("DISEASES",      0.270),
        ("SEL",           0.040),
    ]
    total_w = sum(w for _, w in col_specs)
    col_xs = []
    cx = tb_x0 + 0.010
    usable_w = (tb_x1 - tb_x0) - 0.008
    for _, w in col_specs:
        col_xs.append(cx)
        cx += (w / total_w) * usable_w

    hdr_y = tb_y1 - 0.022
    for (lbl, _), lx in zip(col_specs, col_xs):
        ax.text(lx, hdr_y, lbl, ha="left", va="center",
                fontsize=2.2, color="#888", fontweight="bold",
                transform=T, zorder=4)
    # divider
    div_y = hdr_y - 0.014
    ax.plot([tb_x0 + 0.010, tb_x1 - 0.010], [div_y, div_y],
            color=MGRAY, lw=0.4, transform=T, zorder=4)

    MAGENTA = "#C735C7"
    # ----- 2 regular rows + 1 expanded row -----
    rows_data = [
        ("PRJEB1418",   "2021", "United States",  "4,850", "105", "4,440",
         "V3-V5 (est.)", "ADHD · ASD · Cancer +15"),
        ("PRJNA728511", "2021", "United States",  "3,540", "600", "1,002",
         "V1-V2 (est.)", "C. difficile Infection"),
    ]
    row_h_t = 0.021
    for ri, row in enumerate(rows_data):
        ry = div_y - 0.008 - (ri + 0.5) * row_h_t
        ax.scatter([tb_x0 + 0.006], [ry], s=4, c=MAGENTA,
                   transform=T, zorder=5)
        ax.text(col_xs[0], ry, row[0], ha="left", va="center",
                fontsize=2.2, color="#3B5FA8", transform=T, zorder=4)
        ax.text(col_xs[1], ry, row[1], ha="left", va="center",
                fontsize=2.2, color=INK, transform=T, zorder=4)
        ax.text(col_xs[2], ry, row[2], ha="left", va="center",
                fontsize=2.2, color=INK, transform=T, zorder=4)
        ax.text(col_xs[3], ry, row[3], ha="left", va="center",
                fontsize=2.2, color=INK, transform=T, zorder=4)
        ax.text(col_xs[4], ry, row[4], ha="left", va="center",
                fontsize=2.2, color=GREEN, fontweight="bold",
                transform=T, zorder=4)
        ax.text(col_xs[5], ry, row[5], ha="left", va="center",
                fontsize=2.2, color=RED, fontweight="bold",
                transform=T, zorder=4)
        ax.text(col_xs[6], ry, row[6], ha="left", va="center",
                fontsize=2.0, color="#888", transform=T, zorder=4)
        ax.text(col_xs[7], ry, row[7], ha="left", va="center",
                fontsize=2.0, color=INK, transform=T, zorder=4)
        ax.add_patch(Rectangle((col_xs[8] + 0.010, ry - 0.008),
            0.016, 0.016, facecolor="white", edgecolor=MGRAY,
            lw=0.4, transform=T, zorder=4))
        rd_y = ry - row_h_t * 0.5
        ax.plot([tb_x0 + 0.010, tb_x1 - 0.010], [rd_y, rd_y],
                color=LGRAY, lw=0.3, transform=T, zorder=3)

    # ----- expanded row (PRJNA237362) -----
    exp_ry = div_y - 0.008 - (2 + 0.5) * row_h_t
    # minus icon
    ax.text(tb_x0 + 0.006, exp_ry, "−", ha="center", va="center",
            fontsize=3.2, color=MAGENTA, fontweight="bold",
            transform=T, zorder=5)
    ax.text(col_xs[0], exp_ry, "PRJNA237362", ha="left", va="center",
            fontsize=2.2, color=MAGENTA, fontweight="bold",
            transform=T, zorder=4)
    ax.text(col_xs[1], exp_ry, "2014", ha="left", va="center",
            fontsize=2.2, color=INK, transform=T, zorder=4)
    ax.text(col_xs[2], exp_ry, "United States", ha="left", va="center",
            fontsize=2.2, color=INK, transform=T, zorder=4)
    ax.text(col_xs[3], exp_ry, "1,379", ha="left", va="center",
            fontsize=2.2, color=INK, transform=T, zorder=4)
    ax.text(col_xs[4], exp_ry, "177", ha="left", va="center",
            fontsize=2.2, color=GREEN, fontweight="bold",
            transform=T, zorder=4)
    ax.text(col_xs[5], exp_ry, "440", ha="left", va="center",
            fontsize=2.2, color=RED, fontweight="bold",
            transform=T, zorder=4)
    ax.text(col_xs[6], exp_ry, "V3-V4 (est.)", ha="left", va="center",
            fontsize=2.0, color="#888", transform=T, zorder=4)
    # disease pill chips
    pill_x = col_xs[7]
    for plbl in ["Crohn's Disease", "Ulcerative Colitis"]:
        pw = 0.008 + len(plbl) * 0.0058
        ax.add_patch(FancyBboxPatch((pill_x, exp_ry - 0.010), pw, 0.008,
            boxstyle="round,pad=0.001,rounding_size=0.005",
            facecolor=LGRAY, edgecolor=MGRAY, lw=0.3,
            transform=T, zorder=4))
        ax.text(pill_x + pw/2, exp_ry, plbl, ha="center", va="center",
                fontsize=1.9, color=INK, transform=T, zorder=5)
        pill_x += pw + 0.006

    # ----- meta line -----
    meta_y = exp_ry - 0.024
    ax.plot([tb_x0 + 0.010, tb_x1 - 0.010], [meta_y + 0.012, meta_y + 0.012],
            color=LGRAY, lw=0.3, transform=T, zorder=3)
    meta_parts = [
        ("Samples:", "#888"), ("1,379", INK),
        ("NC:", "#888"), ("177", GREEN),
        ("Disease:", "#888"), ("440", RED),
        ("Year:", "#888"), ("2014", INK),
        ("Countries:", "#888"), ("United States", INK),
        ("Instrument:", "#888"), ("Illumina MiSeq", INK),
        ("NCBI BioProject", MAGENTA),
    ]
    mx = tb_x0 + 0.014
    for txt, col in meta_parts:
        fw = "bold" if col in (INK, GREEN, RED, MAGENTA) else "normal"
        ax.text(mx, meta_y, txt, ha="left", va="center",
                fontsize=2.0, color=col, fontweight=fw, transform=T, zorder=4)
        mx += 0.004 + len(txt) * 0.0080

    # ----- 2×2 detail mini-panels -----
    dt_x0, dt_x1 = tb_x0 + 0.010, tb_x1 - 0.010
    dt_y0, dt_y1 = tb_y0 + 0.008, meta_y - 0.016
    dcol_split = (dt_x0 + dt_x1) / 2 - 0.004
    drow_split = (dt_y0 + dt_y1) / 2 - 0.004

    def _dpanel(x0, y0, x1, y1, title_text):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
            facecolor="white", edgecolor=MGRAY, lw=0.4, transform=T, zorder=3))
        ax.text((x0 + x1)/2, y1 - 0.014, title_text,
                ha="center", va="center", fontsize=2.4, color=INK,
                fontweight="bold", transform=T, zorder=4)

    def _centered_rows(x0, x1, y_top, items, fs=3.2):
        cx = (x0 + x1) / 2
        for i, (lbl, val) in enumerate(items):
            iy = y_top - i * 0.022
            ax.text(cx - 0.006, iy, lbl, ha="right", va="center",
                    fontsize=fs, color=INK, fontweight="bold",
                    transform=T, zorder=4)
            ax.text(cx + 0.006, iy, val, ha="left", va="center",
                    fontsize=fs, color=INK, fontweight="bold",
                    transform=T, zorder=4)

    # Disease Breakdown (top-left)
    _dpanel(dt_x0, drow_split + 0.004, dcol_split, dt_y1, "Disease Breakdown")
    _centered_rows(dt_x0, dcol_split, dt_y1 - 0.048,
                   [("CD", "350"), ("UC", "90")])

    # Country Distribution (top-right)
    _dpanel(dcol_split + 0.008, drow_split + 0.004, dt_x1, dt_y1,
            "Country Distribution")
    _centered_rows(dcol_split + 0.008, dt_x1, dt_y1 - 0.048,
                   [("United States (US)", "1,379")])

    # Age Groups (bottom-left)
    _dpanel(dt_x0, dt_y0, dcol_split, drow_split - 0.004, "Age Groups")
    _centered_rows(dt_x0, dcol_split, drow_split - 0.038,
                   [("Child", "1,379")])

    # Sex Distribution (bottom-right)
    _dpanel(dcol_split + 0.008, dt_y0, dt_x1, drow_split - 0.004,
            "Sex Distribution")
    _centered_rows(dcol_split + 0.008, dt_x1, drow_split - 0.038,
                   [("unknown", "1,257"), ("male", "76"), ("female", "46")],
                   fs=2.8)

def _thumb_similarity(ax):
    """Similarity Search — canonical v2 style.

    Real output from /similarity with pasted query
    (Anaerotruncus 0.1 · Bifidobacterium 0.2 · Senegalimassilia 0.3):
      • 3 matched / 2,996 total genera · Top-K 10
      • All 10 hits Healthy (NC), similarity 47.6% → 39.2%
      • Country mix: 5 India · 2 Sweden · 1 Canada · 1 US · 1 China
    """
    _ui_bg(ax, "white")
    NAVY = "#1F3864"; RED = "#C41435"; INK = "#1A1A1A"
    MGRAY = "#BCC6D6"; LGRAY = "#EEF1F6"
    MAGENTA = "#C735C7"; GREEN = "#2BB573"; ORANGE = "#F0A020"
    T = ax.transAxes

    # ── title + subtitle ────────────────────────────────────────────────
    ax.text(0.500, 0.962, "Similarity Search",
            ha="center", va="center", fontsize=5.6, color=INK,
            fontweight="bold", transform=T)
    ax.text(0.500, 0.922,
            "Query top-K similar samples  ·  Bray-Curtis / Jaccard "
            " ·  154,785 reference profiles",
            ha="center", va="center", fontsize=2.5, color="#666", transform=T)

    # ── stats tiles row ────────────────────────────────────────────────
    tiles = [("Matched genera", "3 / 2,996"),
             ("Top-K", "10"),
             ("Top similarity", "47.6%"),
             ("Mean Top-10", "42.4%")]
    t_x0 = 0.030; t_x1 = 0.982
    t_gap = 0.012
    t_w = (t_x1 - t_x0 - 3 * t_gap) / 4
    for i, (lbl, val) in enumerate(tiles):
        tx = t_x0 + i * (t_w + t_gap)
        ax.add_patch(FancyBboxPatch((tx, 0.852), t_w, 0.056,
            boxstyle="round,pad=0.003,rounding_size=0.006",
            facecolor=LGRAY, edgecolor=MGRAY, lw=0.5,
            transform=T, zorder=3))
        ax.text(tx + 0.012, 0.893, val,
                ha="left", va="center", fontsize=3.6,
                color=INK, fontweight="bold", transform=T)
        ax.text(tx + 0.012, 0.867, lbl,
                ha="left", va="center", fontsize=2.3,
                color="#666", transform=T)

    # ── controls row ────────────────────────────────────────────────────
    cr_y = 0.792
    def _pill(x, w, label, sel=False, italic=False):
        ax.add_patch(FancyBboxPatch((x, cr_y - 0.014), w, 0.028,
            boxstyle="round,pad=0.002,rounding_size=0.005",
            facecolor=NAVY if sel else "white",
            edgecolor=NAVY if sel else MGRAY, lw=0.5,
            transform=T, zorder=3))
        ax.text(x + w/2, cr_y, label, ha="center", va="center",
                fontsize=2.1,
                color="white" if sel else NAVY,
                fontweight="bold" if sel else "normal",
                fontstyle="italic" if italic else "normal",
                transform=T)
    ax.text(0.030, cr_y, "Metric", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    _pill(0.072, 0.076, "Bray-Curtis", sel=True)
    ax.text(0.158, cr_y, "K", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    _pill(0.176, 0.032, "10")
    ax.text(0.216, cr_y, "Disease", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    _pill(0.260, 0.034, "NC")
    ax.text(0.302, cr_y, "Country", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    _pill(0.350, 0.034, "All")
    ax.text(0.392, cr_y, "Age", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    _pill(0.420, 0.034, "All")
    ax.add_patch(FancyBboxPatch((0.470, cr_y - 0.014), 0.082, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor="white", edgecolor=MGRAY, lw=0.5,
        transform=T, zorder=3))
    ax.text(0.511, cr_y, "Upload CSV / TSV", ha="center", va="center",
            fontsize=2.0, color="#555", fontstyle="italic", transform=T)
    # CTA
    cta_w = 0.170
    ax.add_patch(FancyBboxPatch((t_x1 - cta_w, cr_y - 0.016), cta_w, 0.032,
        boxstyle="round,pad=0.002,rounding_size=0.006",
        facecolor=MAGENTA, edgecolor=MAGENTA, lw=0.5, transform=T, zorder=3))
    ax.text(t_x1 - cta_w/2, cr_y, "Search",
            ha="center", va="center", fontsize=2.5, color="white",
            fontweight="bold", transform=T)

    # ── query genera pill strip ────────────────────────────────────────
    fp_y = 0.712
    ax.text(0.030, fp_y, "Query:", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    q_genera = [("Senegalimassilia", 50.0),
                ("Bifidobacterium",  33.3),
                ("Anaerotruncus",    16.7)]
    fp_x = 0.080
    for nm, pct in q_genera:
        lbl = f"{nm}  {pct:.1f}%"
        pw = 0.010 + len(lbl) * 0.0058
        ax.add_patch(FancyBboxPatch((fp_x, fp_y - 0.013), pw, 0.026,
            boxstyle="round,pad=0.001,rounding_size=0.004",
            facecolor="white", edgecolor=ORANGE, lw=0.5,
            transform=T, zorder=3))
        ax.text(fp_x + pw/2, fp_y, lbl, ha="center", va="center",
                fontsize=1.7, color=INK, transform=T)
        fp_x += pw + 0.012
    # current search space hint on right
    ax.text(t_x1, fp_y,
            "search space  ·  Healthy (NC)  ·  all countries  ·  all ages",
            ha="right", va="center", fontsize=1.9, color="#888",
            fontstyle="italic", transform=T)

    # ── 4 tabs ──────────────────────────────────────────────────────────
    tabs = [("Top-10 ranked", True), ("Query heatmap", False),
            ("Country mix", False), ("Age mix", False)]
    tab_x = 0.030
    tab_y = 0.672
    for lbl, sel in tabs:
        tw = 0.008 + len(lbl) * 0.0095
        ax.add_patch(FancyBboxPatch((tab_x, tab_y - 0.014), tw, 0.028,
            boxstyle="round,pad=0.002,rounding_size=0.005",
            facecolor="white",
            edgecolor=MAGENTA if sel else MGRAY,
            lw=0.8 if sel else 0.4, transform=T, zorder=3))
        ax.text(tab_x + tw/2, tab_y, lbl, ha="center", va="center",
                fontsize=2.2, color=MAGENTA if sel else INK,
                fontweight="bold" if sel else "normal", transform=T)
        tab_x += tw + 0.008

    # ── main content: asymmetric ── big ranked bars left, 2 right panels
    rank_x, rank_y = 0.030, 0.040
    rank_w, rank_h = 0.430, 0.584
    right_x = 0.500
    right_w = 0.482
    top_y, bot_y = 0.354, 0.040
    row_h = 0.270

    # ── LEFT: ranked top-10 similarity bars ────────────────────────────
    r_ax = ax.inset_axes([rank_x, rank_y, rank_w, rank_h])
    _ui_bg(r_ax, "white")
    hits = [
        ("#1  PRJNA414540_SRR6182318",  47.6, "Canada", "Adult"),
        ("#2  PRJNA589488_SRR10484245", 43.6, "USA",    "Infant"),
        ("#3  PRJNA510423_SRR14431930", 43.4, "Sweden", "Infant"),
        ("#4  PRJEB38948_ERR4348299",   42.1, "India",  "Infant"),
        ("#5  PRJNA699380_SRR13626223", 42.0, "China",  "Unknown"),
        ("#6  PRJEB38948_ERR4348727",   41.8, "India",  "Infant"),
        ("#7  PRJEB21946_ERR2044530",   40.2, "India",  "Infant"),
        ("#8  PRJEB38948_ERR4347342",   40.2, "India",  "Infant"),
        ("#9  PRJEB38948_ERR4347063",   39.3, "India",  "Infant"),
        ("#10 PRJNA510423_SRR8348606",  39.2, "Sweden", "Infant"),
    ]
    names = [h[0] for h in hits]
    sims  = [h[1] for h in hits]
    ys = np.arange(len(hits))[::-1]
    # gradient color: hotter for higher similarity
    cmap = LinearSegmentedColormap.from_list("sim",
            ["#FFDFA8", "#F0A020", "#C94F2A"])
    cols = [cmap((s - 38.5) / (48.0 - 38.5)) for s in sims]
    r_ax.barh(ys, sims, height=0.68, color=cols,
              edgecolor="white", linewidth=0.4)
    for yi, s in zip(ys, sims):
        r_ax.text(s + 0.4, yi, f"{s:.1f}%", va="center", ha="left",
                  fontsize=2.0, color=INK, fontweight="bold")
    r_ax.set_yticks(ys)
    r_ax.set_yticklabels(names, fontsize=1.9, color=INK,
                         family="monospace")
    r_ax.set_xlim(0, 54)
    r_ax.set_xticks([0, 20, 40])
    r_ax.set_xticklabels(["0", "20", "40"], fontsize=1.9, color="#888")
    r_ax.set_xlabel("similarity (%)", fontsize=2.1, color="#666", labelpad=1)
    r_ax.tick_params(length=0, pad=1)
    for sp in r_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    ax.text(rank_x + rank_w/2, rank_y + rank_h - 0.004,
            "Top-10 ranked hits  ·  all Healthy (NC)  ·  47.6 → 39.2%",
            ha="center", va="bottom", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=2.0))

    # ── RIGHT TOP: query vs matches heatmap ────────────────────────────
    h_ax = ax.inset_axes([right_x, top_y, right_w, row_h])
    _ui_bg(h_ax, "white")
    hm_rows = ["Query", "#1", "#2", "#3", "#4", "#5"]
    hm_cols = ["Senegalimassilia", "Bifidobacterium", "Anaerotruncus"]
    M = np.array([
        [50.0, 33.3, 16.7],
        [14.4, 33.2,  0.0],
        [ 0.0, 45.8, 10.3],
        [12.3, 31.1,  0.0],
        [ 8.8, 37.1,  0.0],
        [ 8.6, 44.5,  0.0],
    ])
    hcmap = LinearSegmentedColormap.from_list("hm",
            ["#FFF4E0", "#F0A020", "#7A2E00"])
    h_ax.imshow(M, cmap=hcmap, aspect="auto", vmin=0, vmax=50)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            tc = "white" if v > 25 else INK
            h_ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                      fontsize=1.8, color=tc, fontweight="bold")
    h_ax.set_xticks(range(3))
    h_ax.set_xticklabels(hm_cols, fontsize=1.7, color=INK,
                         fontstyle="italic", rotation=0)
    h_ax.set_yticks(range(6))
    h_ax.set_yticklabels(hm_rows, fontsize=1.8, color=INK)
    h_ax.tick_params(length=0, pad=1)
    for sp in h_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    ax.text(right_x + right_w/2, top_y + row_h + 0.026,
            "Query vs top-5 matches  ·  relative abundance (%)",
            ha="center", va="bottom", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=2.0))

    # ── RIGHT BOTTOM: country mix bars ─────────────────────────────────
    _c_dy = -0.022
    c_ax = ax.inset_axes([right_x, bot_y + _c_dy, right_w, row_h + _c_dy])
    _ui_bg(c_ax, "white")
    countries = [("India", 5), ("Sweden", 2), ("Canada", 1),
                 ("USA", 1), ("China", 1)]
    cn = [c[0] for c in countries]
    cv = [c[1] for c in countries]
    cys = np.arange(len(countries))[::-1]
    ccols = [NAVY, "#4A6FA8", "#7A99C8", "#B4C4DC", "#D8E0EC"]
    c_ax.barh(cys, cv, height=0.68, color=ccols,
              edgecolor="white", linewidth=0.4)
    for yi, v in zip(cys, cv):
        c_ax.text(v + 0.08, yi, f"{v}", va="center", ha="left",
                  fontsize=2.0, color=INK, fontweight="bold")
    c_ax.set_yticks(cys)
    c_ax.set_yticklabels(cn, fontsize=2.0, color=INK)
    c_ax.set_xlim(0, 6.2)
    c_ax.set_xticks([0, 2, 4, 6])
    c_ax.set_xticklabels(["0", "2", "4", "6"], fontsize=1.9, color="#888")
    c_ax.set_xlabel("samples in Top-10", fontsize=2.1, color="#666",
                    labelpad=1)
    c_ax.tick_params(length=0, pad=1)
    for sp in c_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    ax.text(right_x + right_w/2, bot_y + row_h - 0.018,
            "Top-10 country mix  ·  Infant-dominated",
            ha="center", va="bottom", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=2.0))

def _thumb_metabolism(ax):
    """Metabolic Function Browser — literature-curated functional categories + profile view."""
    _ui_bg(ax, "white")
    NAVY = "#1F3864"; RED = "#C41435"; INK = "#1A1A1A"
    MGRAY = "#BCC6D6"; LGRAY = "#EEF1F6"
    TEAL = "#1A6B82"; GREEN = "#2BB573"; MAGENTA = "#C735C7"
    T = ax.transAxes
    CJK = ["Microsoft YaHei", "SimHei", "Arial"]

    # ── title ────────────────────────────────────────────────────────────────
    ax.text(0.025, 0.968, "Metabolic Function Browser", ha="left", va="center",
            fontsize=5.6, color=INK, fontweight="bold", transform=T)
    ax.text(0.025, 0.935,
            "Explore gut microbiota organized by metabolic role and clinical relevance",
            ha="left", va="center", fontsize=3.0, color="#888", transform=T)
    # disclaimer
    ax.text(0.025, 0.905,
            "Literature-curated functional associations · genus-level patterns, not pathway activity",
            ha="left", va="center", fontsize=2.4, color="#999",
            fontstyle="italic", transform=T)

    # ── search box ───────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.025, 0.858), 0.950, 0.036,
        boxstyle="round,pad=0.002,rounding_size=0.006",
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
    ax.text(0.034, 0.876, "Search category, genus, metabolite, or pathway...",
            ha="left", va="center", fontsize=2.8, color="#AAA", transform=T)

    # ── left sidebar: FUNCTIONAL CATEGORIES (15) ─────────────────────────────
    sb_x0, sb_x1 = 0.012, 0.292
    ax.text(sb_x0 + 0.006, 0.828, "FUNCTIONAL CATEGORIES",
            ha="left", va="center", fontsize=2.6, color="#666",
            fontweight="bold", transform=T)
    # sort dropdown
    ax.add_patch(FancyBboxPatch((sb_x0 + 0.006, 0.790), sb_x1 - sb_x0 - 0.012, 0.028,
        boxstyle="round,pad=0.001,rounding_size=0.004",
        facecolor="white", edgecolor=MGRAY, lw=0.4, transform=T, zorder=3))
    ax.text(sb_x0 + 0.012, 0.804, "Default order  v",
            ha="left", va="center", fontsize=2.4, color="#555", transform=T)

    categories = [
        ("Short-Chain Fatty Acid Producers",   RED,       True),
        ("Bile Acid Metabolizing Bacteria",    "#E8B83A", False),
        ("Tryptophan Metabolizing Bacteria",   "#3CA0E0", False),
        ("Sulfur / H2S Producers",             "#B574D9", False),
        ("TMAO Precursor Producers",           RED,       False),
        ("LPS / Endotoxin Producers",          "#E36B4A", False),
        ("Folate & B12 Producers",             GREEN,     False),
        ("Neurotransmitter-Related Bacteria",  "#B574D9", False),
        ("Mucin-Degrading Bacteria",           "#E8B83A", False),
        ("Amino Acid Fermenters",              "#3CA0E0", False),
        ("Hydrogen & Methane Cycling",         "#8C9BAA", False),
        ("Lactate Producers & Cross-Feeders",  GREEN,     False),
        ("Oxalate-Degrading Bacteria",         "#3CA0E0", False),
        ("Complex Polysaccharide Degraders",   "#E8B83A", False),
        ("Opportunistic Pathobionts",          RED,       False),
    ]
    cat_y0 = 0.770
    row_h_c = (cat_y0 - 0.035) / len(categories)
    for ci, (en, col, sel) in enumerate(categories):
        iy = cat_y0 - (ci + 1) * row_h_c
        if sel:
            ax.add_patch(FancyBboxPatch((sb_x0 + 0.004, iy),
                sb_x1 - sb_x0 - 0.008, row_h_c - 0.003,
                boxstyle="round,pad=0.001,rounding_size=0.004",
                facecolor="#FCE8EC", edgecolor=RED, lw=0.5,
                transform=T, zorder=3))
        ax.scatter([sb_x0 + 0.014], [iy + row_h_c * 0.50], s=4,
                   c=col, edgecolors="none", transform=T, zorder=5)
        ax.text(sb_x0 + 0.022, iy + row_h_c * 0.50, en,
                ha="left", va="center", fontsize=2.4,
                color=RED if sel else INK,
                fontweight="bold" if sel else "normal",
                transform=T, zorder=4)

    # ── right main panel ─────────────────────────────────────────────────────
    rp_x0, rp_x1 = 0.302, 0.975

    # tabs: Profile | Disease Context
    tb_y = 0.822; tb_h = 0.034
    tabs = [("Profile", True), ("Disease Context", False)]
    tx = rp_x0 + 0.008
    for tlbl, sel in tabs:
        tw = 0.098 if tlbl == "Profile" else 0.148
        ax.add_patch(FancyBboxPatch((tx, tb_y), tw, tb_h,
            boxstyle="round,pad=0.001,rounding_size=0.016",
            facecolor=NAVY if sel else "white",
            edgecolor=NAVY, lw=0.5, transform=T, zorder=3))
        ax.text(tx + tw/2, tb_y + tb_h/2, tlbl, ha="center", va="center",
                fontsize=2.8, color="white" if sel else NAVY,
                fontweight="bold", transform=T, zorder=4)
        tx += tw + 0.008

    # category header: icon + name + zh
    ax.scatter([rp_x0 + 0.018], [0.775], s=28, c=RED,
               edgecolors="none", transform=T, zorder=5)
    ax.text(rp_x0 + 0.034, 0.775,
            "Short-Chain Fatty Acid Producers",
            ha="left", va="center", fontsize=4.8, color=INK,
            fontweight="bold", transform=T)

    # description
    ax.text(rp_x0 + 0.008, 0.722,
            "Bacteria that produce short-chain fatty acids (SCFAs) such as butyrate,",
            ha="left", va="center", fontsize=2.4, color="#555", transform=T)
    ax.text(rp_x0 + 0.008, 0.702,
            "propionate, and acetate through fermentation of dietary fiber.",
            ha="left", va="center", fontsize=2.4, color="#555", transform=T)

    # 4 stat boxes
    stat_y = 0.622; sh = 0.068
    stats = [
        ("MATCHED GENERA",    "10",     NAVY),
        ("STRICT NC SAMPLES", "82,106", GREEN),
        ("TESTED DISEASES",   "30",     NAVY),
        ("SIGNIFICANT",       "20",     MAGENTA),
    ]
    stat_gap = 0.010
    sw = ((rp_x1 - rp_x0) - 0.016 - 3 * stat_gap) / 4
    for i, (lbl, num, col) in enumerate(stats):
        sx = rp_x0 + 0.008 + i * (sw + stat_gap)
        ax.add_patch(FancyBboxPatch((sx, stat_y), sw, sh,
            boxstyle="round,pad=0.001,rounding_size=0.004",
            facecolor="white", edgecolor=MGRAY, lw=0.4,
            transform=T, zorder=3))
        ax.add_patch(Rectangle((sx + 0.003, stat_y + sh - 0.005),
            sw - 0.006, 0.004, facecolor=col, edgecolor="none",
            transform=T, zorder=4))
        ax.text(sx + sw/2, stat_y + sh * 0.72, lbl,
                ha="center", va="center", fontsize=2.4,
                color="#666", transform=T, zorder=4)
        ax.text(sx + sw/2, stat_y + sh * 0.30, num,
                ha="center", va="center", fontsize=4.2,
                color=INK, fontweight="bold", transform=T, zorder=4)

    # MEMBER GENERA (10)
    mg_y = 0.560
    ax.text(rp_x0 + 0.008, mg_y, "MEMBER GENERA (10)",
            ha="left", va="center", fontsize=2.4, color="#666",
            fontweight="bold", transform=T)
    genera = ["Faecalibacterium", "Roseburia", "Eubacterium",
              "Bifidobacterium", "Akkermansia", "Coprococcus",
              "Butyrivibrio", "Anaerostipes", "Subdoligranulum",
              "Ruminococcus"]
    gx = rp_x0 + 0.008
    gy = 0.530
    for g in genera:
        gw = 0.008 + len(g) * 0.0075
        if gx + gw > rp_x1 - 0.008:
            gx = rp_x0 + 0.008
            gy -= 0.028
        ax.add_patch(FancyBboxPatch((gx, gy - 0.008), gw, 0.016,
            boxstyle="round,pad=0.001,rounding_size=0.005",
            facecolor=LGRAY, edgecolor=MGRAY, lw=0.3,
            transform=T, zorder=3))
        ax.text(gx + gw/2, gy, g, ha="center", va="center",
                fontsize=2.0, color=INK,
                fontstyle="italic", transform=T, zorder=4)
        gx += gw + 0.006

    # KEY METABOLITES
    km_y = 0.458
    ax.text(rp_x0 + 0.008, km_y, "KEY METABOLITES",
            ha="left", va="center", fontsize=2.4, color="#666",
            fontweight="bold", transform=T)
    metabolites = [("Butyrate", GREEN), ("Propionate", GREEN), ("Acetate", GREEN)]
    mx = rp_x0 + 0.008
    for m, col in metabolites:
        mw = 0.010 + len(m) * 0.0082
        ax.add_patch(FancyBboxPatch((mx, km_y - 0.026), mw, 0.016,
            boxstyle="round,pad=0.001,rounding_size=0.005",
            facecolor="#E6F5EC", edgecolor=col, lw=0.4,
            transform=T, zorder=3))
        ax.text(mx + mw/2, km_y - 0.018, m, ha="center", va="center",
                fontsize=2.1, color=col, fontweight="bold", transform=T, zorder=4)
        mx += mw + 0.008

    # RELATED PATHWAYS
    rp_y = 0.396
    ax.text(rp_x0 + 0.008, rp_y, "RELATED PATHWAYS",
            ha="left", va="center", fontsize=2.4, color="#666",
            fontweight="bold", transform=T)
    pathways = [("Pyruvate fermentation", "KEGG"),
                ("Wood-Ljungdahl", "KEGG"),
                ("Acrylate pathway", None)]
    px = rp_x0 + 0.008
    for lbl, tag in pathways:
        base_w = 0.010 + len(lbl) * 0.0075
        pw = base_w + (0.028 if tag else 0)
        ax.add_patch(FancyBboxPatch((px, rp_y - 0.026), pw, 0.016,
            boxstyle="round,pad=0.001,rounding_size=0.005",
            facecolor=LGRAY, edgecolor=MGRAY, lw=0.3,
            transform=T, zorder=3))
        ax.text(px + 0.006, rp_y - 0.018, lbl,
                ha="left", va="center", fontsize=2.0, color=INK,
                transform=T, zorder=4)
        if tag:
            ax.add_patch(FancyBboxPatch((px + base_w - 0.002, rp_y - 0.024),
                0.024, 0.012,
                boxstyle="round,pad=0.001,rounding_size=0.003",
                facecolor=NAVY, edgecolor="none",
                transform=T, zorder=4))
            ax.text(px + base_w + 0.010, rp_y - 0.018, tag,
                    ha="center", va="center", fontsize=1.7,
                    color="white", fontweight="bold",
                    transform=T, zorder=5)
        px += pw + 0.008

    # CLINICAL RELEVANCE
    cr_y = 0.335
    ax.text(rp_x0 + 0.008, cr_y, "CLINICAL RELEVANCE",
            ha="left", va="center", fontsize=2.4, color="#666",
            fontweight="bold", transform=T)
    ax.text(rp_x0 + 0.008, cr_y - 0.022,
            "Low SCFA producers associated with IBD, colorectal cancer,",
            ha="left", va="center", fontsize=2.1, color="#555", transform=T)
    ax.text(rp_x0 + 0.008, cr_y - 0.040,
            "obesity, and metabolic syndrome.",
            ha="left", va="center", fontsize=2.1, color="#555", transform=T)

    # ── ABUNDANCE BY DISEASE heatmap (left) + DISEASE VS STRICT NC (right) ──
    hm_title_y = 0.258
    ax.text(rp_x0 + 0.008, hm_title_y, "ABUNDANCE BY DISEASE",
            ha="left", va="center", fontsize=2.2, color="#666",
            fontweight="bold", transform=T)
    # heatmap on left half
    hm_w = 0.310
    hm_ax = ax.inset_axes([rp_x0 + 0.080, 0.040, hm_w, 0.180])
    _ui_bg(hm_ax, "white")
    hm_genera = ["Faecalibacterium", "Roseburia", "Bifidobacterium",
                 "Akkermansia", "Ruminococcus"]
    hm_diseases = ["CD", "UC", "CRC", "IBS", "CDI", "T2D", "Obes.",
                   "NAFLD", "PD", "MS", "RA", "Autism"]
    rng_hm = np.random.default_rng(7)
    hm_data = rng_hm.uniform(0, 1, (len(hm_genera), len(hm_diseases))) ** 1.5
    hm_data[0, :4] *= 2.2
    hm_data[2, 3] *= 2.0
    cmap = LinearSegmentedColormap.from_list("mg", ["white", MAGENTA], N=256)
    hm_ax.imshow(hm_data, cmap=cmap, aspect="auto", vmin=0, vmax=2)
    hm_ax.set_xticks(np.arange(len(hm_diseases)))
    hm_ax.set_xticklabels(hm_diseases, fontsize=1.6, rotation=40,
                          ha="right", rotation_mode="anchor")
    hm_ax.set_yticks(np.arange(len(hm_genera)))
    hm_ax.set_yticklabels(hm_genera, fontsize=1.6, fontstyle="italic")
    hm_ax.tick_params(axis="both", length=0, pad=1)
    for sp in hm_ax.spines.values():
        sp.set_linewidth(0.3); sp.set_color(MGRAY)

    # DISEASE VS STRICT NC lollipop on right half
    dc_title_x = rp_x0 + 0.420
    ax.text(dc_title_x, hm_title_y, "DISEASE VS STRICT NC",
            ha="left", va="center", fontsize=2.2, color="#666",
            fontweight="bold", transform=T)
    lg_y = hm_title_y - 0.024
    ax.scatter([dc_title_x + 0.004], [lg_y], s=5, c=MAGENTA,
               transform=T, zorder=5)
    ax.text(dc_title_x + 0.012, lg_y, "↑disease",
            ha="left", va="center", fontsize=2.0, color="#666", transform=T)
    ax.scatter([dc_title_x + 0.070], [lg_y], s=5, c="#5B6FE5",
               transform=T, zorder=5)
    ax.text(dc_title_x + 0.078, lg_y, "↑NC",
            ha="left", va="center", fontsize=2.0, color="#666", transform=T)

    dc_ax = ax.inset_axes([dc_title_x + 0.080, 0.040, 0.230, 0.180])
    _ui_bg(dc_ax, "white")
    dc_diseases = ["Lean ctrl", "HIV-exp. uninf.", "ICU pt.", "AML",
                   "HSCT", "Preterm inf.", "LTAC pt.", "Non-CDI diar.",
                   "HIV inf.", "CRC", "Helminth", "CDI"]
    dc_lfc = [-2.1, -1.7, -1.5, -1.35, -1.30, -1.25,
              -1.15, -1.05, -1.31, -0.85, 0.35, 0.28]
    dc_colors = [MAGENTA if v > 0 else "#5B6FE5" for v in dc_lfc]
    ypos = np.arange(len(dc_diseases))[::-1]
    for yi, v, col in zip(ypos, dc_lfc, dc_colors):
        dc_ax.plot([0, v], [yi, yi], color=col, lw=0.7)
        dc_ax.scatter([v], [yi], s=5, c=col, edgecolors="white", linewidths=0.2)
    dc_ax.axvline(0, color="#888", lw=0.3, linestyle=(0, (1, 1)))
    dc_ax.set_xlim(-2.6, 1.0)
    dc_ax.set_ylim(-0.6, len(dc_diseases) - 0.4)
    dc_ax.set_yticks(ypos)
    dc_ax.set_yticklabels(dc_diseases, fontsize=1.4, color=INK)
    dc_ax.set_xticks([])
    dc_ax.tick_params(axis="y", length=0, pad=1)
    for sp in dc_ax.spines.values():
        sp.set_visible(False)

    # key references footer
    ax.text(rp_x0 + 0.008, -0.004, "KEY REFS:",
            ha="left", va="center", fontsize=2.0, color="#888",
            fontweight="bold", transform=T)
    ax.text(rp_x0 + 0.078, -0.004, "PMID:26963409 · PMID:27259147",
            ha="left", va="center", fontsize=2.0, color=NAVY,
            transform=T)

def _thumb_differential(ax):
    """Differential Analysis — canonical v2 style, all-in-one workspace."""
    _ui_bg(ax, "white")
    T = ax.transAxes
    NAVY = "#1F3864"; RED = "#C41435"; INK = "#1A1A1A"
    MGRAY = "#BCC6D6"; LGRAY = "#EEF1F6"
    MAGENTA = "#C735C7"; BLUE = "#5B6FE5"

    # ── title ────────────────────────────────────────────────────────────────
    ax.text(0.025, 0.968, "Differential Analysis", ha="left", va="center",
            fontsize=5.6, color=INK, fontweight="bold", transform=T)
    ax.text(0.025, 0.935,
            "Sample size, differential taxa, alpha/beta diversity, composition, cross-study evidence",
            ha="left", va="center", fontsize=3.0, color="#888", transform=T)

    # ── Group A / Group B picker cards ───────────────────────────────────────
    gp_y0 = 0.790; gp_h = 0.110
    gp_w = 0.462; gp_gap = 0.026
    groups = [("A", "China", "186"), ("B", "Ireland", "109")]
    for gi, (gname, country, n_s) in enumerate(groups):
        gx0 = 0.025 + gi * (gp_w + gp_gap)
        ax.add_patch(FancyBboxPatch((gx0, gp_y0), gp_w, gp_h,
            boxstyle="round,pad=0.002,rounding_size=0.006",
            facecolor="white", edgecolor=MAGENTA, lw=0.6,
            transform=T, zorder=3))
        ax.text(gx0 + 0.008, gp_y0 + gp_h - 0.012, f"Group {gname}",
                ha="left", va="center", fontsize=3.0, color=MAGENTA,
                fontweight="bold", fontstyle="italic", transform=T)
        ax.text(gx0 + gp_w - 0.008, gp_y0 + gp_h - 0.012, f"n = {n_s}",
                ha="right", va="center", fontsize=2.6, color="#888", transform=T)
        fields = [("Country", country), ("Disease", "CD"),
                  ("Age group", "-- Any --"), ("Sex", "-- Any --")]
        for fi, (flbl, fval) in enumerate(fields):
            fy = gp_y0 + gp_h - 0.034 - fi * 0.018
            ax.text(gx0 + 0.012, fy, flbl, ha="left", va="center",
                    fontsize=2.0, color="#666", transform=T)
            dd_x = gx0 + 0.110
            dd_w = gp_w - 0.122
            ax.add_patch(FancyBboxPatch((dd_x, fy - 0.007), dd_w, 0.014,
                boxstyle="round,pad=0.001,rounding_size=0.003",
                facecolor="#F7F8FB", edgecolor=MGRAY, lw=0.3,
                transform=T, zorder=4))
            ax.text(dd_x + 0.006, fy, fval, ha="left", va="center",
                    fontsize=1.9, color=INK, transform=T, zorder=5)
            ax.text(dd_x + dd_w - 0.006, fy, "v", ha="right", va="center",
                    fontsize=1.9, color="#888", transform=T, zorder=5)

    # ── TAXONOMY / TEST pills + Run Analysis ─────────────────────────────────
    row_y = 0.770
    tax_pills = [("genus", True), ("family", False), ("phylum", False)]
    px = 0.025
    for lbl, sel in tax_pills:
        pw = 0.008 + len(lbl) * 0.0088
        ax.add_patch(FancyBboxPatch((px, row_y - 0.010), pw, 0.018,
            boxstyle="round,pad=0.001,rounding_size=0.009",
            facecolor=MAGENTA if sel else "white",
            edgecolor=MAGENTA if sel else MGRAY, lw=0.4,
            transform=T, zorder=3))
        ax.text(px + pw/2, row_y - 0.001, lbl, ha="center", va="center",
                fontsize=2.2, color="white" if sel else INK,
                fontweight="bold" if sel else "normal", transform=T, zorder=4)
        px += pw + 0.005

    test_x0 = 0.240
    test_pills = [("wilcoxon", True), ("t-test", False),
                  ("LEfSe", False), ("PERMANOVA", False)]
    px = test_x0
    for lbl, sel in test_pills:
        pw = 0.008 + len(lbl) * 0.0088
        ax.add_patch(FancyBboxPatch((px, row_y - 0.010), pw, 0.018,
            boxstyle="round,pad=0.001,rounding_size=0.009",
            facecolor=MAGENTA if sel else "white",
            edgecolor=MAGENTA if sel else MGRAY, lw=0.4,
            transform=T, zorder=3))
        ax.text(px + pw/2, row_y - 0.001, lbl, ha="center", va="center",
                fontsize=2.2, color="white" if sel else INK,
                fontweight="bold" if sel else "normal", transform=T, zorder=4)
        px += pw + 0.005

    # A/B counts chips
    for lbl in ["A: 186", "B: 109"]:
        pw = 0.052
        ax.add_patch(FancyBboxPatch((px, row_y - 0.010), pw, 0.018,
            boxstyle="round,pad=0.001,rounding_size=0.009",
            facecolor=LGRAY, edgecolor=MGRAY, lw=0.4,
            transform=T, zorder=3))
        ax.text(px + pw/2, row_y - 0.001, lbl, ha="center", va="center",
                fontsize=2.0, color="#555", transform=T, zorder=4)
        px += pw + 0.005

    # Run Analysis CTA (right-aligned)
    cta_w = 0.118
    cta_x = 0.975 - cta_w
    ax.add_patch(FancyBboxPatch((cta_x, row_y - 0.011), cta_w, 0.020,
        boxstyle="round,pad=0.001,rounding_size=0.010",
        facecolor=MAGENTA, edgecolor="none", transform=T, zorder=3))
    ax.text(cta_x + cta_w/2, row_y - 0.001, "Run Analysis",
            ha="center", va="center", fontsize=2.6, color="white",
            fontweight="bold", transform=T, zorder=4)

    # ── 4 stat boxes ─────────────────────────────────────────────────────────
    stat_y = 0.682; sh = 0.068
    stats = [
        ("WORKSPACE SUMMARY", "Cn-CD vs Ie-CD", NAVY, 3.6),
        ("MATCHED SAMPLES",   "186 / 109",      NAVY, 4.4),
        ("TOTAL GENERA",      "3,143",          NAVY, 4.6),
        ("SIGNIFICANT GENERA","213",            MAGENTA, 4.8),
    ]
    stat_gap = 0.012
    sw = (0.950 - 3 * stat_gap) / 4
    for i, (lbl, num, col, nfs) in enumerate(stats):
        sx = 0.025 + i * (sw + stat_gap)
        ax.add_patch(FancyBboxPatch((sx, stat_y), sw, sh,
            boxstyle="round,pad=0.001,rounding_size=0.004",
            facecolor="white", edgecolor=MGRAY, lw=0.4, transform=T, zorder=3))
        ax.add_patch(Rectangle((sx + 0.003, stat_y + sh - 0.005),
            sw - 0.006, 0.004, facecolor=col, edgecolor="none",
            transform=T, zorder=4))
        ax.text(sx + sw/2, stat_y + sh * 0.72, lbl,
                ha="center", va="center", fontsize=2.3,
                color="#666", transform=T, zorder=4)
        ax.text(sx + sw/2, stat_y + sh * 0.28, num,
                ha="center", va="center", fontsize=nfs,
                color=INK, fontweight="bold", transform=T, zorder=4)
        if i == 3:
            ax.text(sx + sw/2, stat_y + sh * 0.05, "adj p < 0.05",
                    ha="center", va="center", fontsize=1.6,
                    color="#888", transform=T, zorder=4)

    # ── tabs row (8 tabs, underline style) ───────────────────────────────────
    tab_y = 0.638
    tabs = [("Differential Abundance", True), ("Volcano", False),
            ("Alpha Div.", False), ("Beta PCoA", False),
            ("Composition", False), ("Heatmap", False),
            ("Spearman", False), ("Meta-Analysis", False)]
    tx = 0.025
    for tlbl, sel in tabs:
        tw = 0.006 + len(tlbl) * 0.0088
        if sel:
            ax.add_patch(Rectangle((tx, tab_y - 0.010), tw, 0.0025,
                facecolor=MAGENTA, edgecolor="none", transform=T, zorder=4))
        ax.text(tx + tw/2, tab_y, tlbl, ha="center", va="center",
                fontsize=2.2, color=MAGENTA if sel else "#666",
                fontweight="bold" if sel else "normal", transform=T)
        tx += tw + 0.010

    # ── main content: 2×3 grid — Volcano | Heatmap | Alpha  /  LEfSe | Beta | Spearman
    col_w = 0.295
    col_x = [0.030, 0.345, 0.660]
    row_y = [0.320, 0.050]
    row_h = 0.250
    title_y_top = 0.586
    title_y_bot = 0.316

    # Volcano (top-left)
    v_ax = ax.inset_axes([col_x[0] - 0.018, row_y[0] + 0.018, col_w, row_h])
    _ui_bg(v_ax, "white")
    rng_v = np.random.default_rng(11)
    n = 380
    fc_v = rng_v.normal(0, 2.0, n)
    pv_v = rng_v.uniform(0.2, 3.5, n) + np.abs(fc_v) * 0.6
    ns_mask = ~((np.abs(fc_v) > 1.0) & (pv_v > 1.3))
    v_ax.scatter(fc_v[ns_mask], pv_v[ns_mask], s=3, c="#BBBBBB",
                 alpha=0.55, edgecolors="none")
    sig_pos = ~ns_mask & (fc_v > 0)
    sig_neg = ~ns_mask & (fc_v < 0)
    v_ax.scatter(fc_v[sig_pos], pv_v[sig_pos], s=6, c=BLUE,
                 alpha=0.85, edgecolors="none")
    v_ax.scatter(fc_v[sig_neg], pv_v[sig_neg], s=6, c=MAGENTA,
                 alpha=0.85, edgecolors="none")
    v_ax.axhline(1.3, color="#888", lw=0.4, linestyle=(0, (2, 2)))
    v_ax.axvline(1.0, color="#888", lw=0.4, linestyle=(0, (2, 2)))
    v_ax.axvline(-1.0, color="#888", lw=0.4, linestyle=(0, (2, 2)))
    v_labels = [(-3.6, 4.7, "Faecalibact.", MAGENTA),
                (-2.7, 4.1, "Roseburia",    MAGENTA),
                (-2.2, 3.3, "Blautia",      MAGENTA),
                ( 2.6, 4.5, "Fusobact.",    BLUE),
                ( 3.3, 3.9, "Escherichia",  BLUE),
                ( 2.1, 3.2, "Bacteroides",  BLUE)]
    for x2, y2, lbl, c in v_labels:
        v_ax.scatter([x2], [y2], s=12, c=c, edgecolors="white",
                     linewidths=0.3, zorder=5)
        v_ax.text(x2 + 0.15, y2 + 0.10, lbl, fontsize=1.9,
                  color=c, fontstyle="italic", ha="left", va="bottom")
    v_ax.set_xlabel("")
    ax.text(col_x[0] - 0.018 + col_w/2, row_y[0] + 0.018 - 0.044,
            "log2 fold change", ha="center", va="top",
            fontsize=2.2, color="#666", transform=T)
    v_ax.set_ylabel("-log10(padj)", fontsize=2.2, color="#666", labelpad=1)
    v_ax.set_xlim(-5, 5); v_ax.set_ylim(0, 5.4)
    v_ax.tick_params(labelsize=1.7, length=1, pad=1, colors="#888")
    for sp in v_ax.spines.values():
        sp.set_linewidth(0.3); sp.set_color(MGRAY)
    ax.text(col_x[0] + col_w/2, title_y_top, "Volcano · Cn-CD vs Ie-CD",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=1.2))

    # ── Differential Heatmap (top-center, 3 columns) ─────────────────────────
    # shrink slightly inside its grid cell
    _hm_pad_x = 0.022
    _hm_pad_y = 0.022
    _hm_shift_x = 0.044
    _hm_shift_y = 0.020
    h_ax = ax.inset_axes([col_x[1] + _hm_pad_x + _hm_shift_x,
                          row_y[0] + _hm_pad_y + _hm_shift_y,
                          col_w - 2 * _hm_pad_x, row_h - 2 * _hm_pad_y])
    _ui_bg(h_ax, "white")
    # Real values from /api/diff-analysis (CN-CD vs IE-CD, genus, wilcoxon)
    hm_g = ["Fusobact.","Acinetobact.","Brevundimonas","Hydrogenoph.",
            "Anaerostipes","Acidovorax","Agrobact.","Klebsiella"]
    hm_data = np.array([
        [ 8.86,  0.50,  4.15],
        [ 2.26,  0.00, 10.22],
        [ 0.45,  0.00, 10.44],
        [ 0.91,  0.00, 19.80],
        [ 0.06,  4.12, -6.10],
        [ 0.76,  0.01,  6.30],
        [ 0.17,  0.00, 17.41],
        [ 0.52,  0.49,  0.09],
    ])
    cmap_h = LinearSegmentedColormap.from_list(
        "mb", [MAGENTA, "white", BLUE], N=128)
    # normalize each column to [-1,1] for consistent color
    hm_norm = np.zeros_like(hm_data)
    for cj in range(3):
        col = hm_data[:, cj]
        mx = max(abs(col.max()), abs(col.min()), 1e-6)
        hm_norm[:, cj] = col / mx
    h_ax.imshow(hm_norm, cmap=cmap_h, aspect="auto", vmin=-1, vmax=1)
    h_ax.set_xticks(np.arange(3))
    h_ax.set_xticklabels(["Mean A", "Mean B", "log2FC"],
                         fontsize=1.4, color="#666")
    h_ax.set_yticks(np.arange(len(hm_g)))
    h_ax.set_yticklabels(hm_g, fontsize=1.7, fontstyle="italic", color=INK)
    h_ax.tick_params(axis="both", length=0, pad=1)
    for sp in h_ax.spines.values():
        sp.set_linewidth(0.3); sp.set_color(MGRAY)
    for i in range(len(hm_g)):
        for j in range(3):
            v = hm_data[i, j]
            nv = hm_norm[i, j]
            h_ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                      fontsize=1.1,
                      color="white" if abs(nv) > 0.55 else INK,
                      fontweight="bold")
    ax.text(col_x[1] + col_w/2, title_y_top, "Differential Heatmap · top 8",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T)

    # ── Alpha diversity (top-right): Shannon / Simpson / Chao1 ───────────────
    a_ax = ax.inset_axes([col_x[2], row_y[0], col_w, row_h])
    _ui_bg(a_ax, "white")
    # three metrics at different scales — draw as three mini-stacks with
    # per-column normalized jitter so shapes are comparable
    metrics = [("Shannon", 2.249, 2.266, "p=0.681"),
               ("Simpson", 0.758, 0.806, "p=0.002"),
               ("Chao1",   93.9,  40.9,  "p<1e-300")]
    rng_a = np.random.default_rng(23)
    for mi, (mname, ma, mb, pv) in enumerate(metrics):
        base_x = mi * 3 + 0.5
        # normalize visually so columns share a 0–1 y band
        hi = max(ma, mb) * 1.35
        lo = 0.0
        def _ny(v): return (v - lo) / (hi - lo)
        sd_a = 0.10 * ma + 0.02
        sd_b = 0.10 * mb + 0.02
        a_pts = [_ny(v) for v in rng_a.normal(ma, sd_a, 70)]
        b_pts = [_ny(v) for v in rng_a.normal(mb, sd_b, 70)]
        a_ax.scatter(np.full(70, base_x) + rng_a.uniform(-0.22, 0.22, 70),
                     a_pts, s=1.2, c=BLUE, alpha=0.45, edgecolors="none")
        a_ax.scatter(np.full(70, base_x + 1.0) + rng_a.uniform(-0.22, 0.22, 70),
                     b_pts, s=1.2, c=MAGENTA, alpha=0.45, edgecolors="none")
        a_ax.plot([base_x - 0.28, base_x + 0.28], [_ny(ma)]*2, color=BLUE, lw=1.1)
        a_ax.plot([base_x + 0.72, base_x + 1.28], [_ny(mb)]*2, color=MAGENTA, lw=1.1)
        y_br = 0.92
        a_ax.plot([base_x, base_x + 1.0], [y_br, y_br], color="#666", lw=0.4)
        a_ax.text(base_x + 0.5, y_br + 0.02, pv, ha="center", va="bottom",
                  fontsize=1.5, color="#333")
        a_ax.text(base_x + 0.5, 0.02, mname, ha="center", va="top",
                  fontsize=1.7, color="#333", fontweight="bold")
        # mean-value annotations
        a_ax.text(base_x, _ny(ma) - 0.06, f"{ma:.2f}" if ma < 10 else f"{ma:.0f}",
                  ha="center", va="top", fontsize=1.3, color=BLUE)
        a_ax.text(base_x + 1.0, _ny(mb) - 0.06, f"{mb:.2f}" if mb < 10 else f"{mb:.0f}",
                  ha="center", va="top", fontsize=1.3, color=MAGENTA)
    a_ax.set_xlim(-0.3, 8.3); a_ax.set_ylim(-0.15, 1.05)
    a_ax.set_xticks([]); a_ax.set_yticks([])
    for sp in a_ax.spines.values():
        sp.set_linewidth(0.3); sp.set_color(MGRAY)
    ax.text(col_x[2] + col_w/2, title_y_top, "Alpha diversity",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T)

    # ── LEfSe bar (bottom-left) ──────────────────────────────────────────────
    l_ax = ax.inset_axes([col_x[0], row_y[1] - 0.024, col_w, row_h])
    _ui_bg(l_ax, "white")
    # real LEfSe top hits (Cn-CD=A vs Ie-CD=B); positive = enriched in A (Cn-CD)
    lfse_rows = [
        ("Fusobact.",     6.09, "A"),
        ("Halomonas",     5.43, "A"),
        ("Acinetobact.",  5.19, "A"),
        ("Shewanella",    4.98, "A"),
        ("Blautia",       6.23, "B"),
        ("Anaerostipes",  6.07, "B"),
        ("Faecalibact.",  5.96, "B"),
        ("Mediterranei.", 5.50, "B"),
    ]
    # sort so A on top positive, B on bottom negative
    lfse_rows = sorted(lfse_rows, key=lambda r: (r[2], -r[1]))
    lfse_g = [r[0] for r in lfse_rows]
    lfse_v = [r[1] if r[2] == "A" else -r[1] for r in lfse_rows]
    lfse_c = [BLUE if v > 0 else MAGENTA for v in lfse_v]
    l_ax.barh(np.arange(len(lfse_g)), lfse_v, color=lfse_c,
              height=0.72, edgecolor="white", linewidth=0.3)
    l_ax.axvline(0, color="#666", lw=0.4)
    l_ax.set_yticks(np.arange(len(lfse_g)))
    l_ax.set_yticklabels(lfse_g, fontsize=1.5, fontstyle="italic")
    l_ax.set_xlabel("LDA score  (+ Cn-CD / − Ie-CD)", fontsize=1.6,
                    color="#666", labelpad=1)
    l_ax.set_xticks([-6, -3, 0, 3, 6])
    l_ax.set_xlim(-7, 7)
    l_ax.tick_params(labelsize=1.4, length=1, pad=1, colors="#888")
    for sp in l_ax.spines.values():
        sp.set_linewidth(0.3); sp.set_color(MGRAY)
    ax.text(col_x[0] + col_w/2, title_y_bot, "LEfSe · |LDA| ≥ 2",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T)

    # ── Beta diversity PCoA (bottom-center) ──────────────────────────────────
    b_ax = ax.inset_axes([col_x[1], row_y[1], col_w, row_h])
    _ui_bg(b_ax, "white")
    rng_b = np.random.default_rng(31)
    # Cn-CD cluster
    na = 186
    pa1 = rng_b.normal(-0.05, 0.18, na)
    pa2 = rng_b.normal(0.02, 0.14, na)
    # Ie-CD cluster
    nb = 109
    pb1 = rng_b.normal(0.08, 0.16, nb)
    pb2 = rng_b.normal(-0.03, 0.13, nb)
    b_ax.scatter(pa1, pa2, s=2.5, c=BLUE, alpha=0.65, edgecolors="none")
    b_ax.scatter(pb1, pb2, s=2.5, c=MAGENTA, alpha=0.65, edgecolors="none")
    # 95% ellipses
    from matplotlib.patches import Ellipse
    b_ax.add_patch(Ellipse((-0.05, 0.02), 0.72, 0.56, angle=10,
        facecolor="none", edgecolor=BLUE, lw=0.6, alpha=0.8))
    b_ax.add_patch(Ellipse((0.08, -0.03), 0.64, 0.52, angle=-5,
        facecolor="none", edgecolor=MAGENTA, lw=0.6, alpha=0.8))
    b_ax.set_xlabel("PCo1 (braycurtis) 18.41%",
                    fontsize=1.6, color="#666", labelpad=1)
    b_ax.set_ylabel("PCo2 (braycurtis) 13.25%",
                    fontsize=1.6, color="#666", labelpad=1)
    b_ax.set_xlim(-0.55, 0.55); b_ax.set_ylim(-0.45, 0.45)
    b_ax.tick_params(labelsize=1.4, length=1, pad=1, colors="#888")
    for sp in b_ax.spines.values():
        sp.set_linewidth(0.3); sp.set_color(MGRAY)
    # legend
    b_ax.scatter([-0.50], [0.40], s=4, c=BLUE)
    b_ax.text(-0.46, 0.40, "Cn-CD (n=186)", fontsize=1.4,
              color="#555", va="center")
    b_ax.scatter([-0.50], [0.32], s=4, c=MAGENTA)
    b_ax.text(-0.46, 0.32, "Ie-CD (n=109)", fontsize=1.4,
              color="#555", va="center")
    ax.text(col_x[1] + col_w/2, title_y_bot,
            "Beta PCoA · braycurtis · PERMANOVA p<0.001",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T)

    # ── Spearman correlation matrix (bottom-right) ───────────────────────────
    s_ax = ax.inset_axes([col_x[2], row_y[1], col_w, row_h])
    _ui_bg(s_ax, "white")
    # real Spearman matrix from /api/spearman-analysis (n=295, genus level)
    sp_genera = ["Bacteroides","Shigella","Fusobact.","Faecalibact.",
                 "Blautia","Sutterella","Segatella","Enteroclost.",
                 "Anaerost.","Phascolarct.","Halomonas","Acinetobact."]
    sp_mat = np.array([
        [ 1.00,-0.04, 0.16,-0.01, 0.05, 0.24,-0.32, 0.47, 0.09, 0.27,-0.13,-0.28],
        [-0.04, 1.00, 0.30,-0.38,-0.31,-0.17, 0.05, 0.13,-0.22, 0.00, 0.14, 0.21],
        [ 0.16, 0.30, 1.00,-0.31,-0.46,-0.04, 0.25, 0.17,-0.38, 0.31, 0.27, 0.46],
        [-0.01,-0.38,-0.31, 1.00, 0.46, 0.14, 0.08,-0.18, 0.45, 0.05,-0.07,-0.25],
        [ 0.05,-0.31,-0.46, 0.46, 1.00, 0.21,-0.17, 0.16, 0.60,-0.12,-0.14,-0.36],
        [ 0.24,-0.17,-0.04, 0.14, 0.21, 1.00,-0.06, 0.19, 0.27, 0.11, 0.06,-0.22],
        [-0.32, 0.05, 0.25, 0.08,-0.17,-0.06, 1.00,-0.20,-0.25, 0.17, 0.28, 0.49],
        [ 0.47, 0.13, 0.17,-0.18, 0.16, 0.19,-0.20, 1.00, 0.18, 0.32, 0.00,-0.19],
        [ 0.09,-0.22,-0.38, 0.45, 0.60, 0.27,-0.25, 0.18, 1.00,-0.06,-0.19,-0.41],
        [ 0.27, 0.00, 0.31, 0.05,-0.12, 0.11, 0.17, 0.32,-0.06, 1.00, 0.11, 0.02],
        [-0.13, 0.14, 0.27,-0.07,-0.14, 0.06, 0.28, 0.00,-0.19, 0.11, 1.00, 0.48],
        [-0.28, 0.21, 0.46,-0.25,-0.36,-0.22, 0.49,-0.19,-0.41, 0.02, 0.48, 1.00],
    ])
    ns = len(sp_genera)
    cmap_s = LinearSegmentedColormap.from_list(
        "gr", ["#C41435", "white", "#2BB573"], N=128)
    s_ax.imshow(sp_mat, cmap=cmap_s, aspect="auto", vmin=-1, vmax=1)
    s_ax.set_xticks(np.arange(ns))
    s_ax.set_xticklabels(sp_genera, fontsize=1.2, rotation=50,
                         ha="right", rotation_mode="anchor",
                         fontstyle="italic", color=INK)
    s_ax.set_yticks(np.arange(ns))
    s_ax.set_yticklabels(sp_genera, fontsize=1.2, fontstyle="italic", color=INK)
    s_ax.tick_params(axis="both", length=0, pad=1)
    for sp in s_ax.spines.values():
        sp.set_linewidth(0.3); sp.set_color(MGRAY)
    ax.text(col_x[2] + col_w/2, title_y_bot,
            "Spearman · 295 matched samples",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T)

def _thumb_meta(ax):
    """Cross-Study Meta-Analysis — canonical v2 style.

    Real data from POST /api/cross-study with 7 projects + disease=CD +
    method=wilcoxon + taxonomy_level=genus.  Top-10 consensus markers
    (n_significant=10, I² range 0–93), 10 markers × 7 projects log2FC
    matrix is the actual per_project output (see _meta.json).
    """
    _ui_bg(ax, "white")
    NAVY = "#1F3864"; RED = "#C41435"; INK = "#1A1A1A"
    MGRAY = "#BCC6D6"; LGRAY = "#EEF1F6"
    MAGENTA = "#C735C7"; BLUE = "#5B6FE5"; GREEN = "#2BB573"
    T = ax.transAxes

    # ── title + subtitle ────────────────────────────────────────────────
    ax.text(0.500, 0.962, "Cross-Study Meta-Analysis",
            ha="center", va="center", fontsize=5.6, color=INK,
            fontweight="bold", transform=T)
    ax.text(0.500, 0.922,
            "Cross-study meta-analysis at the genus level with "
            "consistency, heterogeneity, and consensus markers",
            ha="center", va="center", fontsize=2.5, color="#666", transform=T)

    # ── target disease + project search line ────────────────────────────
    ax.text(0.030, 0.878, "Target disease",
            ha="left", va="center", fontsize=2.4, color="#666", transform=T)
    ax.add_patch(FancyBboxPatch((0.128, 0.862), 0.056, 0.030,
        boxstyle="round,pad=0.002,rounding_size=0.006",
        facecolor=NAVY, edgecolor=NAVY, lw=0.4, transform=T, zorder=3))
    ax.text(0.156, 0.878, "CD", ha="center", va="center",
            fontsize=2.5, color="white", fontweight="bold", transform=T)
    ax.text(0.220, 0.878, "Project search",
            ha="left", va="center", fontsize=2.4, color="#666", transform=T)
    ax.add_patch(FancyBboxPatch((0.312, 0.862), 0.500, 0.030,
        boxstyle="round,pad=0.002,rounding_size=0.006",
        facecolor="white", edgecolor=MGRAY, lw=0.4, transform=T, zorder=3))
    ax.text(0.320, 0.878, "Filter by project ID or disease",
            ha="left", va="center", fontsize=2.2, color="#999",
            fontstyle="italic", transform=T)
    ax.text(0.826, 0.878, "12 eligible projects",
            ha="left", va="center", fontsize=2.3, color="#888",
            fontstyle="italic", transform=T)

    # ── project cards row (7 purple selected + 5 gray unselected) ──────
    # Real n_disease / n_control values from /api/cross-study output
    projs_sel = [
        ("PRJNA237362", 350, 177),
        ("PRJNA350277",  11, 553),
        ("PRJNA398187", 109,  64),
        ("PRJNA431126", 186,  38),
        ("PRJNA450540", 234,  22),
        ("PRJNA450340",  42,  46),
        ("PRJEB42155",  119,  19),
    ]
    projs_unsel = [
        ("PRJNA399038", 125),
        ("PRJNA524825", 116),
        ("PRJNA684564", 102),
    ]
    pr_y0 = 0.790; pr_h = 0.062
    pr_x0 = 0.026; pr_x1 = 0.982
    pr_gap = 0.005
    pr_total = len(projs_sel) + len(projs_unsel)
    pr_w = (pr_x1 - pr_x0 - (pr_total - 1) * pr_gap) / pr_total
    for i in range(pr_total):
        px = pr_x0 + i * (pr_w + pr_gap)
        sel = i < len(projs_sel)
        if sel:
            pid, nd, nc = projs_sel[i]
            face = "#B29BE8"; edge = "#8A6FD4"
            txt = "white"; sub_txt = "#F1E8FF"
            sub_label = f"D>{nd}  C>{nc}"
        else:
            pid, n = projs_unsel[i - len(projs_sel)]
            face = "#E5E7EB"; edge = "#C9CDD4"
            txt = "#555"; sub_txt = "#777"
            sub_label = f"n={n}"
        ax.add_patch(FancyBboxPatch((px, pr_y0), pr_w, pr_h,
            boxstyle="round,pad=0.002,rounding_size=0.005",
            facecolor=face, edgecolor=edge, lw=0.5, transform=T, zorder=3))
        ax.text(px + pr_w/2, pr_y0 + pr_h*0.68, pid,
                ha="center", va="center", fontsize=1.8,
                color=txt, fontweight="bold", transform=T)
        ax.text(px + pr_w/2, pr_y0 + pr_h*0.28, sub_label,
                ha="center", va="center", fontsize=1.5,
                color=sub_txt, transform=T)

    # ── summary row: Projects / Disease / Level / Consensus markers  CTA
    sm_y = 0.750
    ax.text(0.030, sm_y, "Projects", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    ax.text(0.098, sm_y, "7", ha="left", va="center",
            fontsize=2.6, color=INK, fontweight="bold", transform=T)
    ax.text(0.170, sm_y, "Disease", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    ax.add_patch(FancyBboxPatch((0.232, sm_y - 0.014), 0.150, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor="white", edgecolor=NAVY, lw=0.5, transform=T, zorder=3))
    ax.text(0.307, sm_y, "Crohn's Disease", ha="center", va="center",
            fontsize=2.1, color=NAVY, transform=T)
    ax.text(0.408, sm_y, "Level", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    ax.add_patch(FancyBboxPatch((0.452, sm_y - 0.014), 0.064, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor="white", edgecolor=NAVY, lw=0.5, transform=T, zorder=3))
    ax.text(0.484, sm_y, "genus", ha="center", va="center",
            fontsize=2.1, color=NAVY, transform=T)
    ax.text(0.540, sm_y, "Consensus markers", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    ax.text(0.684, sm_y, "10", ha="left", va="center",
            fontsize=2.6, color=INK, fontweight="bold", transform=T)
    # Run Meta-Analysis CTA
    cta_w = 0.170
    ax.add_patch(FancyBboxPatch((pr_x1 - cta_w, sm_y - 0.016), cta_w, 0.032,
        boxstyle="round,pad=0.002,rounding_size=0.006",
        facecolor=MAGENTA, edgecolor=MAGENTA, lw=0.5, transform=T, zorder=3))
    ax.text(pr_x1 - cta_w/2, sm_y, "Run Meta-Analysis",
            ha="center", va="center", fontsize=2.5, color="white",
            fontweight="bold", transform=T)

    # ── project detail pill strip (7 pills with D/C counts) ────────────
    pp_y = 0.712
    pp_x = 0.030
    pp_x1 = 0.982
    _labels = [f"{pid} D{nd}/C{nc}" for pid, nd, nc in projs_sel]
    _widths = [0.006 + len(lbl) * 0.0062 for lbl in _labels]
    _gap = (pp_x1 - 0.030 - sum(_widths)) / (len(_labels) - 1)
    for lbl, pw in zip(_labels, _widths):
        ax.add_patch(FancyBboxPatch((pp_x, pp_y - 0.013), pw, 0.026,
            boxstyle="round,pad=0.001,rounding_size=0.004",
            facecolor="white", edgecolor=MAGENTA, lw=0.5,
            transform=T, zorder=3))
        ax.text(pp_x + pw/2, pp_y, lbl, ha="center", va="center",
                fontsize=1.7, color=MAGENTA, transform=T)
        pp_x += pw + _gap

    # ── 5 tabs ──────────────────────────────────────────────────────────
    tabs = [("Forest", True), ("Heatmap", False), ("Consistency", False),
            ("Bubble", False), ("Table", False)]
    tab_x = 0.030
    tab_y = 0.672
    for lbl, sel in tabs:
        tw = 0.008 + len(lbl) * 0.0095
        ax.add_patch(FancyBboxPatch((tab_x, tab_y - 0.014), tw, 0.028,
            boxstyle="round,pad=0.002,rounding_size=0.005",
            facecolor="white",
            edgecolor=MAGENTA if sel else MGRAY,
            lw=0.8 if sel else 0.4, transform=T, zorder=3))
        ax.text(tab_x + tw/2, tab_y, lbl, ha="center", va="center",
                fontsize=2.2, color=MAGENTA if sel else INK,
                fontweight="bold" if sel else "normal", transform=T)
        tab_x += tw + 0.008

    # ── main content grid: 2 rows × 3 cols ──────────────────────────────
    col_w = 0.295
    col_x = [0.030, 0.345, 0.660]
    row_y = [0.338, 0.040]
    row_h = 0.272
    title_y_top = 0.624
    title_y_bot = 0.328

    # Real consensus markers data (from /api/cross-study, sorted by meta_p)
    markers = [
        ("Holdemanella",        -1.280, -1.776, -0.784,   8.6,  "3/7"),
        ("Phascolarct.",        -3.209, -4.643, -1.775,  74.7,  "5/7"),
        ("Clostridium",         -4.040, -6.085, -1.995,  77.4,  "6/7"),
        ("Granulicatella",      +1.543, +0.717, +2.369,  48.7,  "0/5"),
        ("Marvinbryantia",      -1.610, -2.499, -0.721,  92.6,  "4/7"),
        ("Senegalimassilia",    -1.661, -2.592, -0.730,  82.8,  "5/7"),
        ("Romboutsia",          -4.485, -7.010, -1.960,  93.2,  "5/7"),
        ("Barnesiella",         -2.526, -3.989, -1.062,  64.6,  "4/7"),
        ("Proteus",             +0.801, +0.334, +1.267,  46.9,  "2/6"),
        ("Lactiplantibacillus", +0.415, +0.173, +0.657,   0.0,  "0/3"),
    ]
    mk_names = [m[0] for m in markers]
    mk_lfc   = [m[1] for m in markers]
    mk_lo    = [m[2] for m in markers]
    mk_hi    = [m[3] for m in markers]
    mk_i2    = [m[4] for m in markers]
    # per-project log2fc (10 × 7)  — real data; None means missing
    per_proj = np.array([
        [-1.68,-1.01,-1.59,-1.42,-1.92,-3.99,-4.22],
        [-2.69,-1.80,-7.21,-2.26,-5.07,-1.03,-6.88],
        [-3.37, 3.24,-4.31,-2.34,-6.61,-8.39,-5.69],
        [ 0.78, np.nan, np.nan, 0.08, 1.29, 2.19, 2.74],
        [-1.59,-0.18,-2.45,-1.06, 0.28,-11.06,-7.21],
        [-0.20,-0.37,-2.61,-1.54,-2.34,-6.83,-4.57],
        [-3.51,-0.38,-3.41,-4.66,-3.76,-10.95,-6.19],
        [-2.67, 3.96,-3.58,-4.03,-3.52,-1.08,-5.13],
        [ 0.44, 1.80, 0.72, 2.84, 0.45, np.nan, 1.51],
        [ 0.28, np.nan, np.nan, 0.46, np.nan, np.nan, 0.52],
    ])

    # ── Forest plot (top-left) ─────────────────────────────────────────
    _fc_dx = 0.040
    f_ax = ax.inset_axes([col_x[0] + _fc_dx, row_y[0], col_w, row_h])
    _ui_bg(f_ax, "white")
    ys = np.arange(len(markers))[::-1]
    for i, (lo, hi, m, i2) in enumerate(zip(mk_lo, mk_hi, mk_lfc, mk_i2)):
        yy = ys[i]
        c = GREEN if m > 0 else BLUE
        f_ax.plot([lo, hi], [yy, yy], color=c, lw=1.8)
        f_ax.scatter([m], [yy], s=14 + (100 - i2) * 0.12,
                     color=c, edgecolors="white", lw=0.8, zorder=5)
    f_ax.axvline(0, color="#AAA", lw=0.6, ls="--", zorder=1)
    f_ax.set_yticks(ys)
    f_ax.set_yticklabels(mk_names, fontsize=2.8, fontstyle="italic", color=INK)
    f_ax.set_xlim(-8, 4)
    f_ax.set_xticks([-6, -3, 0, 3])
    f_ax.set_xlabel("meta log2FC (95% CI)", fontsize=3.2,
                    color="#666", labelpad=1)
    f_ax.tick_params(axis="x", labelsize=2.6, length=1, pad=1, colors="#888")
    f_ax.tick_params(axis="y", length=0, pad=1)
    for sp in f_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    ax.text(col_x[0] + _fc_dx + col_w/2, title_y_top, "Cross-study forest plot",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=1.2))

    # ── Heatmap (top-center) Per-project log2FC ────────────────────────
    _hm_dx = 0.288
    _hm_dy = 0.038
    h_ax = ax.inset_axes([col_x[1] + _hm_dx, row_y[0] + 0.008 + _hm_dy,
                          col_w - 0.024, row_h - 0.024])
    _ui_bg(h_ax, "white")
    cmap_h = LinearSegmentedColormap.from_list(
        "gn_bl", [GREEN, "white", BLUE], N=128)
    # normalize for display: clip to [-8,8]
    disp = np.clip(per_proj, -8, 8) / 8
    h_ax.imshow(disp, cmap=cmap_h, aspect="auto", vmin=-1, vmax=1)
    h_ax.set_xticks(np.arange(7))
    h_ax.set_xticklabels([p[0] for p in projs_sel], fontsize=1.1,
                         rotation=40, ha="right", rotation_mode="anchor",
                         color="#555")
    h_ax.set_yticks(np.arange(len(markers)))
    h_ax.set_yticklabels(mk_names, fontsize=2.8,
                         fontstyle="italic", color=INK)
    h_ax.tick_params(axis="both", length=0, pad=1)
    for sp in h_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    ax.text(col_x[1] + _hm_dx + (col_w - 0.024)/2, title_y_top + _hm_dy,
            "Per-project log2FC heatmap",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=1.2))

    # ── Consistency bars (bottom-left) ─────────────────────────────────
    c_ax = ax.inset_axes([col_x[0] + _fc_dx, row_y[1], col_w, row_h])
    _ui_bg(c_ax, "white")
    # consistency = n_sig / n_studies (use the "3/7" etc strings)
    cons = []
    for m in markers:
        a, b = m[5].split("/")
        cons.append(int(a) / int(b))
    cys = np.arange(len(markers))[::-1]
    cbar_col = [GREEN if m > 0 else BLUE for m in mk_lfc]
    c_ax.barh(cys, cons, height=0.65, color=cbar_col,
              edgecolor="white", linewidth=0.6)
    for yi, (cv, i2) in enumerate(zip(cons, mk_i2)):
        c_ax.text(cv + 0.02, cys[yi],
                  f"{int(cv*100)}% · I²{int(i2)}%",
                  va="center", ha="left", fontsize=2.6, color="#555")
    c_ax.set_yticks(cys)
    c_ax.set_yticklabels(mk_names, fontsize=2.8,
                         fontstyle="italic", color=INK)
    c_ax.set_xlim(0, 1.38)
    c_ax.set_xticks([0, 0.5, 1.0])
    c_ax.set_xticklabels(["0", "50%", "100%"], fontsize=2.6, color="#888")
    c_ax.tick_params(length=1, pad=1)
    for sp in c_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    ax.text(col_x[0] + _fc_dx + col_w/2, title_y_bot, "Cross-study consistency",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=4.5))

    # ── Bubble view (bottom-center, shifted 1× down-left from top-right)
    _bub_dx = 0.288
    _bub_dy = -0.030
    b_ax = ax.inset_axes([col_x[1] + _bub_dx, row_y[1] + _bub_dy, col_w, row_h])
    _ui_bg(b_ax, "white")
    bys = np.arange(len(markers))[::-1]
    # normalized effect for x position in [0.1, 0.9]
    max_abs = max(abs(x) for x in mk_lfc)
    bxs = [0.5 + (m / max_abs) * 0.4 for m in mk_lfc]
    sizes = [50 + abs(m) * 14 for m in mk_lfc]
    bcolor = [GREEN if m > 0 else BLUE for m in mk_lfc]
    b_ax.axvline(0.5, color="#AAA", lw=0.6, ls="--")
    b_ax.scatter(bxs, bys, s=sizes, c=bcolor,
                 edgecolors="white", lw=1.0, zorder=5)
    b_ax.set_yticks(bys)
    b_ax.set_yticklabels(mk_names, fontsize=2.8,
                         fontstyle="italic", color=INK)
    # n= annotation on right
    n_totals = ["n=1970","n=1970","n=1970","n=1233","n=1970",
                "n=1970","n=1970","n=1970","n=1882","n=889"]
    for i, nt in enumerate(n_totals):
        b_ax.text(1.02, bys[i], nt, va="center", ha="left",
                  fontsize=2.6, color="#777")
    b_ax.set_xlim(0, 1.28)
    b_ax.set_xticks([0.1, 0.5, 0.9])
    b_ax.set_xticklabels(["Control", "0", "Disease"], fontsize=2.6, color="#888")
    b_ax.tick_params(length=1, pad=1)
    for sp in b_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    ax.text(col_x[1] + _bub_dx + col_w/2, title_y_bot + _bub_dy, "Effect-size bubble",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=1.2))


def _thumb_network(ax):
    """Network workspace — canonical v2 style.

    Panels sourced from real UI: association bipartite (62 nodes · 180 edges),
    chord arc diagram, SparCC co-occurrence network with hub genera
    (36 nodes · 58 edges · density 0.092 · 10 modules · 4 hubs · 43/15 pos/neg),
    and disease-vs-control rewiring (20 gained · 95 lost · 58/133 edges).
    """
    _ui_bg(ax, "white")
    NAVY = "#1F3864"; RED = "#C41435"; INK = "#1A1A1A"
    MGRAY = "#BCC6D6"; LGRAY = "#EEF1F6"
    MAGENTA = "#C735C7"; BLUE = "#5B6FE5"; GREEN = "#2BB573"
    ORANGE = "#E07830"; TEAL = "#1A6B82"
    T = ax.transAxes

    # ── title + subtitle ────────────────────────────────────────────────
    ax.text(0.500, 0.962, "Network Analysis",
            ha="center", va="center", fontsize=5.6, color=INK,
            fontweight="bold", transform=T)
    ax.text(0.500, 0.922,
            "Association graphs, chord diagrams, co-occurrence structure, "
            "and disease-vs-control rewiring",
            ha="center", va="center", fontsize=2.5, color="#666", transform=T)

    # ── stats tiles row (62 nodes / 180 edges / 12 disease / 50 genus) ──
    tiles = [("Nodes", "62"), ("Edges", "180"),
             ("Disease nodes", "12"), ("Genus nodes", "50")]
    t_x0 = 0.030; t_x1 = 0.982
    t_gap = 0.012
    t_w = (t_x1 - t_x0 - 3 * t_gap) / 4
    for i, (lbl, val) in enumerate(tiles):
        tx = t_x0 + i * (t_w + t_gap)
        ax.add_patch(FancyBboxPatch((tx, 0.852), t_w, 0.056,
            boxstyle="round,pad=0.003,rounding_size=0.006",
            facecolor=LGRAY, edgecolor=MGRAY, lw=0.5,
            transform=T, zorder=3))
        ax.text(tx + 0.012, 0.893, val,
                ha="left", va="center", fontsize=3.6,
                color=INK, fontweight="bold", transform=T)
        ax.text(tx + 0.012, 0.867, lbl,
                ha="left", va="center", fontsize=2.3,
                color="#666", transform=T)

    # ── controls row : disease + min|r| + method + Run CTA ──────────────
    cr_y = 0.792
    ax.text(0.030, cr_y, "Disease", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    ax.add_patch(FancyBboxPatch((0.088, cr_y - 0.014), 0.050, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor=NAVY, edgecolor=NAVY, lw=0.5, transform=T, zorder=3))
    ax.text(0.113, cr_y, "CD", ha="center", va="center",
            fontsize=2.4, color="white", fontweight="bold", transform=T)
    ax.text(0.156, cr_y, "Min |r|", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    ax.add_patch(FancyBboxPatch((0.210, cr_y - 0.014), 0.044, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor="white", edgecolor=NAVY, lw=0.5, transform=T, zorder=3))
    ax.text(0.232, cr_y, "0.3", ha="center", va="center",
            fontsize=2.2, color=NAVY, transform=T)
    ax.text(0.270, cr_y, "Method", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    # method toggle: Spearman | SparCC (selected)
    ax.add_patch(FancyBboxPatch((0.325, cr_y - 0.014), 0.076, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
    ax.text(0.363, cr_y, "Spearman", ha="center", va="center",
            fontsize=2.1, color="#555", transform=T)
    ax.add_patch(FancyBboxPatch((0.405, cr_y - 0.014), 0.068, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor=MAGENTA, edgecolor=MAGENTA, lw=0.5,
        transform=T, zorder=3))
    ax.text(0.439, cr_y, "SparCC", ha="center", va="center",
            fontsize=2.2, color="white", fontweight="bold", transform=T)
    ax.add_patch(FancyBboxPatch((0.480, cr_y - 0.014), 0.068, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
    ax.text(0.514, cr_y, "Compare", ha="center", va="center",
            fontsize=2.1, color="#555", transform=T)
    # Run CTA
    cta_w = 0.170
    ax.add_patch(FancyBboxPatch((t_x1 - cta_w, cr_y - 0.016), cta_w, 0.032,
        boxstyle="round,pad=0.002,rounding_size=0.006",
        facecolor=MAGENTA, edgecolor=MAGENTA, lw=0.5, transform=T, zorder=3))
    ax.text(t_x1 - cta_w/2, cr_y, "Run Network",
            ha="center", va="center", fontsize=2.5, color="white",
            fontweight="bold", transform=T)

    # ── top hub genera pill strip (from co-occurrence panel) ────────────
    hp_y = 0.712
    hp_x = 0.030
    hubs = ["Faecalibacterium", "Blautia", "Ruminococcus",
            "Agathobacter", "Fusicatenibacter", "Anaerostipes",
            "Gemmiger"]
    _hlbls = [f"{h}" for h in hubs]
    _hwid = [0.006 + len(l) * 0.0063 for l in _hlbls]
    _hgap = (t_x1 - 0.030 - sum(_hwid)) / (len(_hlbls) - 1)
    for lbl, pw in zip(_hlbls, _hwid):
        ax.add_patch(FancyBboxPatch((hp_x, hp_y - 0.013), pw, 0.026,
            boxstyle="round,pad=0.001,rounding_size=0.004",
            facecolor="white", edgecolor=MAGENTA, lw=0.5,
            transform=T, zorder=3))
        ax.text(hp_x + pw/2, hp_y, lbl, ha="center", va="center",
                fontsize=1.7, fontstyle="italic",
                color=MAGENTA, transform=T)
        hp_x += pw + _hgap

    # ── 4 tabs ──────────────────────────────────────────────────────────
    tabs = [("Association", False), ("Chord", False),
            ("Co-occurrence", True), ("Disease vs HC", False)]
    tab_x = 0.030
    tab_y = 0.672
    for lbl, sel in tabs:
        tw = 0.008 + len(lbl) * 0.0095
        ax.add_patch(FancyBboxPatch((tab_x, tab_y - 0.014), tw, 0.028,
            boxstyle="round,pad=0.002,rounding_size=0.005",
            facecolor="white",
            edgecolor=MAGENTA if sel else MGRAY,
            lw=0.8 if sel else 0.4, transform=T, zorder=3))
        ax.text(tab_x + tw/2, tab_y, lbl, ha="center", va="center",
                fontsize=2.2, color=MAGENTA if sel else INK,
                fontweight="bold" if sel else "normal", transform=T)
        tab_x += tw + 0.008

    # ── main content grid: 2 rows × 2 cols ──────────────────────────────
    col_w = 0.440
    col_x = [0.040, 0.515]
    row_y = [0.338, 0.040]
    row_h = 0.272
    title_y_top = 0.624
    title_y_bot = 0.328

    # ── Association bipartite (top-left) ─────────────────────────────
    a_ax = ax.inset_axes([col_x[0], row_y[0], col_w, row_h])
    _ui_bg(a_ax, "white")
    dis_labels = ["IBD", "CRC", "T2D", "UC", "CD", "Ob", "CDiff", "HCC"]
    dn = len(dis_labels)
    theta_d = np.linspace(np.pi * 0.5, np.pi * 2.5, dn, endpoint=False)
    dxr, dyr = np.cos(theta_d) * 0.88, np.sin(theta_d) * 0.88
    gen_labels = ["Faecali.", "Blautia", "Rumino.", "Bactero.",
                  "Bifido.", "Prevot.", "Lacto.", "Akker."]
    gn = len(gen_labels)
    theta_g = np.linspace(0, 2 * np.pi, gn, endpoint=False) + np.pi / 8
    gxr, gyr = np.cos(theta_g) * 0.38, np.sin(theta_g) * 0.38
    rng2 = np.random.default_rng(11)
    for di in range(dn):
        targets = rng2.choice(gn, size=3, replace=False)
        for gi in targets:
            w = rng2.uniform(0.3, 1.4)
            pos = rng2.random() < 0.65
            col = RED if not pos else GREEN
            a_ax.plot([dxr[di], gxr[gi]], [dyr[di], gyr[gi]],
                      color=col, lw=w * 0.35, alpha=0.35, zorder=1)
    a_ax.scatter(dxr, dyr, s=22, c=RED, edgecolors="white", lw=0.4,
                 zorder=5)
    for xi, yi, lbl in zip(dxr, dyr, dis_labels):
        a_ax.text(xi * 1.18, yi * 1.18, lbl, fontsize=2.2,
                  ha="center", va="center", color=RED, fontweight="bold")
    a_ax.scatter(gxr, gyr, s=14, c=NAVY, edgecolors="white", lw=0.4,
                 zorder=5)
    for xi, yi, lbl in zip(gxr, gyr, gen_labels):
        a_ax.text(xi * 1.55, yi * 1.55, lbl, fontsize=1.9, ha="center",
                  va="center", color=NAVY, fontstyle="italic")
    a_ax.set_xlim(-1.35, 1.35); a_ax.set_ylim(-1.35, 1.35)
    a_ax.set_aspect("equal")
    a_ax.set_xticks([]); a_ax.set_yticks([])
    for sp in a_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    ax.text(col_x[0] + col_w/2, title_y_top,
            "Association graph  ·  62 nodes · 180 edges",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=2.0))

    # ── Chord diagram (top-right) ───────────────────────────────────
    c_ax = ax.inset_axes([col_x[1], row_y[0], col_w, row_h])
    _ui_bg(c_ax, "white")
    # 14 arcs around a circle
    arc_labels = ["Faecali.", "Blautia", "Rumino.", "Bactero.", "Bifido.",
                  "Prevot.", "IBD", "CRC", "T2D", "UC", "CD", "Ob",
                  "Lacto.", "Akker."]
    arc_colors = [NAVY, NAVY, NAVY, NAVY, NAVY, NAVY,
                  RED, RED, RED, RED, RED, RED,
                  NAVY, NAVY]
    na = len(arc_labels)
    arc_w = 2 * np.pi / na * 0.85
    for i, (lbl, col) in enumerate(zip(arc_labels, arc_colors)):
        th0 = i * 2 * np.pi / na - arc_w / 2 + np.pi / 2
        th1 = th0 + arc_w
        ts = np.linspace(th0, th1, 30)
        r_in, r_out = 0.88, 1.00
        xs = np.concatenate([np.cos(ts) * r_out,
                             np.cos(ts[::-1]) * r_in])
        ys = np.concatenate([np.sin(ts) * r_out,
                             np.sin(ts[::-1]) * r_in])
        c_ax.fill(xs, ys, color=col, alpha=0.85,
                  edgecolor="white", linewidth=0.4, zorder=4)
        tm = (th0 + th1) / 2
        c_ax.text(np.cos(tm) * 1.14, np.sin(tm) * 1.14, lbl,
                  fontsize=1.7, ha="center", va="center",
                  color=col, rotation=np.degrees(tm) - 90,
                  rotation_mode="anchor", fontstyle="italic"
                  if col == NAVY else "normal")
    # draw curved ribbons (bezier-ish with matplotlib Path)
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    pairs = [(0, 7), (0, 9), (1, 8), (1, 10), (2, 6), (3, 11),
             (4, 7), (5, 10), (12, 8), (13, 6), (0, 10), (1, 7)]
    for a, b in pairs:
        tha = a * 2 * np.pi / na + np.pi / 2
        thb = b * 2 * np.pi / na + np.pi / 2
        pa = (np.cos(tha) * 0.86, np.sin(tha) * 0.86)
        pb = (np.cos(thb) * 0.86, np.sin(thb) * 0.86)
        pth = Path([pa, (0, 0), pb],
                   [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        c_ax.add_patch(PathPatch(pth, facecolor="none",
                                 edgecolor=RED, lw=0.5,
                                 alpha=0.35, zorder=2))
    c_ax.set_xlim(-1.42, 1.42); c_ax.set_ylim(-1.32, 1.32)
    c_ax.set_aspect("equal")
    c_ax.set_xticks([]); c_ax.set_yticks([])
    for sp in c_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    ax.text(col_x[1] + col_w/2, title_y_top,
            "Chord diagram  ·  disease ↔ genus ribbons",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=2.0))

    # ── Co-occurrence network with hubs (bottom-left) ───────────────
    co_ax = ax.inset_axes([col_x[0], row_y[1], col_w, row_h])
    _ui_bg(co_ax, "white")
    rng3 = np.random.default_rng(7)
    n_co = 36
    # module assignments (10 modules)
    mods = rng3.integers(0, 10, size=n_co)
    mod_palette = [NAVY, BLUE, GREEN, ORANGE, MAGENTA,
                   TEAL, "#E04545", "#7A5CF5", "#2AA89B", "#B07B2A"]
    # force-directed-ish layout via random circular shells
    angles = rng3.uniform(0, 2 * np.pi, n_co)
    radii = rng3.uniform(0.25, 1.05, n_co)
    px, py = np.cos(angles) * radii, np.sin(angles) * radii
    # 4 hubs — boost their degree (near center) + bigger size
    hub_idx = [0, 8, 17, 26]
    for h in hub_idx:
        px[h] *= 0.45; py[h] *= 0.45
    # edges: 58 total, mostly intra-module, hubs extra-connected
    edges = []
    for _ in range(44):
        i, j = rng3.integers(0, n_co, size=2)
        if i != j:
            edges.append((i, j, True))
    for h in hub_idx:
        for _ in range(3):
            j = rng3.integers(0, n_co)
            if j != h:
                edges.append((h, j, True))
    # some negative edges
    for _ in range(15):
        i, j = rng3.integers(0, n_co, size=2)
        if i != j:
            edges.append((i, j, False))
    for i, j, pos in edges[:58]:
        col = "#3E9B6B" if pos else "#C75C5C"
        co_ax.plot([px[i], px[j]], [py[i], py[j]],
                   color=col, lw=0.35, alpha=0.45, zorder=1)
    # nodes
    sizes = np.full(n_co, 6.0)
    for h in hub_idx:
        sizes[h] = 22.0
    colors = [mod_palette[m] for m in mods]
    co_ax.scatter(px, py, s=sizes, c=colors,
                  edgecolors="white", lw=0.4, zorder=5)
    # hub ring highlight
    co_ax.scatter(px[hub_idx], py[hub_idx], s=42,
                  facecolors="none", edgecolors="#F2C94C",
                  lw=0.7, zorder=6)
    # hub labels
    hub_names = ["Faecali.", "Blautia", "Rumino.", "Agathob."]
    for h, nm in zip(hub_idx, hub_names):
        co_ax.text(px[h], py[h] - 0.20, nm,
                   fontsize=1.8, ha="center", va="top",
                   color=INK, fontstyle="italic", fontweight="bold")
    co_ax.set_xlim(-1.35, 1.35); co_ax.set_ylim(-1.35, 1.35)
    co_ax.set_aspect("equal")
    co_ax.set_xticks([]); co_ax.set_yticks([])
    for sp in co_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    # inline stats strip (top-left of panel)
    co_ax.text(-1.30, 1.22, "36 nodes · 58 edges · ρ=0.092 · 10 modules · 4 hubs",
               fontsize=1.7, ha="left", va="top", color="#666")
    # pos/neg legend (bottom-right)
    co_ax.plot([0.35, 0.60], [-1.18, -1.18], color="#3E9B6B", lw=0.9)
    co_ax.text(0.64, -1.18, "+ 43", fontsize=1.7, ha="left", va="center",
               color="#3E9B6B")
    co_ax.plot([0.88, 1.13], [-1.18, -1.18], color="#C75C5C", lw=0.9)
    co_ax.text(1.17, -1.18, "− 15", fontsize=1.7, ha="left", va="center",
               color="#C75C5C")
    ax.text(col_x[0] + col_w/2, title_y_bot,
            "Co-occurrence network  ·  SparCC · hub genera",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=2.0))

    # ── Disease vs Control rewiring (bottom-right) ──────────────────
    dv_ax = ax.inset_axes([col_x[1], row_y[1], col_w, row_h])
    _ui_bg(dv_ax, "white")
    # draw 2 side-by-side mini networks + stats
    # left = disease (n=2173), right = healthy (n=3066)
    def _mini_net(cx_, cy_, n, seed, is_disease):
        rng_ = np.random.default_rng(seed)
        a = rng_.uniform(0, 2 * np.pi, n)
        r = rng_.uniform(0.05, 0.38, n)
        xs = cx_ + np.cos(a) * r
        ys = cy_ + np.sin(a) * r
        edge_col = "#C75C5C" if is_disease else "#3E9B6B"
        pairs_ = [(rng_.integers(0, n), rng_.integers(0, n))
                  for _ in range(n + 4)]
        for i, j in pairs_:
            if i != j:
                dv_ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                           color=edge_col, lw=0.28, alpha=0.45, zorder=1)
        mods_ = rng_.integers(0, 6, size=n)
        cols_ = [mod_palette[m] for m in mods_]
        dv_ax.scatter(xs, ys, s=5.5, c=cols_,
                      edgecolors="white", lw=0.25, zorder=5)
    _mini_net(-0.55, 0.20, 12, 21, True)
    _mini_net(+0.55, 0.20, 16, 42, False)
    # panel frame lines
    dv_ax.text(-0.55, 0.68, "Disease (CD)", fontsize=1.9,
               ha="center", va="center", color=RED, fontweight="bold")
    dv_ax.text(-0.55, 0.58, "n=2,173", fontsize=1.7,
               ha="center", va="center", color="#666")
    dv_ax.text(+0.55, 0.68, "Healthy control", fontsize=1.9,
               ha="center", va="center", color=GREEN, fontweight="bold")
    dv_ax.text(+0.55, 0.58, "n=3,066", fontsize=1.7,
               ha="center", va="center", color="#666")
    # stats row across bottom
    stats_dv = [("20", "gained", "#3E9B6B"),
                ("95", "lost", "#C75C5C"),
                ("0",  "sign-switched", "#888"),
                ("58/133", "Dis/Ctrl edges", NAVY)]
    sx = -1.05
    for val, lbl, col in stats_dv:
        dv_ax.text(sx, -0.70, val, fontsize=2.4,
                   ha="left", va="center", color=col, fontweight="bold")
        dv_ax.text(sx, -0.92, lbl, fontsize=1.7,
                   ha="left", va="center", color="#666")
        sx += 0.60
    dv_ax.set_xlim(-1.15, 1.15); dv_ax.set_ylim(-1.10, 0.90)
    dv_ax.set_aspect("auto")
    dv_ax.set_xticks([]); dv_ax.set_yticks([])
    for sp in dv_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    ax.text(col_x[1] + col_w/2, title_y_bot,
            "Disease vs control rewiring",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=2.0))

def _thumb_lifecycle(ax):
    """Lifecycle Microbiome Atlas — canonical v2 style.

    Real data from GET /api/lifecycle (global) and /api/lifecycle-compare?disease=CD:
      • Global: 130,499 samples across 7 named stages (Infant–Centenarian),
        top-10 genera composition per stage, Shannon mean ± sd per stage,
        Kruskal-Wallis η² for top-10 genera.
      • CD cohort: 1,866 samples across 5 available stages.
    """
    _ui_bg(ax, "white")
    NAVY = "#1F3864"; RED = "#C41435"; INK = "#1A1A1A"
    MGRAY = "#BCC6D6"; LGRAY = "#EEF1F6"
    MAGENTA = "#C735C7"; BLUE = "#5B6FE5"; GREEN = "#2BB573"
    T = ax.transAxes

    # ── title + subtitle ────────────────────────────────────────────────
    ax.text(0.500, 0.962, "Lifecycle Microbiome Atlas",
            ha="center", va="center", fontsize=5.6, color=INK,
            fontweight="bold", transform=T)
    ax.text(0.500, 0.922,
            "Stage-stratified taxonomic trajectories and non-parametric "
            "statistics at the cohort level across 7 named life stages",
            ha="center", va="center", fontsize=2.5, color="#666", transform=T)

    # ── stats tiles row ────────────────────────────────────────────────
    tiles = [("Samples", "130,499"), ("Life stages", "7"),
             ("Top genera", "10"), ("Method", "Kruskal-Wallis")]
    t_x0 = 0.030; t_x1 = 0.982
    t_gap = 0.012
    t_w = (t_x1 - t_x0 - 3 * t_gap) / 4
    for i, (lbl, val) in enumerate(tiles):
        tx = t_x0 + i * (t_w + t_gap)
        ax.add_patch(FancyBboxPatch((tx, 0.852), t_w, 0.056,
            boxstyle="round,pad=0.003,rounding_size=0.006",
            facecolor=LGRAY, edgecolor=MGRAY, lw=0.5,
            transform=T, zorder=3))
        ax.text(tx + 0.012, 0.893, val,
                ha="left", va="center", fontsize=3.4,
                color=INK, fontweight="bold", transform=T)
        ax.text(tx + 0.012, 0.867, lbl,
                ha="left", va="center", fontsize=2.3,
                color="#666", transform=T)

    # ── controls row ────────────────────────────────────────────────────
    cr_y = 0.792
    ax.text(0.030, cr_y, "Disease", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    ax.add_patch(FancyBboxPatch((0.088, cr_y - 0.014), 0.050, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor=NAVY, edgecolor=NAVY, lw=0.5, transform=T, zorder=3))
    ax.text(0.113, cr_y, "CD", ha="center", va="center",
            fontsize=2.4, color="white", fontweight="bold", transform=T)
    ax.text(0.156, cr_y, "Country", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    ax.add_patch(FancyBboxPatch((0.214, cr_y - 0.014), 0.044, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor="white", edgecolor=NAVY, lw=0.5, transform=T, zorder=3))
    ax.text(0.236, cr_y, "All", ha="center", va="center",
            fontsize=2.1, color=NAVY, transform=T)
    ax.text(0.274, cr_y, "Top genera", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    ax.add_patch(FancyBboxPatch((0.352, cr_y - 0.014), 0.034, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor="white", edgecolor=NAVY, lw=0.5, transform=T, zorder=3))
    ax.text(0.369, cr_y, "10", ha="center", va="center",
            fontsize=2.1, color=NAVY, transform=T)
    # view toggle: Single / Disease-vs-NC (selected)
    ax.add_patch(FancyBboxPatch((0.398, cr_y - 0.014), 0.072, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
    ax.text(0.434, cr_y, "Single", ha="center", va="center",
            fontsize=2.1, color="#555", transform=T)
    ax.add_patch(FancyBboxPatch((0.476, cr_y - 0.014), 0.098, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor=MAGENTA, edgecolor=MAGENTA, lw=0.5,
        transform=T, zorder=3))
    ax.text(0.525, cr_y, "Disease vs NC", ha="center", va="center",
            fontsize=2.1, color="white", fontweight="bold", transform=T)
    # Run CTA
    cta_w = 0.170
    ax.add_patch(FancyBboxPatch((t_x1 - cta_w, cr_y - 0.016), cta_w, 0.032,
        boxstyle="round,pad=0.002,rounding_size=0.006",
        facecolor=MAGENTA, edgecolor=MAGENTA, lw=0.5, transform=T, zorder=3))
    ax.text(t_x1 - cta_w/2, cr_y, "Run Lifecycle",
            ha="center", va="center", fontsize=2.5, color="white",
            fontweight="bold", transform=T)

    # ── life-stage pill strip (7 stages with n=) ────────────────────────
    sp_y = 0.712
    sp_x = 0.030
    stages = [("Infant", 36518), ("Child", 15841), ("Adolescent", 2070),
              ("Adult", 66303), ("Older Adult", 7739),
              ("Oldest Old", 1616), ("Centenarian", 412)]
    _slbls = [f"{n}  n={c:,}" for n, c in stages]
    _swid = [0.006 + len(l) * 0.0060 for l in _slbls]
    _sgap = (t_x1 - 0.030 - sum(_swid)) / (len(_slbls) - 1)
    for lbl, pw in zip(_slbls, _swid):
        ax.add_patch(FancyBboxPatch((sp_x, sp_y - 0.013), pw, 0.026,
            boxstyle="round,pad=0.001,rounding_size=0.004",
            facecolor="white", edgecolor=MAGENTA, lw=0.5,
            transform=T, zorder=3))
        ax.text(sp_x + pw/2, sp_y, lbl, ha="center", va="center",
                fontsize=1.7, color=MAGENTA, transform=T)
        sp_x += pw + _sgap

    # ── 4 tabs ──────────────────────────────────────────────────────────
    tabs = [("Trajectory", True), ("Shannon", False),
            ("Disease vs NC", False), ("Effect size", False)]
    tab_x = 0.030
    tab_y = 0.672
    for lbl, sel in tabs:
        tw = 0.008 + len(lbl) * 0.0095
        ax.add_patch(FancyBboxPatch((tab_x, tab_y - 0.014), tw, 0.028,
            boxstyle="round,pad=0.002,rounding_size=0.005",
            facecolor="white",
            edgecolor=MAGENTA if sel else MGRAY,
            lw=0.8 if sel else 0.4, transform=T, zorder=3))
        ax.text(tab_x + tw/2, tab_y, lbl, ha="center", va="center",
                fontsize=2.2, color=MAGENTA if sel else INK,
                fontweight="bold" if sel else "normal", transform=T)
        tab_x += tw + 0.008

    # ── main content grid 2×2 ───────────────────────────────────────────
    col_w = 0.440
    _right_dx = 0.065
    col_x = [0.040, 0.515 + _right_dx]
    _bot_dy = -0.050
    row_y = [0.338, 0.040 + _bot_dy]
    row_h = 0.272
    title_y_top = 0.624
    title_y_bot = 0.328 + _bot_dy

    # Real composition matrix (top-10 genera + Other) — from /api/lifecycle
    genera_list = ["Bacteroides", "Bifidobacterium", "Shigella",
                   "Faecalibacterium", "Blautia", "Streptococcus",
                   "Segatella", "Enterococcus", "Klebsiella",
                   "Agathobacter", "Other"]
    comp_g = np.array([
        [ 7.48, 18.39,  8.83,  1.63,  2.14,  3.86,  1.21,  2.16,  4.32,  0.54, 49.45],
        [11.89,  9.71,  5.29,  4.29,  2.52,  4.73,  3.72,  0.62,  1.10,  1.14, 54.99],
        [20.61,  4.39,  4.21,  5.46,  3.70,  0.74,  7.00,  0.55,  0.81,  2.34, 50.20],
        [13.95,  2.20,  2.95,  4.23,  4.21,  2.82,  2.73,  2.76,  0.94,  1.78, 61.44],
        [18.98,  1.96,  6.74,  4.61,  3.90,  1.35,  2.66,  1.40,  1.98,  2.16, 54.26],
        [22.04,  1.90,  7.43,  4.15,  2.66,  1.76,  2.68,  2.24,  2.57,  1.42, 51.16],
        [18.69,  1.98,  4.99,  2.01,  3.69,  2.46,  3.43,  0.44,  2.26,  0.83, 59.22],
    ])  # 7 × 11
    stage_lbls_short = ["Inf", "Chi", "Ado", "Adu", "OlA", "OlO", "Cen"]
    stage_lbls_full  = ["Infant", "Child", "Adoles", "Adult",
                        "Older\nAdult", "Oldest\nOld", "Cente-\nnarian"]
    shannon_mean = [1.373, 1.692, 2.204, 1.817, 2.193, 2.138, 2.420]
    shannon_sd   = [0.787, 0.818, 0.624, 0.944, 0.668, 0.608, 0.680]
    stage_n      = [36518, 15841, 2070, 66303, 7739, 1616, 412]

    # distinct palette for 11 bands
    palette = [
        "#E04545",  # Bacteroides
        "#F2A71B",  # Bifidobacterium
        "#D4C020",  # Shigella
        "#2BB573",  # Faecalibacterium
        "#1A9B9B",  # Blautia
        "#5B6FE5",  # Streptococcus
        "#8657D4",  # Segatella
        "#C735C7",  # Enterococcus
        "#E07830",  # Klebsiella
        "#1F3864",  # Agathobacter
        "#BCC6D6",  # Other
    ]

    # ── Global stacked area (top-left) ─────────────────────────────────
    sa_ax = ax.inset_axes([col_x[0], row_y[0], col_w, row_h])
    _ui_bg(sa_ax, "white")
    # normalize each row to 100 for display
    row_sums = comp_g.sum(axis=1, keepdims=True)
    comp_n = comp_g / row_sums * 100
    xs = np.arange(len(stage_lbls_short))
    baseline = np.zeros_like(xs, dtype=float)
    for gi in range(comp_n.shape[1]):
        yvals = comp_n[:, gi]
        sa_ax.fill_between(xs, baseline, baseline + yvals,
                           color=palette[gi], alpha=0.92,
                           edgecolor="white", linewidth=0.25)
        baseline = baseline + yvals
    sa_ax.set_xlim(0, len(xs) - 1)
    sa_ax.set_ylim(0, 100)
    sa_ax.set_xticks(xs)
    sa_ax.set_xticklabels(stage_lbls_short, fontsize=2.2, color="#555")
    sa_ax.set_yticks([0, 50, 100])
    sa_ax.set_yticklabels(["0", "50", "100%"], fontsize=2.0, color="#888")
    sa_ax.tick_params(length=0, pad=1)
    for sp in sa_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    # inline legend (2 cols × 6 rows) on right side
    lg_x0 = len(xs) - 1 + 0.08
    for gi, gname in enumerate(genera_list):
        col_i, row_i = divmod(gi, 6)
        lx = lg_x0 + col_i * 1.15
        ly = 98 - row_i * 16
        sa_ax.add_patch(plt.Rectangle((lx, ly - 2), 0.25, 4,
                                      color=palette[gi], clip_on=False))
        sa_ax.text(lx + 0.32, ly, gname[:10],
                   fontsize=1.4, ha="left", va="center",
                   color=INK, fontstyle="italic", clip_on=False)
    ax.text(col_x[0] + col_w/2, title_y_top,
            "Global trajectory  ·  top-10 genera composition",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=2.0))

    # ── Shannon trajectory + n per stage (top-right) ───────────────────
    sh_ax = ax.inset_axes([col_x[1], row_y[0], col_w, row_h])
    _ui_bg(sh_ax, "white")
    # sample-count bars (secondary, faint)
    bar_h = np.array(stage_n) / max(stage_n) * 0.50   # top 50% of panel
    sh_ax.bar(xs, bar_h, width=0.65, color=LGRAY,
              edgecolor=MGRAY, linewidth=0.3, zorder=1,
              bottom=0)
    for xi, n_ in zip(xs, stage_n):
        sh_ax.text(xi, (n_ / max(stage_n)) * 0.50 + 0.03,
                   f"n={n_:,}", fontsize=1.5, ha="center",
                   va="bottom", color="#888")
    # Shannon line (main) on secondary y [0,3]
    sh_scale = 3.0
    sh_y = np.array(shannon_mean) / sh_scale * 0.96 + 0.02
    sh_sd_up = (np.array(shannon_mean) + np.array(shannon_sd)) / sh_scale * 0.96 + 0.02
    sh_sd_dn = (np.array(shannon_mean) - np.array(shannon_sd)) / sh_scale * 0.96 + 0.02
    sh_ax.fill_between(xs, sh_sd_dn, sh_sd_up,
                       color=MAGENTA, alpha=0.18, zorder=2)
    sh_ax.plot(xs, sh_y, color=MAGENTA, lw=1.2,
               marker="o", markersize=3.4,
               markerfacecolor="white", markeredgewidth=0.8,
               zorder=4)
    for xi, m in zip(xs, shannon_mean):
        sh_ax.text(xi, m / sh_scale * 0.96 + 0.08,
                   f"{m:.2f}", fontsize=1.6, ha="center",
                   va="bottom", color=MAGENTA, fontweight="bold")
    sh_ax.set_xlim(-0.5, len(xs) - 0.5)
    sh_ax.set_ylim(0, 1.08)
    sh_ax.set_xticks(xs)
    sh_ax.set_xticklabels(stage_lbls_short, fontsize=2.2, color="#555")
    sh_ax.set_yticks([0.02, 0.50, 0.98])
    sh_ax.set_yticklabels(["0", "1.5", "3.0"], fontsize=2.0, color="#888")
    sh_ax.tick_params(length=0, pad=1)
    for sp in sh_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    ax.text(col_x[1] + col_w/2, title_y_top,
            "Shannon diversity  ·  sample counts",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=2.0))

    # ── CD trajectory (bottom-left) — stages with real data ───────────
    cd_ax = ax.inset_axes([col_x[0], row_y[1], col_w, row_h])
    _ui_bg(cd_ax, "white")
    # CD only has 5 stages: Child, Adolescent, Adult, Older_Adult, Oldest_Old
    cd_stage_lbls = ["Chi", "Ado", "Adu", "OlA", "OlO"]
    cd_comp = np.array([
        # [Bacteroides, Bifidobacterium, Shigella, Faecali, Blautia,
        #  Strepto, Segatella, Enterococcus, Klebsiella, Agathobacter, Other]
        [28.40, 11.75, 2.37, 6.89, 2.11, 0.44, 1.47, 0.91, 0.80, 1.04, 44.38],
        [32.41,  5.23, 2.00,10.21, 0.65, 0.74, 0.42, 0.29, 1.18, 0.77, 45.96],
        [22.60,  8.16, 6.53, 3.76, 2.02, 2.34, 1.93, 2.52, 1.79, 1.73, 46.62],
        [16.05, 10.32,14.97, 1.79, 5.18, 2.31, 2.01, 0.17, 1.01, 1.01, 45.17],
        [ 6.58, 11.42, 5.99, 2.25, 4.92, 1.22, 3.45, 0.00, 0.66, 5.92, 57.58],
    ])
    cd_n = [369, 57, 1322, 109, 9]
    cd_xs = np.arange(len(cd_stage_lbls))
    row_sums_c = cd_comp.sum(axis=1, keepdims=True)
    cd_n_comp = cd_comp / row_sums_c * 100
    baseline = np.zeros_like(cd_xs, dtype=float)
    for gi in range(cd_n_comp.shape[1]):
        yvals = cd_n_comp[:, gi]
        cd_ax.fill_between(cd_xs, baseline, baseline + yvals,
                           color=palette[gi], alpha=0.92,
                           edgecolor="white", linewidth=0.25)
        baseline = baseline + yvals
    cd_ax.set_xlim(0, len(cd_xs) - 1)
    cd_ax.set_ylim(0, 100)
    cd_ax.set_xticks(cd_xs)
    cd_ax.set_xticklabels(
        [f"{s}\nn={n}" for s, n in zip(cd_stage_lbls, cd_n)],
        fontsize=1.8, color="#555")
    cd_ax.set_yticks([0, 50, 100])
    cd_ax.set_yticklabels(["0", "50", "100%"], fontsize=2.0, color="#888")
    cd_ax.tick_params(length=0, pad=1)
    for sp in cd_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    # CD header strip
    cd_ax.text(0.02, 96, "CD cohort  ·  n=1,866",
               fontsize=1.8, ha="left", va="top",
               color=RED, fontweight="bold")
    ax.text(col_x[0] + col_w/2, title_y_bot,
            "Disease trajectory  ·  CD vs age stages",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=2.0))

    # ── Kruskal-Wallis effect size (bottom-right) ──────────────────────
    es_ax = ax.inset_axes([col_x[1], row_y[1], col_w, row_h])
    _ui_bg(es_ax, "white")
    kw_data = [
        ("Bifidobacterium",  0.1283),
        ("Agathobacter",     0.1052),
        ("Blautia",          0.0954),
        ("Faecalibacterium", 0.0902),
        ("Shigella",         0.0882),
        ("Bacteroides",      0.0793),
        ("Enterococcus",     0.0560),
        ("Klebsiella",       0.0518),
        ("Streptococcus",    0.0339),
        ("Segatella",        0.0284),
    ]
    kw_names = [k[0] for k in kw_data]
    kw_eta   = [k[1] for k in kw_data]
    es_ys = np.arange(len(kw_data))[::-1]
    es_colors = [palette[genera_list.index(n)] for n in kw_names]
    es_ax.barh(es_ys, kw_eta, height=0.66,
               color=es_colors, edgecolor="white", linewidth=0.5)
    for yi, v in zip(es_ys, kw_eta):
        es_ax.text(v + 0.004, yi, f"{v:.3f}",
                   fontsize=1.7, va="center", ha="left", color="#555")
    es_ax.set_yticks(es_ys)
    es_ax.set_yticklabels([n[:12] for n in kw_names],
                          fontsize=1.9, fontstyle="italic", color=INK)
    es_ax.set_xlim(0, 0.165)
    es_ax.set_xticks([0, 0.05, 0.10, 0.15])
    es_ax.set_xticklabels(["0", "0.05", "0.10", "0.15"],
                          fontsize=1.7, color="#888")
    es_ax.set_xlabel("Kruskal-Wallis η²", fontsize=2.0,
                     color="#666", labelpad=1)
    es_ax.tick_params(length=0, pad=1)
    for sp in es_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    es_ax.text(0.163, len(kw_data) - 0.5, "all p < 1e-300",
               fontsize=1.5, ha="right", va="top", color="#888",
               fontstyle="italic")
    ax.text(col_x[1] + col_w/2, title_y_bot,
            "Effect size  ·  Kruskal-Wallis η² (top-10)",
            ha="center", va="center", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=2.0))
    return

    # ── stacked bar chart ──────────────────────────────────────────────────────
    bar_ax = ax.inset_axes([0.06, 0.08, 0.94, 0.78])
    _ui_bg(bar_ax, "white")
    stages = ["Infant","Child","Adoles","Adult","Older\nAdult","Oldest\nOld","Cente-\nnarian"]
    x = np.arange(len(stages))
    taxa_data = [
        ("Bifidobacterium", "#E04545", [0.35,0.12,0.06,0.04,0.05,0.06,0.08]),
        ("Lactobacillus",   "#E07830", [0.15,0.10,0.06,0.04,0.05,0.06,0.05]),
        ("Blautia",         "#D4C020", [0.10,0.15,0.18,0.22,0.22,0.20,0.18]),
        ("Faecalibacterium","#3AB060", [0.05,0.12,0.16,0.20,0.18,0.15,0.12]),
        ("Ruminococcus",    "#2080C0", [0.08,0.10,0.12,0.14,0.15,0.14,0.12]),
        ("Bacteroides",     "#5545A0", [0.12,0.18,0.22,0.22,0.20,0.20,0.22]),
        ("Other",           "#BBBBBB", [0.15,0.23,0.20,0.14,0.15,0.19,0.23]),
    ]
    bottoms = np.zeros(len(stages))
    for tname, tcol, tvals in taxa_data:
        bar_ax.bar(x, tvals, bottom=bottoms, color=tcol, width=0.80,
                   label=tname, edgecolor="none")
        bottoms += np.array(tvals)
    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(stages, fontsize=4.0)
    bar_ax.set_ylim(0, 1.05)
    bar_ax.set_ylabel("Proportion", fontsize=4.5)
    bar_ax.tick_params(axis='y', labelsize=4.0)
    bar_ax.legend(fontsize=3.2, loc="upper right", ncol=1, framealpha=0.7,
                  handlelength=0.7, handletextpad=0.3, borderpad=0.3)
    for sp in bar_ax.spines.values(): sp.set_linewidth(0.3); sp.set_color("#CCC")
    bar_ax.spines["bottom"].set_visible(True)
    bar_ax.spines["left"].set_visible(True)

    # ── significance stars ─────────────────────────────────────────────────────
    for xi, star in enumerate(["***","**","ns","***","**","*","***"]):
        bar_ax.text(xi, 1.01, star, ha="center", va="bottom", fontsize=4.0,
                    color=RED if star != "ns" else "#AAA")

def _thumb_gbhi(ax):
    """GBHI Health Score — canonical v2 style.

    Real data from GET /api/health-index/reference:
      • NC cohort:     82,106 samples · median=75.4 · mean=57.1 · std=43.5
      • Disease:       72,679 samples · median= 9.8 · mean=31.4 · std=38.4
      • 20-bin histogram across 0–100 scale (NC vs Disease)
      •  5 health-associated genera (log2fc>0) +
        21 disease-associated genera (log2fc<0) from population GMHI fit.
    """
    _ui_bg(ax, "white")
    NAVY = "#1F3864"; RED = "#C41435"; INK = "#1A1A1A"
    MGRAY = "#BCC6D6"; LGRAY = "#EEF1F6"
    MAGENTA = "#C735C7"; BLUE = "#5B6FE5"; GREEN = "#2BB573"
    ORANGE = "#F0A020"; AMBER = "#E0B030"
    T = ax.transAxes

    # ── title + subtitle ────────────────────────────────────────────────
    ax.text(0.500, 0.962, "GBHI Health Score",
            ha="center", va="center", fontsize=5.6, color=INK,
            fontweight="bold", transform=T)
    ax.text(0.500, 0.922,
            "Gut Microbiome Health Index · per-sample 0–100 score "
            "from genus-level abundance (Gupta et al. 2020)",
            ha="center", va="center", fontsize=2.5, color="#666", transform=T)

    # ── stats tiles row ────────────────────────────────────────────────
    tiles = [("NC samples", "82,106"),
             ("Disease samples", "72,679"),
             ("Feature genera", "5 + 21"),
             ("LOCO AUC", "0.78")]
    t_x0 = 0.030; t_x1 = 0.982
    t_gap = 0.012
    t_w = (t_x1 - t_x0 - 3 * t_gap) / 4
    for i, (lbl, val) in enumerate(tiles):
        tx = t_x0 + i * (t_w + t_gap)
        ax.add_patch(FancyBboxPatch((tx, 0.852), t_w, 0.056,
            boxstyle="round,pad=0.003,rounding_size=0.006",
            facecolor=LGRAY, edgecolor=MGRAY, lw=0.5,
            transform=T, zorder=3))
        ax.text(tx + 0.012, 0.893, val,
                ha="left", va="center", fontsize=3.6,
                color=INK, fontweight="bold", transform=T)
        ax.text(tx + 0.012, 0.867, lbl,
                ha="left", va="center", fontsize=2.3,
                color="#666", transform=T)

    # ── controls row ────────────────────────────────────────────────────
    cr_y = 0.792
    ax.text(0.030, cr_y, "Reference", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    ax.add_patch(FancyBboxPatch((0.098, cr_y - 0.014), 0.050, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor=NAVY, edgecolor=NAVY, lw=0.5, transform=T, zorder=3))
    ax.text(0.123, cr_y, "NC", ha="center", va="center",
            fontsize=2.4, color="white", fontweight="bold", transform=T)
    ax.text(0.168, cr_y, "CV", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    ax.add_patch(FancyBboxPatch((0.202, cr_y - 0.014), 0.060, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor="white", edgecolor=NAVY, lw=0.5, transform=T, zorder=3))
    ax.text(0.232, cr_y, "LOCO", ha="center", va="center",
            fontsize=2.1, color=NAVY, transform=T)
    ax.text(0.282, cr_y, "Scale", ha="left", va="center",
            fontsize=2.3, color="#666", transform=T)
    ax.add_patch(FancyBboxPatch((0.326, cr_y - 0.014), 0.058, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor="white", edgecolor=NAVY, lw=0.5, transform=T, zorder=3))
    ax.text(0.355, cr_y, "0–100", ha="center", va="center",
            fontsize=2.1, color=NAVY, transform=T)
    # upload csv toggle (inactive) + calculate CTA
    ax.add_patch(FancyBboxPatch((0.404, cr_y - 0.014), 0.108, 0.028,
        boxstyle="round,pad=0.002,rounding_size=0.005",
        facecolor="white", edgecolor=MGRAY, lw=0.5, transform=T, zorder=3))
    ax.text(0.458, cr_y, "Upload CSV / TSV", ha="center", va="center",
            fontsize=2.0, color="#555", fontstyle="italic", transform=T)
    # Calculate CTA
    cta_w = 0.170
    ax.add_patch(FancyBboxPatch((t_x1 - cta_w, cr_y - 0.016), cta_w, 0.032,
        boxstyle="round,pad=0.002,rounding_size=0.006",
        facecolor=MAGENTA, edgecolor=MAGENTA, lw=0.5, transform=T, zorder=3))
    ax.text(t_x1 - cta_w/2, cr_y, "Calculate Index",
            ha="center", va="center", fontsize=2.5, color="white",
            fontweight="bold", transform=T)

    # ── feature-genera pill strip ──────────────────────────────────────
    fp_y = 0.712
    fp_x = 0.030
    feat_genera = ["Anaerotruncus", "Bifidobacterium",
                   "Senegalimassilia", "Segatella",
                   "Lacticaseibacillus", "Rothia",
                   "Enterococcus"]
    feat_color = [GREEN, GREEN, GREEN, GREEN, RED, RED, RED]
    _flbls = feat_genera
    _fwid = [0.006 + len(l) * 0.0063 for l in _flbls]
    _fgap = (t_x1 - 0.030 - sum(_fwid)) / (len(_flbls) - 1)
    for lbl, pw, col in zip(_flbls, _fwid, feat_color):
        ax.add_patch(FancyBboxPatch((fp_x, fp_y - 0.013), pw, 0.026,
            boxstyle="round,pad=0.001,rounding_size=0.004",
            facecolor="white", edgecolor=col, lw=0.5,
            transform=T, zorder=3))
        ax.text(fp_x + pw/2, fp_y, lbl, ha="center", va="center",
                fontsize=1.7, fontstyle="italic", color=col,
                transform=T)
        fp_x += pw + _fgap

    # ── 4 tabs ──────────────────────────────────────────────────────────
    tabs = [("Gauge", True), ("Distribution", False),
            ("Health features", False), ("Disease features", False)]
    tab_x = 0.030
    tab_y = 0.672
    for lbl, sel in tabs:
        tw = 0.008 + len(lbl) * 0.0095
        ax.add_patch(FancyBboxPatch((tab_x, tab_y - 0.014), tw, 0.028,
            boxstyle="round,pad=0.002,rounding_size=0.005",
            facecolor="white",
            edgecolor=MAGENTA if sel else MGRAY,
            lw=0.8 if sel else 0.4, transform=T, zorder=3))
        ax.text(tab_x + tw/2, tab_y, lbl, ha="center", va="center",
                fontsize=2.2, color=MAGENTA if sel else INK,
                fontweight="bold" if sel else "normal", transform=T)
        tab_x += tw + 0.008

    # ── main content: asymmetric — big gauge left, 2 right panels ──────
    gauge_x, gauge_y = 0.005, 0.040
    gauge_w, gauge_h = 0.370, 0.584
    right_x = 0.430
    right_w = 0.552
    top_y, bot_y = 0.354, 0.040
    row_h = 0.270

    # ── Big gauge (LEFT, spans full content height) ────────────────────
    g_ax = ax.inset_axes([gauge_x, gauge_y, gauge_w, gauge_h])
    _ui_bg(g_ax, "white")
    # colored arc segments: red(0-40) · amber(40-70) · green(70-100)
    for (s0, s1, c) in [(0, 40, "#C75C5C"),
                        (40, 70, ORANGE),
                        (70, 100, "#2BB573")]:
        a0 = np.pi * (1 - s0 / 100)
        a1 = np.pi * (1 - s1 / 100)
        th = np.linspace(a0, a1, 80)
        r_out, r_in = 1.00, 0.70
        g_ax.fill(
            np.concatenate([np.cos(th) * r_out,
                            np.cos(th[::-1]) * r_in]),
            np.concatenate([np.sin(th) * r_out,
                            np.sin(th[::-1]) * r_in]),
            color=c, alpha=0.92, linewidth=0, zorder=2)
    # major tick marks around arc
    for pct in [0, 20, 40, 60, 80, 100]:
        a = np.pi * (1 - pct / 100)
        g_ax.plot([np.cos(a) * 1.02, np.cos(a) * 1.10],
                  [np.sin(a) * 1.02, np.sin(a) * 1.10],
                  color="#555", lw=0.6, zorder=3)
        g_ax.text(np.cos(a) * 1.20, np.sin(a) * 1.20,
                  str(pct), fontsize=3.0, ha="center", va="center",
                  color="#555")
    # needle to 75.4
    score = 75.4
    score_a = np.pi * (1 - score / 100)
    g_ax.annotate("",
                  xy=(np.cos(score_a) * 0.70, np.sin(score_a) * 0.70),
                  xytext=(0.0, 0.0),
                  arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4),
                  zorder=6)
    g_ax.add_patch(Circle((0, 0), 0.06, facecolor=INK, zorder=7))
    # big score text
    g_ax.text(0.0, -0.30, "75.4", ha="center", va="center",
              fontsize=13, fontweight="bold", color=NAVY)
    g_ax.text(0.0, -0.55, "NC median score", ha="center", va="center",
              fontsize=3.0, color="#666")
    # NC stats block
    g_ax.text(-1.05, -0.84,
              "n = 82,106\nmean = 57.1\nstd = 43.5",
              fontsize=2.4, ha="left", va="top",
              color=NAVY,
              bbox=dict(facecolor="white", edgecolor=MGRAY,
                        boxstyle="round,pad=0.15", linewidth=0.4))
    # Disease stats block
    g_ax.text(1.05, -0.84,
              "n = 72,679\nmean = 31.4\nmed = 9.8",
              fontsize=2.4, ha="right", va="top",
              color=RED,
              bbox=dict(facecolor="white", edgecolor=MGRAY,
                        boxstyle="round,pad=0.15", linewidth=0.4))
    g_ax.set_xlim(-1.40, 1.40)
    g_ax.set_ylim(-1.25, 1.18)
    g_ax.set_aspect("equal")
    g_ax.set_xticks([]); g_ax.set_yticks([])
    for sp in g_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    ax.text(gauge_x + gauge_w/2, gauge_y + gauge_h - 0.004,
            "Population gauge  ·  NC median  ·  LOCO-CV AUC 0.78",
            ha="center", va="bottom", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=2.0))

    # ── Distribution histogram (RIGHT TOP) ─────────────────────────────
    dist_ax = ax.inset_axes([right_x, top_y, right_w, row_h])
    _ui_bg(dist_ax, "white")
    # Real 20-bin histogram from /api/health-index/reference
    nc_counts = [18728, 4673, 4334, 1863, 1439, 1227, 1072, 950, 934, 949,
                 991, 1037, 1049, 1064, 935, 899, 1060, 1320, 2006, 35849]
    ds_counts = [27958, 9302, 4827, 2718, 1992, 1607, 1403, 1231, 1203,
                 1188, 1185, 1189, 1150, 1074, 938, 610, 804, 929, 1498,
                 12776]
    bin_x = np.arange(20)
    w = 0.42
    dist_ax.bar(bin_x - w/2, nc_counts, width=w, color=NAVY,
                alpha=0.78, edgecolor="white", linewidth=0.3,
                label="NC")
    dist_ax.bar(bin_x + w/2, ds_counts, width=w, color=RED,
                alpha=0.75, edgecolor="white", linewidth=0.3,
                label="Disease")
    dist_ax.axvline((75.4 / 5) - 0.5, color=NAVY,
                    lw=0.8, ls="--", alpha=0.8)
    dist_ax.text((75.4 / 5) - 0.5, max(nc_counts) * 0.92,
                 "median 75.4",
                 fontsize=1.8, ha="left", va="center", color=NAVY,
                 fontweight="bold")
    dist_ax.set_xticks([0, 5, 10, 15, 19])
    dist_ax.set_xticklabels(["0", "25", "50", "75", "100"],
                            fontsize=2.2, color="#555")
    dist_ax.set_yticks([0, 10000, 20000, 30000])
    dist_ax.set_yticklabels(["0", "10k", "20k", "30k"],
                            fontsize=2.0, color="#888")
    dist_ax.set_xlabel("GBHI score", fontsize=2.2, color="#666",
                       labelpad=1)
    dist_ax.tick_params(length=0, pad=1)
    dist_ax.legend(fontsize=2.0, loc="upper center",
                   framealpha=0.7, handletextpad=0.3, borderpad=0.3,
                   ncol=2)
    for sp in dist_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    ax.text(right_x + right_w/2, top_y + row_h - 0.004,
            "Population distribution  ·  NC vs Disease (n=154,785)",
            ha="center", va="bottom", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=2.0))

    # ── Feature genera bars (RIGHT BOTTOM) ─────────────────────────────
    _feat_dx = 0.060
    feat_ax = ax.inset_axes([right_x + _feat_dx, bot_y, right_w - _feat_dx, row_h])
    _ui_bg(feat_ax, "white")
    # Health-associated (green, log2fc > 0) + Disease-associated (red, log2fc < 0)
    health = [
        ("Anaerotruncus",    +1.81),
        ("Bifidobacterium",  +1.33),
        ("Senegalimassilia", +0.47),
        ("Segatella",        +0.33),
        ("Asaccharobacter",  +0.30),
    ]
    disease = [
        ("Lacticaseibacillus", -2.17),
        ("Rothia",             -2.09),
        ("Enterococcus",       -1.75),
        ("Schaalia",           -1.61),
        ("Staphylococcus",     -1.32),
    ]
    all_feat = health + disease
    names = [n for n, _ in all_feat]
    vals  = [v for _, v in all_feat]
    cols  = [GREEN if v > 0 else RED for v in vals]
    ys = np.arange(len(all_feat))[::-1]
    feat_ax.barh(ys, vals, height=0.66, color=cols,
                 edgecolor="white", linewidth=0.5)
    for yi, (nm, v) in zip(ys, all_feat):
        off = 0.06 if v > 0 else -0.06
        feat_ax.text(v + off, yi, f"{v:+.2f}",
                     fontsize=1.6, va="center",
                     ha="left" if v > 0 else "right",
                     color="#555")
    feat_ax.axvline(0, color="#999", lw=0.4, zorder=2)
    feat_ax.set_yticks(ys)
    feat_ax.set_yticklabels(names, fontsize=1.9,
                            fontstyle="italic", color=INK)
    feat_ax.set_xlim(-2.7, 2.3)
    feat_ax.set_xticks([-2, -1, 0, 1, 2])
    feat_ax.set_xticklabels(["-2", "-1", "0", "+1", "+2"],
                            fontsize=1.8, color="#888")
    feat_ax.set_xlabel("log2FC (disease / NC)",
                       fontsize=2.0, color="#666", labelpad=1)
    feat_ax.tick_params(length=0, pad=1)
    for sp in feat_ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(MGRAY)
    ax.text(right_x + right_w/2, bot_y + row_h + 0.010,
            "Top GMHI feature genera  ·  5 health + 5 disease",
            ha="center", va="bottom", fontsize=2.0, color="#555",
            fontweight="bold", transform=T, zorder=50).set_bbox(
        dict(facecolor="white", edgecolor="none", pad=2.0))

def _thumb_download(ax):
    """Download + REST API — download icon + format pills + endpoint count."""
    _ui_bg(ax, "white")
    NAVY = "#253D6E"; RED = "#C41435"; FOREST = "#00836F"; TEAL = "#1A6B82"
    T = ax.transAxes

    # ── title ──────────────────────────────────────────────────────────────────
    ax.text(0.03, 0.97, "Download  +  REST API  ·  46 endpoints",
            ha="left", va="top", fontsize=5.5, color=FOREST, fontweight="bold", transform=T)

    # ── download section header ────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.03, 0.85), 0.45, 0.07,
        boxstyle="round,pad=0.005", facecolor="#E8F4F0",
        edgecolor=FOREST, lw=0.6, transform=T, zorder=3))
    ax.text(0.255, 0.885, "Download Data", ha="center", va="center",
            fontsize=4.5, color=FOREST, fontweight="bold", transform=T)
    ax.add_patch(FancyBboxPatch((0.52, 0.85), 0.45, 0.07,
        boxstyle="round,pad=0.005", facecolor="#EAF0F8",
        edgecolor=NAVY, lw=0.6, transform=T, zorder=3))
    ax.text(0.745, 0.885, "REST API Docs", ha="center", va="center",
            fontsize=4.5, color=NAVY, fontweight="bold", transform=T)

    # ── download icon ──────────────────────────────────────────────────────────
    ax.add_patch(Circle((0.255, 0.63), 0.12, facecolor="#E8F4F0",
                        edgecolor=FOREST, lw=1.0, transform=T, zorder=3))
    ax.annotate("", xy=(0.255, 0.55), xytext=(0.255, 0.68),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=FOREST, lw=1.8), zorder=4)
    ax.plot([0.18, 0.33], [0.535, 0.535], color=FOREST, lw=2.2,
            solid_capstyle="round", transform=T)

    # ── format pills ──────────────────────────────────────────────────────────
    fmts = [("CSV", "#3AB060", 0.04), ("SVG", NAVY, 0.39), ("JSON", TEAL, 0.74)]
    for lbl, col, xp in fmts:
        ax.add_patch(FancyBboxPatch((xp, 0.42), 0.31, 0.09,
            boxstyle="round,pad=0.008", facecolor=col,
            edgecolor="none", alpha=0.88, transform=T, zorder=3))
        ax.text(xp+0.155, 0.465, lbl, ha="center", va="center",
                fontsize=6.0, color="white", fontweight="bold", transform=T)

    # ── download item list ─────────────────────────────────────────────────────
    items = [
        ("Summary statistics  (CSV)", FOREST),
        ("Disease profiles  (CSV/JSON)", NAVY),
        ("Species profiles  (TSV)", TEAL),
        ("Genus catalog  (CSV)", FOREST),
    ]
    for ii, (ilbl, icol) in enumerate(items):
        iy = 0.34 - ii * 0.075
        ax.add_patch(FancyBboxPatch((0.03, iy), 0.93, 0.062,
            boxstyle="round,pad=0.003", facecolor="#F8FAFB",
            edgecolor="#D0D8E8", lw=0.4, transform=T, zorder=2))
        ax.text(0.06, iy + 0.031, ilbl, ha="left", va="center",
                fontsize=4.0, color=icol, transform=T)
        ax.add_patch(FancyBboxPatch((0.83, iy + 0.010), 0.12, 0.040,
            boxstyle="round,pad=0.002", facecolor=icol,
            edgecolor="none", alpha=0.80, transform=T, zorder=3))
        ax.text(0.890, iy + 0.030, "↓ Get", ha="center", va="center",
                fontsize=3.5, color="white", transform=T)

    # ── OpenAPI badge ─────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.60, 0.03), 0.36, 0.07,
        boxstyle="round,pad=0.005", facecolor="#F0A020",
        edgecolor="none", transform=T, zorder=3))
    ax.text(0.78, 0.065, "OpenAPI 3.0", ha="center", va="center",
            fontsize=4.0, color="white", fontweight="bold", transform=T)

def _thumb_about(ax):
    """About & Cite — paper title, DOI box, authors, CC-BY badge."""
    _ui_bg(ax, "white")
    NAVY = "#253D6E"; FOREST = "#00836F"; TEAL = "#1A6B82"
    T = ax.transAxes

    # ── title ──────────────────────────────────────────────────────────────────
    ax.text(0.03, 0.97, "About & Cite  ·  authors · license",
            ha="left", va="top", fontsize=5.5, color=FOREST, fontweight="bold", transform=T)

    # ── paper citation card ────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.03, 0.60), 0.94, 0.28,
        boxstyle="round,pad=0.010", facecolor="#EBF4F0",
        edgecolor=TEAL, lw=0.8, transform=T, zorder=2))
    ax.text(0.50, 0.83, "GutBiomeDB: An Integrated Human",
            ha="center", va="center", fontsize=5.2, color="#1A1A1A",
            fontweight="bold", transform=T)
    ax.text(0.50, 0.77, "Gut Microbiome Atlas",
            ha="center", va="center", fontsize=5.2, color="#1A1A1A",
            fontweight="bold", transform=T)
    ax.text(0.50, 0.70, "Zhai J., Dai C., et al.  ·  2026",
            ha="center", va="center", fontsize=4.2, color="#666",
            fontstyle="italic", transform=T)
    ax.text(0.50, 0.64, "gutbiomedb.online  ·  contact: cdai@cmu.edu.cn",
            ha="center", va="center", fontsize=3.4, color="#888", transform=T)

    # ── preprint / availability box ────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.03, 0.44), 0.94, 0.13,
        boxstyle="round,pad=0.008", facecolor=NAVY,
        edgecolor="none", transform=T, zorder=3))
    ax.text(0.50, 0.505, "Preprint forthcoming  ·  data & code open-access",
            ha="center", va="center", fontsize=4.6, color="white",
            fontweight="bold", transform=T)

    # ── stats row ─────────────────────────────────────────────────────────────
    stat_items = [("168,464", "samples"), ("224", "diseases"), ("482", "studies")]
    for si, (num, lbl) in enumerate(stat_items):
        sx = 0.07 + si * 0.31
        ax.add_patch(FancyBboxPatch((sx, 0.29), 0.27, 0.11,
            boxstyle="round,pad=0.005", facecolor="#F4F7FB",
            edgecolor="#C8D4E8", lw=0.5, transform=T, zorder=3))
        ax.text(sx+0.135, 0.358, num, ha="center", va="center",
                fontsize=5.5, color=NAVY, fontweight="bold", transform=T)
        ax.text(sx+0.135, 0.305, lbl, ha="center", va="center",
                fontsize=3.5, color="#888", transform=T)

    # ── CC-BY badge ────────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.25, 0.14), 0.50, 0.11,
        boxstyle="round,pad=0.008", facecolor=FOREST,
        edgecolor="none", transform=T, zorder=3))
    ax.text(0.50, 0.195, "CC-BY 4.0  ·  Open Access",
            ha="center", va="center", fontsize=4.8, color="white",
            fontweight="bold", transform=T)

    # ── cite button ────────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.25, 0.03), 0.50, 0.09,
        boxstyle="round,pad=0.005", facecolor="#F4F7FB",
        edgecolor=NAVY, lw=0.6, transform=T, zorder=3))
    ax.text(0.50, 0.075, "Copy Citation  ·  BibTeX / RIS",
            ha="center", va="center", fontsize=3.8, color=NAVY, transform=T)

_THUMB_FN = {
    "browse_phenotype_real":      _thumb_phenotype,
    "browse_disease_real":        _thumb_disease,
    "browse_search_real":         _thumb_search,
    "browse_studies_real":        _thumb_studies,
    "browse_similarity_v7":       _thumb_similarity,
    "browse_metabolism_real":     _thumb_metabolism,
    "analytics_differential_v11": _thumb_differential,
    "analytics_meta_v11":         _thumb_meta,
    "analytics_network_v11":      _thumb_network,
    "analytics_lifecycle_v11":    _thumb_lifecycle,
    "analytics_gbhi_v11":         _thumb_gbhi,
    "export_download_v7":         _thumb_download,
    "export_about_v7":            _thumb_about,
}

def render_thumb(key, dpi=600):
    """Render an inline thumbnail, save to thumbs/ and return as numpy array."""
    fn = _THUMB_FN.get(key)
    if fn is None:
        return None
    fig_t, ax_t = plt.subplots(figsize=(6.0, 5.0))
    fig_t.patch.set_facecolor("white")
    plt.rcParams.update({"font.family":"Arial","font.sans-serif":["Arial","DejaVu Sans"]})
    fn(ax_t)
    # post-scale every text artist (walk entire figure, catch all Text instances)
    import matplotlib.text as _mtext
    FONT_SCALE = 2.2
    for t in fig_t.findobj(_mtext.Text):
        if t.get_text():
            t.set_fontsize(t.get_fontsize() * FONT_SCALE)
    os.makedirs(THUMB_DIR, exist_ok=True)
    out_path = os.path.join(THUMB_DIR, key + ".png")
    fig_t.savefig(out_path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.04,
                  facecolor="white")
    buf = io.BytesIO()
    fig_t.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.04,
                  facecolor="white")
    buf.seek(0)
    img = mpimg.imread(buf)
    plt.close(fig_t)
    return img

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_thumb(name):
    p = os.path.join(THUMB_DIR, name + ".png")
    if os.path.exists(p):
        return mpimg.imread(p)
    return render_thumb(name)

def draw_luxe_shadow(ax, x, y, w, h, rounding=0.16):
    # far halo
    ax.add_patch(FancyBboxPatch(
        (x-0.07, y-0.13), w+0.14, h+0.14,
        boxstyle=f"round,pad=0,rounding_size={rounding+0.05}",
        facecolor="#0A1528", edgecolor="none", alpha=0.055, zorder=1))
    # near shadow
    ax.add_patch(FancyBboxPatch(
        (x+0.028, y-0.048), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor="#0A1528", edgecolor="none", alpha=0.13, zorder=2))

def draw_luxe_band(ax, x, y, w, h, pal):
    # Thin top highlight line + bottom rule
    ax.plot([x+0.02, x+w-0.02], [y+h-0.02, y+h-0.02],
            color=pal["border"], lw=0.85, alpha=0.55, zorder=0.1)
    ax.plot([x+0.02, x+w-0.02], [y+0.02, y+0.02],
            color=pal["border"], lw=0.35, alpha=0.22, zorder=0.1)

def draw_row_header(ax, x0, x1, y_top, row_key, number, label, sublabel, method_line):
    """Horizontal inline row header — [large number] [color bar] LABEL  subtitle · method ..... rule"""
    pal = V4[row_key]
    y_c = y_top - ROW_HEADER_H*0.55
    # large number (navy heavy)
    ax.text(x0, y_c, number, ha="left", va="center",
            fontsize=48, fontweight="black", color=pal["tag_bg"],
            family="Arial", zorder=5)
    # vertical color bar
    bar_x = x0 + 0.52
    ax.plot([bar_x, bar_x], [y_c-0.24, y_c+0.24],
            color=pal["tag_bg"], lw=2.8, solid_capstyle="round", zorder=5)
    # LABEL uppercase (vertically centered in header band)
    ax.text(bar_x+0.14, y_c, label.upper(),
            ha="left", va="center", fontsize=16.0, fontweight="bold",
            color=pal["tag_bg"], zorder=5)
    # sublabel  ·  method  — centered in right margin, away from card badge top area
    ax.text((x0 + x1) / 2.0, y_top - 0.05,
            f"{sublabel}   ·   {method_line}",
            ha="center", va="top", fontsize=14.5, color=MUTE,
            fontstyle="italic", zorder=5)
    ax.plot([x1-0.02, x1-0.02], [y_c-0.08, y_c+0.08],
            color=pal["tag_bg"], lw=0.9, alpha=0.55, zorder=4)

def draw_tag_pill(ax, cx, cy, row_key, number, label, sublabel):
    pal = V4[row_key]
    # gradient pill
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    lighter = "#%02x%02x%02x" % tuple(min(255, int(int(pal["tag_bg"][i:i+2], 16)*1.25))
                                       for i in (1, 3, 5))
    cmap = LinearSegmentedColormap.from_list("t", [lighter, pal["tag_bg"]])
    # shadow
    ax.add_patch(FancyBboxPatch(
        (cx-TAG_W/2+0.02, cy-TAG_H/2-0.04), TAG_W, TAG_H,
        boxstyle="round,pad=0,rounding_size=0.22",
        facecolor="#0A1528", edgecolor="none", alpha=0.22, zorder=2))
    # fill
    ax.imshow(grad, extent=(cx-TAG_W/2, cx+TAG_W/2, cy-TAG_H/2, cy+TAG_H/2),
              aspect="auto", cmap=cmap, zorder=3, interpolation="lanczos")
    # mask to rounded pill via FancyBboxPatch border
    ax.add_patch(FancyBboxPatch(
        (cx-TAG_W/2, cy-TAG_H/2), TAG_W, TAG_H,
        boxstyle="round,pad=0,rounding_size=0.22",
        facecolor="none", edgecolor=pal["tag_bg"], linewidth=1.6, zorder=4))
    # number
    ax.text(cx, cy+0.20, number, ha="center", va="center",
            fontsize=30, fontweight="black", color=pal["tag_fg"], zorder=5)
    # thin divider
    ax.plot([cx-TAG_W*0.36, cx+TAG_W*0.36], [cy-0.12, cy-0.12],
            color=pal["tag_fg"], lw=0.6, alpha=0.45, zorder=5)
    # label
    ax.text(cx, cy-0.30, label, ha="center", va="center",
            fontsize=9.2, fontweight="bold", color=pal["tag_fg"],
            family="Arial", zorder=5)
    # sublabel
    ax.text(cx, cy-0.50, sublabel, ha="center", va="center",
            fontsize=6.3, color=pal["tag_fg"], alpha=0.78, zorder=5,
            fontstyle="italic")

def draw_card(ax, x, y, w, h, mod):
    row, title, hero, detail, thumb_key, crop, badge = mod
    pal = V4[row]
    # shadow
    draw_luxe_shadow(ax, x, y, w, h, rounding=0.14)
    # white card
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.14",
        facecolor="white", edgecolor=pal["border"], linewidth=0.9, zorder=3))
    # left accent bar — signature "Cell/Nature card" look
    ax.add_patch(Rectangle(
        (x+0.03, y+0.06), 0.05, h-0.12,
        facecolor=pal["border"], edgecolor="none", alpha=0.95, zorder=4))
    # thin top hairline
    ax.plot([x+0.12, x+w-0.12], [y+h-0.05, y+h-0.05],
            color=pal["border"], lw=0.5, alpha=0.32, zorder=4)
    # corner method badge (pill)
    if badge:
        bw = 0.07*len(badge) + 0.36
        bh = 0.30
        bx = x + w - bw*0.55
        by = y + h - bh*0.55
        ax.add_patch(FancyBboxPatch(
            (bx, by), bw, bh,
            boxstyle="round,pad=0,rounding_size=0.10",
            facecolor=pal["border"], edgecolor="none", alpha=0.92, zorder=6))
        ax.text(bx+bw/2, by+bh/2, badge, ha="center", va="center",
                fontsize=7.8, color="white", fontweight="bold", zorder=7)

    # --- thumbnail region: top 69% of card (v5.2: eliminated dead space) ---
    t_top_frac = 0.975
    t_bot_frac = 0.285
    t_left_frac = 0.055
    t_right_frac = 0.945
    tx0 = x + w*t_left_frac
    tx1 = x + w*t_right_frac
    ty0 = y + h*t_bot_frac
    ty1 = y + h*t_top_frac
    img = load_thumb(thumb_key)
    if img is not None:
        # if it's a real screenshot (crop hint != None), letterbox-fit with crop
        if crop is not None:
            H, W = img.shape[:2]
            l, t, r, b = crop
            sub = img[int(H*t):int(H*b), int(W*l):int(W*r)]
            ax.imshow(sub, extent=(tx0, tx1, ty0, ty1), aspect="auto", zorder=5,
                      interpolation="lanczos")
            # subtle inner frame for screenshots
            ax.add_patch(FancyBboxPatch(
                (tx0, ty0), tx1-tx0, ty1-ty0,
                boxstyle="round,pad=0,rounding_size=0.04",
                facecolor="none", edgecolor=pal["border"], linewidth=0.8,
                alpha=0.55, zorder=6))
        else:
            ax.imshow(img, extent=(tx0, tx1, ty0, ty1), aspect="auto", zorder=5,
                      interpolation="lanczos")

    # --- bottom text region (29% of card) ---
    # hairline separator
    ax.plot([x+0.10, x+w-0.10], [y+h*0.265, y+h*0.265],
            color=pal["border"], lw=0.5, alpha=0.3, zorder=6)
    # title
    ax.text(x+w/2, y+h*0.200, title, ha="center", va="center",
            fontsize=9.8, fontweight="bold", color=pal["ink"],
            linespacing=1.10, zorder=7)
    # hero
    ax.text(x+w/2, y+h*0.088, hero, ha="center", va="center",
            fontsize=13.2, fontweight="bold", color=pal["hero"], zorder=7)
    # detail
    ax.text(x+w/2, y+h*0.028, detail, ha="center", va="center",
            fontsize=7.2, color=MUTE, zorder=7)

def draw_flow(ax, y_top, y_bot, label=""):
    """v5: thin dotted lines + endpoint circles, replacing triangular funnel"""
    xc = LEFT_PAD + row_usable_w()*0.5
    # three parallel dotted lines
    for dx in (-0.90, 0.0, 0.90):
        ax.plot([xc+dx, xc+dx], [y_top-0.04, y_bot+0.04],
                color="#6A7A90", lw=0.7, ls=(0, (1, 2.6)),
                alpha=0.62, zorder=3)
    # bottom arrow circle
    ax.add_patch(Circle((xc, y_bot+0.04), 0.07,
                        facecolor="#3B5A82", edgecolor="white",
                        linewidth=0.8, zorder=5))
    # top endpoint tick
    ax.plot([xc-0.10, xc+0.10], [y_top-0.03, y_top-0.03],
            color="#3B5A82", lw=1.2, zorder=5)
    if label:
        lbl_dx = 5.60
        ax.text(xc + lbl_dx, (y_top+y_bot)/2, label,
                ha="left", va="center", fontsize=10.5,
                fontweight="bold", color="#3B5A82",
                fontstyle="italic", zorder=6)
        # decorative horizontal extension line
        ax.plot([xc + lbl_dx - 0.15, xc + lbl_dx - 0.03],
                [(y_top+y_bot)/2, (y_top+y_bot)/2],
                color="#3B5A82", lw=0.8, alpha=0.7, zorder=6)

# ── Draw ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.set_aspect("auto")
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Top bar v5 ──────────────────────────────────────────────────────────────
ax.text(LEFT_PAD, FIG_H - 0.40, "a", fontsize=24, fontweight="black", color="#1A1A1A")
# main title
ax.text(FIG_W/2, FIG_H - 0.45,
        "GutBiomeDB",
        ha="center", va="center", fontsize=38, fontweight="black",
        color="#1A1A1A")
ax.text(FIG_W/2, FIG_H - 0.82,
        "an integrated gut microbiome atlas  ·  13 modules  ·  browse → analyse → export",
        ha="center", va="center", fontsize=13.5, fontstyle="italic", color=MUTE)
# gold hairline
ax.plot([LEFT_PAD+0.4, FIG_W-RIGHT_PAD-0.4], [FIG_H-1.05, FIG_H-1.05],
        color=GOLD, lw=1.2, alpha=0.82)
# four metric pills in a row
metric_pills = [
    ("168,464", "samples"),
    ("224",     "diseases"),
    ("482",     "BioProjects"),
    ("72",      "countries"),
]
pill_w, pill_h = 2.40, 0.44
pill_gap = 0.28
total_pills_w = len(metric_pills)*pill_w + (len(metric_pills)-1)*pill_gap
px0 = (FIG_W - total_pills_w)/2
py = FIG_H - 1.45
for i, (num, lab) in enumerate(metric_pills):
    x = px0 + i*(pill_w + pill_gap)
    ax.add_patch(FancyBboxPatch(
        (x, py), pill_w, pill_h,
        boxstyle="round,pad=0,rounding_size=0.16",
        facecolor="white", edgecolor=NAVY, linewidth=0.9, zorder=3))
    ax.text(x+0.22, py+pill_h/2, num, ha="left", va="center",
            fontsize=16, fontweight="black", color=NAVY, zorder=4)
    ax.text(x+pill_w-0.18, py+pill_h/2, lab, ha="right", va="center",
            fontsize=10.0, color=MUTE, fontstyle="italic", zorder=4)

# ── Row headers (inline, replacing large pill badges) ────────────────────────
row_x0 = LEFT_PAD
row_x1 = FIG_W - RIGHT_PAD
draw_row_header(ax, row_x0, row_x1, HEADER_Y_TOP["browse"],    "browse",
                "5", "Browsing",  "entry points",
                "Phenotype · Disease · Taxa · Study · Metabolism")
draw_row_header(ax, row_x0, row_x1, HEADER_Y_TOP["analytics"], "analytics",
                "6", "Analytics", "stats modules",
                "Similarity · Differential · Meta · Network · Lifecycle · GBHI")
draw_row_header(ax, row_x0, row_x1, HEADER_Y_TOP["export"],    "export",
                "2", "Export",    "outlets",
                "REST API · CSV · SVG · PNG · DOI · CC-BY")

# ── Cards ────────────────────────────────────────────────────────────────────
def row_cards(row_key, n_cards):
    usable = row_usable_w()
    cw = CARD_W_ROW[row_key]
    total = n_cards * cw + (n_cards-1) * GAP_X
    start = LEFT_PAD + (usable - total) / 2
    return [start + i*(cw + GAP_X) for i in range(n_cards)]

browse_mods    = [m for m in MODULES if m[0] == "browse"]
analytics_mods = [m for m in MODULES if m[0] == "analytics"]
export_mods    = [m for m in MODULES if m[0] == "export"]

for mods, row_key in [(browse_mods, "browse"),
                       (analytics_mods, "analytics"),
                       (export_mods, "export")]:
    xs = row_cards(row_key, len(mods))
    cy = ROW_Y[row_key]
    for x, mod in zip(xs, mods):
        draw_card(ax, x, cy, CARD_W_ROW[row_key], CARD_H_ROW[row_key], mod)

# ── Flow connectors between bands ───────────────────────────────────────────
draw_flow(ax, BAND_Y["browse"] - 0.02,
          HEADER_Y_TOP["analytics"] - ROW_HEADER_H - 0.02, label="drill-in")
draw_flow(ax, BAND_Y["analytics"] - 0.02,
          HEADER_Y_TOP["export"] - ROW_HEADER_H - 0.02, label="export")

# ── Save ─────────────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    out = os.path.join(OUTPUT_DIR, f"fig1c.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.1,
                facecolor="white")
    print(f"[OK] {out}")
plt.close(fig)
print("v4 render done")
