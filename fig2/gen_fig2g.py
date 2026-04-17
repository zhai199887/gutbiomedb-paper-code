# -*- coding: utf-8 -*-
"""
fig1e_circular: CD vs NC SparCC Co-occurrence Network
ggClusterNet/igraph style — circular node layout
Reads from pre-computed cache; does NOT rerun FastSpar.
"""
import os, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import networkx as nx
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
CACHE_JSON = r"E:\tasks\screenshots\fig1e\fig2g_sparcc_data.json"
OUT_DIR    = r"E:\tasks\screenshots\fig2g"
OUT_PNG    = os.path.join(OUT_DIR, "fig2g.png")
OUT_PDF    = os.path.join(OUT_DIR, "fig2g.pdf")

# Parameters (must match cache)
MIN_ABS_R   = 0.30
FDR_THRESH  = 0.05
MAX_SAMPLES = 3000
TOP_GENERA  = 50

# ─── Phylum-grouped gradient colors ──────────────────────────────────────────
# Each phylum → distinct color family (dark → light range)
# Genera within the same phylum share the family but get evenly-spaced shades
_PHYLUM_COLOR_RANGES = {
    # Bacillota / Firmicutes → TEAL
    "Bacillota":               ("#1A6B82", "#B2E8EF"),
    "Firmicutes":              ("#1A6B82", "#B2E8EF"),
    # Bacteroidota / Bacteroidetes → NAVY
    "Bacteroidota":            ("#253D6E", "#8EB3D5"),
    "Bacteroidetes":           ("#253D6E", "#8EB3D5"),
    # Pseudomonadota / Proteobacteria → GREEN-TEAL
    "Pseudomonadota":          ("#00836F", "#91D1C2"),
    "Proteobacteria":          ("#00836F", "#91D1C2"),
    # Actinomycetota / Actinobacteria → SALMON
    "Actinomycetota":          ("#D4735A", "#F7B89A"),
    "Actinobacteria":          ("#D4735A", "#F7B89A"),
    # Fusobacteriota / Fusobacteria → LAVENDER-NAVY
    "Fusobacteriota":          ("#5271A4", "#B3BDDA"),
    "Fusobacteria":            ("#5271A4", "#B3BDDA"),
    # Thermodesulfobacteriota → DARK PURPLE
    "Thermodesulfobacteriota": ("#4A148C", "#CE93D8"),
    # Verrucomicrobiota / Verrucomicrobia → CORAL
    "Verrucomicrobiota":       ("#C41435", "#F9C0B5"),
    "Verrucomicrobia":         ("#C41435", "#F9C0B5"),
    # Default → steel-gray
    "Other":                   ("#424242", "#BDBDBD"),
}


def _gradient_colors(color_dark, color_light, n):
    """Return n hex colors evenly spaced from dark to light."""
    if n <= 0:
        return []
    d = np.array(mcolors.to_rgb(color_dark))
    l = np.array(mcolors.to_rgb(color_light))
    if n == 1:
        rgb = ((d + l) / 2 * 255).clip(0, 255).astype(int)
        return [f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"]
    colors = []
    for i in range(n):
        t = i / (n - 1)
        rgb = ((d * (1 - t) + l * t) * 255).clip(0, 255).astype(int)
        colors.append(f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}")
    return colors


# ─── Qualitative palette for large phyla (≥ LARGE_PHY_THRESH genera) ─────────
# Round-robin across hue families (teal→navy→green→warm→lavender) so adjacent
# genera always have visually distinct colors while staying in the consistent style.
LARGE_PHY_THRESH = 8
_QUAL_PALETTE = [
    "#4DBBD5",  # bright teal
    "#3C5488",  # navy
    "#00A087",  # dark green-teal
    "#F39B7F",  # salmon
    "#8491B4",  # lavender
    "#1DBD9E",  # bright green
    "#E8836A",  # coral-orange
    "#2B8FA3",  # medium teal
    "#6484B8",  # medium blue
    "#56D0B8",  # medium green
    "#D4735A",  # deep salmon
    "#6ECCD8",  # light teal
    "#5271A4",  # dark-medium blue
    "#91D1C2",  # mint
    "#F7B89A",  # light salmon
    "#0D5C70",  # dark teal
    "#7A9DC8",  # light navy
    "#3CCAB0",  # medium-bright green
    "#B3BDDA",  # light lavender
    "#37A8BC",  # teal-bright
    "#4A6BA8",  # medium navy
    "#72D8C2",  # light green
    "#C47C60",  # warm brown-orange
    "#5EC3D2",  # light-medium teal
    "#9CA8C8",  # medium lavender
    "#A8DDD4",  # very light mint
    "#253D6E",  # dark navy
    "#EBA080",  # light orange
    "#5BA4A4",  # teal-gray
    "#8EB3D5",  # steel blue
]


def build_genus_colors(all_genera, genus_to_phylum=None):
    """
    Color assignment:
    - Large phyla (≥ LARGE_PHY_THRESH genera): qualitative round-robin palette
      so each genus has a clearly different color from its neighbors.
    - Small phyla (< threshold): single-family gradient for easy visual grouping.
    Arc bracket colors remain phylum-specific regardless.
    """
    if genus_to_phylum is None:
        genus_to_phylum = {}
    phylum_genera: dict = {}
    for g in sorted(set(all_genera)):
        phy = genus_to_phylum.get(g, "Other")
        phylum_genera.setdefault(phy, []).append(g)
    result = {}
    for phy, genera in phylum_genera.items():
        genera_sorted = sorted(genera)
        if len(genera_sorted) >= LARGE_PHY_THRESH:
            # Qualitative: cycle through multi-hue palette
            for i, g in enumerate(genera_sorted):
                result[g] = _QUAL_PALETTE[i % len(_QUAL_PALETTE)]
        else:
            # Gradient: single hue family, easy to see as a group
            dark, light = _PHYLUM_COLOR_RANGES.get(phy, _PHYLUM_COLOR_RANGES["Other"])
            shades = _gradient_colors(dark, light, len(genera_sorted))
            for g, c in zip(genera_sorted, shades):
                result[g] = c
    return result


def make_genus_legend_handles(genus_colors, genus_to_phylum=None):
    """Return mpatches list for genus legend, sorted by phylum then genus name."""
    from matplotlib.patches import Patch
    if genus_to_phylum:
        items = sorted(genus_colors.items(),
                       key=lambda x: (genus_to_phylum.get(x[0], "Other"), x[0]))
    else:
        items = sorted(genus_colors.items())
    return [Patch(facecolor=c, label=g, linewidth=0.3, edgecolor="#cccccc")
            for g, c in items]

POS_COLOR = "#3CB4E5"
NEG_COLOR = "#E05C5C"


# ─── Circular layout ──────────────────────────────────────────────────────────

def circular_layout(nodes, offset_angle=0.0):
    """Place nodes evenly on unit circle; return {node: (x,y)}."""
    n = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + offset_angle
    return {node: (np.cos(a), np.sin(a)) for node, a in zip(nodes, angles)}


def _node_order(G):
    """Sort nodes: group by phylum, then by degree desc inside phylum."""
    phylum_deg = {}
    for n in G.nodes:
        phy = G.nodes[n].get("phylum", "Other")
        phylum_deg.setdefault(phy, []).append((G.degree(n), n))
    # Sort phyla by total degree (most connected first)
    phy_order = sorted(phylum_deg, key=lambda p: sum(d for d, _ in phylum_deg[p]),
                       reverse=True)
    ordered = []
    for phy in phy_order:
        for _, n in sorted(phylum_deg[phy], reverse=True):
            ordered.append(n)
    return ordered


# ─── Phylum bracket arcs (ggClusterNet style) ────────────────────────────────

def _draw_phylum_arcs(ax, nodes, phylum_of, r_arc=1.30, tick_len=0.06,
                      label_fontsize=13):
    """
    Draw curved bracket arcs per phylum outside the node circle.
    nodes must be in the same order passed to circular_layout (offset=π/2).
    Each phylum gets a colored arc + label at midpoint.
    """
    n_total = len(nodes)
    if n_total < 2:
        return

    angle_step = 2 * np.pi / n_total
    offset = np.pi / 2  # must match circular_layout call

    # Identify consecutive phylum runs (nodes are already grouped by _node_order)
    phylum_runs = []
    current_phy = None
    start_i = 0
    for i, n in enumerate(nodes):
        phy = phylum_of.get(n, "Other")
        if phy != current_phy:
            if current_phy is not None:
                phylum_runs.append((current_phy, start_i, i - 1))
            current_phy = phy
            start_i = i
    if current_phy is not None:
        phylum_runs.append((current_phy, start_i, n_total - 1))

    def _arc_color(phy):
        """Darker shade of phylum family for arc line."""
        dark, _ = _PHYLUM_COLOR_RANGES.get(phy, _PHYLUM_COLOR_RANGES["Other"])
        return dark

    for phy, si, ei in phylum_runs:
        n_phy = ei - si + 1
        arc_col = _arc_color(phy)

        # Angular span for this run (add small padding)
        a_s = offset + si * angle_step - angle_step * 0.45
        a_e = offset + ei * angle_step + angle_step * 0.45

        if n_phy == 1:
            # Singleton: short tick + label (pushed further out to avoid overlap)
            a_mid = offset + si * angle_step
            ax.plot([(r_arc - 0.025) * np.cos(a_mid),
                     (r_arc + tick_len * 0.8) * np.cos(a_mid)],
                    [(r_arc - 0.025) * np.sin(a_mid),
                     (r_arc + tick_len * 0.8) * np.sin(a_mid)],
                    color=arc_col, lw=2.2, zorder=6)
            r_lab  = r_arc + 0.60
            lx_p   = r_lab * np.cos(a_mid)
            ly_p   = r_lab * np.sin(a_mid)
            ha_p   = "left" if np.cos(a_mid) >= 0 else "right"
            ax.text(lx_p, ly_p, phy,
                    fontsize=label_fontsize - 1, ha=ha_p, va="center",
                    color=arc_col, fontfamily="Arial", fontweight="bold",
                    zorder=8,
                    bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                              edgecolor=arc_col, linewidth=1.0, alpha=0.95))
        else:
            # Arc
            theta = np.linspace(a_s, a_e, 150)
            ax.plot(r_arc * np.cos(theta), r_arc * np.sin(theta),
                    color=arc_col, lw=5.0, solid_capstyle="butt", zorder=6)
            # End ticks
            for a in [a_s, a_e]:
                ax.plot([(r_arc - 0.02) * np.cos(a),
                         (r_arc + tick_len) * np.cos(a)],
                        [(r_arc - 0.02) * np.sin(a),
                         (r_arc + tick_len) * np.sin(a)],
                        color=arc_col, lw=2.0, zorder=6)

            # Phylum label at arc midpoint — opaque white bbox with colored border
            a_mid = offset + (si + ei) / 2.0 * angle_step
            r_lab = r_arc + 0.50
            lx_p  = r_lab * np.cos(a_mid)
            ly_p  = r_lab * np.sin(a_mid)
            ha_p  = "left" if np.cos(a_mid) >= 0 else "right"
            ax.text(lx_p, ly_p, phy,
                    fontsize=label_fontsize, ha=ha_p, va="center",
                    color=arc_col, fontfamily="Arial", fontweight="bold",
                    zorder=8,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor=arc_col, linewidth=1.2, alpha=0.96))


# ─── Draw one panel ───────────────────────────────────────────────────────────

def draw_circular(ax, result, title, subtitle, genus_colors=None):
    taxa  = result.get("taxa", [])
    edges = result.get("edges", [])

    if not taxa:
        ax.text(0, 0, "No data", ha="center", va="center", fontsize=11)
        return

    # Build graph
    G = nx.Graph()
    for t in taxa:
        G.add_node(t["taxon"],
                   phylum=t.get("phylum", "Other"),
                   mean_abundance=t.get("mean_abundance", 0))

    pos_elist, neg_elist = [], []
    for e in edges:
        r = e.get("r", 0)
        G.add_edge(e["source"], e["target"], r=r)
        (pos_elist if r >= 0 else neg_elist).append((e["source"], e["target"]))

    G.remove_nodes_from(list(nx.isolates(G)))
    if len(G) == 0:
        ax.text(0, 0, "No connected nodes", ha="center", va="center"); return

    # Node ordering for circular layout (grouped by phylum)
    nodes = _node_order(G)
    pos   = circular_layout(nodes, offset_angle=np.pi / 2)

    # ── Scale up the circle so nodes are more spread out ──────────────────────
    # SCALE only enlarges the circle (node spacing grows); node visual size is FIXED.
    # For no overlap: node_diameter(fixed) < node_spacing(=2π*SCALE/N) → SCALE > 1.97
    SCALE = 3.20
    pos   = {n: (x * SCALE, y * SCALE) for n, (x, y) in pos.items()}

    # Node attributes
    deg     = dict(G.degree())
    max_deg = max(deg.values()) or 1
    abund   = {n: G.nodes[n]["mean_abundance"] for n in nodes}
    ab_vals = np.array(list(abund.values()))
    ab_min, ab_max = ab_vals.min(), ab_vals.max()

    def norm_ab(n):
        return (abund[n] - ab_min) / max(ab_max - ab_min, 1e-9)

    def node_radius(n):
        """Node size is FIXED (not scaled with circle radius).
        Circle grows via SCALE; node diameter stays constant so nodes don't overlap
        once SCALE > 2*(0.085+0.145)/( 2π/N ) ≈ 2.52."""
        return 0.085 + (deg[n] / max_deg * 0.60 + norm_ab(n) * 0.40) * 0.145

    # ── Draw edges ────────────────────────────────────────────────────────────
    all_r = [abs(G.edges[e]["r"]) for e in G.edges]
    max_r = max(all_r) if all_r else 1

    def draw_edge(u, v, color, alpha, lw, ls):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        # Shorten edge so it doesn't overlap node circles
        ru = node_radius(u); rv = node_radius(v)
        d  = np.hypot(x1 - x0, y1 - y0)
        if d < 1e-6: return
        dx, dy = (x1 - x0) / d, (y1 - y0) / d
        ax.plot([x0 + dx * ru, x1 - dx * rv],
                [y0 + dy * ru, y1 - dy * rv],
                color=color, lw=lw, alpha=alpha,
                linestyle=ls, solid_capstyle="round",
                zorder=1)

    # Negative (drawn first = behind)
    for u, v in neg_elist:
        if G.has_edge(u, v):
            w = 1.1 + abs(G.edges[u, v]["r"]) / max_r * 3.5
            draw_edge(u, v, NEG_COLOR, 0.50, w, "--")

    # Positive
    for u, v in pos_elist:
        if G.has_edge(u, v):
            w = 1.1 + abs(G.edges[u, v]["r"]) / max_r * 3.5
            draw_edge(u, v, POS_COLOR, 0.45, w, "-")

    # ── Draw nodes ────────────────────────────────────────────────────────────
    for n in nodes:
        x, y   = pos[n]
        r      = node_radius(n)
        color  = genus_colors.get(n, "#91D1C2") if genus_colors else "#91D1C2"
        # Glow
        circle_glow = plt.Circle((x, y), r * 1.5,
                                  color=color, alpha=0.12, zorder=2)
        ax.add_patch(circle_glow)
        # Main circle
        circle = plt.Circle((x, y), r,
                             facecolor=color, edgecolor="white",
                             linewidth=2.0, alpha=0.93, zorder=3)
        ax.add_patch(circle)

    # ── Labels: ALL nodes — radial rotation so names never overlap ──────────
    for n in nodes:
        x, y   = pos[n]
        r_node = node_radius(n)
        mag    = np.hypot(x, y)
        pad    = r_node + 0.09
        lx     = x / mag * (mag + pad)
        ly     = y / mag * (mag + pad)
        angle_rad = np.arctan2(y, x)
        angle_deg = np.degrees(angle_rad)

        # Radial text: rotate label to point outward; flip on left half
        if np.cos(angle_rad) >= 0:      # right half → read left-to-right
            rot = angle_deg
            ha  = "left"
        else:                            # left half → flip 180° to stay readable
            rot = angle_deg + 180
            ha  = "right"

        ax.text(lx, ly, n,
                fontsize=30, fontfamily="Arial", ha=ha, va="center",
                color="#000000", fontweight="normal", zorder=5,
                rotation=rot, rotation_mode="anchor",
                bbox=dict(boxstyle="round,pad=0.05", facecolor="white",
                          edgecolor="none", alpha=0.72))

    # ── Phylum bracket arcs ───────────────────────────────────────────────────
    phylum_of = {n: G.nodes[n].get("phylum", "Other") for n in nodes}
    _draw_phylum_arcs(ax, nodes, phylum_of, r_arc=SCALE + 4.80, tick_len=0.14,
                      label_fontsize=26)

    # ── Title & stats ─────────────────────────────────────────────────────────
    n_pos = sum(1 for u, v in pos_elist if G.has_edge(u, v))
    n_neg = sum(1 for u, v in neg_elist if G.has_edge(u, v))
    ax.set_title(f"{title}\n{subtitle}",
                 fontsize=52, fontweight="bold",
                 fontfamily="Arial", pad=12, color="#000000")

    stats = f"n={len(G)} genera  ·  {n_pos} positive  ·  {n_neg} negative"
    # Stats text at TOP of axes (between title and arc labels) — fixed position
    ax.text(0.5, 0.97, stats, transform=ax.transAxes,
            ha="center", va="top",
            fontsize=36, color="#000000", fontfamily="Arial")

    # ── Clean frame ────────────────────────────────────────────────────────
    # margin=5.50: arc labels (r_arc+0.50=8.70) fit within xlim (±8.70)
    # ylim asymmetric: more top space → circle sits lower in panel
    margin = 5.50
    ax.set_xlim(-(SCALE + margin), (SCALE + margin))
    # Symmetric ylim: circle centered, slightly raised (+1.20 vs +1.20)
    ax.set_ylim(-(SCALE + margin + 1.20), (SCALE + margin + 1.20))
    ax.set_aspect("equal")
    ax.axis("off")


# ─── Main figure ──────────────────────────────────────────────────────────────

def make_figure():
    print(f"[Cache] Loading {CACHE_JSON}")
    with open(CACHE_JSON, encoding="utf-8") as f:
        results = json.load(f)

    # Unified genus color map across both panels
    all_genera = (
        [t["taxon"] for t in results.get("CD", {}).get("taxa", [])] +
        [t["taxon"] for t in results.get("NC", {}).get("taxa", [])]
    )
    # Build genus → phylum lookup for gradient coloring
    genus_to_phylum: dict = {}
    for group_data in results.values():
        for t in group_data.get("taxa", []):
            genus_to_phylum[t["taxon"]] = t.get("phylum", "Other")
    genus_colors = build_genus_colors(all_genera, genus_to_phylum)

    fig, axes = plt.subplots(1, 2, figsize=(52, 24),
                             gridspec_kw={"wspace": 0.18})
    fig.patch.set_facecolor("white")

    draw_circular(axes[0], results.get("CD", {}),
                  title="Crohn's Disease network",
                  subtitle="CD · n=2173",
                  genus_colors=genus_colors)

    draw_circular(axes[1], results.get("NC", {}),
                  title="Healthy-control network",
                  subtitle=f"NC · n={MAX_SAMPLES}",
                  genus_colors=genus_colors)

    # ── Genus legend + edge legend ─────────────────────────────────────────────
    genus_handles = make_genus_legend_handles(genus_colors, genus_to_phylum)
    edge_handles = [
        Line2D([0], [0], color=POS_COLOR, lw=1.8,
               label="Positive correlation (ρ > 0)"),
        Line2D([0], [0], color=NEG_COLOR, lw=1.8, linestyle="--",
               label="Negative correlation (ρ < 0)"),
    ]
    # Legend: pushed further below axes bottom to add white gap under the plot
    fig.legend(handles=genus_handles + edge_handles,
               loc="lower center", ncol=8, fontsize=34,
               frameon=True, bbox_to_anchor=(0.5, -0.14),
               framealpha=0.96, edgecolor="#dddddd",
               prop={"family": "Arial", "size": 34},
               labelcolor="#000000")

    method_note = (
        f"SparCC (FastSpar v1.0.0) · |ρ| ≥ {MIN_ABS_R} · FDR < {FDR_THRESH} (BH correction) · 100 bootstraps"
    )
    # Method note: just below the legend
    fig.text(0.5, -0.22, method_note, ha="center", fontsize=36,
             color="#000000", fontfamily="Arial", style="italic")

    # axes occupy top 74% of figure; reserve more bottom space so legend doesn't crowd axes
    plt.tight_layout(rect=[0, 0.26, 1, 1])

    os.makedirs(OUT_DIR, exist_ok=True)
    # pad_inches adds extra white margin below so legend sits in clear space
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.9, facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.9, facecolor="white")
    plt.close(fig)
    print(f"\n[Done] Saved:\n  PNG: {OUT_PNG}\n  PDF: {OUT_PDF}")


if __name__ == "__main__":
    make_figure()
    print("[Figure] Complete.")
