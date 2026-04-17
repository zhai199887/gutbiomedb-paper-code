"""One-shot rebuilder for v6_cache.npz used by fit_gbhi_universal / LOCO.

Reconstructs the cache from the live backend sources:
  - genus_matrix  : rel. abundance matrix from api.get_abundance()
  - sample_key    : row keys
  - project       : BioProject per row (from metadata)
  - nc_mask       : strict NC mask from _strict_nc_mask()

Output: E:/tasks/screenshots/fig1g/v6_cache.npz
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, r"E:/microbiomap_clone/compendium_website")
sys.path.insert(0, r"E:/microbiomap_clone/compendium_website/api")
from api.main import get_abundance, get_metadata, _strict_nc_mask, get_project_column

OUT = r"E:/tasks/screenshots/fig1g/v6_cache.npz"
INFORM_COLS = [f"inform{i}" for i in range(12)]

print("[cache] loading abundance + metadata ...", flush=True)
ab = get_abundance()
me = get_metadata()
print(f"[cache] abundance shape={ab.shape}  meta rows={len(me)}", flush=True)

# align rows: abundance indexed by sample_key, metadata has sample_key column
me = me.set_index("sample_key", drop=False)
common = me.index.intersection(ab.index)
print(f"[cache] common rows: {len(common)}", flush=True)
me = me.loc[common]
ab = ab.loc[common]

pcol = get_project_column(me)
print(f"[cache] project column = {pcol}", flush=True)
proj = me[pcol].astype(str).values

nc = _strict_nc_mask(me, INFORM_COLS).values.astype(bool)
print(f"[cache] NC={int(nc.sum())}  non-NC={int((~nc).sum())}", flush=True)

G = ab.values.astype(np.float32)
print(f"[cache] genus_matrix shape={G.shape}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(
    OUT,
    genus_matrix=G,
    nc_mask=nc,
    project=proj,
    sample_key=np.array([str(x) for x in common], dtype=object),
    genus_columns=np.array([str(c) for c in ab.columns], dtype=object),
)
print(f"[cache] wrote {OUT}  size={os.path.getsize(OUT)/1e6:.1f} MB", flush=True)
