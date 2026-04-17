"""Fig1 correlation heatmap — Phase 1 data prep.

Stratified-sample n=3000 from GutBiomeDB metadata; build host-var matrix,
phylum-level and genus-level abundance matrices.

Project rule: life-stage source MUST be metadata `age_group` column directly.
The host-var "Age" column carries the 1..7 ordinal life-stage code (Infant=1,
Child=2, Adolescent=3, Adult=4, Older_Adult=5, Oldest_Old=6, Centenarian=7).
Numeric `age` column is NOT used. Samples with age_group=Unknown are dropped.

Outputs (in same dir as this script):
  host_vars.csv
  phylum_matrix.csv
  genus_matrix.csv
  sample_ids.txt
"""
import os
import numpy as np
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH = r"E:\microbiomap_clone\data\metadata.csv"
ABUND_PATH = r"E:\microbiomap_clone\data\unfiltered_abundance.csv"
N_SAMPLE = 3000
SEED = 42

rng = np.random.default_rng(SEED)

print("[1/5] Load metadata ...")
meta = pd.read_csv(META_PATH, low_memory=False)
print("  raw shape:", meta.shape)

# Keep only rows with all required fields.
# Project rule: disease label source = inform0-11 (inform-all must NOT be used).
inform_cols = [f"inform{i}" for i in range(12)]
need = ["srr", "project", "AMPLICON", "length", "age_group", "sex", "iso"] + inform_cols
meta = meta[need].copy()
meta = meta.dropna(subset=["srr", "project", "AMPLICON", "length", "sex", "iso"])

def disease_from_inform(row):
    """0 = healthy (any inform0-11 == NC with no non-NC labels) / 1 = disease (has non-NC label).
    All-empty inform0-11 treated as unknown -> NaN, dropped downstream."""
    vals = [str(row[c]).strip().upper() for c in inform_cols if pd.notna(row[c])]
    if not vals:
        return np.nan
    non_nc = [v for v in vals if v != "NC" and v != ""]
    if non_nc:
        return 1
    return 0

meta["disease_status"] = meta.apply(disease_from_inform, axis=1)
meta = meta.dropna(subset=["disease_status"])
meta["disease_status"] = meta["disease_status"].astype(int)

# Life stage from age_group column (project rule: never derive from numeric `age`)
LIFE_STAGES = ["Infant","Child","Adolescent","Adult","Older_Adult","Oldest_Old","Centenarian"]
_STAGE_ORD  = {s: i+1 for i, s in enumerate(LIFE_STAGES)}
meta["age_group"] = meta["age_group"].astype(str).str.strip()
meta = meta[meta["age_group"].isin(LIFE_STAGES)].copy()
meta["stage_ord"] = meta["age_group"].map(_STAGE_ORD).astype(int)

# Sex encode
meta["sex_num"] = (meta["sex"].astype(str).str.lower() == "female").astype(int)

# Length numeric
meta["length_num"] = pd.to_numeric(meta["length"], errors="coerce")
meta = meta.dropna(subset=["length_num"])

# Country -> integer code (Gower-style, treat as categorical distance later);
# for Pearson, we use samples-per-country rank as a pseudo-continuous proxy.
country_rank = meta["iso"].value_counts().rank(method="dense").to_dict()
meta["country_code"] = meta["iso"].map(country_rank)

# BioProject -> rank code
proj_rank = meta["project"].value_counts().rank(method="dense").to_dict()
meta["project_code"] = meta["project"].map(proj_rank)

# Amplicon V-region -> ordinal code by frequency
amp_rank = meta["AMPLICON"].value_counts().rank(method="dense").to_dict()
meta["amplicon_code"] = meta["AMPLICON"].map(amp_rank)

print("  after clean:", meta.shape)

print("[2/5] Stratified sample n=%d by disease_status x continent-like iso-bucket ..." % N_SAMPLE)
# Stratify by disease_status + top-10 countries, else 'OTHER'
top10 = meta["iso"].value_counts().head(10).index
meta["stratum"] = meta["disease_status"].astype(str) + "_" + meta["iso"].where(meta["iso"].isin(top10), "OTHER")

groups = meta.groupby("stratum")
frac = N_SAMPLE / len(meta)
picked = []
for _, g in groups:
    n = max(1, int(round(len(g) * frac)))
    n = min(n, len(g))
    picked.append(g.sample(n=n, random_state=SEED))
samp = pd.concat(picked).sample(n=min(N_SAMPLE, sum(len(p) for p in picked)), random_state=SEED)
samp = samp.reset_index(drop=True)
print("  sampled:", samp.shape)

# Host-vars matrix: 7 columns. "Life stage" carries the 1..7 age_group ordinal.
host = pd.DataFrame({
    "LifeStage": samp["stage_ord"].values,
    "Sex": samp["sex_num"].values,
    "Disease": samp["disease_status"].values,
    "Country": samp["country_code"].values,
    "BioProject": samp["project_code"].values,
    "V_region": samp["amplicon_code"].values,
    "SeqLength": samp["length_num"].values,
}, index=samp["srr"].values)
host.index.name = "sample_id"

print("[3/5] Load abundance (this reads 1.6 GB, takes ~1-2 min) ...")
# Abundance rownames look like "PROJECT_SRR". Build lookup key accordingly.
samp["lookup"] = samp["project"].astype(str) + "_" + samp["srr"].astype(str)
wanted = set(samp["lookup"].values)
chunks = []
for i, chunk in enumerate(pd.read_csv(ABUND_PATH, chunksize=20000, low_memory=False)):
    hit = chunk[chunk["rownames"].astype(str).isin(wanted)]
    if len(hit):
        chunks.append(hit)
    if (i + 1) % 5 == 0:
        print(f"  chunk {i+1} cumulative hits={sum(len(c) for c in chunks)}")
ab = pd.concat(chunks, ignore_index=True)
ab = ab.set_index("rownames")
# Reindex to sampled order; drop samples with no abundance row
found = [s for s in samp["lookup"].values if s in ab.index]
ab = ab.loc[found]
# Re-align host and samp
samp_by_key = samp.set_index("lookup").loc[found]
host.index = samp.set_index("srr").index  # original srr index
host = pd.DataFrame(host.loc[samp_by_key["srr"].values].values,
                    index=found, columns=host.columns)
host.index.name = "sample_id"
samp = samp_by_key.reset_index()
print("  abundance matched:", ab.shape)

# Relative abundance (row sums -> 1)
row_sums = ab.sum(axis=1).replace(0, np.nan)
ab_rel = ab.div(row_sums, axis=0).fillna(0.0)

print("[4/5] Build Phylum-level matrix (aggregate genera by phylum prefix) ...")
def phylum_of(col):
    parts = col.split(".")
    if len(parts) >= 2:
        return parts[1]
    return "Unknown"

phylum_map = {c: phylum_of(c) for c in ab_rel.columns}
phy = ab_rel.T.groupby(phylum_map).sum().T
# Keep top-15 phyla by mean abundance
top_phy = phy.mean().sort_values(ascending=False).head(15).index
phy = phy[top_phy]
print("  phyla kept:", list(phy.columns))

print("[5/5] Build Genus matrix (top 500 by mean rel-abund) ...")
top_genus = ab_rel.mean().sort_values(ascending=False).head(500).index
gen = ab_rel[top_genus]
# Shorten column names to the last part after '.'
gen.columns = [c.split(".")[-1] for c in gen.columns]

host.to_csv(os.path.join(OUT_DIR, "host_vars.csv"))
phy.to_csv(os.path.join(OUT_DIR, "phylum_matrix.csv"))
gen.to_csv(os.path.join(OUT_DIR, "genus_matrix.csv"))
with open(os.path.join(OUT_DIR, "sample_ids.txt"), "w") as f:
    f.write("\n".join(host.index.astype(str)))

print("DONE.")
print("  host_vars:", host.shape, "->", os.path.join(OUT_DIR, "host_vars.csv"))
print("  phylum:   ", phy.shape, "->", os.path.join(OUT_DIR, "phylum_matrix.csv"))
print("  genus:    ", gen.shape, "->", os.path.join(OUT_DIR, "genus_matrix.csv"))
