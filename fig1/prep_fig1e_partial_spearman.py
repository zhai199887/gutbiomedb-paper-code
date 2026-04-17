"""Fig 1e data prep — partial Spearman of genus vs disease/life-stage.

Project rule: life-stage stratification MUST read metadata `age_group` column directly.
Numeric `age` column is NOT used anywhere (neither as covariate nor as stage source).
Unknown / missing age_group samples are dropped.

Outputs (in ./data/):
  genus_abund.csv      samples x top-50 genera   (log10 rel-abund)
  pheno.csv            samples x (disease10 + stage7 + covars)
  r_matrix.csv         50 genera x (20 disease + 7 life stage) partial Spearman r
  p_matrix.csv         same shape, raw p (two-sided)
  q_matrix.csv         BH-FDR adjusted p per column
  group_col.csv        column -> Type {Disease, Life stage}
  group_row.csv        genus  -> Phylum

Covariates (5 per project rule):
  When focal is disease    : country, stage_ord, sex, SeqLength, V_region
  When focal is life stage : country, sex, SeqLength, V_region, disease_any
                             (swap stage->disease to avoid circularity)
"""
import os, json, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# whitelist: canonical diseases only (ontology non-Control, minus blacklist)
ONTO = json.load(open(r"E:\microbiomap_clone\compendium_website\api\disease_ontology.json",
                      "r", encoding="utf-8"))
_BL_PAT = ("PATIENT","EXPOSED","TRANSPLANT","PRETERM","CHICKENPOX","SKIN_CONDITION",
           "LACTOSE","MALIGNANCY","LEUKEMIA","ICU","LTACH","NON-CDI")
DISEASE_WHITELIST = {
    k.upper(): v["standard_name"]
    for k,v in ONTO.items()
    if v.get("category") not in ("Control",) and
       not any(b in k.upper() for b in _BL_PAT)
}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
META  = r"E:\microbiomap_clone\data\metadata.csv"
ABUND = r"E:\microbiomap_clone\data\unfiltered_abundance.csv"
N_SAMPLE = 10000
N_GENUS = 50
N_DISEASE = 20
SEED = 42
rng = np.random.default_rng(SEED)

# ---------------- metadata ----------------
print("[1/6] metadata")
inform_cols = [f"inform{i}" for i in range(12)]
need = ["srr","project","AMPLICON","length","age_group","sex","iso"] + inform_cols
meta = pd.read_csv(META, low_memory=False, usecols=need)
meta = meta.dropna(subset=["srr","project","AMPLICON","length","sex","iso"])

# Life-stage stratification: read age_group column directly (project rule).
# Drop Unknown / missing — those samples don't participate in life-stage analysis.
STAGES = ["Infant","Child","Adolescent","Adult","Older_Adult","Oldest_Old","Centenarian"]
_STAGE_ORD = {s: i+1 for i, s in enumerate(STAGES)}
meta["age_group"] = meta["age_group"].astype(str).str.strip()
meta = meta[meta["age_group"].isin(STAGES)].copy()
meta["stage"]     = meta["age_group"]
meta["stage_ord"] = meta["stage"].map(_STAGE_ORD).astype(int)
meta["sex_num"]    = (meta["sex"].astype(str).str.lower()=="female").astype(int)
meta["length_num"] = pd.to_numeric(meta["length"], errors="coerce")
meta = meta.dropna(subset=["length_num"])

# disease labels (inform0-11 aggregated, per project rule)
def _disease_set(row):
    vals = [str(row[c]).strip().upper() for c in inform_cols if pd.notna(row[c])]
    return [v for v in vals if v and v!="NC" and v in DISEASE_WHITELIST]
meta["disease_set"] = meta.apply(_disease_set, axis=1)
# disease_any uses the FULL inform aggregation (not whitelist-filtered)
# so covariate "any disease" stays meaningful even for samples whose disease is non-canonical
def _any_disease(row):
    vals = [str(row[c]).strip().upper() for c in inform_cols if pd.notna(row[c])]
    return int(any(v and v!="NC" for v in vals))
meta["disease_any"] = meta.apply(_any_disease, axis=1)

# ordinal codes for technical covariates
for c,s in [("iso","country_code"),("project","project_code"),("AMPLICON","amplicon_code")]:
    rk = meta[c].value_counts().rank(method="dense").to_dict()
    meta[s] = meta[c].map(rk)

# curated top-10 headline diseases (platform flagship conditions).
# Order = typical clinical grouping (GI -> metabolic -> infectious -> immune)
CURATED = [
    "UC", "CD", "IBD", "COLORECTAL_CANCER", "ADENOMA", "IBS",
    "OBESITY", "T2D", "T1D", "NAFLD", "NASH",
    "C_DIFFICILE_INFECTION", "HIV", "HBV",
    "SEASONAL_ALLERGIES", "ACID_REFLUX", "ASTHMA", "CELIAC",
    "AUTISM", "ASD", "MULTIPLE_SCLEROSIS", "PARKINSON", "ALZHEIMER",
    "RA", "SLE", "PSORIASIS", "ATOPIC_DERMATITIS",
    "CKD", "PCOS", "AUTOIMMUNE", "MDD", "DEPRESSION", "ANXIETY",
    "COPD", "HYPERTENSION", "COVID_19",
]
# Keep only those with >=100 samples in filtered metadata
exploded = meta.explode("disease_set").dropna(subset=["disease_set"])
avail = exploded["disease_set"].value_counts().to_dict()
top_diseases = [d for d in CURATED if avail.get(d, 0) >= 100][:N_DISEASE]
# backfill from remaining top-counted diseases if curated list came up short
if len(top_diseases) < N_DISEASE:
    extra = [d for d,n in sorted(avail.items(), key=lambda kv: -kv[1])
             if n >= 100 and d not in top_diseases][: N_DISEASE - len(top_diseases)]
    top_diseases = top_diseases + extra
print("  curated diseases:", [(d, avail.get(d,0)) for d in top_diseases])

# stratified sample n=5000 by (disease_any x top10 country)
top10_iso = meta["iso"].value_counts().head(10).index
meta["stratum"] = meta["disease_any"].astype(str)+"_"+meta["iso"].where(meta["iso"].isin(top10_iso),"OTHER")
groups = meta.groupby("stratum")
frac = N_SAMPLE/len(meta)
picked=[]
for _,g in groups:
    n = min(len(g), max(1,int(round(len(g)*frac))))
    picked.append(g.sample(n=n, random_state=SEED))
samp = pd.concat(picked).sample(n=min(N_SAMPLE,sum(len(p) for p in picked)),
                                 random_state=SEED).reset_index(drop=True)
print("  sampled:", samp.shape)

# ---------------- abundance ----------------
print("[2/6] abundance (1-2 min)")
samp["lookup"] = samp["project"].astype(str)+"_"+samp["srr"].astype(str)
wanted = set(samp["lookup"])
chunks=[]
for i,ch in enumerate(pd.read_csv(ABUND, chunksize=20000, low_memory=False)):
    hit = ch[ch["rownames"].astype(str).isin(wanted)]
    if len(hit): chunks.append(hit)
    if (i+1)%10==0: print(f"  chunk {i+1} hits={sum(len(c) for c in chunks)}")
ab = pd.concat(chunks, ignore_index=True).set_index("rownames")
found = [s for s in samp["lookup"] if s in ab.index]
ab = ab.loc[found]
samp = samp.set_index("lookup").loc[found].reset_index()
print("  matched:", ab.shape)

# rel abund, top-50 genera by mean
row_sum = ab.sum(axis=1).replace(0,np.nan)
ab_rel = ab.div(row_sum,axis=0).fillna(0.0)
top_g = ab_rel.mean().sort_values(ascending=False).head(N_GENUS).index
gen = ab_rel[top_g]
# walk taxonomy list; skip placeholder/generic tokens
_PLACEHOLDER = {"na","nan","","none","unknown","group","sedis","incertae",
                "clade","uncultured","gut","metagenome","bacterium","candidatus"}
def _clean_tokens(c):
    return [p for p in c.split(".")
            if p and p.lower() not in _PLACEHOLDER and not p.lower().startswith("unclassified")]
# Normalize old phylum names to modern ones (match fig1e naming)
_PHY_CANON = {
    "Firmicutes":      "Bacillota",
    "Bacteroidetes":   "Bacteroidota",
    "Proteobacteria":  "Pseudomonadota",
    "Actinobacteria":  "Actinomycetota",
    "Fusobacteria":    "Fusobacteriota",
    "Verrucomicrobia": "Verrucomicrobiota",
    "Euryarchaeota":   "Methanobacteriota",
}
def _phylum(c):
    parts = c.split(".")
    if len(parts) >= 2 and parts[1] and parts[1].lower() not in _PLACEHOLDER:
        return _PHY_CANON.get(parts[1], parts[1])
    return "Other"
def _label(c):
    toks = _clean_tokens(c)
    if not toks:
        return "Unclassified"
    last = toks[-1]
    # if last token is too short or numeric-heavy, prepend previous real token
    def _weak(t):
        return (len(t) <= 4) or sum(ch.isdigit() for ch in t) >= max(1, len(t)//2)
    if _weak(last) and len(toks) >= 2:
        return f"{toks[-2]}_{last}"
    return last
raw_labels = [_label(c) for c in gen.columns]
phy_of     = [_phylum(c) for c in gen.columns]
# dedupe
seen = {}
uniq = []
for lab in raw_labels:
    if lab in seen:
        seen[lab]+=1; uniq.append(f"{lab}_{seen[lab]}")
    else:
        seen[lab]=1; uniq.append(lab)
gen.columns = uniq
gen_log = np.log10(gen.values + 1e-6)
gen_df = pd.DataFrame(gen_log, index=samp["srr"].values, columns=gen.columns)
gen_df.index.name="sample_id"

# phylum map for row annotation (keyed on new unique labels)
row_ann = pd.DataFrame({"Phylum": phy_of, "Genus": list(gen.columns)},
                       index=gen.columns)
row_ann.index.name = "RowID"

# --- genus color palette copied from fig1e/gen_fig1e_circular.py -----------
_PHY_RANGES = {
    "Bacillota":               ("#1A6B82", "#B2E8EF"),
    "Bacteroidota":            ("#253D6E", "#8EB3D5"),
    "Pseudomonadota":          ("#00836F", "#91D1C2"),
    "Actinomycetota":          ("#D4735A", "#F7B89A"),
    "Fusobacteriota":          ("#5271A4", "#B3BDDA"),
    "Thermodesulfobacteriota": ("#4A148C", "#CE93D8"),
    "Verrucomicrobiota":       ("#C41435", "#F9C0B5"),
    "Methanobacteriota":       ("#8B7355", "#D7C4A8"),
    "Other":                   ("#424242", "#BDBDBD"),
}
_QUAL = ["#4DBBD5","#3C5488","#00A087","#F39B7F","#8491B4","#1DBD9E",
         "#E8836A","#2B8FA3","#6484B8","#56D0B8","#D4735A","#6ECCD8",
         "#5271A4","#91D1C2","#F7B89A","#0D5C70","#7A9DC8","#3CCAB0",
         "#B3BDDA","#37A8BC","#4A6BA8","#72D8C2","#C47C60","#5EC3D2",
         "#9CA8C8","#A8DDD4","#253D6E","#EBA080","#5BA4A4","#8EB3D5"]
LARGE_PHY_THRESH = 8
def _hex(rgb):
    r,g,b = [int(round(c*255)) for c in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"
def _gradient(dark, light, n):
    import matplotlib.colors as mc
    d = np.array(mc.to_rgb(dark)); l = np.array(mc.to_rgb(light))
    if n == 1: return [_hex(((d+l)/2))]
    return [_hex(d*(1-t) + l*t) for t in np.linspace(0,1,n)]
# group genera by phylum, sorted
by_phy = {}
for g, p in zip(gen.columns, phy_of):
    by_phy.setdefault(p, []).append(g)
genus_color = {}
for p, gs in by_phy.items():
    gs_sorted = sorted(gs)
    if len(gs_sorted) >= LARGE_PHY_THRESH:
        for i, g in enumerate(gs_sorted):
            genus_color[g] = _QUAL[i % len(_QUAL)]
    else:
        d, l = _PHY_RANGES.get(p, _PHY_RANGES["Other"])
        for g, c in zip(gs_sorted, _gradient(d, l, len(gs_sorted))):
            genus_color[g] = c
genus_color_df = pd.DataFrame(
    {"Genus": list(gen.columns),
     "Color": [genus_color[g] for g in gen.columns]})

# ---------------- pheno matrix ----------------
print("[3/6] pheno")
pheno = pd.DataFrame(index=samp["srr"].values)
pheno.index.name="sample_id"
# disease one-hot (stratify sample already done; use whitelist membership)
for d in top_diseases:
    pheno[f"D_{d}"] = samp["disease_set"].apply(lambda s: int(d in s)).values

# drop diseases with too few positives in the final sample (< 30)
min_pos = 30
drop_d = [d for d in top_diseases if pheno[f"D_{d}"].sum() < min_pos]
for d in drop_d:
    pheno.drop(columns=[f"D_{d}"], inplace=True)
top_diseases = [d for d in top_diseases if d not in drop_d]
print(f"  kept {len(top_diseases)} diseases with >={min_pos} positives in sample")
# age stage one-hot
for st in STAGES:
    pheno[f"A_{st}"] = (samp["stage"].values==st).astype(int)
# covariates
pheno["stage_ord"]     = samp["stage_ord"].values
pheno["sex_num"]       = samp["sex_num"].values
pheno["country_code"]  = samp["country_code"].values
pheno["amplicon_code"] = samp["amplicon_code"].values
pheno["length_num"]    = samp["length_num"].values
pheno["disease_any"]   = samp["disease_any"].values

# ---------------- partial spearman ----------------
print("[4/6] partial Spearman")
def _partial_r(y, x, Z):
    """residuals-based partial correlation on ranks (=Spearman partial)."""
    # rank transform
    yr = stats.rankdata(y); xr = stats.rankdata(x)
    Zr = np.column_stack([stats.rankdata(Z[:,j]) for j in range(Z.shape[1])])
    Zr = np.column_stack([np.ones(len(yr)), Zr])
    # OLS residuals
    def _resid(v, M):
        beta,*_ = np.linalg.lstsq(M,v,rcond=None); return v - M@beta
    ey = _resid(yr, Zr); ex = _resid(xr, Zr)
    r,p = stats.pearsonr(ey, ex)
    return r,p

focal_cols = [c for c in pheno.columns if c.startswith("D_") or c.startswith("A_")]
COV_DIS = ["stage_ord","sex_num","country_code","length_num","amplicon_code"]
COV_AGE = ["disease_any","sex_num","country_code","length_num","amplicon_code"]

R = np.full((len(gen_df.columns), len(focal_cols)), np.nan)
P = np.full_like(R, np.nan)
for j,g in enumerate(gen_df.columns):
    for k,f in enumerate(focal_cols):
        covs = COV_AGE if f.startswith("A_") else COV_DIS
        Z = pheno[covs].to_numpy(dtype=float)
        y = gen_df[g].to_numpy(dtype=float)
        x = pheno[f].to_numpy(dtype=float)
        if np.std(x)==0 or np.std(y)==0:
            continue
        try:
            R[j,k],P[j,k] = _partial_r(y, x, Z)
        except Exception:
            pass
    if (j+1)%10==0: print(f"  genus {j+1}/{len(gen_df.columns)}")

# pretty column labels
pretty_cols = []
for f in focal_cols:
    if f.startswith("D_"):
        key = f[2:]
        pretty_cols.append(DISEASE_WHITELIST.get(key, key.title()))
    else:
        pretty_cols.append(f[2:].replace("_", " "))
r_df = pd.DataFrame(R, index=gen_df.columns, columns=pretty_cols)
p_df = pd.DataFrame(P, index=gen_df.columns, columns=pretty_cols)

# BH-FDR per column
q_df = p_df.copy()
for c in q_df.columns:
    v = q_df[c].values
    mask = ~np.isnan(v)
    if mask.sum()>0:
        _,adj,_,_ = multipletests(v[mask], method="fdr_bh")
        v2 = v.copy(); v2[mask]=adj; q_df[c]=v2

col_types = ["Disease"]*len(top_diseases) + ["Life stage"]*7
group_col = pd.DataFrame({"Type": col_types}, index=pretty_cols)
group_col.index.name="Variable"

# ---------------- save ----------------
print("[5/6] save")
os.makedirs(OUT, exist_ok=True)
gen_df.to_csv(os.path.join(OUT,"genus_abund.csv"))
pheno.to_csv(os.path.join(OUT,"pheno.csv"))
r_df.to_csv(os.path.join(OUT,"r_matrix.csv"))
p_df.to_csv(os.path.join(OUT,"p_matrix.csv"))
q_df.to_csv(os.path.join(OUT,"q_matrix.csv"))
group_col.to_csv(os.path.join(OUT,"group_col.csv"))
row_ann.to_csv(os.path.join(OUT,"group_row.csv"))
genus_color_df.to_csv(os.path.join(OUT,"genus_colors.csv"), index=False)

print("[6/6] DONE.")
print("  r_matrix:", r_df.shape, " range", np.nanmin(R), "to", np.nanmax(R))
print("  sig cells (q<0.05):", int((q_df<0.05).sum().sum()))
