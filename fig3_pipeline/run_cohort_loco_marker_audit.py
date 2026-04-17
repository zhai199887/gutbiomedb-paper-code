"""
Pure-meta audit for cohort-LOCO: re-run only the marker-selection stage of
run_cohort_loco_oof.py for all 482 BioProjects, record per-fold AND per-disease
MH / MN counts, dump to gbhi_loco_marker_audit.npz.

No LR fit, no scoring. Much faster than OOF (~5-8s per fold).
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
_LOG_PATH = r"E:\tasks\screenshots\fig1g\run_cohort_loco_marker_audit.log"
_log_fh = open(_LOG_PATH, "w", encoding="utf-8", buffering=1)
class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, x):
        for s in self.s:
            try: s.write(x); s.flush()
            except Exception: pass
    def flush(self):
        for s in self.s:
            try: s.flush()
            except Exception: pass
sys.stdout = _Tee(sys.stdout, _log_fh)
sys.stderr = _Tee(sys.stderr, _log_fh)
sys.path.insert(0, r"E:\microbiomap_clone\compendium_website\api")
import numpy as np, pandas as pd
from scipy.stats import norm
from main import _inform_label_mask

CACHE = r"E:\tasks\screenshots\fig1g\v6_cache.npz"
META  = r"E:\microbiomap_clone\data\metadata.csv"
OUT   = r"E:\tasks\screenshots\fig1g\gbhi_loco_marker_audit.npz"

DIS9 = ["c_difficile_infection","CD","UC","rheumatoid arthritis",
        "HIV","adenoma","obesity","IBS","colorectal_cancer"]
MIN_N=5; MIN_PROJ=4; TOPN=25; QT=0.1; GT=0.2; AB_EPS=1e-3

print("[audit] loading cache + meta ...", flush=True)
c = np.load(CACHE, allow_pickle=True)
G = c["genus_matrix"].astype(np.float32)
nc = c["nc_mask"]; proj = np.array([str(x) for x in c["project"]])
n, p = G.shape

meta = pd.read_csv(META, low_memory=False)
meta["_amp"] = meta["AMPLICON"].fillna("NA").astype(str).str.lower().str.strip()
meta["_iso"] = meta["iso"].fillna("NA").astype(str).str.upper().str.strip()
meta["_age"] = meta["age_group"].fillna("Unknown").astype(str).str.strip()
meta["_sex"] = meta["sex"].fillna("unknown").astype(str).str.lower().str.strip()
meta["_len"] = np.log10(meta["length"].fillna(meta["length"].median()).astype(float)+1)
amp_keep = sorted([k for k,v in meta["_amp"].value_counts().items() if v>=500])+["OTHER"]
iso_keep = sorted([k for k,v in meta["_iso"].value_counts().items() if v>=200])+["OTHER"]
meta["_amp"] = meta["_amp"].where(meta["_amp"].isin(amp_keep[:-1]), other="OTHER")
meta["_iso"] = meta["_iso"].where(meta["_iso"].isin(iso_keep[:-1]), other="OTHER")
age_keep = sorted(meta["_age"].unique().tolist())
sex_keep = sorted(meta["_sex"].unique().tolist())
pairs = ([("_amp",v) for v in amp_keep[1:]]+[("_iso",v) for v in iso_keep[1:]]
        +[("_age",v) for v in age_keep[1:]]+[("_sex",v) for v in sex_keep[1:]])
cov = np.zeros((n,len(pairs)), np.float32)
for i,(pre,val) in enumerate(pairs):
    cov[:,i] = (meta[pre].astype(str).values == val).astype(np.float32)
lenv = meta["_len"].values.astype(np.float32).reshape(-1,1)
cov_feat = np.column_stack([cov, lenv])
Xf = np.column_stack([np.ones(n,np.float32), cov_feat])

print("[audit] global residualization ...", flush=True)
logG = np.log10(G+AB_EPS).astype(np.float32)
beta = np.linalg.solve(Xf.T@Xf, Xf.T@logG)
R = (logG - Xf@beta).astype(np.float32)
del logG

def pool_re(eff):
    if len(eff)<MIN_PROJ: return None
    gs = np.array([e[0] for e in eff]); vs = np.array([e[1] for e in eff])
    w = 1/vs; g_fe = (w*gs).sum()/w.sum()
    Q = (w*(gs-g_fe)**2).sum(); df = len(gs)-1
    Cw = w.sum()-(w**2).sum()/w.sum()
    t2 = max(0,(Q-df)/Cw) if Cw>0 else 0
    w2 = 1/(vs+t2); g2 = (w2*gs).sum()/w2.sum()
    se = 1/np.sqrt(w2.sum()); z = g2/se if se>0 else 0
    return g2, se, 2*(1-norm.cdf(abs(z))), t2, len(gs)

def bh(ps):
    nn = len(ps); o = np.argsort(ps); r = np.asarray(ps)[o]
    q = r*nn/np.arange(1,nn+1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(nn); out[o] = np.clip(q,0,1); return out

def hedges(Rmat, y, projv):
    out = {}
    for pp in np.unique(projv):
        m = projv==pp; yp = y[m]
        if yp.sum()<MIN_N or (~yp).sum()<MIN_N: continue
        x1 = Rmat[m][yp]; x0 = Rmat[m][~yp]
        n1, n0 = len(x1), len(x0)
        sp2 = ((n1-1)*x1.var(0,ddof=1)+(n0-1)*x0.var(0,ddof=1))/(n1+n0-2)
        sp2 = np.where(sp2<=0, np.nan, sp2)
        dd = (x1.mean(0)-x0.mean(0))/np.sqrt(sp2)
        J = 1-3/(4*(n1+n0)-9); g = J*dd
        v = (n1+n0)/(n1*n0)+g**2/(2*(n1+n0))
        out[pp] = (g.astype(np.float64), v.astype(np.float64))
    return out

def sel(gpp):
    if len(gpp)<MIN_PROJ: return [],[]
    pl = list(gpp.keys())
    gs = np.stack([gpp[p][0] for p in pl]); vs = np.stack([gpp[p][1] for p in pl])
    pooled = []
    for j in range(gs.shape[1]):
        v = np.isfinite(gs[:,j])&np.isfinite(vs[:,j])&(vs[:,j]>0)
        if v.sum()<MIN_PROJ: pooled.append((np.nan,np.nan,np.nan,0,0)); continue
        r = pool_re(list(zip(gs[v,j], vs[v,j])))
        pooled.append(r if r else (np.nan,np.nan,np.nan,0,0))
    gP = np.array([pp[0] for pp in pooled]); psP = np.array([pp[2] for pp in pooled])
    ok = np.isfinite(psP); qs = np.full_like(psP, np.nan); qs[ok] = bh(psP[ok])
    keep = np.isfinite(gP)&(np.abs(gP)>=GT)&(qs<QT)
    ip = np.where(keep&(gP>0))[0]; inn = np.where(keep&(gP<0))[0]
    ip = ip[np.argsort(-gP[ip])][:TOPN]; inn = inn[np.argsort(gP[inn])][:TOPN]
    return ip.tolist(), inn.tolist()

dmasks = {}
for d in DIS9:
    dm = _inform_label_mask(meta, d)
    dmasks[d] = dm.values if hasattr(dm,"values") else np.asarray(dm)

all_projects = sorted(np.unique(proj).tolist())
n_proj = len(all_projects)
print(f"[audit] iterating {n_proj} BioProjects", flush=True)

# per-fold records
rec_cohort        = []                    # (n_proj,) str
rec_n_hold        = np.zeros(n_proj, np.int32)
rec_union_n       = np.zeros(n_proj, np.int32)
rec_mh_n          = np.zeros(n_proj, np.int32)
rec_mn_n          = np.zeros(n_proj, np.int32)
rec_overlap_n     = np.zeros(n_proj, np.int32)
rec_per_dis_mh    = np.zeros((n_proj, len(DIS9)), np.int32)
rec_per_dis_mn    = np.zeros((n_proj, len(DIS9)), np.int32)
rec_per_dis_k     = np.zeros((n_proj, len(DIS9)), np.int32)
rec_elapsed       = np.zeros(n_proj, np.float32)

t_start = time.time()
for fi, hold_proj in enumerate(all_projects):
    t0 = time.time()
    excl = (proj == hold_proj)
    rec_cohort.append(hold_proj)
    rec_n_hold[fi] = int(excl.sum())

    mh_u = set(); mn_u = set()
    for di, dd in enumerate(DIS9):
        dm = dmasks[dd] & (~excl)
        usable_tr = [pp for pp in all_projects
                     if pp != hold_proj
                     and ((proj==pp) & nc & (~excl)).sum() >= MIN_N
                     and ((proj==pp) & dm).sum() >= MIN_N]
        rec_per_dis_k[fi, di] = len(usable_tr)
        tm = np.isin(proj, usable_tr) & (nc|dm) & (~excl)
        if tm.sum() < 20 or len(usable_tr) < MIN_PROJ: continue
        gpp = hedges(R[tm], dm[tm], proj[tm])
        mh, mn = sel(gpp)
        rec_per_dis_mh[fi, di] = len(mh)
        rec_per_dis_mn[fi, di] = len(mn)
        mh_u.update(mh); mn_u.update(mn)

    mh_set = set(mh_u); mn_set = set(mn_u)
    union  = mh_set | mn_set
    rec_mh_n[fi]      = len(mh_set)
    rec_mn_n[fi]      = len(mn_set)
    rec_overlap_n[fi] = len(mh_set & mn_set)
    rec_union_n[fi]   = len(union)
    rec_elapsed[fi]   = time.time() - t0

    if (fi+1) % 20 == 0 or fi < 3:
        elapsed = time.time() - t_start
        eta = elapsed / (fi+1) * (n_proj - fi - 1)
        print(f"[audit] {fi+1}/{n_proj} {hold_proj}  n_hold={rec_n_hold[fi]}  "
              f"MH={rec_mh_n[fi]} MN={rec_mn_n[fi]} overlap={rec_overlap_n[fi]} "
              f"union={rec_union_n[fi]}  [{rec_elapsed[fi]:.1f}s ETA {eta/60:.1f}m]",
              flush=True)

print(f"\n[audit] total time: {(time.time()-t_start)/60:.1f} min", flush=True)
print(f"[audit] MH  range: {rec_mh_n.min()}..{rec_mh_n.max()}  median {int(np.median(rec_mh_n))}", flush=True)
print(f"[audit] MN  range: {rec_mn_n.min()}..{rec_mn_n.max()}  median {int(np.median(rec_mn_n))}", flush=True)
print(f"[audit] union range: {rec_union_n.min()}..{rec_union_n.max()}  median {int(np.median(rec_union_n))}", flush=True)

np.savez(OUT,
         cohort        = np.array(rec_cohort, dtype=object),
         n_hold        = rec_n_hold,
         mh_n          = rec_mh_n,
         mn_n          = rec_mn_n,
         overlap_n     = rec_overlap_n,
         union_n       = rec_union_n,
         per_dis_mh    = rec_per_dis_mh,
         per_dis_mn    = rec_per_dis_mn,
         per_dis_k     = rec_per_dis_k,
         elapsed_s     = rec_elapsed,
         diseases      = np.array(DIS9, dtype=object))
print(f"[audit] saved {OUT}")
