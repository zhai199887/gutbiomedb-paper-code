"""
Cohort-level leave-one-out with per-sample OOF softmax for universal GBHI.

For EACH BioProject p in the full compendium (not just per-disease usable subset):
  - excl = (proj == p)
  - Re-select markers on training cohorts only (~excl), iterating all 9 diseases
  - Re-fit 10-class multinomial LR on training samples (NC + 9 diseases,
    excluding p)
  - Score EVERY held-out sample in p (NC + 9 diseases + unseen diseases + unknown)
  - Write P[n,10] into oof_probs[te, :] and record oof_fold[te] = p

Output: gbhi_universal_oof.npz with keys
  - oof_probs   : (n, 10) float32, NaN for samples never scored OOF
  - oof_fold    : (n,)   U64,     held-out BioProject string, "" if never scored
  - classes     : (10,)  U40,     class names
  - sample_key  : (n,)   U64,     sample identifiers from v6_cache
  - proj        : (n,)   U64,     BioProject per sample
  - per_fold    : list of dicts with diagnostics
"""
import os, sys, json, time, pickle
sys.stdout.reconfigure(encoding="utf-8")
# redirect print to a file inside Python so we don't depend on shell redirect
_LOG_PATH = r"E:\tasks\screenshots\fig1g\run_cohort_loco_oof.log"
_log_fh = open(_LOG_PATH, "w", encoding="utf-8", buffering=1)
class _Tee:
    def __init__(self, *streams): self.s = streams
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from main import _inform_label_mask

CACHE = r"E:\tasks\screenshots\fig1g\v6_cache.npz"
META  = r"E:\microbiomap_clone\data\metadata.csv"
OUT   = r"E:\tasks\screenshots\fig1g\gbhi_universal_oof.npz"

CLASSES = ["NC","c_difficile_infection","CD","UC","rheumatoid arthritis",
           "HIV","adenoma","obesity","IBS","colorectal_cancer"]
DIS9 = CLASSES[1:]
MIN_N=5; MIN_PROJ=4; TOPN=25; QT=0.1; GT=0.2; AB_EPS=1e-3
N_CLS = len(CLASSES)
NC_CI = 0

print("[oof] loading cache + meta ...", flush=True)
c = np.load(CACHE, allow_pickle=True)
G = c["genus_matrix"].astype(np.float32)
nc = c["nc_mask"]; proj = np.array([str(x) for x in c["project"]])
_idkey = "sample_key" if "sample_key" in c.files else ("sample_id" if "sample_id" in c.files else None)
samp_ids = np.array([str(x) for x in c[_idkey]]) if _idkey else np.array([f"row_{i}" for i in range(G.shape[0])])
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

print("[oof] global residualization ...", flush=True)
logG = np.log10(G+AB_EPS).astype(np.float32)
beta = np.linalg.solve(Xf.T@Xf, Xf.T@logG)
R = (logG - Xf@beta).astype(np.float32)
del logG
# Pre-compute Shannon entropy BEFORE releasing G
print("[oof] pre-computing Shannon entropy for psi ...", flush=True)
_gp = np.clip(G/100.0, 1e-12, 1.0)
_H_precomp = -((_gp)*np.log(_gp)).sum(1).astype(np.float32)
del _gp
print(f"[oof] entropy done, H range=[{_H_precomp.min():.2f}, {_H_precomp.max():.2f}]", flush=True)
# Convert G to boolean presence (uint8) — saves ~2.3GB (3.1GB float32 → 0.8GB uint8)
G_bool = (G > 0).astype(np.uint8)
del G
print(f"[oof] G released, G_bool={G_bool.nbytes/1e9:.1f}GB", flush=True)

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

def gupta_psi_fast(mh, mn):
    """Uses pre-computed _H_precomp and G_bool (uint8 presence matrix)."""
    EPS = 1e-5
    if not mh and not mn: return np.zeros(G_bool.shape[0], np.float32)
    H = _H_precomp
    def rp(idx):
        if not idx: return 1.0
        pr = G_bool[:,idx].sum(1); v = pr[pr>0]
        return float(np.median(v)) if len(v) else 1.0
    Rh = rp(mh); Rn = rp(mn)
    ph = (G_bool[:,mh].sum(1)/Rh)*H if mh else np.full(G_bool.shape[0], EPS)
    pn = (G_bool[:,mn].sum(1)/Rn)*H if mn else np.full(G_bool.shape[0], EPS)
    return np.log10((ph+EPS)/(pn+EPS)).astype(np.float32)

# disease masks
dmasks = {}
for d in DIS9:
    dm = _inform_label_mask(meta, d)
    dmasks[d] = dm.values if hasattr(dm,"values") else np.asarray(dm)

# accumulators — try to resume from checkpoint
CKPT = OUT.replace(".npz", ".ckpt.npz")
_resume_from = 0
if os.path.exists(CKPT):
    _ck = np.load(CKPT, allow_pickle=True)
    oof_probs = _ck["oof_probs"].copy()
    oof_fold  = list(_ck["oof_fold"])
    per_fold  = list(_ck["per_fold"])
    _resume_from = int(_ck["last_fold"])
    del _ck
    print(f"[oof] RESUMING from checkpoint at fold {_resume_from}", flush=True)
else:
    oof_probs = np.full((n, N_CLS), np.nan, dtype=np.float32)
    oof_fold  = np.full(n, "", dtype=object)
    per_fold  = []

_all_projects = sorted(np.unique(proj).tolist())

# --- training-pool filter: BioProjects with >=1 NC or >=1 disease sample ---
tp_set = set()
for _p in _all_projects:
    pm = (proj == _p)
    if (pm & nc).sum() >= 1:
        tp_set.add(_p); continue
    for _d in DIS9:
        if (pm & dmasks[_d]).sum() >= 1:
            tp_set.add(_p); break
all_projects = sorted(tp_set)
ext_projects = sorted(set(_all_projects) - tp_set)
print(f"[oof] training-pool: {len(all_projects)}, pure-external: {len(ext_projects)}, total: {len(_all_projects)}, n={n:,}", flush=True)

# --- pre-compute per-(disease, project) Hedges g & v vectors ---
# These only depend on data WITHIN each project, so they are fold-invariant.
print("[oof] pre-computing per-(disease, project) Hedges stats ...", flush=True)
_pre_hedges = {}  # key = (disease, project) -> (g_vec, v_vec)
_pre_nc_count = {}  # key = project -> NC count
_pre_dis_count = {}  # key = (disease, project) -> disease count
for pp in all_projects:
    pm = (proj == pp)
    _pre_nc_count[pp] = int((pm & nc).sum())
    for dd in DIS9:
        dm = dmasks[dd]
        _pre_dis_count[(dd, pp)] = int((pm & dm).sum())
        # compute hedges for this (disease, project) if enough samples
        y_dis = dm[pm]; y_nc = nc[pm]
        # hedges needs disease vs NC within this project
        mask_usable = y_dis | y_nc
        if y_dis[mask_usable].sum() < MIN_N or y_nc[mask_usable].sum() < MIN_N:
            continue
        Rpp = R[pm][mask_usable]
        ypp = y_dis[mask_usable]
        x1 = Rpp[ypp]; x0 = Rpp[~ypp]
        n1, n0 = len(x1), len(x0)
        sp2 = ((n1-1)*x1.var(0, ddof=1) + (n0-1)*x0.var(0, ddof=1)) / (n1+n0-2)
        sp2 = np.where(sp2 <= 0, np.nan, sp2)
        dd_val = (x1.mean(0) - x0.mean(0)) / np.sqrt(sp2)
        J = 1 - 3/(4*(n1+n0)-9)
        g = J * dd_val
        v = (n1+n0)/(n1*n0) + g**2/(2*(n1+n0))
        _pre_hedges[(dd, pp)] = (g.astype(np.float64), v.astype(np.float64))
print(f"[oof] pre-computed {len(_pre_hedges)} (disease, project) Hedges entries", flush=True)

t_start = time.time()
for fi, hold_proj in enumerate(all_projects):
    if fi < _resume_from:
        continue  # skip already-completed folds
    t0 = time.time()
    _tf = time.time()
    excl = (proj == hold_proj)
    n_hold = int(excl.sum())

    # need at least some training data
    if (~excl).sum() < 100:
        per_fold.append({"fold": fi, "cohort": hold_proj, "n_hold": n_hold,
                         "skipped": "insufficient_train"})
        print(f"[oof] {fi+1}/{len(all_projects)} {hold_proj}  n_hold={n_hold}  SKIP (insufficient train)", flush=True)
        continue

    # marker selection across all 9 diseases — use pre-computed Hedges (no matrix re-scan)
    _t_sel = time.time()
    mh_u = set(); mn_u = set()
    for dd in DIS9:
        usable_tr = [pp for pp in all_projects
                     if pp != hold_proj
                     and _pre_nc_count[pp] >= MIN_N
                     and _pre_dis_count.get((dd, pp), 0) >= MIN_N]
        if len(usable_tr) < MIN_PROJ: continue
        # look up pre-computed hedges, skip projects without enough data
        gpp = {pp: _pre_hedges[(dd, pp)] for pp in usable_tr if (dd, pp) in _pre_hedges}
        if len(gpp) < MIN_PROJ: continue
        mh, mn = sel(gpp); mh_u.update(mh); mn_u.update(mn)
    _t_sel = time.time() - _t_sel
    mh_u = sorted(mh_u); mn_u = sorted(mn_u)
    uidx = sorted(set(mh_u) | set(mn_u))
    print(f"[oof] fold {fi+1} sel={_t_sel:.1f}s markers={len(uidx)}", flush=True)
    if len(uidx) < 5:
        per_fold.append({"fold": fi, "cohort": hold_proj, "n_hold": n_hold,
                         "skipped": "few_markers", "n_markers": len(uidx)})
        print(f"[oof] {fi+1}/{len(all_projects)} {hold_proj}  few markers ({len(uidx)})  SKIP", flush=True)
        continue

    # psi on selected markers (computed for all n so we can also transform held-out)
    _t_psi = time.time()
    psi = np.nan_to_num(gupta_psi_fast(mh_u, mn_u), nan=0.0, posinf=0.0, neginf=0.0)
    _t_psi = time.time() - _t_psi
    _t_z = time.time()
    Z = np.column_stack([R[:,uidx], cov_feat, psi.reshape(-1,1)]).astype(np.float32)
    _t_z = time.time() - _t_z

    # labels: NC=0, 9 diseases=1..9, rest=-1 (excluded from training)
    y = np.full(n, -1, np.int8); y[nc & (~excl)] = 0
    for ci, dd in enumerate(DIS9, start=1):
        y[dmasks[dd] & (~excl) & (y==-1)] = ci
    tr = (y >= 0)

    if tr.sum() < 50:
        per_fold.append({"fold": fi, "cohort": hold_proj, "n_hold": n_hold,
                         "skipped": "tiny_train", "n_train": int(tr.sum())})
        print(f"[oof] {fi+1}/{len(all_projects)} {hold_proj}  tiny train ({int(tr.sum())})  SKIP", flush=True)
        continue

    # check we have at least 2 classes present
    uniq_y = np.unique(y[tr])
    if len(uniq_y) < 2:
        per_fold.append({"fold": fi, "cohort": hold_proj, "n_hold": n_hold,
                         "skipped": "single_class"})
        print(f"[oof] {fi+1}/{len(all_projects)} {hold_proj}  single class  SKIP", flush=True)
        continue

    _t_lr = time.time()
    sc = StandardScaler(); Zs_tr = sc.fit_transform(Z[tr])
    clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                             class_weight="balanced", tol=1e-3)
    clf.fit(Zs_tr, y[tr].astype(int))
    _t_lr = time.time() - _t_lr

    # score ALL held-out samples (te = excl), regardless of label
    te = excl
    Zs_te = sc.transform(Z[te])
    P_te = clf.predict_proba(Zs_te)  # (n_hold, k) where k = len(clf.classes_)

    # map back to full 10-class matrix using clf.classes_
    P_full = np.zeros((n_hold, N_CLS), dtype=np.float32)
    for k_idx, c_idx in enumerate(clf.classes_):
        P_full[:, int(c_idx)] = P_te[:, k_idx]

    oof_probs[te, :] = P_full
    fold_arr = np.array([hold_proj]*n_hold, dtype=object)
    for i_global, i_local in zip(np.where(te)[0], range(n_hold)):
        oof_fold[i_global] = hold_proj

    per_fold.append({
        "fold": fi, "cohort": hold_proj, "n_hold": n_hold,
        "n_train": int(tr.sum()),
        "n_markers": len(uidx),
        "n_classes_trained": int(len(uniq_y)),
        "elapsed_s": round(time.time()-t0, 1),
    })
    if (fi+1) % 10 == 0 or fi < 5:
        elapsed = time.time() - t_start
        eta = elapsed / (fi+1) * (len(all_projects) - fi - 1)
        print(f"[oof] {fi+1}/{len(all_projects)} {hold_proj}  "
              f"n_hold={n_hold}  n_tr={int(tr.sum())}  k={len(uidx)}  "
              f"[psi={_t_psi:.1f}s Z={_t_z:.1f}s LR={_t_lr:.1f}s total={time.time()-t0:.1f}s  ETA {eta/60:.1f}m]", flush=True)

    # checkpoint every 10 folds — if we crash we don't lose everything
    if (fi+1) % 10 == 0:
        ckpt = OUT.replace(".npz", ".ckpt.npz")
        np.savez(ckpt,
                 oof_probs=oof_probs,
                 oof_fold=np.array([str(x) for x in oof_fold], dtype=object),
                 classes=np.array(CLASSES, dtype=object),
                 sample_key=samp_ids,
                 proj=proj,
                 per_fold=np.array(per_fold, dtype=object),
                 last_fold=fi+1)
        print(f"[oof]   checkpoint saved @ fold {fi+1} -> {ckpt}", flush=True)

# --- external scoring deferred — model will be retrained on held-out data ---
print(f"\n[oof] NOTE: {len(ext_projects)} pure-external BioProjects NOT scored (separate model rebuild required)", flush=True)

# summary
n_scored = np.isfinite(oof_probs[:, NC_CI]).sum()
print(f"\n[oof] scored {n_scored:,}/{n:,} samples OOF "
      f"({100*n_scored/n:.1f}%)", flush=True)
print(f"[oof] total time: {(time.time()-t_start)/60:.1f} min", flush=True)

oof_fold_str = np.array([str(x) for x in oof_fold], dtype=object)
np.savez(OUT,
         oof_probs = oof_probs,
         oof_fold  = oof_fold_str,
         classes   = np.array(CLASSES, dtype=object),
         sample_key= samp_ids,
         proj      = proj,
         per_fold  = np.array(per_fold, dtype=object))
print(f"[oof] saved {OUT}")
