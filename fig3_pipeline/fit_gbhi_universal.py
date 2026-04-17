"""
Universal GBHI — one multinomial LR for all 9 diseases + NC.

Training set  : all 82,106 NC  +  16,741 (9-disease) samples  = ~99k
Hold-out pool : 69,617 other-disease / unlabeled samples (for LODO)

Output formula (frozen, single pkl):
    Health(s) = P(NC | s) * 100
    where P(c|s) = softmax(W @ z_tilde(s) + b)[c]
    z(s) = [ R_{MH_u ∪ MN_u}(s),  C_frozen(s),  psi_u(s) ]
    z_tilde = (z - mu) / sigma
"""
import os, sys, json, pickle, time
sys.path.insert(0, r"E:\microbiomap_clone\compendium_website\api")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
from main import _inform_label_mask

MIN_N=5; MIN_PROJ=4; TOPN=25; QT=0.1; GT=0.2

def _pool_re(eff):
    if len(eff)<MIN_PROJ: return None
    gs=np.array([e[0] for e in eff]); vs=np.array([e[1] for e in eff])
    w=1/vs; g_fe=(w*gs).sum()/w.sum()
    Q=(w*(gs-g_fe)**2).sum(); df=len(gs)-1
    Cw=w.sum()-(w**2).sum()/w.sum()
    t2=max(0,(Q-df)/Cw) if Cw>0 else 0
    w2=1/(vs+t2); g2=(w2*gs).sum()/w2.sum()
    se=1/np.sqrt(w2.sum()); z=g2/se if se>0 else 0
    return g2,se,2*(1-norm.cdf(abs(z))),t2,len(gs)
def _bh(ps):
    n=len(ps); o=np.argsort(ps); r=np.asarray(ps)[o]
    q=r*n/np.arange(1,n+1); q=np.minimum.accumulate(q[::-1])[::-1]
    out=np.empty(n); out[o]=np.clip(q,0,1); return out
def _hedges(Rmat,y,projv):
    out={}
    for p in np.unique(projv):
        m=projv==p; yp=y[m]
        if yp.sum()<MIN_N or (~yp).sum()<MIN_N: continue
        x1=Rmat[m][yp]; x0=Rmat[m][~yp]
        n1,n0=len(x1),len(x0)
        sp2=((n1-1)*x1.var(0,ddof=1)+(n0-1)*x0.var(0,ddof=1))/(n1+n0-2)
        sp2=np.where(sp2<=0,np.nan,sp2)
        dd=(x1.mean(0)-x0.mean(0))/np.sqrt(sp2)
        J=1-3/(4*(n1+n0)-9); g=J*dd
        v=(n1+n0)/(n1*n0)+g**2/(2*(n1+n0))
        out[p]=(g.astype(np.float64),v.astype(np.float64))
    return out
def _select_markers(gpp):
    if len(gpp)<MIN_PROJ: return [],[]
    pl=list(gpp.keys())
    gs=np.stack([gpp[p][0] for p in pl]); vs=np.stack([gpp[p][1] for p in pl])
    pooled=[]
    for j in range(gs.shape[1]):
        v=np.isfinite(gs[:,j])&np.isfinite(vs[:,j])&(vs[:,j]>0)
        if v.sum()<MIN_PROJ: pooled.append((np.nan,np.nan,np.nan,0,0)); continue
        r=_pool_re(list(zip(gs[v,j],vs[v,j])))
        pooled.append(r if r else (np.nan,np.nan,np.nan,0,0))
    g_p=np.array([p[0] for p in pooled]); ps_p=np.array([p[2] for p in pooled])
    ok=np.isfinite(ps_p); qs=np.full_like(ps_p,np.nan); qs[ok]=_bh(ps_p[ok])
    keep=np.isfinite(g_p)&(np.abs(g_p)>=GT)&(qs<QT)
    ip=np.where(keep&(g_p>0))[0]; inn=np.where(keep&(g_p<0))[0]
    ip=ip[np.argsort(-g_p[ip])][:TOPN]; inn=inn[np.argsort(g_p[inn])][:TOPN]
    return ip.tolist(),inn.tolist()

CACHE    = r"E:\tasks\screenshots\fig1g\v6_cache.npz"
META_CSV = r"E:\microbiomap_clone\data\metadata.csv"
MODEL_DIR= r"E:\tasks\screenshots\fig1g\gbhi_models"
PER_DIS  = MODEL_DIR           # per-disease pkl folder (source of mh/mn)
OUT_PKL  = os.path.join(MODEL_DIR, "gbhi_universal.pkl")
OUT_LOCO = r"E:\tasks\screenshots\fig1g\gbhi_universal_loco.json"
OUT_COEF = r"E:\tasks\screenshots\fig1g\gbhi_universal_coefs.csv"

CLASSES = ["NC","c_difficile_infection","CD","UC","rheumatoid arthritis",
           "HIV","adenoma","obesity","IBS","colorectal_cancer"]
DIS9    = CLASSES[1:]
AB_EPS  = 1e-3

t0=time.time()
print("[univ] loading cache + meta …", flush=True)
c=np.load(CACHE, allow_pickle=True)
G=c["genus_matrix"].astype(np.float32)
nc_mask=c["nc_mask"]
proj_all=np.array([str(x) for x in c["project"]])
n_samp, n_genus = G.shape
meta=pd.read_csv(META_CSV, low_memory=False)

# ─── covariate schema freeze (required for 160k inference) ───
print("[univ] freezing covariate schema …", flush=True)
meta["_amp"]=meta["AMPLICON"].fillna("NA").astype(str).str.lower().str.strip()
meta["_iso"]=meta["iso"].fillna("NA").astype(str).str.upper().str.strip()
meta["_age"]=meta["age_group"].fillna("Unknown").astype(str).str.strip()
meta["_sex"]=meta["sex"].fillna("unknown").astype(str).str.lower().str.strip()
meta["_len"]=np.log10(meta["length"].fillna(meta["length"].median()).astype(float)+1)

# rare-category collapse (frozen thresholds)
AMP_KEEP_MIN=500; ISO_KEEP_MIN=200
amp_vc=meta["_amp"].value_counts(); amp_keep=sorted(amp_vc[amp_vc>=AMP_KEEP_MIN].index.tolist())
iso_vc=meta["_iso"].value_counts(); iso_keep=sorted(iso_vc[iso_vc>=ISO_KEEP_MIN].index.tolist())
meta["_amp"]=meta["_amp"].where(meta["_amp"].isin(amp_keep), other="OTHER")
meta["_iso"]=meta["_iso"].where(meta["_iso"].isin(iso_keep), other="OTHER")
if "OTHER" not in amp_keep: amp_keep.append("OTHER")
if "OTHER" not in iso_keep: iso_keep.append("OTHER")

age_keep=sorted(meta["_age"].unique().tolist())
sex_keep=sorted(meta["_sex"].unique().tolist())

# deterministic (prefix, value) pairs — drop_first per category
dummy_pairs = (
    [("_amp", v) for v in amp_keep[1:]]
  + [("_iso", v) for v in iso_keep[1:]]
  + [("_age", v) for v in age_keep[1:]]
  + [("_sex", v) for v in sex_keep[1:]]
)
dummy_columns = [f"{pre}={val}" for pre,val in dummy_pairs]

def build_covar_matrix(df):
    out=np.zeros((len(df), len(dummy_pairs)), dtype=np.float32)
    for i,(pre,val) in enumerate(dummy_pairs):
        out[:,i]=(df[pre].astype(str).values==val).astype(np.float32)
    return out

covar_mat = build_covar_matrix(meta)             # (n, d)
len_vec   = meta["_len"].values.astype(np.float32).reshape(-1,1)
cov_feat  = np.column_stack([covar_mat, len_vec])   # (n, d+1)
X_full    = np.column_stack([np.ones(n_samp,np.float32), cov_feat])  # intercept

print(f"[univ]   covariate columns frozen: {len(dummy_columns)+1} ({len(dummy_columns)} dummies + length)", flush=True)
print(f"[univ]   amp cats={len(amp_keep)}, iso cats={len(iso_keep)}, age cats={len(age_keep)}, sex cats={len(sex_keep)}", flush=True)

# ─── β̂ on full 168k (global covariate residualization) ───
print("[univ] fitting beta on full 168k …", flush=True)
logG = np.log10(G+AB_EPS).astype(np.float32)
beta = np.linalg.solve(X_full.T @ X_full, X_full.T @ logG)          # (d+2, p)
R    = (logG - X_full @ beta).astype(np.float32)
del logG
print(f"[univ]   residual matrix R: {R.shape}", flush=True)

# ─── run per-disease Hedges' g + DL meta for marker selection, then union ───
print("[univ] running per-disease marker selection (9 diseases)…", flush=True)
mh_union=set(); mn_union=set(); per_dis_info={}
for d in DIS9:
    dm_raw=_inform_label_mask(meta,d)
    dm=dm_raw.values if hasattr(dm_raw,"values") else np.asarray(dm_raw)
    usable=[p for p in np.unique(proj_all)
            if ((proj_all==p)&nc_mask).sum()>=MIN_N and ((proj_all==p)&dm).sum()>=MIN_N]
    tm=np.isin(proj_all,usable)&(nc_mask|dm)
    if tm.sum()<20 or len(usable)<MIN_PROJ:
        print(f"[univ]   {d}: skipped (insufficient cohorts)", flush=True)
        per_dis_info[d]={"n_mh":0,"n_mn":0,"skipped":True}
        continue
    gpp=_hedges(R[tm], dm[tm], proj_all[tm])
    mh,mn=_select_markers(gpp)
    mh_union.update(mh); mn_union.update(mn)
    per_dis_info[d]={"n_mh":len(mh),"n_mn":len(mn),"n_usable_cohorts":len(usable)}
    print(f"[univ]   {d:30s}  MH={len(mh):>3} MN={len(mn):>3} cohorts={len(usable)}", flush=True)
mh_u=sorted(mh_union); mn_u=sorted(mn_union)
union_idx=sorted(set(mh_u)|set(mn_u))
print(f"[univ]   |MH_u|={len(mh_u)}  |MN_u|={len(mn_u)}  overlap={len(set(mh_u)&set(mn_u))}  total={len(union_idx)}", flush=True)

# ─── Gupta psi using union markers (shared across classes) ───
def gupta_psi(Graw, mh, mn):
    EPS=1e-5
    if not mh and not mn: return np.full(Graw.shape[0], 0.0, np.float32)
    gp=np.clip(Graw/100.0, 1e-12, 1.0)
    H=-(gp*np.log(gp)).sum(1)
    def rp(idx):
        if not idx: return 1.0
        pres=(Graw[:,idx]>0).sum(1); v=pres[pres>0]
        return float(np.median(v)) if len(v) else 1.0
    Rh=rp(mh); Rn=rp(mn)
    psi_h=((Graw[:,mh]>0).sum(1)/Rh)*H if mh else np.full(Graw.shape[0], EPS)
    psi_n=((Graw[:,mn]>0).sum(1)/Rn)*H if mn else np.full(Graw.shape[0], EPS)
    return np.log10((psi_h+EPS)/(psi_n+EPS)).astype(np.float32)

print("[univ] computing universal psi for all 168k …", flush=True)
psi_all=np.nan_to_num(gupta_psi(G, mh_u, mn_u), nan=0.0, posinf=0.0, neginf=0.0)

# ─── build feature matrix Z for all 168k ───
#     z = [ R[:,union_idx], cov_feat, psi ]
Z_all = np.column_stack([R[:,union_idx], cov_feat, psi_all.reshape(-1,1)])
print(f"[univ]   Z shape (all): {Z_all.shape}", flush=True)

# ─── label vector: NC=0, 9 diseases=1..9, everything else=-1 (excluded) ───
print("[univ] assigning class labels …", flush=True)
y_full=np.full(n_samp, -1, dtype=np.int8)
y_full[nc_mask]=0
for ci, d in enumerate(DIS9, start=1):
    m=_inform_label_mask(meta, d)
    m=m.values if hasattr(m,"values") else np.asarray(m)
    # if a sample already got a disease label, keep the first (priority by DIS9 order)
    avail=(y_full==-1)|(y_full==0)     # overwrite NC only if it also has disease? No — NC & disease overlap=0 confirmed
    take=m & (y_full<=0)
    # but overlap=0 so simpler:
    y_full[m & (y_full==-1)] = ci
# any sample still -1 = "other disease / unlabeled" → hold out
train_mask = y_full>=0
holdout_mask = y_full==-1

for ci,nm in enumerate(CLASSES):
    print(f"[univ]   class {ci} {nm:30s}: {int((y_full==ci).sum()):>7,}", flush=True)
print(f"[univ]   train total   : {int(train_mask.sum()):,}", flush=True)
print(f"[univ]   holdout total : {int(holdout_mask.sum()):,}", flush=True)

# ─── fit StandardScaler + Multinomial LR on training set ───
print("[univ] fitting StandardScaler …", flush=True)
Z_tr=Z_all[train_mask]; y_tr=y_full[train_mask].astype(int)
sc=StandardScaler(); Z_tr_s=sc.fit_transform(Z_tr)

print("[univ] fitting multinomial LR (class_weight=balanced, C=1.0) …", flush=True)
clf=LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                       class_weight="balanced", n_jobs=-1)
clf.fit(Z_tr_s, y_tr)
print(f"[univ]   converged. coef shape: {clf.coef_.shape}  classes: {clf.classes_.tolist()}", flush=True)

# ─── in-sample diagnostics (per-class NC-vs-disease AUROC) ───
print("[univ] in-sample diagnostics …", flush=True)
probs_tr = clf.predict_proba(Z_tr_s)
pnc_tr   = probs_tr[:,0]
diag={}
for ci,d in enumerate(DIS9, start=1):
    m_dis = (y_tr==ci)
    m_nc  = (y_tr==0)
    if m_dis.sum()<5: continue
    # AUC of "P(disease class ci)" separating disease vs NC  (not P(NC))
    pdi = probs_tr[:,ci]
    y_bin = np.concatenate([np.ones(m_dis.sum()), np.zeros(m_nc.sum())])
    s_bin = np.concatenate([pdi[m_dis],          pdi[m_nc]])
    try:
        auc = roc_auc_score(y_bin, s_bin)
    except Exception:
        auc = float("nan")
    # also 1-P(NC) as alt score
    s_bin2 = np.concatenate([1-pnc_tr[m_dis], 1-pnc_tr[m_nc]])
    auc_hs = roc_auc_score(y_bin, s_bin2)
    diag[d]={"n_dis":int(m_dis.sum()),
             "in_sample_auc_class": float(auc),
             "in_sample_auc_1mPNC": float(auc_hs)}
    print(f"[univ]   {d:30s}  n={int(m_dis.sum()):>5}  AUC(class)={auc:.3f}  AUC(1-PNC)={auc_hs:.3f}", flush=True)

# ─── holdout sanity: P(NC) distribution for 69k other-disease samples ───
print("[univ] holdout LODO sanity …", flush=True)
Z_ho_s = sc.transform(Z_all[holdout_mask])
probs_ho = clf.predict_proba(Z_ho_s)
pnc_ho = probs_ho[:,0]
pnc_nc = pnc_tr[y_tr==0]
summary_lodo={
    "n_holdout": int(holdout_mask.sum()),
    "n_nc_train": int((y_tr==0).sum()),
    "pnc_holdout_mean": float(pnc_ho.mean()),
    "pnc_holdout_median": float(np.median(pnc_ho)),
    "pnc_nc_mean": float(pnc_nc.mean()),
    "pnc_nc_median": float(np.median(pnc_nc)),
}
print(f"[univ]   NC train P(NC) median    : {summary_lodo['pnc_nc_median']:.3f}", flush=True)
print(f"[univ]   holdout  P(NC) median    : {summary_lodo['pnc_holdout_median']:.3f}", flush=True)

# ─── persist the frozen formula ───
print("[univ] persisting gbhi_universal.pkl …", flush=True)
model_blob={
    "version": "gbhi_universal_v1",
    "train_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "class_names": CLASSES,
    "classes_idx": clf.classes_.tolist(),
    "W": clf.coef_,                   # (10, k)
    "b": clf.intercept_,              # (10,)
    "sc_mean": sc.mean_,
    "sc_scale": sc.scale_,
    "beta": beta,                     # (d+2, p) for residualization
    "mh_union": mh_u,
    "mn_union": mn_u,
    "union_idx": union_idx,
    "freeze": {
        "amp_keep": amp_keep,
        "iso_keep": iso_keep,
        "age_keep": age_keep,
        "sex_keep": sex_keep,
        "dummy_columns": dummy_columns,
        "dummy_pairs": dummy_pairs,
        "intercept_first": True,
        "len_last": True,
    },
    "n_genus": int(n_genus),
    "per_dis_info": per_dis_info,
    "n_train": int(train_mask.sum()),
    "n_holdout": int(holdout_mask.sum()),
    "diagnostics": diag,
    "lodo_summary": summary_lodo,
}
with open(OUT_PKL,"wb") as f: pickle.dump(model_blob,f)
print(f"[univ] saved {OUT_PKL}  ({os.path.getsize(OUT_PKL)/1024:.1f} KB)", flush=True)

# coef csv — coefficient export
coef_rows=[]
feature_names=(
    [f"R[{i}]" for i in union_idx]
    + dummy_columns
    + ["log10_length"]
    + ["psi_universal"]
)
for ci,cname in enumerate(CLASSES):
    for fj,fname in enumerate(feature_names):
        coef_rows.append({"class":cname,"feature":fname,"weight":float(clf.coef_[ci,fj])})
    coef_rows.append({"class":cname,"feature":"__intercept__","weight":float(clf.intercept_[ci])})
pd.DataFrame(coef_rows).to_csv(OUT_COEF, index=False)
print(f"[univ] saved {OUT_COEF}", flush=True)

with open(OUT_LOCO,"w",encoding="utf-8") as f:
    json.dump({"diagnostics":diag,"lodo_summary":summary_lodo,
               "n_train":int(train_mask.sum()),"n_holdout":int(holdout_mask.sum()),
               "classes":CLASSES}, f, indent=2)
print(f"[univ] saved {OUT_LOCO}", flush=True)
print(f"[univ] done in {time.time()-t0:.1f}s", flush=True)
