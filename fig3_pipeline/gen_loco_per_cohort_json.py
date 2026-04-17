"""Generate gbhi_universal_loco_per_cohort.json from OOF data."""
import sys, json
import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, r"E:\microbiomap_clone\compendium_website\api")
from main import _inform_label_mask
import pandas as pd

CACHE   = r"E:\tasks\screenshots\fig1g\v6_cache.npz"
META    = r"E:\microbiomap_clone\data\metadata.csv"
OOF_NPZ = r"E:\tasks\screenshots\fig1g\gbhi_universal_oof.npz"
OUT     = r"E:\tasks\screenshots\fig1g\gbhi_universal_loco_per_cohort.json"

CLASSES = ["NC","c_difficile_infection","CD","UC","rheumatoid arthritis",
           "HIV","adenoma","obesity","IBS","colorectal_cancer"]
DIS9 = CLASSES[1:]

c = np.load(CACHE, allow_pickle=True)
nc_mask = c["nc_mask"]
proj = np.array([str(x) for x in c["project"]])

oof = np.load(OOF_NPZ, allow_pickle=True)
P = oof["oof_probs"].astype(np.float32)
OOF_CLASSES = [str(x) for x in oof["classes"]]
NC_CI = OOF_CLASSES.index("NC")
P_nc = P[:, NC_CI]
dys = 1.0 - P_nc
has_oof = np.isfinite(P_nc)

meta = pd.read_csv(META, low_memory=False)

def auroc(score, y):
    y = np.asarray(y).astype(bool)
    pos = score[y]; neg = score[~y]
    if len(pos) < 2 or len(neg) < 2: return None
    u, _ = mannwhitneyu(pos, neg, alternative="two-sided")
    a = u / (len(pos) * len(neg))
    return float(max(a, 1 - a))

result = {"per_disease": {}}
for dis in DIS9:
    dm = _inform_label_mask(meta, dis)
    dm = dm.values if hasattr(dm, "values") else np.asarray(dm)

    per_cohort = []
    all_projects = sorted(np.unique(proj).tolist())
    for pp in all_projects:
        pm = (proj == pp) & has_oof
        n_nc = int((pm & nc_mask).sum())
        n_dis = int((pm & dm).sum())
        if n_nc < 2 or n_dis < 2:
            continue
        sel = pm & (nc_mask | dm)
        auc = auroc(dys[sel], dm[sel])
        per_cohort.append({
            "cohort": pp,
            "n_nc": n_nc,
            "n_dis": n_dis,
            "auc_universal": auc
        })

    valid_aucs = [co["auc_universal"] for co in per_cohort if co["auc_universal"] is not None]
    auc_mean = float(np.mean(valid_aucs)) if valid_aucs else None

    result["per_disease"][dis] = {
        "skipped": False,
        "auc_mean": auc_mean,
        "per_cohort": per_cohort
    }
    print(f"{dis:30s}  cohorts={len(per_cohort):>3}  mean_auc={auc_mean}", flush=True)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1, default=str)
print(f"saved {OUT}")
