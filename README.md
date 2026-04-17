# GutBiomeDB — Manuscript Figure Code

Custom Python scripts used to produce **all main and Extended Data figures** of the GutBiomeDB paper (Nature Methods, under review).

> Zhai J, Dai C, *et al.* **GutBiomeDB: an integrative platform for 168,464 human gut metagenomes across 48 diseases and 4,680 microbial genera.** *Nature Methods* (under review, 2026).

This repository contains **only** the figure-generation code. The GutBiomeDB **platform source code** (FastAPI backend, React frontend, disease-ontology database) lives in a separate repository:

- **Platform repo:** https://github.com/zhai199887/GutBiomeDB
- **Live platform:** https://gutbiomedb.online

---

## Data dependency (important)

Many scripts here import from the platform backend — for example:

```python
from main import _inform_label_mask, get_abundance_matrix
from api.main import DISEASE_ONTOLOGY
```

To reproduce figures you need **either**:

1. Clone the platform repo alongside this one so both live under a shared parent directory and add it to `PYTHONPATH`, **or**
2. Query the deployed REST API at https://gutbiomedb.online/api (slower, but no local setup).

Raw read counts, curated metadata, per-sample GBHI scores, and all Source-Data spreadsheets are available through the platform. The 168,464 × 4,680 analytical matrix and `_disk_cache/` are the **single source of truth** for every number cited in the paper.

---

## Environment

- Python ≥ 3.10
- Install dependencies:

```bash
pip install -r requirements.txt
```

`playwright` is only needed for Fig 1c thumbnails — run `playwright install chromium` once after `pip install`.

---

## Directory layout

```
fig1/               Figure 1 panels (platform overview + covariate coupling)
fig2/               Figure 2 panels (Crohn's disease case study)
fig3/               Figure 3 panels (universal GBHI LOCO validation)
fig3_pipeline/      Upstream OOF + marker-audit scripts for Fig 3 and ED 1–3
ExtendedDataFig1/   External-cohort GBHI validation
ExtendedDataFig2/   Per-disease × per-cohort LOCO grid
ExtendedDataFig3/   OOF score distributions across 9 training diseases
```

### `fig1/` — Figure 1 (platform overview)
| Script | Panel |
|---|---|
| `gen_fig1b_worldmaps.py` | b (choropleth, 72 countries) |
| `capture_thumbs*.py` + `gen_fig1c_platform_overview.py` | c (12-module thumbnails) |
| `prep_fig1d_mantel_corr.py` | d (Mantel coupling, n=1,500, seed=42) |
| `prep_fig1e_partial_spearman.py` | e (partial Spearman ρ, n=10,000, seed=42, BH-FDR) |
| `stitch_fig1.py` | final composite |

### `fig2/` — Figure 2 (Crohn's disease case study)
Scripts `gen_fig2a.py` ~ `gen_fig2f.py` produce the published panels a–f. `gen_fig2.py` assembles the composite. `gen_fig2g.py` / `gen_fig2h.py` are exploratory panels **retained for reviewer transparency** but not part of the published figure.

### `fig3/` — Figure 3 (universal GBHI LOCO validation)
`gen_fig3a.py` ~ `gen_fig3f.py` plus `gen_fig3_composite.py` (final 3×2 composite).

### `fig3_pipeline/` — upstream compute (run first)
| Script | Produces |
|---|---|
| `_build_v6_cache.py` | `v6_cache.npz` from backend matrices |
| `fit_gbhi_universal.py` | Universal softmax GBHI classifier |
| `run_cohort_loco_oof.py` | `gbhi_universal_oof.npz` (OOF probabilities) |
| `run_cohort_loco_marker_audit.py` | Supplementary Table 8 |
| `gen_loco_per_cohort_json.py` | `gbhi_universal_loco_per_cohort.json` |

---

## Reproduction order

```
1.  fig3_pipeline/_build_v6_cache.py
2.  fig3_pipeline/fit_gbhi_universal.py
3.  fig3_pipeline/run_cohort_loco_oof.py
4.  fig3_pipeline/run_cohort_loco_marker_audit.py
5.  fig3_pipeline/gen_loco_per_cohort_json.py

6.  fig1/prep_fig1d_mantel_corr.py
7.  fig1/prep_fig1e_partial_spearman.py
8.  fig1/capture_thumbs_light.py
9.  fig1/gen_fig1b_worldmaps.py
10. fig1/gen_fig1c_platform_overview.py
11. fig1/stitch_fig1.py

12. fig2/gen_fig2*.py

13. fig3/gen_fig3[a-f].py
14. fig3/gen_fig3_composite.py

15. ExtendedDataFig1/figS1_gbhi_validation.py
16. ExtendedDataFig2/figS2_gbhi_loco.py
17. ExtendedDataFig3/gen_figS3_all9.py
```

---

## External codebases cited but not bundled

The comparative analyses in Methods cite these public repositories. They are referenced in the manuscript and not redistributed:

- **metaml** (Pasolli et al., 2016) — https://github.com/SegataLab/metaml
- **GMHI2** (Chang et al., 2024) — https://github.com/jaeyunsung/GMHI2_2023

---

## Citation

If you use this code, please cite:

```bibtex
@article{zhai2026gutbiomedb,
  title   = {GutBiomeDB: an integrative platform for 168{,}464 human gut metagenomes across 48 diseases and 4{,}680 microbial genera},
  author  = {Zhai, Jinxia and Dai, Cong and others},
  journal = {Nature Methods},
  year    = {2026},
  note    = {Under review}
}
```

---

## Contact

- **Corresponding author:** Cong Dai
- **First author:** Jinxia Zhai — zhaijinxia07@gmail.com
- **Platform issues:** https://github.com/zhai199887/GutBiomeDB/issues
- **Paper-code issues:** https://github.com/zhai199887/gutbiomedb-paper-code/issues

---

## License

[MIT](LICENSE) — Copyright © 2026 Jinxia Zhai, Cong Dai, and contributors.
