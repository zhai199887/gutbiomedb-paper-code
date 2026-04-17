# GutBiomeDB paper code

Python scripts that produce the figures and source-data tables in the
GutBiomeDB manuscript.

The live website and the backend/frontend source are in a separate repo:
https://github.com/zhai199887/GutBiomeDB (deployed at https://gutbiomedb.online).

## Setup

    pip install -r requirements.txt

Python 3.10+. Fig 1c thumbnails additionally need a Chromium install:

    playwright install chromium

Most scripts import directly from the platform backend, e.g.

    from main import _inform_label_mask
    from api.main import get_abundance, get_metadata

Clone https://github.com/zhai199887/GutBiomeDB next to this repo and put
its `api/` folder on `PYTHONPATH`, or query the REST endpoints at
https://gutbiomedb.online/api. Per-sample counts, curated metadata and
GBHI scores are all served through the platform.

## Layout

    fig1/               Figure 1 panels
    fig2/               Figure 2 panels (Crohn's disease case study)
    fig3/               Figure 3 panels (universal GBHI LOCO validation)
    fig3_pipeline/      upstream compute: the OOF probability matrix and
                        cohort-LOCO marker-audit tables consumed by
                        Fig 3 and Extended Data Fig 1-3
    ExtendedDataFig1/   external-cohort GBHI validation
    ExtendedDataFig2/   per-disease x per-cohort LOCO grid
    ExtendedDataFig3/   OOF score distributions across the nine training diseases

`fig2/gen_fig2g.py` and `fig2/gen_fig2h.py` are exploratory panels that
did not make it into the published figure; they are kept for transparency.

## Run order

    fig3_pipeline/_build_v6_cache.py
    fig3_pipeline/fit_gbhi_universal.py
    fig3_pipeline/run_cohort_loco_oof.py
    fig3_pipeline/run_cohort_loco_marker_audit.py
    fig3_pipeline/gen_loco_per_cohort_json.py

    fig1/prep_fig1d_mantel_corr.py
    fig1/prep_fig1e_partial_spearman.py
    fig1/capture_thumbs_light.py
    fig1/gen_fig1b_worldmaps.py
    fig1/gen_fig1c_platform_overview.py
    fig1/stitch_fig1.py

    fig2/gen_fig2a.py .. fig2/gen_fig2f.py
    fig2/gen_fig2.py                    # composite

    fig3/gen_fig3a.py .. fig3/gen_fig3f.py
    fig3/gen_fig3_composite.py

    ExtendedDataFig1/figS1_gbhi_validation.py
    ExtendedDataFig2/figS2_gbhi_loco.py
    ExtendedDataFig3/gen_figS3_all9.py

## External tools

Two public codebases are cited in the Methods but not redistributed:

    metaml   https://github.com/SegataLab/metaml        (Pasolli et al., 2016)
    GMHI2    https://github.com/jaeyunsung/GMHI2_2023   (Chang et al., 2024)

## Citation

Zhai J, Li Y, Liu J, Su X, Cui R, Zheng D, Sun Y, Yu J, Dai C.
GutBiomeDB: an integrated human gut microbiome database of 168,464 samples.
Manuscript, 2026.

    @unpublished{zhai2026gutbiomedb,
      title  = {GutBiomeDB: an integrated human gut microbiome database of 168,464 samples},
      author = {Zhai, Jinxia and Li, Yingjie and Liu, Jiameng and Su, Xinyi
                and Cui, Runze and Zheng, Dianyu and Sun, Yuhan and Yu, Jingsheng
                and Dai, Cong},
      year   = {2026},
      note   = {Manuscript}
    }

## Contact

Jinxia Zhai     zhaijinxia07@gmail.com
Cong Dai (PI)   cdai@cmu.edu.cn

## License

MIT (see LICENSE).
