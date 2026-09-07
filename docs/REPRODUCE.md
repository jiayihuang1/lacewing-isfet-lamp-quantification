# Reproducing the dissertation results

This document gives the end-to-end command sequence for reproducing
every number in the report, starting from **raw chip data** — the
`.bin`-file trees that the Lacewing platform captures. No pre-built
caches assumed.

## Prerequisites

- Python 3.12
- PyTorch 2.x with CUDA (for training) or CPU-only (evaluation only)
- Imperial HPC access if running the full experiment suite
- Two clones of the `titan-signal-processing` package (available on
  request from the Centre for Bio-Inspired Technology at Imperial):
  the main-branch clone for CoV, plus the `Matthew_Multi` branch for
  KP multi-Vref chips
- Raw chip data + paired plate-qLAMP exports (same source as titan)

## Step-by-step

### 1. Environment setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

The `-e .` step installs the `lacewing` package itself. Without it,
every `python -m lacewing.…` command below fails with
`ModuleNotFoundError: No module named 'lacewing'`.

### 2. Set environment variables

Point the three location env vars at your local copies. All commands
below expect them to be set.

    export LACEWING_TITAN_PATH=/path/to/titan-signal-processing
    export LACEWING_TITAN_MULTI_PATH=/path/to/titan-signal-processing-multi
    export LACEWING_DATA_ROOT=/path/to/parent/of/Data

`LACEWING_DATA_ROOT` points at the *parent* of `Data/`. If your tree
is `/foo/bar/Data/24_CoV_Quantification/…`, set it to `/foo/bar`.

### 3. Lay out the raw `Data/` folder

The code expects this exact structure under `$LACEWING_DATA_ROOT`:

    Data/
    ├── 24_CoV_Quantification/          # SARS-CoV-2 dataset
    │   ├── 1e5/D20240821_E01_C07_F4500KHz_U_1e5/     # 5 dose chips
    │   ├── 1e6/D20240821_E01_C09_F4500KHz_U_1e6/
    │   ├── 1e7/D20240820_E02_C03_F4500KHz_U_1e7/
    │   ├── 1e8/D20240822_E02_C00_F4500KHz_U_1e8/
    │   ├── 1e9/D20240822_E02_C00_F4500KHz_U_1e9/
    │   └── Others/                     # NTC + multiplex chips
    │       ├── D20240719_E03_C44_F4500KHz_U_COV_SD/  # winning SD test chip
    │       └── D20240719_E05_C48_F4500KHz_U_COV_SD/  # backup SD test chip
    │       └── … (~30 more NTC-only + multiplex chips)
    │
    └── Concentration Data Experiment/  # K. pneumoniae dataset
        ├── D20260812_E00_C00_F4500KHz_U_KP_conc_01/
        ├── D20260813_E00_C00_F4500KHz_U_DDM_KP_con_02/
        ├── D20260820_E00_C00_F4500KHz_U_DDM_KP_conc_03/
        ├── D20260823_E00_C00_F4500KHz_U_DDM_KP_04_05/
        ├── D20260827_E00_C00_F4500KHz_U_DDM_KP_Conc_05/
        ├── Chip 1.xlsx                 # paired LC96 qLAMP export (chip 1)
        ├── Chip 2.xlsx                 # paired LC96 qLAMP export (chip 2)
        ├── Chip Order.xlsx             # well-layout convention
        └── plate_features.json         # parsed plate anchors

Each chip folder contains ~50 `.bin` files (find_active, vref_sweep,
evaluate_pixel, vs_calibration, gain, readout) plus PNG diagnostics.
Standard Lacewing capture format.

Well-layout conventions (which well is which sample) are documented in
[`DATASETS.md`](DATASETS.md) — CoV is a fixed 6-well layout, KP is a
10-well grid with concentrations mapped per row.

### 4. Build preprocessing caches from raw

Every ML method reads from a per-dataset `.npz` cache. Cache-building
runs the titan chip loader + the MAD-ABCD data-quality filter + the
spatA3 spatial averaging on every chip, and writes the result as a
single `.npz` per configuration.

#### 4a. SARS-CoV-2 caches

Three commands (run each in order — later stages read the earlier
cache):

    # (i) baseline: deployed pipeline only (layer D of the outlier filter)
    python -m lacewing.classification.data.build_filtered_dataset

    # (ii) +MAD data-quality filter (layers A + B + C + D, k=1.5)
    python -m lacewing.classification.data.build_filtered_dataset_mad \
        --scope all --layers ABCD --k 1.5

    # (iii) +spatA3 spatial averaging (RQ1 WINNER)
    python -m lacewing.classification.data.build_spatial_filtered_dataset \
        --input dataset_all_filt_abcd_ntcRaw_madk1p5 --order 3

Each command reads the raw `.bin` files under
`$LACEWING_DATA_ROOT/Data/24_CoV_Quantification/` and writes one
`.npz` under
`src/lacewing/classification/data/cache/`.

HPC one-liner that runs (ii) and (iii) in one job:

    qsub jobs/build_cache_cov_full_preproc.pbs

Wall-clock: ~5 h on HPC per full run of steps ii+iii (~150 chip loads
via titan). Each cache lands as e.g.
`dataset_all_filt_abcd_ntcRaw_madk1p5_spatA3.npz` (~250 MB).

The RQ2/RQ3 regression pipeline needs one more cache — a
regression-labelled version of the same spatA3 cache. This one
attaches per-pixel TTP labels from the paired plate qLAMP export:

    python -m lacewing.quantification.regression_shared.data.build_regression_cache

Output: `src/lacewing/quantification/regression_shared/data/cache/regress_all_filt_abcd_ntcRaw_madk1p5_spatA3.npz`.

#### 4b. K. pneumoniae caches

KP uses a different chip loader (`titan-signal-processing-multi`) and
a different well-layout convention (10 wells, per-row concentrations).

    # (i) preprocess all 5 KP chips through the MAD-ABCD + spatA3 pipeline
    python -m lacewing.quantification.conc_data.process_conc_chips

    # (ii) build 5 per-fold LOCO caches (leave-one-chip-out cross-validation)
    python -m lacewing.quantification.conc_data.build_regression_cache
    python -m lacewing.quantification.conc_data.build_classification_cache

Outputs land in
`src/lacewing/quantification/regression_shared/data/cache/regress_conc_loco{1..5}.npz`
(regression, one per fold) and
`src/lacewing/classification/data/cache/dataset_conc*.npz`
(classification).

Wall-clock: ~2 h on HPC (5 chip loads plus per-fold cache assembly).

#### 4c. Verify caches

    ls -lh src/lacewing/classification/data/cache/
    ls -lh src/lacewing/quantification/regression_shared/data/cache/

You should see one `.npz` per cache flavour, sized ~100-300 MB each.

### 5. RQ1: classification

    # a. 9-arch catalogue on the deployed pipeline (Fig. 5.3)
    qsub jobs/rq1_cov_classifier_catalogue.pbs

    # b. Preprocessing ablation deployed → +MAD → +spatA3 (Tab. 5.1)
    qsub jobs/rq1_cov_preproc_ablation.pbs

    # c. Per-layer A/AB/ABC/ABCD ablation (Appendix)
    qsub jobs/rq1_cov_layer_ablation.pbs

    # d. KP RQ1 (5-fold LOCO + preprocessing ablation)
    qsub jobs/rq1_kp_loco.pbs
    qsub jobs/rq1_kp_preproc_ablation_deployed.pbs
    qsub jobs/rq1_kp_preproc_ablation_mad.pbs

Verify results land in `src/lacewing/classification/results/`.

### 6. RQ2: onset regression

Rule-based extractors first (no training):

    python -m lacewing.quantification.runners.run_ttp        # threshold-derivative
    python -m lacewing.quantification.runners.run_sdm_cy0    # SDM + Cy₀

Then the five ML framings on SARS-CoV-2:

    qsub jobs/rq2_cov_fa.pbs   # F-A probability-density regression
    qsub jobs/rq2_cov_fb.pbs   # F-B sliding-window classifier (winner)
    qsub jobs/rq2_cov_fc.pbs   # F-C sliding-window regression (dead branch)
    qsub jobs/rq2_cov_fd.pbs   # F-D 4-class segmentation
    qsub jobs/rq2_cov_fe.pbs   # F-E full-trace BiGRU

Aggregate scoreboards land in `results/scoreboards/rq2_cov_*.csv`.

### 7. RQ3: joint SSL-pretrained framework

Two stages: SSL pretraining, then joint fine-tuning.

    qsub jobs/rq3_cov_ssl_pretrain.pbs    # first (produces the SSL encoder)
    qsub jobs/rq3_cov_ssl_finetune.pbs    # depends on the above
    qsub jobs/rq3_cov_joint_sweep.pbs     # full A1-A5 × SSL sweep

Winner: **A1 α=0.5 + contrastive 50-ep at 1.99 min mean per-well MAE**
(4/5 seeds), see `results/scoreboards/rq3_cov.csv`.

### 8. KP cross-dataset evaluation

    # SSL pretraining on the KP unlabelled pool
    qsub jobs/rq3_kp_ssl_pretrain.pbs

    # 5-fold LOCO for each of the 6 methods
    for method in fb fb_ssl a1_no_ssl a1_ssl a3_no_ssl a3_ssl; do
        qsub jobs/rq3_kp_${method}_fold4.pbs
        qsub jobs/rq3_kp_${method}_folds1235.pbs
    done

Winner: **A3 α=0.5 + contrastive SSL 100-ep at 1.47 min mean chip-side
MAE** across the 5 LOCO folds, see `results/scoreboards/rq3_kp_5fold.csv`.

### 9. Interactive viewers

    # Systematic-TTP labelling viewer (plate-anchored per-well TTP)
    python -m lacewing.quantification.build_systematic_ttp_viewer
    open src/lacewing/quantification/output/systematic_ttp_viewer.html

    # KP preprocessing pipeline viewer
    python -m lacewing.preprocessing.data_quality.build_kp_preproc_viewer_data
    open src/lacewing/preprocessing/data_quality/kp_preproc_viewer.html

    # Extractor comparison viewer
    python -m lacewing.quantification.build_extractor_comparison_viewer
    open src/lacewing/quantification/output/extractor_comparison_viewer.html

## Total wall-clock estimate

On a single L40S GPU running on Imperial HPC:

- Cache building (CoV + KP): ~2 h
- RQ1 classifier catalogue: ~4 h
- RQ1 preprocessing ablations: ~6 h
- RQ2 framings: ~12 h
- RQ3 SSL pretraining + joint sweep: ~20 h
- KP SSL + 5-fold LOCO (12 jobs × ~2 h, parallelised): ~4 h wall-clock

Total: ~2-3 days of HPC compute if run in parallel with reasonable
queue times.
