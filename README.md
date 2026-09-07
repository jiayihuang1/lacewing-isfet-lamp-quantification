# Lacewing ISFET-LAMP Quantification

ML for amplification onset detection on the Lacewing ISFET-LAMP diagnostic
platform. MSc AI dissertation, Imperial College London, 2026.

This repository is a companion to the dissertation report. Every
result and figure in the report is reproducible from the code and
result files listed below.

## Package layout

The Python package is `src/lacewing/`, split into four subpackages,
each with its own README describing what it does and how to use it:

| Subpackage | Answers | README |
|---|---|---|
| [`preprocessing/`](src/lacewing/preprocessing/) | MAD-ABCD data-quality filter + spatA spatial averaging. Turns raw pixels into ML-ready signals. | [preprocessing README](src/lacewing/preprocessing/README.md) |
| [`classification/`](src/lacewing/classification/) | RQ1: 9-arch classifier catalogue, preprocessing ablations, KP 5-fold LOCO. | [classification README](src/lacewing/classification/README.md) |
| [`quantification/`](src/lacewing/quantification/) | RQ2: classical + F-A…F-E onset regression. RQ3: A1–A5 joint architectures + SSL. KP 5-fold LOCO. Chip pipeline for raw-chip processing. | [quantification README](src/lacewing/quantification/README.md) |
| [`features/`](src/lacewing/features/) | Sigmoid-fitting helpers used by the amplitude-viewer notebooks. | — |

If you're new to the repo, start by skimming the subpackage READMEs
in the order preprocessing → classification → quantification —
that's also the order data flows through them.

## Reproducing the report

### Main-body results (Category A)

| Report section | Claim | Code entry point | Job script | Result artefact |
|---|---|---|---|---|
| §RQ1 Tab. 5.1 | Deployed → +MAD → +spatA3 lifts CoV per-pixel accuracy from ~0.85–0.95 to ≥0.988 across 8 of 9 non-collapsing architectures | [`src/lacewing/classification/experiments/run_exp8.py`](src/lacewing/classification/experiments/run_exp8.py) | [`jobs/rq1_cov_preproc_ablation.pbs`](jobs/rq1_cov_preproc_ablation.pbs) | (per-run outputs regenerable via `qsub`) |
| §RQ1 Fig. 5.3 | 9-arch RQ1 classifier catalogue on SD test chips | [`src/lacewing/classification/core/train.py`](src/lacewing/classification/core/train.py) via [`experiments/run_exp6.py`](src/lacewing/classification/experiments/run_exp6.py) | [`jobs/rq1_cov_classifier_catalogue.pbs`](jobs/rq1_cov_classifier_catalogue.pbs) | (per-run outputs regenerable) |
| §NewData Tab. 8.X | KP per-arch preprocessing ablation (7 arch × 3 states × 5 chips) | [`src/lacewing/classification/experiments/w18_rq1_uti_only.py`](src/lacewing/classification/experiments/w18_rq1_uti_only.py) | [`jobs/rq1_kp_preproc_ablation_deployed.pbs`](jobs/rq1_kp_preproc_ablation_deployed.pbs) + [`_mad.pbs`](jobs/rq1_kp_preproc_ablation_mad.pbs) + [`rq1_kp_loco.pbs`](jobs/rq1_kp_loco.pbs) | (per-run outputs regenerable) |
| §RQ2 Tab. 6.1 | Deployed threshold-derivative TTP: 3.20 min MAE (baseline) | [`src/lacewing/quantification/methods/ttp_threshold_derivative.py`](src/lacewing/quantification/methods/ttp_threshold_derivative.py) | — (rule-based, no training) | [`results/scoreboards/rq2_cov_fb.csv`](results/scoreboards/rq2_cov_fb.csv) |
| §RQ2 Tab. 6.1 | Cy₀ tangent-at-inflection: 2.72 min MAE (strongest classical) | [`src/lacewing/quantification/methods/cy0.py`](src/lacewing/quantification/methods/cy0.py) | — | as above |
| §RQ2 Tab. 6.1 | SDM: 2.34 min MAE | [`src/lacewing/quantification/methods/sdm.py`](src/lacewing/quantification/methods/sdm.py) | — | as above |
| §RQ2 Tab. 6.2 | F-B U-Net 2.61 min per-well MAE on SD test chips | [`src/lacewing/quantification/methods/sliding_window_cls.py`](src/lacewing/quantification/methods/sliding_window_cls.py) | [`jobs/rq2_cov_fb.pbs`](jobs/rq2_cov_fb.pbs) | [`results/scoreboards/rq2_cov_fb.csv`](results/scoreboards/rq2_cov_fb.csv) |
| §RQ2 Tab. 6.2 | F-A 2.56 min per-well MAE (0/5 seeds passing) | [`src/lacewing/quantification/methods/pdf_regression.py`](src/lacewing/quantification/methods/pdf_regression.py) | [`jobs/rq2_cov_fa.pbs`](jobs/rq2_cov_fa.pbs) | [`results/scoreboards/rq2_cov_fa.csv`](results/scoreboards/rq2_cov_fa.csv) |
| §RQ3 Tab. 7.4 | A1 + contrastive SSL 50-ep winner at 1.99 min per-well MAE (4/5 seeds) on CoV | [`src/lacewing/quantification/methods/joint_cls_quant.py`](src/lacewing/quantification/methods/joint_cls_quant.py) + [`ssl_pretrain.py`](src/lacewing/quantification/methods/ssl_pretrain.py) | [`jobs/rq3_cov_ssl_pretrain.pbs`](jobs/rq3_cov_ssl_pretrain.pbs) → [`rq3_cov_ssl_finetune.pbs`](jobs/rq3_cov_ssl_finetune.pbs) | [`results/scoreboards/rq3_cov.csv`](results/scoreboards/rq3_cov.csv) |
| §RQ3 Tab. 7.4 | A3 + contrastive SSL 100-ep secondary at 2.47 min (5/5 seeds) on CoV | as above (`joint_cls_quant.py` `--arch a3`) | [`jobs/rq3_cov_joint_sweep.pbs`](jobs/rq3_cov_joint_sweep.pbs) | as above |
| §NewData Tab. 8.3 | A3 + SSL KP winner at 1.47 min mean chip-side MAE across 5 LOCO folds | as above | [`jobs/rq3_kp_a3_ssl_fold4.pbs`](jobs/rq3_kp_a3_ssl_fold4.pbs) + [`_folds1235.pbs`](jobs/rq3_kp_a3_ssl_folds1235.pbs) | [`results/scoreboards/rq3_kp_5fold.csv`](results/scoreboards/rq3_kp_5fold.csv) |
| §NewData Tab. 8.3 | F-B (no SSL) KP second-place at 1.86 min | [`sliding_window_cls.py`](src/lacewing/quantification/methods/sliding_window_cls.py) on KP LOCO | [`jobs/rq3_kp_fb_fold4.pbs`](jobs/rq3_kp_fb_fold4.pbs) + [`_folds1235.pbs`](jobs/rq3_kp_fb_folds1235.pbs) | as above |

### Appendix / documented negative-result methods (Category B)

| Report section | Method | Why kept | Code | Result |
|---|---|---|---|---|
| §RQ2 F-C dead-branch para | Sliding-window regression, Spearman ρ sign-flipped every seed | Documented failure evidence | [`src/lacewing/quantification/methods/sliding_window_reg.py`](src/lacewing/quantification/methods/sliding_window_reg.py) | [`jobs/rq2_cov_fc.pbs`](jobs/rq2_cov_fc.pbs) |
| §RQ2 F-D + §Conclusion future work | 4-class per-timestep segmentation | Retained for future work | [`src/lacewing/quantification/methods/unet_segmentation.py`](src/lacewing/quantification/methods/unet_segmentation.py) | [`jobs/rq2_cov_fd.pbs`](jobs/rq2_cov_fd.pbs) |
| §RQ2 F-E para | Full-trace BiGRU regression | Underperforms; evidence for F-B choice | [`src/lacewing/quantification/methods/full_trace_bigru.py`](src/lacewing/quantification/methods/full_trace_bigru.py) | [`jobs/rq2_cov_fe.pbs`](jobs/rq2_cov_fe.pbs) |
| §RQ3 sweep | A2 / A4 / A5 joint architectures | Losing variants in RQ3 sweep | [`joint_cls_quant.py`](src/lacewing/quantification/methods/joint_cls_quant.py) `--arch a2/a4/a5` | [`jobs/rq3_cov_joint_sweep.pbs`](jobs/rq3_cov_joint_sweep.pbs) |
| §RQ1 layer ablation | Layer A / AB / ABC / ABCD preproc ablation | Evidence for what each MAD-ABCD layer contributes | [`experiments/run_exp7.py`](src/lacewing/classification/experiments/run_exp7.py) via [`rq1_cov_layer_ablation.pbs`](jobs/rq1_cov_layer_ablation.pbs) | (per-run outputs regenerable) |
| §Conclusion "failure modes" | 2D-CNN spectrogram + autoencoder classifiers | Dropped-from-catalogue evidence | [`models/cnn2d_spectrogram.py`](src/lacewing/classification/models/cnn2d_spectrogram.py) + [`autoencoder.py`](src/lacewing/classification/models/autoencoder.py) | as `run_exp6.py` above |

## Interactive viewers

Each viewer is a standalone Python entry point that writes a
self-contained HTML file into its subpackage's `output/` folder
(created on the fly if it doesn't exist). Open the resulting `.html`
in any browser — no server needed.

### Systematic-TTP labelling viewer (primary)

The one your supervisor liked — per-well plate-anchor + refined TTP
overlay for every chip.

    python -m lacewing.quantification.build_systematic_ttp_viewer
    open src/lacewing/quantification/output/systematic_ttp_viewer.html

Source: [`src/lacewing/quantification/build_systematic_ttp_viewer.py`](src/lacewing/quantification/build_systematic_ttp_viewer.py)

### KP preprocessing pipeline viewer

Shows the deployed → +MAD → +spatA3 signal for every well of every
KP chip, so you can eyeball what each preprocessing stage did.

Prerequisites: three KP CoV-augmented caches must exist. Build them
first (once, via `qsub jobs/rq1_kp_preproc_ablation_deployed.pbs` +
`_mad.pbs` for the two ablation stages, plus
`lacewing.quantification.conc_data.build_classification_cache` for
the winner):

    python -m lacewing.quantification.conc_data.build_classification_cache_ablation deployed
    python -m lacewing.quantification.conc_data.build_classification_cache_ablation mad
    python -m lacewing.quantification.conc_data.build_classification_cache

Then build the viewer JSONs and open the HTML:

    python -m lacewing.preprocessing.data_quality.build_kp_preproc_viewer_data
    open src/lacewing/preprocessing/data_quality/kp_preproc_viewer.html

Source: [`src/lacewing/preprocessing/data_quality/build_kp_preproc_viewer_data.py`](src/lacewing/preprocessing/data_quality/build_kp_preproc_viewer_data.py)

### Extractor-comparison viewer

Side-by-side qLAMP vs chip signal for the three classical extractors
(threshold-derivative, Cy₀, SDM) on one worst-case chip well.

    python -m lacewing.quantification.build_extractor_comparison_viewer
    open src/lacewing/quantification/output/extractor_comparison_viewer.html

Source: [`src/lacewing/quantification/build_extractor_comparison_viewer.py`](src/lacewing/quantification/build_extractor_comparison_viewer.py)

All three viewers need raw chip data + `LACEWING_TITAN_PATH` set
(they call the titan chip loader on demand).

## Installation

    pip install -r requirements.txt
    pip install -e .

The `-e .` step installs the `lacewing` package itself (editable mode),
so `python -m lacewing.xxx` commands resolve. Without it every
`python -m lacewing.…` invocation in the READMEs will fail with
`ModuleNotFoundError: No module named 'lacewing'`.

To run the raw-chip preprocessing pipeline (cache builders,
`chip_pipeline/process_chip.py`), clone the `titan-signal-processing`
package separately (available on request from the Centre for
Bio-Inspired Technology at Imperial College London) and point three
environment variables at it:

    # Path to the CoV titan clone (main branch)
    export LACEWING_TITAN_PATH=/path/to/titan-signal-processing

    # Path to the Matthew_Multi titan clone (only needed for KP dataset
    # cache building and chip_pipeline on multi-Vref chips)
    export LACEWING_TITAN_MULTI_PATH=/path/to/titan-signal-processing-multi

    # Path to the raw Data/ tree (defaults to <repo-root>/Data if unset)
    export LACEWING_DATA_ROOT=/path/to/parent/of/Data

## Data

Raw ISFET chip recordings and paired plate-qLAMP exports are held by
the Centre for Bio-Inspired Technology at Imperial College London and
are available on reasonable request. See [`docs/DATASETS.md`](docs/DATASETS.md)
for the exact `Data/` folder layout the code expects and how to
build all the intermediate caches from the raw `.bin` files.

## Running a single experiment locally

The joint framework is a two-stage flow: SSL pretraining produces an
encoder checkpoint, then joint fine-tuning loads it via
`--pretrained-encoder`. Dataset choice comes from `--cache-stem`.

    # Stage 1: SSL contrastive pretraining on the CoV unlabelled pool
    python -m lacewing.quantification.methods.ssl_pretrain \
        --objective contrastive --backbone unet \
        --epochs 50 --batch_size 64 --lr 1e-3 --seed 0

    # Stage 2: RQ3 CoV winner — A1 + contrastive SSL
    python -m lacewing.quantification.methods.joint_cls_quant \
        --arch a1 --alpha 0.5 --seed 0 \
        --window 120 --stride 10 --epochs 30 --lr 1e-3 --batch_size 256 \
        --pretrained-encoder path/to/pretrained.pt

    # KP 5-fold LOCO winner: A3 + contrastive SSL, fold 4
    python -m lacewing.quantification.methods.joint_cls_quant \
        --arch a3 --alpha 0.5 --seed 0 \
        --cache-stem regress_conc_loco4 \
        --window 120 --stride 10 --epochs 30 --lr 1e-3 --batch_size 256 \
        --pretrained-encoder path/to/kp_pretrained.pt

Full flag reference: `python -m lacewing.quantification.methods.joint_cls_quant --help`.

## Full reproduction on HPC

    # Preprocessing caches first
    qsub jobs/build_cache_cov_full_preproc.pbs
    qsub jobs/build_cache_spatA.pbs

    # RQ1 (SARS-CoV-2 + KP)
    qsub jobs/rq1_cov_classifier_catalogue.pbs
    qsub jobs/rq1_cov_preproc_ablation.pbs
    qsub jobs/rq1_kp_loco.pbs
    qsub jobs/rq1_kp_preproc_ablation_deployed.pbs
    qsub jobs/rq1_kp_preproc_ablation_mad.pbs

    # RQ2 (SARS-CoV-2)
    qsub jobs/rq2_cov_fa.pbs
    qsub jobs/rq2_cov_fb.pbs
    qsub jobs/rq2_cov_fd.pbs
    qsub jobs/rq2_cov_fe.pbs

    # RQ3 (SARS-CoV-2)
    qsub jobs/rq3_cov_ssl_pretrain.pbs   # first (produces the SSL encoder)
    qsub jobs/rq3_cov_ssl_finetune.pbs   # depends on the above
    qsub jobs/rq3_cov_joint_sweep.pbs    # full A1-A5 × SSL sweep

    # RQ3 (KP 5-fold LOCO)
    qsub jobs/rq3_kp_ssl_pretrain.pbs
    for method in fb fb_ssl a1_no_ssl a1_ssl a3_no_ssl a3_ssl; do
        qsub jobs/rq3_kp_${method}_fold4.pbs
        qsub jobs/rq3_kp_${method}_folds1235.pbs
    done

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the full recipe with
intermediate checks.
