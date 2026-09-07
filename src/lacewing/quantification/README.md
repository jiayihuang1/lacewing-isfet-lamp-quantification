# Quantification (RQ2 + RQ3)

Answers RQ2 (*how early can we detect the amplification onset?*) and
RQ3 (*does a joint classification+regression model with SSL
pretraining transfer to a new organism?*).

Delivers:
- Classical baselines (threshold-derivative, Cy₀, SDM) on both
  qLAMP and Lacewing chip data — RQ2 Tab. 6.1
- Five ML framings F-A…F-E, of which F-B (sliding-window classifier
  U-Net) is the winner — RQ2 Tab. 6.2
- Joint architectures A1–A5, of which A1+SSL is the CoV winner and
  A3+SSL is the KP 5-fold LOCO winner — RQ3 Tab. 7.4, Ch. 8 Tab. 8.3

---

## Folder layout

```
quantification/
├── methods/                    # RQ2 + RQ3 model implementations
│   ├── ttp_threshold_derivative.py    # baseline (deployed rule)
│   ├── cy0.py                         # Cy₀ tangent-at-inflection
│   ├── sdm.py                         # Second-Derivative Maximum
│   ├── pdf_regression.py              # F-A: probability-density regression
│   ├── sliding_window_cls.py          # F-B: sliding-window classifier (winner)
│   ├── sliding_window_reg.py          # F-C: sliding-window regression (Cat B)
│   ├── unet_segmentation.py           # F-D: 4-class per-timestep segmentation (Cat B)
│   ├── full_trace_bigru.py            # F-E: full-trace BiGRU (Cat B)
│   ├── joint_cls_quant.py             # A1–A5 joint architectures + fine-tuning
│   └── ssl_pretrain.py                # SSL pretraining (contrastive + masked recon)
├── conc_data/                  # KP dataset ingestion + 5-fold LOCO caches
│   ├── process_conc_chips.py          # raw KP chip → preprocessed
│   ├── build_regression_cache.py      # 5 per-fold .npz caches for RQ3 KP
│   ├── build_classification_cache.py  # KP RQ1 cache
│   └── parse_book_xlsx.py, parse_chip_xlsx.py
├── chip_pipeline/              # raw-chip → per-well signal (shared by all RQs)
│   ├── process_chip.py                # main entry: raw .bin → per-well trace
│   ├── compute_chip_ttps.py           # rule-based TTP extraction
│   ├── fit_offset_calibration.py      # chip→plate temporal offset
│   ├── parse_lc96_features.py         # LC96 plate-qLAMP export parser
│   ├── manual_label_ttp_ui.py         # interactive labelling helper
│   └── w18_preprocessing_ablation.py  # KP preprocessing ablation driver
├── eval/                       # scoreboards + metrics
│   ├── fa_extraction_sweep.py         # F-A σ threshold sweep
│   ├── fb_extraction_sweep.py         # F-B k-threshold sweep
│   ├── fd_extraction_sweep.py         # F-D boundary metrics
│   ├── cls_side_eval.py               # joint-framework classification-side eval
│   └── extraction_rules.py            # rule-based baselines for the scoreboard
├── plotting/                   # figures for the report + presentation
├── runners/                    # per-experiment run drivers
├── notebooks/                  # analysis + reproduction notebooks
├── viewer/                     # interactive quant viewers (regenerable)
├── systematic_ttp.py                  # plate-anchored per-well TTP labeller
├── build_systematic_ttp_viewer.py     # supervisor's favourite viewer
├── build_extractor_comparison_viewer.py
├── build_kp_preproc_ablation.py       # KP preprocessing ablation aggregate
└── validate_systematic_ttp_on_uti.py
```

---

## The four processing stages

Raw chip data flows through four stages before it can produce a
TTP number:

1. **Raw acquisition** — chip captures `.bin` chunks
   (~5–50 MB per chip)
2. **Chip pipeline** (`chip_pipeline/process_chip.py`) — decode,
   linearise, filter, per-well aggregate → per-well fluorescence-like
   traces
3. **Cache builders** (`conc_data/build_*.py`, or classification
   equivalents in [`../classification/data/`](../classification/data/))
   → per-experiment `.npz`
4. **Method** — ingest cache, produce a TTP prediction per well
   (rule-based or ML)

Stages 1–2 need `titan-signal-processing` (see top-level README).
Stages 3–4 run entirely on the cache.

---

## Typical usage

### 1. Rule-based baselines (no training)

Two runner scripts under `runners/` invoke the three classical
extractors on every chip and write per-chip / per-well TTP tables:

```bash
# Threshold-derivative TTP (the deployed baseline)
python -m lacewing.quantification.runners.run_ttp

# Cy₀ tangent-at-inflection + SDM second-derivative maximum
python -m lacewing.quantification.runners.run_sdm_cy0
```

Reported numbers (CoV SD test): threshold-derivative 3.20 min MAE,
Cy₀ 2.72, SDM 2.34. Outputs land in
`src/lacewing/quantification/results/{ttp,sdm_cy0}/`.

### 2. RQ2 ML framings

```bash
# F-B (winner): sliding-window classifier, ANN backbone, subsampled for speed
python -m lacewing.quantification.methods.sliding_window_cls \
    --seed 0 --epochs 60 --backbone ann \
    --window 120 --stride 10 --k_thr 0.7

# F-A: probability-density regression, CNN1D backbone
python -m lacewing.quantification.methods.pdf_regression \
    --seed 0 --epochs 40 --backbone cnn1d

# Cat-B: F-C, F-D, F-E
python -m lacewing.quantification.methods.sliding_window_reg --seed 0 --epochs 40
python -m lacewing.quantification.methods.unet_segmentation --seed 0 --epochs 40
python -m lacewing.quantification.methods.full_trace_bigru --seed 0 --epochs 40
```

Full flag reference: `python -m lacewing.quantification.methods.sliding_window_cls --help` etc.

HPC alternative: `qsub jobs/rq2_cov_{fa,fb,fc,fd,fe}.pbs`.

### 3. RQ3 joint framework

Two-stage: SSL pretraining produces an encoder checkpoint, then joint
fine-tuning loads it via `--pretrained-encoder`. Dataset choice comes
from `--cache-stem`.

```bash
# Stage 1: SSL contrastive pretraining on the CoV unlabelled pool
python -m lacewing.quantification.methods.ssl_pretrain \
    --objective contrastive --backbone unet \
    --epochs 50 --batch_size 64 --lr 1e-3 --seed 0

# Stage 2: RQ3 CoV winner — A1 α=0.5 + contrastive SSL 50-ep
python -m lacewing.quantification.methods.joint_cls_quant \
    --arch a1 --alpha 0.5 --seed 0 \
    --window 120 --stride 10 --epochs 30 --lr 1e-3 --batch_size 256 \
    --pretrained-encoder path/to/pretrained.pt

# RQ3 CoV secondary — A3 α=0.5 + contrastive SSL 100-ep
python -m lacewing.quantification.methods.joint_cls_quant \
    --arch a3 --alpha 0.5 --seed 0 \
    --window 120 --stride 10 --epochs 30 --lr 1e-3 --batch_size 256 \
    --pretrained-encoder path/to/pretrained.pt
```

Full flag reference: `python -m lacewing.quantification.methods.joint_cls_quant --help`.

HPC: `qsub jobs/rq3_cov_ssl_pretrain.pbs` then
`qsub jobs/rq3_cov_ssl_finetune.pbs` or `jobs/rq3_cov_joint_sweep.pbs`
for the full A1-A5 × SSL sweep.

### 4. KP 5-fold LOCO cross-dataset evaluation

Each of the 5 KP chips is held out once as the test set; the other 4
form the training set. Six methods swept: F-B, F-B+SSL, A1, A1+SSL,
A3, A3+SSL.

```bash
# Build the 5 per-fold caches (once)
python -m lacewing.quantification.conc_data.build_regression_cache
python -m lacewing.quantification.conc_data.build_classification_cache

# HPC: submit all 12 array jobs (6 methods × 2 fold-groups)
qsub jobs/rq3_kp_ssl_pretrain.pbs      # first
for method in fb fb_ssl a1_no_ssl a1_ssl a3_no_ssl a3_ssl; do
    qsub jobs/rq3_kp_${method}_fold4.pbs
    qsub jobs/rq3_kp_${method}_folds1235.pbs
done
```

Winner: A3+SSL at 1.47 min mean chip-side MAE across the 5 folds
(Tab. 8.3).

### 5. Scoreboards

Aggregate per-run outputs into the final CSV scoreboards cited from
the report:

```bash
python -m lacewing.quantification.eval.fb_extraction_sweep \
    --output results/scoreboards/rq2_cov_fb.csv

python -m lacewing.quantification.eval.cls_side_scoreboard \
    --output results/scoreboards/rq3_cov_cls_side.csv
```

The definitive scoreboards live in
[`results/scoreboards/`](../../../results/scoreboards/) at the repo
root.

---

## Interactive viewers

```bash
# Systematic-TTP labelling viewer (plate-anchored per-well TTP)
python -m lacewing.quantification.build_systematic_ttp_viewer
open src/lacewing/quantification/output/systematic_ttp_viewer.html

# Extractor comparison viewer (qLAMP vs chip, three extractors side-by-side)
python -m lacewing.quantification.build_extractor_comparison_viewer
open src/lacewing/quantification/output/extractor_comparison_viewer.html
```

---

## Per-run artefacts

Every joint / SSL / F-* training run writes to
`methods/results/<experiment>/<seed>/`:

- `config.yaml` — full CLI args
- `metrics.csv` — per-epoch train/val loss
- `predictions.npz` — per-well predictions + truth + chip/well IDs
- `checkpoints/best.pt`, `checkpoints/last.pt`
- `train.log`, `git_state.txt`, `env.txt`
- Diagnostic plots

Rule-based baselines only produce a single CSV row in the scoreboard.
