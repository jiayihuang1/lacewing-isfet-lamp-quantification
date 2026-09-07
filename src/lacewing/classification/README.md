# Classification (RQ1)

Answers RQ1: *Can pixel-level classifiers reliably decide amplification
positive vs. negative on Lacewing ISFET-LAMP chips?*

Delivers the 9-arch classifier catalogue (Fig. 5.3), the preprocessing
ablation (Tab. 5.1), and the KP 5-fold LOCO cross-dataset evaluation
(Ch. 8).

---

## Folder layout

```
classification/
├── core/
│   ├── train.py                # train one (model, split, fold, seed)
│   ├── evaluate.py             # pixel + well-level metrics
│   ├── paths.py                # canonical paths (needs titan-signal-processing)
│   ├── seeding.py              # deterministic seeds
│   └── logging_utils.py        # per-run config + git/env snapshots
├── data/
│   ├── ntc_inventory.py        # inventory NTC wells in Others/
│   ├── build_dataset.py        # per-pixel cache, no preprocessing
│   ├── build_filtered_dataset.py       # + layer-D outlier filter (deployed)
│   ├── build_filtered_dataset_mad.py   # + MAD-ABCD data-quality filter
│   ├── build_spatial_filtered_dataset.py  # + spatA spatial averaging (WINNER)
│   ├── build_spatial_pooled_dataset.py    # + spatB tile pooling (Cat B)
│   └── dataset.py              # torch Dataset + train/val/test splits
├── models/                     # 9 PyTorch architectures
│   ├── ann.py
│   ├── cnn1d.py
│   ├── fcn.py
│   ├── resnet.py
│   ├── inception_time.py
│   ├── gru.py
│   ├── cnn_gru_par.py          # winner arch (parallel CNN + GRU)
│   ├── cnn_transformer_par.py
│   ├── cnn2d_spectrogram.py    # Cat-B: 2D-CNN on STFT
│   └── autoencoder.py          # Cat-B: anomaly-style classifier
├── experiments/
│   ├── run_exp6.py             # 9-arch catalogue (Fig. 5.3)
│   ├── run_exp7.py             # per-layer A/AB/ABC/ABCD ablation (Cat B)
│   ├── run_exp8.py             # deployed → +MAD → +spatA3 aggregate (Tab. 5.1)
│   ├── run_exp9.py             # follow-up robustness runs
│   └── w18_rq1_uti_only.py     # KP per-arch preprocessing ablation
├── reporting/                  # aggregation + plotting
└── results/                    # per-run outputs (gitignored)
```

---

## Typical usage

### 1. Build the preprocessing caches

Each cache produces one `.npz` file that later stages consume. The
report's winning pipeline is **MAD-ABCD + spatA3**, built with two
commands.

```bash
# a. Baseline: deployed (layer D only)
python -m lacewing.classification.data.build_filtered_dataset

# b. +MAD data-quality filter (layer A + B + C + D)
python -m lacewing.classification.data.build_filtered_dataset_mad \
    --scope all --layers ABCD --k 1.5

# c. +spatA3 spatial averaging (WINNER)
python -m lacewing.classification.data.build_spatial_filtered_dataset \
    --input dataset_all_filt_abcd_ntcRaw_madk1p5 --order 3

# HPC alternative (builds a, b, c in one job):
qsub jobs/build_cache_cov_full_preproc.pbs
```

Each cache lands in `data/cache/` as e.g.
`dataset_all_filt_abcd_ntcRaw_madk1p5_spatA3.npz`.

### 2. Train one classifier

```bash
python -m lacewing.classification.core.train \
    --experiment my_run \
    --cache dataset_all_filt_abcd_ntcRaw_madk1p5_spatA3 \
    --model cnn_gru_par \
    --split chip \
    --fold 1e7 \
    --seed 0 \
    --epochs 40
```

Useful flags:

```
--model {ann,cnn1d,fcn,resnet,inception,autoencoder,cnn2d_spec,
         transformer,transformer_patch,cnn_transformer_par,cnn_transformer_seq,
         gru,cnn_gru_par,cnn_gru_seq}
--cache <cache basename>        # cache to load from data/cache/
--split {random,chip,chipkfold,sd_test}
--fold FOLD                     # --split chip: {1e5..1e9} or KP chip ID;
                                # --split chipkfold: fold index 0..K-1;
                                # --split sd_test: comma-separated test-chip folder names
--seed N
--epochs N                      # default 40
--device {cpu,cuda}
```

### 3. Run a full sweep

The RQ1 sweeps that produce report figures/tables are wrapped as
`experiments/*.py` scripts and submitted via PBS:

| Sweep | Report claim | Job script |
|---|---|---|
| 9-arch catalogue on deployed pipeline | Fig. 5.3 | `jobs/rq1_cov_classifier_catalogue.pbs` |
| Deployed → +MAD → +spatA3 aggregate | Tab. 5.1 | `jobs/rq1_cov_preproc_ablation.pbs` |
| Per-layer A/AB/ABC/ABCD ablation | Appendix | `jobs/rq1_cov_layer_ablation.pbs` |
| KP 5-fold LOCO catalogue | Ch. 8 | `jobs/rq1_kp_loco.pbs` |

### 4. Aggregate results

Each experiment writes one row per completed run to
`results/<experiment>/per_method_metrics.csv`. Aggregate with:

```bash
python -m lacewing.classification.reporting.report \
    --experiment rq1_cov_classifier_catalogue
```

---

## Per-run artefacts (paper trail)

`core/train.py` creates one directory per run under
`results/<experiment>/runs/<run_id>/`:

| File | Purpose |
|---|---|
| `config.yaml` | All CLI args + sample counts |
| `metrics.csv` | Per-epoch train/val loss + accuracy |
| `test_metrics.json` | Final pixel + well-level test metrics |
| `predictions.npz` | Test predictions + truth + chip/well/pixel IDs |
| `checkpoints/best.pt`, `last.pt` | Model weights |
| `git_state.txt`, `env.txt` | Git SHA + `pip freeze` snapshot |
| `cm_pixel.png`, `cm_well.png`, `train_curves.png` | Diagnostic plots |

---

## Reproducibility

- All seeds (Python, NumPy, PyTorch CPU+CUDA, CUDNN) set via
  `core.seeding.seed_everything(seed)`.
- Caches are deterministic from raw `Data/` + `ntc_inventory.csv`.
- Every run captures git SHA + dirty-file list + `pip freeze`.
- Class imbalance handled with `pos_weight` in BCE loss (computed
  per-fold from train counts).
