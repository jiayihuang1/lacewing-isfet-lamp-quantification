# Preprocessing

Raw ISFET pixel voltages are noisy: dead pixels, saturated pixels,
electrode drift, spatial gradients, and pixel-level flicker all mask
the biochemical amplification signal. This subpackage implements the
data-quality filter (MAD-ABCD) and spatial averaging (spatA) stages
that turn raw pixels into ML-ready signals.

The final RQ1 winning pipeline is **deployed → +MAD-ABCD → +spatA3**,
which lifts per-pixel CoV classification accuracy from ~0.85–0.95 to
≥0.988 across 8 of 9 non-collapsing architectures (Tab. 5.1).

---

## Folder layout

```
preprocessing/
├── data_quality/               # MAD-ABCD pixel-quality filter
│   ├── filter_core.py          # the 4 layers (A/B/C/D)
│   ├── spatial_coords.py       # helper: pixel row/col → well-relative coords
│   ├── build_filter_pipeline.py       # orchestrates a full filter pass
│   ├── build_kp_preproc_viewer_data.py  # export per-well viewer data (JSON)
│   ├── analyse_misclassification_stats.py
│   ├── diagnose_layer_c.py            # layer-C tuning diagnostic
│   └── export_training_chip_distributions.py
└── spatial_filter/             # spatA spatial averaging
    └── spatial_smooth.py       # spatA0/1/2/3 (kernel size 0/3/5/7)
```

---

## The MAD-ABCD filter

Four sequential per-pixel layers. Each uses a **Median Absolute
Deviation (MAD)** threshold with configurable `k` (report default:
`k=1.5`).

| Layer | Filters | Rejects if... |
|---|---|---|
| **A** | Baseline mean | pixel's per-well-relative baseline mean is > k·MAD from the well median |
| **B** | Baseline std | pixel's baseline std is > k·MAD from the well median |
| **C** | Time-domain slope | pixel's slope over the pre-reaction window is > k·MAD from the well median |
| **D** | Post-reaction saturation | pixel voltage exceeds the safe range at any time (this is the "deployed" filter already in production) |

Rejection at any layer marks the pixel bad; downstream stages see
only survivors.

**Retention** (on SARS-CoV-2, `k=1.5`): 69% of pixels retained on the
5 training chips, 14% on the 2 SD test chips, 54% pooled across all 7
chips.

Full deep-dive: report §Data.pixel-quality, Fig. 3.

---

## spatA — spatial averaging

Once per-pixel gating is done, each surviving pixel is averaged with
its immediate spatial neighbours using a small kernel. Order `N`
means the pixel plus its N nearest neighbours by (row, col)
distance. Reduces pixel-level flicker without blurring biological
signal.

| Config | Kernel | Notes |
|---|---|---|
| spatA0 | pixel only (no averaging) | baseline |
| spatA1 | pixel + 3 neighbours | mild smoothing |
| spatA2 | pixel + 5 neighbours | medium |
| **spatA3** | pixel + 7 neighbours | **RQ1 winner** |

`spatial_filter/spatial_smooth.py` implements all four. spatB
(alternative spatial band-pass) was tested and dropped as a losing
variant — see report Appendix.

---

## Typical usage

You rarely call these modules directly — the classification cache
builders drive them (see [`classification/README.md`](../classification/README.md)):

```bash
# Build the MAD-ABCD-filtered cache (calls data_quality/ internals)
python -m lacewing.classification.data.build_filtered_dataset_mad \
    --scope all --layers ABCD --k 1.5

# Layer on spatA3 spatial averaging (calls spatial_filter/ internals)
python -m lacewing.classification.data.build_spatial_filtered_dataset \
    --input dataset_all_filt_abcd_ntcRaw_madk1p5 --order 3
```

`build_spatial_filtered_dataset` builds the **spatA** family
(spatial averaging — the RQ1 winner). Its sibling
`build_spatial_pooled_dataset` builds the **spatB** family (tile
pooling), which was tested and dropped.

To generate the per-well **preprocessing pipeline viewer** (browser
HTML showing before/after for each well of each KP chip):

```bash
python -m lacewing.preprocessing.data_quality.build_kp_preproc_viewer_data
open src/lacewing/preprocessing/data_quality/kp_preproc_viewer.html
```

To generate the layer-C collateral diagnostic (per-chip figures
showing which pixels layer C drops that the classifier always gets
right — sanity check for layer-C tuning), first build the layer-C
filter dump JSONs, then run the diagnostic:

```bash
python -m lacewing.preprocessing.data_quality.build_filter_pipeline
python -m lacewing.preprocessing.data_quality.diagnose_layer_c
```

---

## Choosing `k`

Report Tab. 5.1 explores `k ∈ {1.5, 1.645, 2.0}`. `k=1.5` is the
winner on CoV: it strips the most noise without sacrificing enough
positive-signal pixels to hurt classification. For KP, `k=1.5`
transfers directly with no re-tuning (see Ch. 8).
