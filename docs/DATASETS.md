# Datasets

## Two datasets used in the project

### SARS-CoV-2 dataset (seven chips)

- 5 dose-response training chips (`1e5`, `1e6`, `1e7`, `1e8`, `1e9`
  copies per reaction), Lacewing platform
- 2 SD (serial-dilution) multiplex held-out test chips

Raw pixel recordings are `.bin` chunks accompanied by `params.json`
and `titan-run.log`. Standard Lacewing capture format.

### K. pneumoniae dataset (five chips)

- 5 chips (`260812_KP_01`, `260813_KP_DDM_02`, `260820_KP_DDM_03`,
  `260823_KP_DDM_04`, `260827_KP_DDM_05`)
- Each chip: 10 wells (8 amp+ at 10³–10⁶ copies/reaction + 1 PTC +
  1 NTC)
- Paired plate-qLAMP export (LC96) for chips 1 and 2 only; chips 3-5
  use a pooled per-concentration plate anchor.

## Well layout and labelling

Chip-side data has no built-in well-content annotations. Every well is
just index 0..N by hardware convention; **you** have to say which well
was loaded with which sample. Two chip form factors, two conventions.

### SARS-CoV-2 chips (6 wells)

Fixed layout, defined once in
[`src/lacewing/classification/core/paths.py`](../src/lacewing/classification/core/paths.py):

| Well | Content |
|---|---|
| 0 | positive sample (E1-R1) |
| 1 | positive sample (E1-R2) |
| 2 | positive sample (E2-R1) |
| 3 | positive sample (E2-R2) |
| 4 | PTC — positive template control |
| 5 | NTC — no-template control |

For the 5 dose-response training chips, all 5 positive wells on one
chip share the same concentration (that chip's dose — `1e5`, `1e6`,
etc.). The `FINAL_CHIPS` dict in `paths.py` maps concentration →
chip folder.

For SD (serial-dilution multiplex) test chips, wells 0..4 span all 5
concentrations in one chip and well 5 is NTC. Held out from training.

For **any other chip** under `Data/24_CoV_Quantification/Others/`
(NTC-only chips, extra multiplex chips), the well identity is
recovered by the **NTC-inventory workflow**:

    python -m lacewing.classification.data.ntc_inventory

writes `ntc_inventory.csv` with one row per chip: classification
(`mixed` / `ntc_only` / `non_ntc` / `unknown`), NTC well indices,
folder-name hints. Defaults are heuristic (well 5 = NTC unless the
folder name suggests otherwise). **Hand-edit the CSV before running
`build_dataset.py`** — the builder only picks negatives from rows
whose classification is `ntc_only` or `mixed` with valid NTC
indices.

An example `ntc_inventory.csv` ships in
[`src/lacewing/classification/data/ntc_inventory.csv`](../src/lacewing/classification/data/ntc_inventory.csv)
— review + edit for your run.

### K. pneumoniae chips (10 wells)

Fixed layout, defined in
[`src/lacewing/quantification/conc_data/process_conc_chips.py`](../src/lacewing/quantification/conc_data/process_conc_chips.py):

| Row | Left well | Right well |
|---|---|---|
| 0 | well 0 · 10⁶ copies | well 1 · 10⁵ copies |
| 1 | well 2 · 10⁴ copies | well 3 · 10³ copies |
| 2 | well 4 · 10⁶ copies | well 5 · 10⁵ copies |
| 3 | well 6 · 10⁴ copies | well 7 · 10³ copies |
| 4 | well 8 · PTC or NTC | well 9 · NTC or PTC |

Row 4 flips between chips: **Chip 1 = PTC/NTC**, **Chip 2 = NTC/PTC**.
Chips 3–5 (`_DDM_` naming) follow one of the two conventions; check
the `_build_chip_configs()` list.

Each amp-positive concentration (10³–10⁶) appears on **two wells per
chip**. Both wells share the same plate qLAMP anchor (see below).

### KP chip-tag naming

The 5 KP chips have systematic tags in `conc_data`:

| chip_tag | raw folder | date | qLAMP source |
|---|---|---|---|
| `conc_260812_KP_01` | `D20260812_E00_C00_F4500KHz_U_KP_conc_01` | 2026-08-12 | per-chip LC96 export (Chip 1.xlsx) |
| `conc_260813_KP_DDM_02` | `D20260813_E00_C00_F4500KHz_U_DDM_KP_con_02` | 2026-08-13 | per-chip LC96 export (Chip 2.xlsx) |
| `conc_260820_KP_DDM_03` | `D20260820_E00_C00_F4500KHz_U_DDM_KP_conc_03` | 2026-08-20 | pooled Chip1+Chip2 per-concentration mean |
| `conc_260823_KP_DDM_04` | `D20260823_E00_C00_F4500KHz_U_DDM_KP_04_05` | 2026-08-23 | pooled |
| `conc_260827_KP_DDM_05` | `D20260827_E00_C00_F4500KHz_U_DDM_KP_Conc_05` | 2026-08-27 | pooled |

### TTP labelling flow

1. **Plate qLAMP → per-well TTP anchor.** For each amp-positive
   concentration, the LC96 xlsx export gives a Cq. Report convention:
   `plate_TTP_min = 0.5 · Cq`. Implemented in
   [`parse_chip_xlsx.py`](../src/lacewing/quantification/conc_data/parse_chip_xlsx.py)
   for KP and in the CoV data-loading side of `run_ttp.py`.

2. **Stage 1 rule variants.** From the plate curve alone,
   [`systematic_ttp.py`](../src/lacewing/quantification/systematic_ttp.py)
   computes 4 candidate anchors (V1 closest-first-derivative-peak,
   V4 first zero-crossing of second derivative, V6 Cy₀ after plate
   TTP, V7 threshold-derivative crossing). V1 is the winner from the
   report ablation.

3. **Stage 2 per-pixel refinement.** For each pixel in each amp-positive
   well, [`build_per_pixel_labels.py`](../src/lacewing/quantification/conc_data/build_per_pixel_labels.py)
   cross-correlates the pixel's smoothed signal against the well-mean
   over a ±5 min window around the anchor, and shifts the anchor to
   the argmax. NTC and PTC wells get no label (they're excluded from
   the regression targets). Output: `per_pixel_labels.csv`, one row
   per (chip, well, pixel).

## Access

Raw chip data is not shipped with this repository (~5.7 GB total).

**For academic access**, contact the Centre for Bio-Inspired
Technology at Imperial College London.

## What you need on disk

The pipeline builds every intermediate cache from the raw `.bin` chip
recordings, so the only thing you need to lay out is the raw `Data/`
folder plus the two paired plate-qLAMP xlsx files for KP:

    Data/
    ├── 24_CoV_Quantification/                       # SARS-CoV-2
    │   ├── 1e5/D20240821_E01_C07_F4500KHz_U_1e5/    # 5 dose-response chips
    │   ├── 1e6/D20240821_E01_C09_F4500KHz_U_1e6/
    │   ├── 1e7/D20240820_E02_C03_F4500KHz_U_1e7/
    │   ├── 1e8/D20240822_E02_C00_F4500KHz_U_1e8/
    │   ├── 1e9/D20240822_E02_C00_F4500KHz_U_1e9/
    │   └── Others/                                  # ~30 NTC + multiplex chips
    │       ├── D20240719_E03_C44_F4500KHz_U_COV_SD/ # primary SD test chip
    │       └── … (see `ntc_inventory.csv` for the full list)
    │
    └── Concentration Data Experiment/               # K. pneumoniae
        ├── D20260812_E00_C00_F4500KHz_U_KP_conc_01/
        ├── D20260813_E00_C00_F4500KHz_U_DDM_KP_con_02/
        ├── D20260820_E00_C00_F4500KHz_U_DDM_KP_conc_03/
        ├── D20260823_E00_C00_F4500KHz_U_DDM_KP_04_05/
        ├── D20260827_E00_C00_F4500KHz_U_DDM_KP_Conc_05/
        ├── Chip 1.xlsx                              # paired LC96 qLAMP (chip 1)
        ├── Chip 2.xlsx                              # paired LC96 qLAMP (chip 2)
        ├── Chip Order.xlsx                          # well-layout definition
        └── plate_features.json                      # parsed plate anchors

Each chip folder contains ~50 `.bin` files (`find_active`,
`vref_sweep`, `evaluate_pixel`, `vs_calibration`, `gain`, `readout`)
plus PNG diagnostics — standard Lacewing capture output.

Set `LACEWING_DATA_ROOT` to the **parent** of `Data/` (see
[`REPRODUCE.md`](REPRODUCE.md) step 2).

## Building caches from raw

Once the raw `Data/` tree is in place, the cache-build steps in
[`REPRODUCE.md`](REPRODUCE.md) regenerate every `.npz` cache the ML
methods need:

    # CoV: builds deployed → +MAD → +spatA3 caches in one go (~1 h on HPC)
    qsub jobs/build_cache_cov_full_preproc.pbs

    # KP: builds 5 per-fold LOCO caches
    python -m lacewing.quantification.conc_data.build_regression_cache
    python -m lacewing.quantification.conc_data.build_classification_cache

Caches land in:

    src/lacewing/classification/data/cache/                       # CoV
    src/lacewing/quantification/regression_shared/data/cache/     # CoV regression + KP LOCO
    src/lacewing/quantification/methods/cache/                    # merged CoV+UTI variants
