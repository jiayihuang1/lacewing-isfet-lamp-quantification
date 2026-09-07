# Data-quality analysis

Diagnostics that ask, of the exp6 test set: *do the pixels the
classifiers misclassify look statistically different from the pixels
they get right?*

If yes, the statistic on which they differ is a candidate signal for a
preprocessing filter that would remove the failure cases before
training.  If no, the failures are not pixel-quality issues, and
preprocessing is not the right lever.

## Method

For every (chip, well, pixel) triple in the exp6 test set (21,058
pixels across the two held-out SD chips), count how many of the 21
model x seed runs got it wrong.

- `always_correct`     n_wrong = 0
- `sometimes_wrong`    1 <= n_wrong < 21
- `always_wrong`       n_wrong = 21

Then compute per-pixel statistics on the raw 450-sample trace
(dynamic range, baseline-segment std, absolute end-to-end drift,
max-abs, mean-abs) and compare distributions across buckets,
broken down by chip and by true label.

A Cohen's d effect size is computed between `always_correct` and
`always_wrong` to surface statistics where the two populations
clearly separate.

## Run

```
cd "Individual Project"
.venv/bin/python -m Analysis.preprocessing.data_quality.analyse_misclassification_stats
```

CPU-only, runs in well under a minute on a laptop using files already
in `Analysis/classification/results/exp6_final_to_sd/runs/*/predictions.npz`.

## Outputs

`Analysis/preprocessing/data_quality/results/`:

- `bucket_counts.csv` -- pixel counts per (chip, label, bucket)
- `separability.csv` -- per-(chip, label, statistic) medians and
  Cohen's d for `always_correct` vs `always_wrong`
- `hist_overall.png` / `box_overall.png` -- bucket comparison across
  all test pixels
- `hist_<chip>_label<0|1>.png` / `box_<chip>_label<0|1>.png` --
  bucket comparison broken down by chip and class

## NTC-referenced four-layer filter

Once the separability analysis above identifies the statistics that
distinguish failures from correctly-classified pixels, the chip-
relative four-layer filter ports those observations into a
preprocessing step.  Pure filter math lives in
[`filter_core.py`](filter_core.py); the diagnostic / viewer pipeline
that scores each pixel against the exp6 failure analysis lives in
[`build_filter_pipeline.py`](build_filter_pipeline.py).

The four layers, all calibrated **per chip** with no hard-coded
voltage thresholds:

- **Layer A (amplitude floor).**  Drop pixels whose dynamic range
  falls below the NTC well's lower percentile of dynamic range
  (default 5th).  Catches pixels that are flatter than even the
  flattest NTC pixels on that chip.
- **Layer B (shape floor).**  Drop pixels whose net slope or trace
  minimum is below the NTC well's lower percentile (default 5th).
  Catches downward-going traces that the chip's NTC drift cannot
  account for.
- **Layer C (kNN bad-neighbour count).**  Drop a pixel if &ge; 4 of
  its 8 nearest spatial neighbours (same well) were dropped by
  Layer A or B.  Catches pixels in regions where the surrounding
  hardware is misbehaving.
- **Layer D (individual disagreement, median variant).**  For each
  surviving pixel, the L2 distance from its trace to the elementwise
  *median* of its 8 nearest spatial neighbours' traces.  Drop pixels
  above the per-well 95th percentile cutoff.  The median (rather
  than mean) reference makes this layer robust to a single
  contaminating neighbour.

Layers C and D use each pixel's (row, col) on the chip's pixel grid;
the coordinates are recovered by reloading each chip via titan
([`spatial_coords.py`](spatial_coords.py)).

### Diagnostic viewer

```
.venv/bin/python -m Analysis.preprocessing.data_quality.build_filter_pipeline
```

Optional flags::

    --pct-amp        percentile of NTC dyn_range for Layer A (default 5)
    --pct-shape      percentile of NTC net_slope / trace_min for Layer B (default 5)
    --c-k            kNN size for Layer C (default 8)
    --c-bad-frac     min fraction of bad neighbours for Layer C (default 0.5)
    --d-k            kNN size for Layer D (default 8)
    --d-pct          per-well percentile cutoff for Layer D (default 95)
    --downsample     time-axis stride to keep JSON small (default 1)

Outputs land in `filter_data/`: one JSON per chip + `manifest.json`
with the overall recall/collateral table.

### HTML viewer

`filter_viewer.html` renders the six pipeline steps side by side
per (chip, well), with two parallel rows:

- **Top row (trace overlays):**  Step 0 (raw) | after A | after B |
  after C | after D | final.  Traces are colour-coded by which layer
  dropped them (purple = A, teal = B, orange = C, red = D), or grey
  if kept.  Dropped traces are drawn at higher alpha so you can see
  what each layer removed.
- **Bottom row (spatial heatmaps):**  same six steps, but as a
  heatmap on the pixel's (row, col) grid.

Layer D has three sub-strategies selectable from a dropdown: `mean`,
`median`, `kth`.  Hover any trace or spatial pixel to see exactly
which threshold was tripped, in mathematical detail.

To open the viewer (browsers block local fetch otherwise):

```
cd "Individual Project"
.venv/bin/python -m http.server --directory Analysis/preprocessing/data_quality 8765
```

then visit http://localhost:8765/filter_viewer.html

### MAD viewer

A second viewer, `filter_viewer_mad.html`, renders exactly the same
six-panel pipeline but with the MAD-based adaptive threshold rule
instead of the percentile cut.  It has a top-level dropdown to
switch between three pre-built MAD configurations:
- `filter_data_mad_k1p5/`   (k=1.5, ≈7th percentile on a clean Gaussian)
- `filter_data_mad_k1p645/` (k=1.645, = 5th percentile)
- `filter_data_mad_k2p0/`   (k=2.0, ≈2.3rd percentile)

Build the three MAD data sets:

```
.venv/bin/python -m Analysis.preprocessing.data_quality.build_filter_pipeline --rule mad --k-mad 1.5
.venv/bin/python -m Analysis.preprocessing.data_quality.build_filter_pipeline --rule mad --k-mad 1.645
.venv/bin/python -m Analysis.preprocessing.data_quality.build_filter_pipeline --rule mad --k-mad 2.0
```

Then open http://localhost:8765/filter_viewer_mad.html and pick a
`k` from the dropdown.  Switching `k` re-loads the manifest and
chip data; the chip / well / Layer D strategy selectors stay where
they were.

## Experiment: ablation of each filter layer on exp6 accuracy

The `exp7` family of experiments retrains the five well-behaved
time-domain classifiers (ANN, 1D-DCNN, FCN, ResNet1D, InceptionTime)
on filtered training caches and reports accuracy on the same two
held-out SD chips used by exp6.  Eight experiments compare:

- **positive-well filtering**: A only / A+B / A+B+C / A+B+C+D
- **NTC well treatment**: leave untouched / clean with Layer D
  (also affects the A/B chip-relative thresholds since they are
  recomputed on the cleaned NTC distribution)

Build the eight filtered caches once (CPU-only, ~10-20 s/chip on a
laptop, ~10-20 minutes total per cache for the all-data scope):

```
.venv/bin/python -m Analysis.classification.experiments.run_exp7 --build-all-caches
```

Then submit the PBS array on Imperial RCS:

```
qsub jobs/exp7_sweep.pbs
```

The array runs 8 experiments x 5 models x 3 seeds = 120 GPU jobs.
Each cell takes 5-15 min on an RCS GPU, so the full sweep finishes
in ~2-4 hours wall-clock at 8-16 concurrent jobs (depending on the
queue allocation).

See [`run_exp7.py`](../../classification/experiments/run_exp7.py)
for the experiment configuration table and the cache-stem naming
convention.
