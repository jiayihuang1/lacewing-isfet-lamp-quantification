"""Data · KP classification cache builder. [Cat A] Report §NewData.

Build classification cache for RQ1 experiments on the KP-conc chips.

Output: Analysis/classification/data/cache/dataset_conc.npz

Schema (matches lacewing.classification.data.build_dataset):
    X         (N, T)   float32 — per-pixel spatA3 trace
    y         (N,)     uint8   — 0 = amp-negative (NTC), 1 = amp-positive (well 0-7 + PTC)
    chip_id   (N,)     <U64
    well_id   (N,)     uint8
    pixel_id  (N,)     uint16

Notes:
- Labels are derived from Chip Order.xlsx: wells 0-7 = conc replicates (amp+),
  well 8/9 = PTC/NTC (per-chip PTC/NTC ordering varies — see ChipConfig).
- The PTC well is treated as amp-positive (it IS a positive control, the
  reaction should amplify). NTC is amp-negative.
- KP-conc alone gives a ~7.3:1 pos:neg imbalance. To reduce that, we
  ALSO pull all NTC pixels from the shipped SARS-CoV-2 classification
  cache (dataset_all_filt_abcd_ntcRaw_madk1p5_spatA3) as additional
  amp-negatives. amp-positive pixels from SARS-CoV-2 are NOT added
  (would introduce cross-chemistry confound on the amp+ side).
- Downstream: pass to train.py via --cache dataset_conc --split sd_test
  --fold <chip_folder_name> to hold out a single chip for LOCO evaluation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from lacewing.quantification.conc_data.process_conc_chips import (  # noqa: E402
    _build_chip_configs,
)
from lacewing.quantification.chip_pipeline import process_chip as pipe  # noqa: E402
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[3]
CACHE_DIR = LACEWING_PKG_DIR / "classification" / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = CACHE_DIR / "dataset_conc.npz"

# Shipped SARS-CoV-2 classification cache we borrow NTC pixels from
# (spatA3 preprocessing matches what we run on the KP-conc chips).
COV_CACHE_PATH = CACHE_DIR / "dataset_all_filt_abcd_ntcRaw_madk1p5_spatA3.npz"


def build_chip_pixels(cfg):
    """Return (X, y, chip_id, well_id, pixel_id) arrays for one chip."""
    print(f"\n--- {cfg.chip_tag} ---")
    if not cfg.chip_dir.exists():
        raise SystemExit(f"chip_dir not found: {cfg.chip_dir}")

    old = (pipe.WELL_LABELS, pipe.WELL_LOG10, pipe.NTC_WELLS, pipe.TRIM_SEARCH_MAX_MIN)
    try:
        pipe.WELL_LABELS = cfg.well_labels
        pipe.WELL_LOG10  = cfg.well_log10
        pipe.NTC_WELLS   = tuple(cfg.ntc_wells)
        pipe.TRIM_SEARCH_MAX_MIN = cfg.trim_search_max_min
        result = pipe.process_chip(cfg.chip_dir, save_per_pixel=True)
    finally:
        pipe.WELL_LABELS, pipe.WELL_LOG10, pipe.NTC_WELLS, pipe.TRIM_SEARCH_MAX_MIN = old

    xs, ys, chips, wells, pixels = [], [], [], [], []
    for w in result.wells:
        if w.per_pixel_spat is None or w.per_pixel_spat.shape[0] == 0:
            continue
        n = w.per_pixel_spat.shape[0]
        # amp+ / amp- label: NTC = 0, everything else = 1
        y_val = 0 if w.well in cfg.ntc_wells else 1
        xs.append(w.per_pixel_spat.astype(np.float32, copy=False))
        ys.append(np.full(n, y_val, dtype=np.uint8))
        chips.append(np.full(n, cfg.chip_tag, dtype="<U64"))
        wells.append(np.full(n, w.well, dtype=np.uint8))
        pixels.append(np.arange(n, dtype=np.uint16))
        print(f"  well {w.well:>2} ({w.label:>7}):  n_pix={n:>4}  y={y_val}")

    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(chips, axis=0),
        np.concatenate(wells, axis=0),
        np.concatenate(pixels, axis=0),
    )


def main():
    all_X, all_y, all_chip, all_well, all_pix = [], [], [], [], []
    print("=" * 70)
    print("Building classification cache for all 5 KP-conc chips")
    print("=" * 70)
    for cfg in _build_chip_configs():
        X, y, chip, well, pix = build_chip_pixels(cfg)
        all_X.append(X); all_y.append(y); all_chip.append(chip)
        all_well.append(well); all_pix.append(pix)

    widths = {i: x.shape[1] for i, x in enumerate(all_X)}
    if len(set(widths.values())) > 1:
        min_w = min(widths.values())
        all_X = [x[:, :min_w] for x in all_X]
        print(f"\n[note] truncating all X to width={min_w} to match shortest chip")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    chip_id = np.concatenate(all_chip, axis=0)
    well_id = np.concatenate(all_well, axis=0)
    pixel_id = np.concatenate(all_pix, axis=0)

    n_pos_kp = int((y == 1).sum())
    n_neg_kp = int((y == 0).sum())
    print(f"\nKP-conc only: N={len(X):,} pixels  ({n_pos_kp:,} pos / {n_neg_kp:,} neg)  "
          f"ratio pos:neg = {n_pos_kp/max(n_neg_kp,1):.2f}:1")

    # -------- Augment with SARS-CoV-2 NTC pixels (amp- only) --------
    if COV_CACHE_PATH.exists():
        print(f"\nLoading SARS-CoV-2 cache for NTC augmentation: {COV_CACHE_PATH.name}")
        cov = np.load(COV_CACHE_PATH, allow_pickle=False)
        cov_X = cov["X"]
        cov_y = cov["y"]
        cov_chip = cov["chip_id"]
        cov_well = cov["well_id"]
        cov_pix = cov["pixel_id"]
        neg_mask = cov_y == 0
        n_cov_neg = int(neg_mask.sum())
        print(f"  SARS-CoV-2 NTC pixels available: {n_cov_neg:,}")

        # X-width may differ (different chip readout length). Pad or truncate
        # to match KP-conc width so we can concatenate.
        kp_T = X.shape[1]
        cov_T = cov_X.shape[1]
        if cov_T > kp_T:
            print(f"  truncating COV X from width {cov_T} to {kp_T}")
            cov_X = cov_X[:, :kp_T]
        elif cov_T < kp_T:
            print(f"  right-padding COV X from width {cov_T} to {kp_T} (zeros)")
            pad = np.zeros((cov_X.shape[0], kp_T - cov_T), dtype=cov_X.dtype)
            cov_X = np.concatenate([cov_X, pad], axis=1)

        X = np.concatenate([X, cov_X[neg_mask].astype(np.float32)], axis=0)
        y = np.concatenate([y, np.zeros(n_cov_neg, dtype=y.dtype)], axis=0)
        chip_id = np.concatenate([chip_id, cov_chip[neg_mask].astype(chip_id.dtype)], axis=0)
        well_id = np.concatenate([well_id, cov_well[neg_mask].astype(well_id.dtype)], axis=0)
        pixel_id = np.concatenate([pixel_id, cov_pix[neg_mask].astype(pixel_id.dtype)], axis=0)
    else:
        print(f"\n[warn] SARS-CoV-2 cache not at {COV_CACHE_PATH} — skipping NTC augmentation")

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"\nFinal: N={len(X):,} pixels  ({n_pos:,} pos / {n_neg:,} neg)  "
          f"ratio pos:neg = {n_pos/max(n_neg,1):.2f}:1")
    print(f"       X.shape = {X.shape}  X.dtype = {X.dtype}")
    print(f"       unique chips = {sorted(np.unique(chip_id).tolist())}")

    np.savez_compressed(
        OUT_PATH,
        X=X, y=y, chip_id=chip_id, well_id=well_id, pixel_id=pixel_id,
    )
    print(f"\nwrote {OUT_PATH}  ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
