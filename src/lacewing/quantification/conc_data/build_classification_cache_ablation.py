"""Data · KP classification cache builder for 3-state preprocessing ablation. [Cat A] Report §NewData RQ1 preproc ablation.

Build classification caches for the KP preprocessing ablation.

Mirrors the SARS-CoV-2 preprocessing ablation of Table 5.2:
  - deployed  : Layer A only, no spatA -> dataset_conc_deployed.npz
  - +MAD      : Layers ABCD (k=1.5), no spatA -> dataset_conc_mad.npz
  - +MAD+spatA3: existing dataset_conc.npz (built by build_classification_cache.py)

Each ablation cache is augmented with COV NTC pixels from the matching
SARS-CoV-2 cache (deployed COV NTCs augment deployed KP, etc.), so the
comparison stays apples-to-apples across preprocessing states.

Run (twice — once per ablation state):

    python -m lacewing.quantification.conc_data.build_classification_cache_ablation deployed
    python -m lacewing.quantification.conc_data.build_classification_cache_ablation mad
"""
from __future__ import annotations

import argparse
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

# Ablation state -> (KP pipe filter layers, KP spatA order,
#                    COV NTC cache stem, output KP cache name)
STATES = {
    "deployed": {
        "layers":       "A",
        "spat_order":   0,
        "cov_cache":    "dataset_all_filt_a_ntcRaw.npz",
        "out_name":     "dataset_conc_deployed.npz",
    },
    "mad": {
        "layers":       "ABCD",
        "spat_order":   0,
        "cov_cache":    "dataset_all_filt_abcd_ntcRaw.npz",
        "out_name":     "dataset_conc_mad.npz",
    },
}


def build_chip_pixels(cfg, layers: str, spat_order: int):
    """Run the KP titan pipeline with given filter layers and spat order.

    We read per_pixel_qc for both ablation states because with spat_order=0
    the spatial-averaging step is a no-op — per_pixel_spat == per_pixel_qc.
    per_pixel_qc is unambiguously "post-QC, pre-spatA" for either state.
    """
    print(f"\n--- {cfg.chip_tag}   (layers={layers}, spat_order={spat_order}) ---")
    if not cfg.chip_dir.exists():
        raise SystemExit(f"chip_dir not found: {cfg.chip_dir}")

    old = (pipe.WELL_LABELS, pipe.WELL_LOG10, pipe.NTC_WELLS,
           pipe.TRIM_SEARCH_MAX_MIN, pipe.FILTER_LAYERS, pipe.SPAT_ORDER)
    try:
        pipe.WELL_LABELS = cfg.well_labels
        pipe.WELL_LOG10  = cfg.well_log10
        pipe.NTC_WELLS   = tuple(cfg.ntc_wells)
        pipe.TRIM_SEARCH_MAX_MIN = cfg.trim_search_max_min
        pipe.FILTER_LAYERS = layers
        pipe.SPAT_ORDER    = spat_order
        result = pipe.process_chip(cfg.chip_dir, save_per_pixel=True)
    finally:
        (pipe.WELL_LABELS, pipe.WELL_LOG10, pipe.NTC_WELLS,
         pipe.TRIM_SEARCH_MAX_MIN, pipe.FILTER_LAYERS, pipe.SPAT_ORDER) = old

    xs, ys, chips, wells, pixels = [], [], [], [], []
    for w in result.wells:
        if w.per_pixel_qc is None or w.per_pixel_qc.shape[0] == 0:
            continue
        n = w.per_pixel_qc.shape[0]
        y_val = 0 if w.well in cfg.ntc_wells else 1
        xs.append(w.per_pixel_qc.astype(np.float32, copy=False))
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


def build_one_state(state: str):
    st = STATES[state]
    layers = st["layers"]
    spat_order = st["spat_order"]
    cov_cache_path = CACHE_DIR / st["cov_cache"]
    out_path = CACHE_DIR / st["out_name"]

    print("=" * 74)
    print(f"KP preprocessing ablation state: {state}")
    print(f"  layers={layers}  spat_order={spat_order}")
    print(f"  COV NTC augmentation cache: {cov_cache_path.name}")
    print(f"  Output: {out_path.name}")
    print("=" * 74)

    all_X, all_y, all_chip, all_well, all_pix = [], [], [], [], []
    for cfg in _build_chip_configs():
        X, y, chip, well, pix = build_chip_pixels(cfg, layers, spat_order)
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
    print(f"\nKP-conc only: N={len(X):,} pixels  "
          f"({n_pos_kp:,} pos / {n_neg_kp:,} neg)")

    # COV NTC augmentation from the matching-preprocessing cache
    if cov_cache_path.exists():
        print(f"\nLoading matching-preprocessing SARS-CoV-2 cache for NTC augmentation: "
              f"{cov_cache_path.name}")
        cov = np.load(cov_cache_path, allow_pickle=False)
        cov_X = cov["X"]
        cov_y = cov["y"]
        cov_chip = cov["chip_id"]
        cov_well = cov["well_id"]
        cov_pix = cov["pixel_id"]
        neg_mask = cov_y == 0
        n_cov_neg = int(neg_mask.sum())
        print(f"  SARS-CoV-2 NTC pixels available: {n_cov_neg:,}")

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
        print(f"\n[warn] SARS-CoV-2 cache not at {cov_cache_path} — "
              "skipping NTC augmentation")

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"\nFinal: N={len(X):,} pixels  ({n_pos:,} pos / {n_neg:,} neg)  "
          f"ratio pos:neg = {n_pos/max(n_neg,1):.2f}:1")
    print(f"       X.shape = {X.shape}  X.dtype = {X.dtype}")

    np.savez_compressed(out_path, X=X, y=y, chip_id=chip_id, well_id=well_id, pixel_id=pixel_id)
    print(f"\nwrote {out_path}  ({out_path.stat().st_size:,} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", choices=sorted(STATES.keys()),
                    help="Which ablation state to build.")
    args = ap.parse_args()
    build_one_state(args.state)


if __name__ == "__main__":
    main()
