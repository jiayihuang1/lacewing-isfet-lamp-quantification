"""Data · KP regression cache builder. [Cat A] Report §NewData.

Build per-pixel regression caches for the 5 Concentration Data chips.

Label recipe (per-well cy0_after_plate broadcast to pixels):

  Per amp-positive well w on chip C:
    plate_ttp  = qLAMP TTP from plate_features.json
                 - Chips 1/2 use per-chip plate
                 - Chips 3/4/5 use pooled_chip1_chip2 mean
    y_ttp_min  = cy0_after_plate(well_mean_signal, plate_ttp_min = plate_ttp)
                 -- one anchor per well, broadcast to every pixel of that well.

  Per PTC / NTC well:
    y_ttp_min = NaN
    (kept in the cache so RQ1 classifier training has negative examples;
     dropped by the regression harness which requires ~np.isnan(y_ttp_min))

  Per-pixel cross-correlation shift: TESTED (see build_per_pixel_labels.py)
     and found to add zero variety after the MAD-ABCD filter, so labels
     are per-well broadcast — same effective information content as
     the SARS-CoV-2 regression cache, differing only in the anchor
     (cy0_after_plate here vs raw plate qLAMP on SARS-CoV-2).

Output: 5 caches, one per LOCO fold, at

  Analysis/regression/data/cache/regress_conc_loco{TEST_CHIP_IDX}.npz

with schema matching Analysis/regression/data/build_regression_cache.py:

    X          (N, T)       float32 — per-pixel trace, spatA3 output
    y_ttp_min  (N,)         float32 — cy0_after_plate for that well, NaN if PTC/NTC
    chip_id    (N,)         <U64    — chip folder name
    well_id    (N,)         uint8   — well index 0..9
    pixel_id   (N,)         uint16  — pixel index within active set
    split      (N,)         uint8   — 0=train pool (4 chips), 1=test (LOCO chip)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from lacewing.quantification.conc_data.process_conc_chips import (  # noqa: E402
    _build_chip_configs,
    PLATE_FEATURES_JSON,
    _LOG10_TO_KP,
)
from lacewing.quantification.chip_pipeline import process_chip as pipe  # noqa: E402
from lacewing.quantification.systematic_ttp import compute_all_variants  # noqa: E402
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[3]
CACHE_DIR = LACEWING_PKG_DIR / "regression" / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _plate_ttp_for_well(cfg, plate_per_conc, w_idx):
    log10 = cfg.well_log10.get(w_idx)
    if log10 is None or not np.isfinite(log10):
        return None
    kp_label = _LOG10_TO_KP.get(float(log10))
    if kp_label is None:
        return None
    entry = plate_per_conc.get(kp_label)
    if entry is None:
        return None
    return float(entry.get("ttp_min"))


def build_chip_pixels(cfg):
    """Return dict with per-pixel X, y_ttp_min, chip_id, well_id, pixel_id for one chip.

    Loads chip data, runs the full titan pipeline (BS + MAD-ABCD + spatA3)
    with save_per_pixel=True, and computes cy0_after_plate per amp-positive
    well as the training label (broadcast to every pixel of that well).
    NTC / PTC pixels get y=NaN.
    """
    print(f"\n--- {cfg.chip_tag} ---")
    if not cfg.chip_dir.exists():
        raise SystemExit(f"chip_dir not found: {cfg.chip_dir}")

    plate_data = json.loads(PLATE_FEATURES_JSON.read_text())
    plate_per_conc = plate_data[cfg.plate_key]["per_conc_kp"]

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

        plate_ttp = _plate_ttp_for_well(cfg, plate_per_conc, w.well)
        if plate_ttp is None:
            # PTC / NTC — keep pixels with NaN label for RQ1 classification.
            y_val = float("nan")
        else:
            variants = compute_all_variants(
                w.mean_after_spat, w.time_min, plate_ttp_min=plate_ttp,
            )
            v = variants["cy0_after_plate"].ttp_min
            if not np.isfinite(v):
                print(f"  well {w.well} ({w.label}): cy0_after_plate=NaN → labelling as NaN")
                y_val = float("nan")
            else:
                y_val = float(v)

        n = w.per_pixel_spat.shape[0]
        xs.append(w.per_pixel_spat.astype(np.float32, copy=False))
        ys.append(np.full(n, y_val, dtype=np.float32))
        chips.append(np.full(n, cfg.chip_tag, dtype="<U64"))
        wells.append(np.full(n, w.well, dtype=np.uint8))
        pixels.append(np.arange(n, dtype=np.uint16))
        print(f"  well {w.well:>2} ({w.label:>7}): n_pix={n:>4}  "
              f"plate={plate_ttp if plate_ttp is not None else 'NA':>8}  "
              f"cy0_after_plate={y_val if np.isfinite(y_val) else 'NaN':>8}")

    return {
        "X":         np.concatenate(xs, axis=0) if xs else np.zeros((0, 0), np.float32),
        "y_ttp_min": np.concatenate(ys, axis=0) if ys else np.zeros((0,), np.float32),
        "chip_id":   np.concatenate(chips, axis=0) if chips else np.array([], dtype="<U64"),
        "well_id":   np.concatenate(wells, axis=0) if wells else np.array([], dtype=np.uint8),
        "pixel_id":  np.concatenate(pixels, axis=0) if pixels else np.array([], dtype=np.uint16),
    }


def main():
    configs = _build_chip_configs()

    # Step 1: build per-chip pixel dicts once (expensive titan pipeline call).
    per_chip = {}
    print(f"\n{'=' * 70}")
    print("STEP 1: run titan pipeline on all 5 chips (save_per_pixel=True)")
    print(f"{'=' * 70}")
    for cfg in configs:
        per_chip[cfg.chip_tag] = build_chip_pixels(cfg)

    # Sanity: X width must match across chips.
    widths = {ct: d["X"].shape[1] for ct, d in per_chip.items()}
    if len(set(widths.values())) > 1:
        print(f"\n[warn] X widths differ across chips: {widths}")
        # Truncate all to the minimum width.
        min_w = min(widths.values())
        for ct in per_chip:
            per_chip[ct]["X"] = per_chip[ct]["X"][:, :min_w]
        print(f"       Truncating all X to width={min_w}")

    # Step 2: write 5 LOCO caches (each chip serves as test once).
    print(f"\n{'=' * 70}")
    print("STEP 2: write LOCO caches (one per test chip)")
    print(f"{'=' * 70}")

    chip_tags_in_order = [cfg.chip_tag for cfg in configs]
    for test_idx, test_chip_tag in enumerate(chip_tags_in_order, start=1):
        parts = []
        splits = []
        for cfg in configs:
            d = per_chip[cfg.chip_tag]
            n = d["X"].shape[0]
            splits.append(np.full(n, 1 if cfg.chip_tag == test_chip_tag else 0,
                                  dtype=np.uint8))
            parts.append(d)
        cache = {
            "X":         np.concatenate([p["X"] for p in parts], axis=0),
            "y_ttp_min": np.concatenate([p["y_ttp_min"] for p in parts], axis=0),
            "chip_id":   np.concatenate([p["chip_id"] for p in parts], axis=0),
            "well_id":   np.concatenate([p["well_id"] for p in parts], axis=0),
            "pixel_id":  np.concatenate([p["pixel_id"] for p in parts], axis=0),
            "split":     np.concatenate(splits, axis=0),
        }
        out_path = CACHE_DIR / f"regress_conc_loco{test_idx}.npz"
        np.savez_compressed(out_path, **cache)
        n_tr = int((cache["split"] == 0).sum())
        n_te = int((cache["split"] == 1).sum())
        m_tr_labelled = (cache["split"] == 0) & ~np.isnan(cache["y_ttp_min"])
        m_te_labelled = (cache["split"] == 1) & ~np.isnan(cache["y_ttp_min"])
        print(f"  fold {test_idx} (test = {test_chip_tag}):")
        print(f"    total px = {cache['X'].shape[0]}  X.width = {cache['X'].shape[1]}")
        print(f"    train pool: {n_tr} px  ({int(m_tr_labelled.sum())} labelled amp+)")
        print(f"    test:       {n_te} px  ({int(m_te_labelled.sum())} labelled amp+)")
        print(f"    unique y values (train pool amp+): "
              f"{sorted(set(np.round(cache['y_ttp_min'][m_tr_labelled], 3).tolist()))}")
        print(f"    -> wrote {out_path}  ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
