"""A2b — compute per-well chip TTPs via argmax(gradient(well_mean_smoothed)).

Picks ONE chip TTP per amp-positive well by finding the argmax of the smoothed
first derivative WITHIN a search window that skips the thermal-settle bump.

No thresholds. No fitted models. No baseline heuristics. Just:
    smoothed = savgol(well_mean, window=15, order=3)
    d1       = np.gradient(smoothed)
    valid    = (t >= 2 min) & (t <= end - 5 min)
    ttp_idx  = np.argmax(d1[valid])
    chip_TTP = time_min[ttp_idx]

Output:
    Analysis/quantification/chip_pipeline/output/chip_ttps.json
    - per-well chip TTPs across all 4 UTI chips
    - plus each well's plate TTP (from A1) for immediate pairing
    - includes the smoothed signal + derivative arrays for the visualiser

Then A2b visualiser reads this JSON + renders per-well diagnostic plots so the
user can spot bad picks before committing to the calibration.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from lacewing.quantification.chip_pipeline import process_chip as pipe
from lacewing.quantification.chip_pipeline import process_all_uti_chips as multi
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[3]
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "chip_ttps.json"

# Search window for chip TTP (in minutes since chip start).
SEARCH_START_MIN = 2.0    # skip thermal-settle bump
SEARCH_END_BUFFER_MIN = 5.0   # skip last 5 min of run

# Smoothing (Savitzky-Golay window in samples; SAMPLES_PER_MIN = 15 → window=15 ≈ 1 min).
SG_WINDOW = 15
SG_POLY = 3


def _smooth_savgol(y: np.ndarray, window: int = SG_WINDOW, order: int = SG_POLY) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    if len(y) < window:
        window = max(3, len(y) - (len(y) + 1) % 2)
    if window < 3:
        return y.copy()
    return savgol_filter(y, window_length=window, polyorder=min(order, window - 1))


def compute_chip_ttp(time_min: np.ndarray, well_mean: np.ndarray) -> dict:
    """Compute chip TTP + supporting arrays for one well.

    Returns dict with:
        chip_ttp_min: float or None (None if window is degenerate)
        chip_ttp_idx: int index into time_min
        smoothed_signal: (T,) np.ndarray
        derivative:      (T,) np.ndarray  (np.gradient — same length)
        search_mask:     (T,) bool array of samples considered
    """
    T = len(time_min)
    smoothed = _smooth_savgol(well_mean)
    d1 = np.gradient(smoothed)

    end_min = time_min[-1] - SEARCH_END_BUFFER_MIN
    search_mask = (time_min >= SEARCH_START_MIN) & (time_min <= end_min)

    if not search_mask.any():
        return {
            "chip_ttp_min": None,
            "chip_ttp_idx": None,
            "smoothed_signal": smoothed,
            "derivative": d1,
            "search_mask": search_mask,
        }

    # Argmax of derivative within window
    d1_window = np.where(search_mask, d1, -np.inf)
    ttp_idx = int(np.argmax(d1_window))
    ttp_min = float(time_min[ttp_idx])

    return {
        "chip_ttp_min": ttp_min,
        "chip_ttp_idx": ttp_idx,
        "smoothed_signal": smoothed,
        "derivative": d1,
        "search_mask": search_mask,
    }


def _plate_features_for_well(cfg: multi.ChipConfig, well_idx: int) -> dict | None:
    """Return a plate-features dict for this well, or None if no LC96 data."""
    if not cfg.plate_features.exists():
        return None
    plate_data = json.loads(cfg.plate_features.read_text())
    wells = plate_data.get("wells", {})
    positions = cfg.lc96_well_map.get(well_idx, [])

    # Collect per-replicate arrays
    replicates = []
    for pos in positions:
        w = wells.get(pos)
        if w is None:
            continue
        if w.get("Cq_takeoff_min") is None:
            continue
        replicates.append({
            "position": pos,
            "cycles": w.get("cycles"),
            "fluor": w.get("fluor"),
            "Cq_start_min": w.get("Cq_start_min"),
            "Cq_takeoff_min": w.get("Cq_takeoff_min"),
            "Cq_plateau_min": w.get("Cq_plateau_min"),
            "is_positive": w.get("is_positive"),
        })

    if not replicates:
        return None

    takeoffs = [r["Cq_takeoff_min"] for r in replicates if r["Cq_takeoff_min"] is not None]
    plate_ttp_mean = float(np.mean(takeoffs)) if takeoffs else None
    return {
        "replicates": replicates,
        "plate_ttp_mean_min": plate_ttp_mean,
    }


def main() -> None:
    all_wells: list[dict] = []

    for cfg in multi.CHIPS:
        print(f"\n{'=' * 70}\n[A2b] {cfg.chip_tag}\n{'=' * 70}")
        if not cfg.chip_dir.exists():
            print(f"  SKIP (chip_dir not found)")
            continue

        # Process chip (only need well-mean + per-pixel_spat — but process_chip already gives
        # us the well-mean via mean_after_spat, which is what we want for TTP picking).
        old_labels = pipe.WELL_LABELS
        old_log10  = pipe.WELL_LOG10
        old_ntc    = pipe.NTC_WELLS
        old_trim   = pipe.TRIM_SEARCH_MAX_MIN
        try:
            pipe.WELL_LABELS = cfg.well_labels
            pipe.WELL_LOG10  = cfg.well_log10
            pipe.NTC_WELLS   = tuple(cfg.ntc_wells)
            pipe.TRIM_SEARCH_MAX_MIN = cfg.trim_search_max_min
            result = pipe.process_chip(cfg.chip_dir, save_per_pixel=False)
        finally:
            pipe.WELL_LABELS = old_labels
            pipe.WELL_LOG10  = old_log10
            pipe.NTC_WELLS   = old_ntc
            pipe.TRIM_SEARCH_MAX_MIN = old_trim

        for w in result.wells:
            log10 = w.log10_conc if np.isfinite(w.log10_conc) else None
            well_mean = w.mean_after_spat

            if well_mean is None or np.all(np.isnan(well_mean)):
                print(f"  well {w.well:>2} {w.label:<25s}  SKIP (no well-mean signal)")
                continue

            picked = compute_chip_ttp(w.time_min, well_mean)
            plate = _plate_features_for_well(cfg, w.well)

            record = {
                "chip_tag": cfg.chip_tag,
                "well_id": w.well,
                "well_label": w.label,
                "log10_conc": log10,
                "is_ntc": w.well in cfg.ntc_wells,
                "n_kept_pixels": w.n_kept,
                "time_min": w.time_min.tolist(),
                "well_mean_spat": well_mean.tolist(),
                "smoothed_signal": picked["smoothed_signal"].tolist(),
                "derivative": picked["derivative"].tolist(),
                "chip_ttp_min": picked["chip_ttp_min"],
                "chip_ttp_idx": picked["chip_ttp_idx"],
                "plate": plate,
                "search_start_min": SEARCH_START_MIN,
                "search_end_buffer_min": SEARCH_END_BUFFER_MIN,
            }
            all_wells.append(record)

            plate_str = (f"plate_TTP={plate['plate_ttp_mean_min']:.2f}" if plate and plate["plate_ttp_mean_min"] is not None
                          else "plate_TTP=None")
            print(f"  well {w.well:>2} {w.label:<25s}  chip_TTP="
                  f"{picked['chip_ttp_min']:.2f}min  ({plate_str})")

    OUT_JSON.write_text(json.dumps({
        "search_start_min": SEARCH_START_MIN,
        "search_end_buffer_min": SEARCH_END_BUFFER_MIN,
        "sg_window": SG_WINDOW,
        "sg_poly": SG_POLY,
        "wells": all_wells,
    }))

    n_with_plate = sum(1 for w in all_wells if w["plate"] and w["plate"]["plate_ttp_mean_min"] is not None)
    print(f"\n[ok] wrote {OUT_JSON}")
    print(f"     {len(all_wells)} wells across {len(multi.CHIPS)} chips")
    print(f"     {n_with_plate} wells have matched plate TTP (usable for calibration fit)")


if __name__ == "__main__":
    main()
