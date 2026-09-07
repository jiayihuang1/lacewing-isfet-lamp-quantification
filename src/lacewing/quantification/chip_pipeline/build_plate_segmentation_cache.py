"""Build a per-pixel segmentation cache for F-D retraining (A4 revised).

Label construction (Option C — anchored on user's manual chip TTPs):

  Per well w (with a saved manual chip TTP + matching plate features):
    user_TTP = manual chip TTP (from manual_chip_ttps.json)
    plate_start   = mean of plate replicates' Cq_start_min
    plate_takeoff = mean of plate replicates' Cq_takeoff_min
    plate_plateau = mean of plate replicates' Cq_plateau_min

    Relative plate spacings:
      dt_start_to_takeoff   = plate_takeoff - plate_start
      dt_takeoff_to_plateau = plate_plateau - plate_takeoff

    Chip boundaries on the well-mean's time axis:
      b1_wellmean = user_TTP - dt_start_to_takeoff        (baseline → drift)
      b2_wellmean = user_TTP                              (drift → rising)
      b3_wellmean = user_TTP + dt_takeoff_to_plateau      (rising → plateau)

  Per pixel p in well w:
    pixel_shift[p] = argmax lag of cross-correlate(pixel_trace, well_mean_trace)
      (constrained to a plate-anchored window around user_TTP so the
       correlation doesn't lock onto the thermal-settle bump)
    Each chip boundary shifted by pixel_shift[p]:
      b_i_pixel_p = b_i_wellmean + pixel_shift[p]

  Broadcast to a per-timestep 4-class label array (int8).

Result: labels use manual TTP as the anchor + plate for relative spacing.
No calibration curve needed. Per-pixel timing variance recovered via
cross-correlation.

Output: Analysis/quantification/methods/results/uti_seg_manual_labels_v1/cache.npz
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import correlate

from lacewing.quantification.chip_pipeline import process_chip as pipe
from lacewing.quantification.chip_pipeline import process_all_uti_chips as multi
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


CLS_BASELINE = 0
CLS_DRIFT = 1
CLS_RISING = 2
CLS_PLATEAU = 3

PROJECT_ROOT = DATA_ROOT  # was: parents[3]
OUT_DIR = LACEWING_PKG_DIR / "quantification" / "methods" / "results" / "uti_seg_manual_labels_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANUAL_LABELS_JSON = Path(__file__).resolve().parent / "output" / "manual_chip_ttps.json"

SAMPLES_PER_MIN = 15   # matches training-time cache convention (T=450 over 30 min)


# ---------------------------------------------------------------------------
# Cross-correlation with plate-anchored search window
# ---------------------------------------------------------------------------

def compute_pixel_shift(pixel_trace: np.ndarray, well_mean_trace: np.ndarray,
                        anchor_sample: int, half_window_samples: int,
                        max_lag_samples: int) -> int:
    """Cross-correlation lag between pixel_trace and well_mean_trace,
    restricted to a window around `anchor_sample` (to avoid locking on
    thermal-settle bumps at t=0).

    Uses Z-score normalisation (subtract mean, divide by std) before the
    correlation so that weak / noisy pixels aren't dominated by absolute
    amplitude noise. Returns lag = 0 if either signal has effectively zero
    variance in the window (safer than a spurious pick).

    Returns integer sample lag such that pixel_trace shifted RIGHT by `shift`
    best aligns with well_mean_trace. Positive shift → pixel amplifies LATER.
    """
    n = len(pixel_trace)
    lo = max(0, anchor_sample - half_window_samples)
    hi = min(n, anchor_sample + half_window_samples)
    if hi - lo < 5:
        return 0

    # Slice both signals to the window.
    p = pixel_trace[lo:hi].astype(np.float64, copy=False)
    m = well_mean_trace[lo:hi].astype(np.float64, copy=False)
    p = np.nan_to_num(p, nan=0.0)
    m = np.nan_to_num(m, nan=0.0)

    # Z-score normalise both (centre + scale by std). If either has essentially
    # no variance in the window, the pixel doesn't carry usable shape info here.
    p_mean, m_mean = float(p.mean()), float(m.mean())
    p_std, m_std = float(p.std(ddof=0)), float(m.std(ddof=0))
    eps = 1e-9
    if p_std < eps or m_std < eps:
        return 0
    p = (p - p_mean) / p_std
    m = (m - m_mean) / m_std

    xcorr = correlate(p, m, mode="full")
    lags = np.arange(-(len(m) - 1), len(p))
    valid = np.abs(lags) <= max_lag_samples
    if not valid.any():
        return 0
    return int(lags[valid][np.argmax(xcorr[valid])])


def build_pixel_labels(n_samples: int, b1: int, b2: int, b3: int) -> np.ndarray:
    """Per-timestep 4-class labels from 3 boundary sample indices (clipped)."""
    b1 = int(max(0, min(n_samples, b1)))
    b2 = int(max(b1, min(n_samples, b2)))
    b3 = int(max(b2, min(n_samples, b3)))
    labels = np.empty(n_samples, dtype=np.int8)
    labels[:b1]   = CLS_BASELINE
    labels[b1:b2] = CLS_DRIFT
    labels[b2:b3] = CLS_RISING
    labels[b3:]   = CLS_PLATEAU
    return labels


# ---------------------------------------------------------------------------
# Plate features helper
# ---------------------------------------------------------------------------

def _plate_spacings_for_well(cfg: multi.ChipConfig, well_idx: int) -> tuple[float | None, float | None]:
    """Return (dt_start_to_takeoff, dt_takeoff_to_plateau) in minutes,
    computed as the mean across LC96 replicates for this chip well.
    """
    if not cfg.plate_features.exists():
        return None, None
    data = json.loads(cfg.plate_features.read_text())
    wells_dict = data.get("wells", {})
    positions = cfg.lc96_well_map.get(well_idx, [])

    starts, takeoffs, plateaus = [], [], []
    for pos in positions:
        w = wells_dict.get(pos)
        if w is None:
            continue
        if w.get("Cq_start_min") is not None:   starts.append(w["Cq_start_min"])
        if w.get("Cq_takeoff_min") is not None: takeoffs.append(w["Cq_takeoff_min"])
        if w.get("Cq_plateau_min") is not None: plateaus.append(w["Cq_plateau_min"])

    if not takeoffs:
        return None, None
    takeoff = float(np.mean(takeoffs))
    dt_st = (takeoff - float(np.mean(starts))) if starts else None
    dt_tp = (float(np.mean(plateaus)) - takeoff) if plateaus else None
    return dt_st, dt_tp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not MANUAL_LABELS_JSON.exists():
        raise SystemExit(f"Missing {MANUAL_LABELS_JSON}. Run the manual labelling UI first.")
    manual = json.loads(MANUAL_LABELS_JSON.read_text())["labels"]

    print(f"Loaded {len(manual)} manual labels from {MANUAL_LABELS_JSON}")

    # Aggregated containers
    all_X: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_chip_tag: list[str] = []
    all_well_id: list[int] = []
    all_log10: list[float] = []
    all_pixel_shift: list[int] = []
    all_user_ttp: list[float] = []
    all_b1, all_b2, all_b3 = [], [], []
    all_reliable: list[bool] = []
    time_min_ref = None

    per_well_stats: list[dict] = []

    for cfg in multi.CHIPS:
        # Skip chips not in manual labels — that includes the NC test chip.
        chip_keys = [k for k in manual if k.startswith(f"{cfg.chip_tag}__")]
        if not chip_keys:
            print(f"\n[SKIP] {cfg.chip_tag}: no manual labels")
            continue

        print(f"\n{'=' * 70}\n[A4] Processing {cfg.chip_tag}\n{'=' * 70}")
        if not cfg.chip_dir.exists():
            print(f"  [SKIP] chip_dir not found")
            continue

        # Process chip with per-pixel data
        old_labels = pipe.WELL_LABELS
        old_log10  = pipe.WELL_LOG10
        old_ntc    = pipe.NTC_WELLS
        old_trim   = pipe.TRIM_SEARCH_MAX_MIN
        try:
            pipe.WELL_LABELS = cfg.well_labels
            pipe.WELL_LOG10  = cfg.well_log10
            pipe.NTC_WELLS   = tuple(cfg.ntc_wells)
            pipe.TRIM_SEARCH_MAX_MIN = cfg.trim_search_max_min
            result = pipe.process_chip(cfg.chip_dir, save_per_pixel=True)
        finally:
            pipe.WELL_LABELS = old_labels
            pipe.WELL_LOG10  = old_log10
            pipe.NTC_WELLS   = old_ntc
            pipe.TRIM_SEARCH_MAX_MIN = old_trim

        if time_min_ref is None:
            time_min_ref = result.wells[0].time_min.astype(np.float32)
        n_samples = len(time_min_ref)

        # Cross-correlation constraints:
        #   half_window: search ±5 min around the user's TTP for the amplification region.
        #     Widened from 3 to 5 min so the correlation has more amplification signal
        #     to lock onto — especially matters for wells where the rise extends beyond
        #     the ±3 min band or where the take-off is somewhat off from user_TTP.
        #   max_lag:     allow per-pixel shifts up to ±5 min from well-mean.
        #     Widened from ±3 min because 07-28 wells were pegged at ±3 min ceiling.
        HALF_WINDOW_MIN = 5.0
        MAX_LAG_MIN = 5.0
        half_window_samples = int(round(HALF_WINDOW_MIN * SAMPLES_PER_MIN))
        max_lag_samples = int(round(MAX_LAG_MIN * SAMPLES_PER_MIN))

        for w in result.wells:
            key = f"{cfg.chip_tag}__well{w.well}"
            label = manual.get(key)
            if label is None:
                print(f"  well {w.well:>2} {w.label:<25s}  SKIP (no manual label)")
                continue
            if label.get("is_no_amp") or label.get("chip_ttp_min") is None:
                print(f"  well {w.well:>2} {w.label:<25s}  SKIP (marked no-amp)")
                continue
            if w.per_pixel_qc is None or len(w.per_pixel_qc) == 0:
                print(f"  well {w.well:>2} {w.label:<25s}  SKIP (0 kept pixels)")
                continue

            user_ttp_min = float(label["chip_ttp_min"])
            log10 = w.log10_conc if np.isfinite(w.log10_conc) else None

            # Plate spacings for THIS well
            dt_start, dt_plateau = _plate_spacings_for_well(cfg, w.well)
            # Fallbacks if a spacing is missing:
            #   - start_to_takeoff: fall back to 3 min (typical LAMP)
            #   - takeoff_to_plateau: fall back to 10 min
            if dt_start is None:
                dt_start = 3.0
                print(f"      [fallback] using dt_start=3.0 min for well {w.well}")
            if dt_plateau is None:
                dt_plateau = 10.0
                print(f"      [fallback] using dt_plateau=10.0 min for well {w.well}")

            # Chip boundaries on well-mean axis (in samples)
            b1_min = user_ttp_min - dt_start
            b2_min = user_ttp_min
            b3_min = user_ttp_min + dt_plateau
            b1_wellmean = int(round(b1_min * SAMPLES_PER_MIN))
            b2_wellmean = int(round(b2_min * SAMPLES_PER_MIN))
            b3_wellmean = int(round(b3_min * SAMPLES_PER_MIN))

            # Cross-correlate every pixel against the well-mean
            pixel_traces_qc = w.per_pixel_qc          # pre-spatA3 for xcorr (has per-pixel timing)
            pixel_traces_spat = w.per_pixel_spat      # post-spatA3 for X (matches training input)
            well_mean_qc = pixel_traces_qc.mean(axis=0)
            n_pix = pixel_traces_qc.shape[0]

            shifts = np.empty(n_pix, dtype=np.int32)
            labels_arr = np.empty((n_pix, n_samples), dtype=np.int8)
            b1_arr = np.empty(n_pix, dtype=np.int32)
            b2_arr = np.empty(n_pix, dtype=np.int32)
            b3_arr = np.empty(n_pix, dtype=np.int32)

            for i in range(n_pix):
                shift = compute_pixel_shift(
                    pixel_traces_qc[i], well_mean_qc,
                    anchor_sample=b2_wellmean,
                    half_window_samples=half_window_samples,
                    max_lag_samples=max_lag_samples,
                )
                shifts[i] = shift
                b1_i = b1_wellmean + shift
                b2_i = b2_wellmean + shift
                b3_i = b3_wellmean + shift
                labels_arr[i] = build_pixel_labels(n_samples, b1_i, b2_i, b3_i)
                b1_arr[i] = b1_i
                b2_arr[i] = b2_i
                b3_arr[i] = b3_i

            reliable = w.well not in cfg.lc96_unreliable
            all_X.append(pixel_traces_spat.astype(np.float32, copy=False))
            all_labels.append(labels_arr)
            all_chip_tag.extend([cfg.chip_tag] * n_pix)
            all_well_id.extend([w.well] * n_pix)
            all_log10.extend([log10 if log10 is not None else float("nan")] * n_pix)
            all_pixel_shift.extend(shifts.tolist())
            all_user_ttp.extend([user_ttp_min] * n_pix)
            all_b1.extend(b1_arr.tolist())
            all_b2.extend(b2_arr.tolist())
            all_b3.extend(b3_arr.tolist())
            all_reliable.extend([reliable] * n_pix)

            print(f"  well {w.well:>2} {w.label:<25s}  n={n_pix:>4}  "
                  f"user_TTP={user_ttp_min:5.2f}min  "
                  f"b1/b2/b3 (well-mean samples)={b1_wellmean}/{b2_wellmean}/{b3_wellmean}  "
                  f"shift range=[{shifts.min():+d}, {shifts.max():+d}]  "
                  f"{'⚠unreliable' if not reliable else ''}")

            per_well_stats.append({
                "chip_tag": cfg.chip_tag, "well": w.well, "label": w.label,
                "n_kept": w.n_kept, "n_labelled": n_pix,
                "user_ttp_min": user_ttp_min,
                "dt_start_to_takeoff_min": dt_start,
                "dt_takeoff_to_plateau_min": dt_plateau,
                "b1_wellmean_samples": b1_wellmean,
                "b2_wellmean_samples": b2_wellmean,
                "b3_wellmean_samples": b3_wellmean,
                "shift_min": int(shifts.min()),
                "shift_max": int(shifts.max()),
                "shift_mean": float(shifts.mean()),
                "shift_std": float(shifts.std()),
                "reliable": reliable,
            })

    if not all_X:
        raise SystemExit("No labellable pixels found.")

    X = np.concatenate(all_X, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    chip_tag = np.array(all_chip_tag, dtype="U40")
    well_id = np.array(all_well_id, dtype=np.int32)
    log10_conc = np.array(all_log10, dtype=np.float32)
    pixel_shift = np.array(all_pixel_shift, dtype=np.int32)
    user_ttp_min = np.array(all_user_ttp, dtype=np.float32)
    b1_arr = np.array(all_b1, dtype=np.int32)
    b2_arr = np.array(all_b2, dtype=np.int32)
    b3_arr = np.array(all_b3, dtype=np.int32)
    is_reliable = np.array(all_reliable, dtype=bool)

    metadata = {
        "samples_per_min": SAMPLES_PER_MIN,
        "n_samples": int(X.shape[1]),
        "chip_tags": sorted(set(all_chip_tag)),
        "n_wells_labelled": len(per_well_stats),
        "per_well_stats": per_well_stats,
        "cls_labels": {"0": "baseline", "1": "drift", "2": "rising", "3": "plateau"},
        "label_construction": (
            "Per well: user_TTP defines drift→rising boundary. Baseline→drift = "
            "user_TTP - plate's (Cq_takeoff - Cq_start) spacing. "
            "Rising→plateau = user_TTP + plate's (Cq_plateau - Cq_takeoff) spacing. "
            "Per pixel: shift = argmax lag of cross-correlate(pixel_qc_trace, well_mean_qc_trace) "
            "within a ±3 min window around user_TTP. All three boundaries shifted by the pixel_shift."
        ),
    }

    out_npz = OUT_DIR / "cache.npz"
    out_meta = OUT_DIR / "metadata.json"

    np.savez_compressed(
        out_npz,
        X=X,
        labels=labels,
        chip_tag=chip_tag,
        well_id=well_id,
        log10_conc=log10_conc,
        pixel_shift=pixel_shift,
        user_ttp_min=user_ttp_min,
        is_reliable=is_reliable,
        b_baseline_end=b1_arr,
        b_drift_end=b2_arr,
        b_rising_end=b3_arr,
        time_min=time_min_ref,
    )
    out_meta.write_text(json.dumps(metadata, indent=2))

    print(f"\n[ok] wrote {out_npz}  ({out_npz.stat().st_size / 1e6:.1f} MB)")
    print(f"[ok] wrote {out_meta}")
    print(f"     N_pixels total: {X.shape[0]}")
    print(f"     N_reliable: {int(is_reliable.sum())}   N_unreliable: {int((~is_reliable).sum())}")
    print(f"     N_chips: {len(set(chip_tag))}   N_wells labelled: {len(per_well_stats)}")
    for cls, name in [(0, "baseline"), (1, "drift"), (2, "rising"), (3, "plateau")]:
        pct = (labels == cls).mean() * 100
        print(f"     class {cls} {name:>8}: {pct:.1f}% of all timesteps")


if __name__ == "__main__":
    main()
