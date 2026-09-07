"""Data · systematic plate-anchored per-well TTP labelling procedure with per-pixel cross-correlation refinement. [Cat A] Report §Data (systematic labelling).

Systematic (rule-based) well-TTP labelling — plate-anchored variants.

All variants share the same core recipe:
    1. Take the plate TTP as an anchor (LC96 Cq × 0.5 min).
    2. Search the CHIP signal only in the window [plate_TTP, plate_TTP + Δ].
    3. Pick a specific signal feature in that window as the chip TTP.

Variants:

    V1  closest_peak_d1     — 1st-derivative local peak CLOSEST to plate_TTP
                              (this is what compute_systematic_ttp defaults to)
    V4  zero_cross_d2       — first zero-crossing of 2nd derivative (+→−)
                              — geometrically the inflection point
    V6  cy0_after_plate     — cy0 tangent-at-inflection intercept, restricted
                              to the post-plate search window
    V7  thr_deriv_after_plate — MATLAB threshold-derivative baseline
                              (0.4-threshold on smoothed d1), restricted to
                              the post-plate search window

V1 is preserved as the default `compute_systematic_ttp()` for backwards
compatibility; the umbrella entry-point is `compute_all_variants()` which
returns a dict of results keyed by variant name.

V6 and V7 reuse the same numerical formulas as
    lacewing.quantification.chip_pipeline.process_chip.extract_cy0
    lacewing.quantification.chip_pipeline.process_chip.extract_ttp_threshold
but with `search_start_min = plate_ttp_min` instead of the pipeline's default
99.5%-thermal-settling start.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, savgol_filter


DEFAULT_SG_WINDOW = 21
DEFAULT_SG_POLY = 3
DEFAULT_SEARCH_WINDOW_MIN = 15.0
DEFAULT_MIN_PROMINENCE_FRAC = 0.05

# For V6/V7 (classical extractors): the heavy centered-moving-average smoothing
# (span=100) shifts the d1 peak EARLIER than the plate TTP on most wells, so a
# hard gate at `plate_ttp_min` finds no peaks. We allow the peak to be up to
# CLASSICAL_PRE_PLATE_MARGIN_MIN before the anchor — this preserves the "plate
# is the reference" idea while accommodating the physical reality that the chip
# reads the amplification earlier than the LC96 plate does.
CLASSICAL_PRE_PLATE_MARGIN_MIN = 2.0
CLASSICAL_THRESHOLD_FRAC = 0.4
CLASSICAL_SMOOTH_ORDER = 100
CY0_BASELINE_WINDOW_MIN = 1.0

VARIANTS = ("closest_peak_d1", "zero_cross_d2", "cy0_after_plate", "thr_deriv_after_plate")


@dataclass
class SystematicTTPResult:
    ttp_min: float | None
    peak_deriv_value: float | None
    smoothed_signal: np.ndarray
    first_derivative: np.ndarray
    all_candidate_times_min: list[float]
    reason: str


# ----------------------------- shared helpers -----------------------------
def _prep(signal: np.ndarray, time_min: np.ndarray, sg_window: int, sg_poly: int):
    """Return (smoothed, d1, d2). Returns (None, None, None) if too short."""
    signal = np.asarray(signal, dtype=np.float64)
    time_min = np.asarray(time_min, dtype=np.float64)
    if signal.shape != time_min.shape:
        raise ValueError(f"signal shape {signal.shape} != time shape {time_min.shape}")
    if signal.size < sg_window:
        return None, None, None
    smoothed = savgol_filter(signal, window_length=sg_window,
                             polyorder=min(sg_poly, sg_window - 1))
    d1 = np.gradient(smoothed, time_min)
    d2 = np.gradient(d1, time_min)
    return smoothed, d1, d2


def _empty_result(smoothed, d1, reason: str) -> SystematicTTPResult:
    return SystematicTTPResult(
        ttp_min=None, peak_deriv_value=None,
        smoothed_signal=smoothed if smoothed is not None else np.zeros(0),
        first_derivative=d1 if d1 is not None else np.zeros(0),
        all_candidate_times_min=[],
        reason=reason,
    )


def _smooth_centered(signal: np.ndarray, span: int) -> np.ndarray:
    """Centered moving average, matching pipeline's smooth_centered()."""
    if span <= 1:
        return signal.copy()
    n = len(signal)
    half = (span - 1) // 2
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.mean(signal[lo:hi])
    return out


# ----------------------------- V1: closest peak of d1 -----------------------------
def _v1_closest_peak_d1(smoothed, d1, time_min, plate_ttp_min, search_window_min,
                        min_prominence_frac):
    prominence_floor = float(np.max(np.abs(d1))) * min_prominence_frac
    peak_idx, _ = find_peaks(d1, prominence=prominence_floor)
    if peak_idx.size == 0:
        return _empty_result(smoothed, d1, "no derivative peaks above prominence floor")

    peak_times = time_min[peak_idx]
    all_candidates = peak_times.tolist()
    valid_mask = (peak_times >= plate_ttp_min) & (peak_times <= plate_ttp_min + search_window_min)
    if not valid_mask.any():
        return _empty_result(smoothed, d1,
                             f"no d1 peaks in [plate={plate_ttp_min:.2f}, +{search_window_min:.1f}]")

    valid_peak_times = peak_times[valid_mask]
    valid_peak_idx = peak_idx[valid_mask]
    chosen_local = int(np.argmin(np.abs(valid_peak_times - plate_ttp_min)))
    chosen_idx = int(valid_peak_idx[chosen_local])
    return SystematicTTPResult(
        ttp_min=float(time_min[chosen_idx]),
        peak_deriv_value=float(d1[chosen_idx]),
        smoothed_signal=smoothed, first_derivative=d1,
        all_candidate_times_min=all_candidates,
        reason=f"V1 closest d1 peak at t={time_min[chosen_idx]:.2f} min "
               f"({len(valid_peak_times)} candidate(s) in window)",
    )


# ----------------------------- V4: first zero-crossing of d2 (+→−) -----------------------------
def _v4_zero_cross_d2(smoothed, d1, d2, time_min, plate_ttp_min, search_window_min):
    lo = plate_ttp_min
    hi = plate_ttp_min + search_window_min
    in_window = (time_min >= lo) & (time_min <= hi)
    if not in_window.any():
        return _empty_result(smoothed, d1,
                             f"no samples in [plate={lo:.2f}, +{search_window_min:.1f}]")

    idx_window = np.where(in_window)[0]
    d2w = d2[idx_window]
    sign = np.sign(d2w)
    diffs = np.diff(sign)
    cross_local = np.where(diffs < 0)[0]
    if cross_local.size == 0:
        return _empty_result(smoothed, d1, "no d2 zero-crossings (+→−) in window")

    first_local = int(cross_local[0])
    i0 = int(idx_window[first_local])
    i1 = int(idx_window[first_local + 1])
    y0, y1 = d2[i0], d2[i1]
    t0, t1 = time_min[i0], time_min[i1]
    if y1 == y0:
        ttp = float(t0)
    else:
        frac = y0 / (y0 - y1)
        ttp = float(t0 + frac * (t1 - t0))
    return SystematicTTPResult(
        ttp_min=ttp, peak_deriv_value=float(d1[i0]),
        smoothed_signal=smoothed, first_derivative=d1,
        all_candidate_times_min=[float(time_min[int(idx_window[c])]) for c in cross_local],
        reason=f"V4 first d2 zero-crossing (+→−) at t={ttp:.2f} min",
    )


# ----------------------------- V6: cy0 after plate -----------------------------
def _v6_cy0_after_plate(raw_signal, smoothed_sg, d1, time_min, plate_ttp_min,
                        search_window_min, pre_plate_margin=CLASSICAL_PRE_PLATE_MARGIN_MIN):
    """Cy0 tangent-at-inflection intercept, restricted to [plate−margin, plate+Δ].

    Matches lacewing.quantification.chip_pipeline.process_chip.extract_cy0
    numerically — uses the classical centered-moving-average smoothing
    (SMOOTH_ORDER=100) rather than the SG smoothing V1/V4 use, so the values
    match the deck's `ttp_cy0` when the anchor is the same.
    """
    n = len(raw_signal)
    lo = plate_ttp_min - pre_plate_margin
    hi = plate_ttp_min + search_window_min
    idx_start = max(0, int(np.searchsorted(time_min, lo)))
    idx_end   = int(np.searchsorted(time_min, hi))
    if idx_end - idx_start < 10:
        return _empty_result(smoothed_sg, d1,
                             f"V6: too few samples in [plate−{pre_plate_margin:.1f}, +{search_window_min:.1f}]")

    sig_smooth = _smooth_centered(np.asarray(raw_signal, dtype=np.float64),
                                   CLASSICAL_SMOOTH_ORDER)
    diff_smooth = _smooth_centered(np.diff(sig_smooth), CLASSICAL_SMOOTH_ORDER)

    search = diff_smooth[idx_start:min(idx_end, len(diff_smooth))]
    if len(search) < 10:
        return _empty_result(smoothed_sg, d1, "V6: search slice too small")
    peaks, props = find_peaks(search, width=1, prominence=1e-5)
    if peaks.size == 0:
        return _empty_result(smoothed_sg, d1, "V6: no d1 peaks in window")

    best = int(np.argmax(search[peaks] * props["widths"]))
    peak_loc = int(peaks[best] + idx_start)

    # Baseline: median of smoothed signal from t=0 up to the search anchor.
    # (Was: fixed 1-min pre-anchor window with min 3 samples — that failed on
    # very fast wells like Chip 1's 1e6 where the search anchor sits at ~0.14 min
    # and there are only 2 samples before it. Using all pre-anchor samples with
    # a minimum of 2 recovers those wells without changing behaviour elsewhere.)
    baseline_end_idx = idx_start
    baseline_start_idx = max(0, int(np.searchsorted(time_min, lo - CY0_BASELINE_WINDOW_MIN)))
    if baseline_end_idx - baseline_start_idx < 2:
        baseline_start_idx = 0
    if baseline_end_idx - baseline_start_idx < 2:
        return _empty_result(smoothed_sg, d1, "V6: baseline window too small (no pre-anchor samples)")
    y_base = float(np.median(sig_smooth[baseline_start_idx:baseline_end_idx]))

    dt = float(time_min[1] - time_min[0])
    if dt <= 0:
        return _empty_result(smoothed_sg, d1, "V6: non-positive dt")
    m = float(diff_smooth[peak_loc]) / dt
    if not np.isfinite(m) or abs(m) < 1e-6:
        return _empty_result(smoothed_sg, d1, "V6: degenerate tangent slope")
    t0 = float(time_min[peak_loc])
    y0 = float(sig_smooth[peak_loc])
    ttp = t0 + (y_base - y0) / m
    return SystematicTTPResult(
        ttp_min=float(ttp),
        peak_deriv_value=float(diff_smooth[peak_loc]),
        smoothed_signal=smoothed_sg, first_derivative=d1,
        all_candidate_times_min=[float(time_min[int(p + idx_start)]) for p in peaks],
        reason=f"V6 cy0 tangent at inflection t={t0:.2f} intercepts baseline at t={ttp:.2f} min "
               f"(search [{lo:.2f}, {hi:.2f}])",
    )


# ----------------------------- V7: thr_deriv after plate -----------------------------
def _v7_thr_deriv_after_plate(raw_signal, smoothed_sg, d1, time_min, plate_ttp_min,
                              search_window_min, threshold_frac,
                              pre_plate_margin=CLASSICAL_PRE_PLATE_MARGIN_MIN):
    """MATLAB 0.4-threshold on smoothed d1, restricted to [plate−margin, plate+Δ]."""
    lo = plate_ttp_min - pre_plate_margin
    hi = plate_ttp_min + search_window_min
    idx_start = max(0, int(np.searchsorted(time_min, lo)))
    idx_end   = int(np.searchsorted(time_min, hi))
    if idx_end - idx_start < 10:
        return _empty_result(smoothed_sg, d1,
                             f"V7: too few samples in [plate−{pre_plate_margin:.1f}, +{search_window_min:.1f}]")

    signal = np.asarray(raw_signal, dtype=np.float64)
    diff_raw = np.diff(signal)
    sig_smooth = _smooth_centered(signal, CLASSICAL_SMOOTH_ORDER)
    diff_smooth = _smooth_centered(np.diff(sig_smooth), CLASSICAL_SMOOTH_ORDER)

    search_slice = diff_smooth[idx_start:min(idx_end, len(diff_smooth))]
    if len(search_slice) < 10:
        return _empty_result(smoothed_sg, d1, "V7: search slice too small")

    peaks, props = find_peaks(search_slice, width=1, prominence=1e-5)
    if peaks.size == 0:
        return _empty_result(smoothed_sg, d1, "V7: no d1 peaks in window")

    best = int(np.argmax(search_slice[peaks] * props["widths"]))
    peak_loc = int(peaks[best] + idx_start)
    peak_min = float(time_min[peak_loc])

    # 0.4-threshold on the smoothed d1, applied to RAW d1 crossing (MATLAB port)
    region_smooth = diff_smooth[idx_start:peak_loc + 1]
    if len(region_smooth) < 2:
        return _empty_result(smoothed_sg, d1, "V7: pre-peak region too small")
    thr_val = float(region_smooth.min()
                    + threshold_frac * (diff_smooth[peak_loc] - region_smooth.min()))

    below = np.where(diff_raw[idx_start:peak_loc + 1] < thr_val)[0]
    if below.size == 0:
        return _empty_result(smoothed_sg, d1, "V7: no crossing below threshold")
    ttp = float(time_min[idx_start + below[-1]])
    return SystematicTTPResult(
        ttp_min=ttp,
        peak_deriv_value=float(diff_smooth[peak_loc]),
        smoothed_signal=smoothed_sg, first_derivative=d1,
        all_candidate_times_min=[peak_min],
        reason=f"V7 thr_deriv crossing (frac={threshold_frac}) at t={ttp:.2f} min "
               f"(d1 peak at t={peak_min:.2f}, search [{lo:.2f}, {hi:.2f}])",
    )


# ----------------------------- public API -----------------------------
def compute_systematic_ttp(
    signal: np.ndarray,
    time_min: np.ndarray,
    plate_ttp_min: float | None,
    *,
    sg_window: int = DEFAULT_SG_WINDOW,
    sg_poly: int = DEFAULT_SG_POLY,
    search_window_min: float = DEFAULT_SEARCH_WINDOW_MIN,
    min_prominence_frac: float = DEFAULT_MIN_PROMINENCE_FRAC,
) -> SystematicTTPResult:
    """V1 (closest 1st-derivative peak ≥ plate TTP). Backwards-compatible default."""
    smoothed, d1, _ = _prep(signal, time_min, sg_window, sg_poly)
    if smoothed is None:
        return _empty_result(None, None,
                             f"signal too short (< sg_window {sg_window})")
    if plate_ttp_min is None or not np.isfinite(plate_ttp_min):
        return _empty_result(smoothed, d1, "no plate TTP available")
    return _v1_closest_peak_d1(smoothed, d1, np.asarray(time_min, dtype=np.float64),
                               float(plate_ttp_min), search_window_min,
                               min_prominence_frac)


def compute_all_variants(
    signal: np.ndarray,
    time_min: np.ndarray,
    plate_ttp_min: float | None,
    *,
    sg_window: int = DEFAULT_SG_WINDOW,
    sg_poly: int = DEFAULT_SG_POLY,
    search_window_min: float = DEFAULT_SEARCH_WINDOW_MIN,
    min_prominence_frac: float = DEFAULT_MIN_PROMINENCE_FRAC,
    threshold_frac: float = CLASSICAL_THRESHOLD_FRAC,
) -> dict[str, SystematicTTPResult]:
    """Run all 4 variants at once. Returns {variant_name: SystematicTTPResult}."""
    smoothed, d1, d2 = _prep(signal, time_min, sg_window, sg_poly)
    out: dict[str, SystematicTTPResult] = {}
    if smoothed is None:
        for name in VARIANTS:
            out[name] = _empty_result(None, None,
                                       f"signal too short (< sg_window {sg_window})")
        return out
    if plate_ttp_min is None or not np.isfinite(plate_ttp_min):
        for name in VARIANTS:
            out[name] = _empty_result(smoothed, d1, "no plate TTP available")
        return out

    tm = np.asarray(time_min, dtype=np.float64)
    raw = np.asarray(signal, dtype=np.float64)
    p = float(plate_ttp_min)
    out["closest_peak_d1"]        = _v1_closest_peak_d1(smoothed, d1, tm, p,
                                                        search_window_min, min_prominence_frac)
    out["zero_cross_d2"]          = _v4_zero_cross_d2(smoothed, d1, d2, tm, p,
                                                       search_window_min)
    out["cy0_after_plate"]        = _v6_cy0_after_plate(raw, smoothed, d1, tm, p,
                                                         search_window_min)
    out["thr_deriv_after_plate"]  = _v7_thr_deriv_after_plate(raw, smoothed, d1, tm, p,
                                                               search_window_min, threshold_frac)
    return out


def second_derivative(signal: np.ndarray, time_min: np.ndarray,
                      *, sg_window: int = DEFAULT_SG_WINDOW,
                      sg_poly: int = DEFAULT_SG_POLY) -> tuple[np.ndarray, np.ndarray]:
    """Convenience helper: return (smoothed, d2) for viewer plotting."""
    smoothed, _, d2 = _prep(signal, time_min, sg_window, sg_poly)
    return smoothed, d2


def _demo() -> None:
    t = np.linspace(0, 30, 601)
    signal = 1.0 / (1 + np.exp(-(t - 10) * 1.5)) + 0.02 * np.random.RandomState(0).randn(len(t))
    results = compute_all_variants(signal, t, plate_ttp_min=9.5)
    for name, r in results.items():
        ttp = "None" if r.ttp_min is None else f"{r.ttp_min:.2f}"
        print(f"  {name:24s} → {ttp:>6s}   {r.reason}")


if __name__ == "__main__":
    _demo()
