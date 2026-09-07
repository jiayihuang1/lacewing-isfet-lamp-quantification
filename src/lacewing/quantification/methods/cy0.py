"""RQ2 · Cy₀ tangent-at-inflection computing extractor. [Cat A] Report §RQ2 Tab. 6.1.

Cy0 tangent-at-inflection landmark.

Reads the x-axis intercept of the tangent line drawn at the
first-derivative maximum.  The earliest of the classical
amplification-curve landmarks: it estimates the time at which an
idealised early-exponential phase would have started.

Not previously reported on ISFET-LAMP data; classical in qPCR
(Guescini 2008, Rutledge/Guescini 2013).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from lacewing.quantification.common import (
    SEARCH_END_MIN,
    smooth_centered,
)


CY0_BASELINE_WINDOW_MIN = 1.0
# Matches the plate-labelling pipeline's Cy0 (systematic_ttp CLASSICAL_SMOOTH_ORDER)
# and the deployed threshold-derivative TTP. The module-level SMOOTH_ORDER=10 in
# common.py is the MATLAB reference constant used by the plate-processing pipeline
# and is deliberately not reused here.
CY0_SMOOTH_ORDER = 100


def extract_cy0(
    time_min: np.ndarray,
    signal: np.ndarray,
    search_start_min: float | None = None,
    search_end_min: float = SEARCH_END_MIN,
    smooth_order: int = CY0_SMOOTH_ORDER,
    baseline_window_min: float = CY0_BASELINE_WINDOW_MIN,
) -> float:
    """Tangent-at-inflection intercept time (minutes from file start).

    Uses the median of the smoothed signal over the
    ``baseline_window_min`` minutes immediately **before** the peak
    search window starts.  Rationale:

      - the trace's first ~2 min is firmware-settle artefact, not a
        no-amplification reading;
      - the peak search window starts at PEAK_SEARCH_START_MIN
        because amplification *can* begin past that point;
      - therefore the cleanest baseline anchor is the 1 min ending
        at the search-window start: past the settling transient,
        before any plausible reaction.

    NaN if the first-derivative peak can't be identified or the
    local slope is too small for a stable intercept.
    """
    if search_start_min is None:
        search_start_min = float(time_min[0])

    idx_start = int(np.searchsorted(time_min, search_start_min))
    idx_end = int(np.searchsorted(time_min, search_end_min))

    sig_smooth = smooth_centered(signal, smooth_order)
    diff_smooth = smooth_centered(np.diff(sig_smooth), smooth_order)

    search_d1 = diff_smooth[idx_start:idx_end]
    if len(search_d1) < 10:
        return float("nan")
    peaks_d1, props_d1 = find_peaks(search_d1, width=1, prominence=0.00001)
    if len(peaks_d1) == 0:
        return float("nan")
    best_d1 = int(np.argmax(search_d1[peaks_d1] * props_d1["widths"]))
    peak_loc_d1 = int(peaks_d1[best_d1] + idx_start)
    t_peak = float(time_min[peak_loc_d1])
    f_peak = float(sig_smooth[peak_loc_d1])

    dt = float(time_min[1] - time_min[0])
    if dt <= 0:
        return float("nan")
    slope_per_min = float(diff_smooth[peak_loc_d1]) / dt
    if not np.isfinite(slope_per_min) or abs(slope_per_min) < 1e-6:
        return float("nan")

    # Baseline window = [search_start - baseline_window_min, search_start].
    # Anchored just before the peak search window begins so we are
    # past the firmware settling transient (typically ~2 min) but
    # before any plausible reaction onset.
    baseline_start_idx = int(np.searchsorted(
        time_min, search_start_min - baseline_window_min))
    baseline_start_idx = max(baseline_start_idx, 0)
    baseline_end_idx = max(idx_start, baseline_start_idx + 1)
    baseline = float(np.median(sig_smooth[baseline_start_idx:baseline_end_idx]))

    cy0 = t_peak - (f_peak - baseline) / slope_per_min
    return float(cy0)
