"""RQ2 · threshold-derivative TTP computing extractor (deployed baseline). [Cat A] Report §RQ2 Tab. 6.1.

Threshold-derivative TTP extraction (the existing Lacewing baseline).

Reads the time at which the smoothed first derivative crosses a fixed
fraction (0.4) of its peak height above the local baseline.  The
landmark located is the rising flank of the first-derivative peak,
back-shifted from the peak by the threshold heuristic.

Faithful port of the MATLAB Lacewing_Readout_DNA_Quant.m pipeline.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from lacewing.quantification.common import (
    SEARCH_END_MIN,
    THRESHOLD_FRAC,
    smooth_centered,
)


# Matches the deployed MATLAB Lacewing readout, the plate-labelling pipeline's
# Cy0 (CLASSICAL_SMOOTH_ORDER), and this project's SDM and Cy0 defaults. The
# module-level SMOOTH_ORDER=10 in common.py is the MATLAB reference constant
# for the plate-processing pipeline and is deliberately not reused here.
TTP_SMOOTH_ORDER = 100


def extract_ttp(
    time_min: np.ndarray,
    signal: np.ndarray,
    search_start_min: float | None = None,
    search_end_min: float = SEARCH_END_MIN,
    smooth_order: int = TTP_SMOOTH_ORDER,
    threshold_frac: float = THRESHOLD_FRAC,
) -> tuple[float, float]:
    """Returns ``(ttp_min, peak_min)``; either may be NaN.

    ``ttp_min`` is the threshold-crossing time (the deployed Lacewing
    readout).  ``peak_min`` is the time of the first-derivative max
    on the same signal, returned for downstream analyses that want
    both landmarks.
    """
    if search_start_min is None:
        search_start_min = float(time_min[0])

    idx_start = int(np.searchsorted(time_min, search_start_min))
    idx_end = int(np.searchsorted(time_min, search_end_min))

    # 1. Derivatives (smoothed and raw).
    diff_raw = np.diff(signal)
    sig_smooth = smooth_centered(signal, smooth_order)
    diff_smooth = smooth_centered(np.diff(sig_smooth), smooth_order)

    search_slice = diff_smooth[idx_start:idx_end]
    if len(search_slice) < 10:
        return float("nan"), float("nan")

    # 2. Peak picking (width * height scoring).
    peaks, props = find_peaks(search_slice, width=1, prominence=0.00001)
    if len(peaks) == 0:
        return float("nan"), float("nan")

    best = int(np.argmax(search_slice[peaks] * props["widths"]))
    peak_loc = int(peaks[best] + idx_start)
    peak_time_abs = float(time_min[peak_loc])

    # 3. Take-off (0.4-threshold) crossing on the raw derivative.
    region_smooth = diff_smooth[idx_start:peak_loc + 1]
    thr = float(region_smooth.min() + threshold_frac *
                (diff_smooth[peak_loc] - region_smooth.min()))

    below = np.where(diff_raw[idx_start:peak_loc + 1] < thr)[0]
    if len(below) == 0:
        return float("nan"), peak_time_abs

    ttp_absolute = float(time_min[idx_start + below[-1]])
    return ttp_absolute, peak_time_abs
