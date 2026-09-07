"""RQ2 · SDM second-derivative-maximum computing extractor. [Cat A] Report §RQ2 Tab. 6.1.

Second-Derivative Maximum (SDM) landmark.

Reads the time at which the smoothed second derivative peaks on the
rising flank of the amplification curve.  Geometrically earlier than
the first-derivative maximum: SDM is at the start of the exponential
phase, where the curvature of the sigmoid is steepest.

Not previously reported on ISFET-LAMP data; classical in qPCR
(Richards-fit literature, Spiess 2008/2016).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from lacewing.quantification.common import (
    SEARCH_END_MIN,
    smooth_centered,
)


# Matches the plate-labelling pipeline's Cy0 (systematic_ttp CLASSICAL_SMOOTH_ORDER)
# and the deployed threshold-derivative TTP. The module-level SMOOTH_ORDER=10 in
# common.py is the MATLAB reference constant used by the plate-processing pipeline
# and is deliberately not reused here.
SDM_SMOOTH_ORDER = 100


def extract_sdm(
    time_min: np.ndarray,
    signal: np.ndarray,
    search_start_min: float | None = None,
    search_end_min: float = SEARCH_END_MIN,
    smooth_order: int = SDM_SMOOTH_ORDER,
) -> float:
    """Time (minutes from file start) of the smoothed second
    derivative's peak on the rising flank.  NaN if no rising-flank
    second-derivative peak is found.
    """
    if search_start_min is None:
        search_start_min = float(time_min[0])

    idx_start = int(np.searchsorted(time_min, search_start_min))
    idx_end = int(np.searchsorted(time_min, search_end_min))

    sig_smooth = smooth_centered(signal, smooth_order)
    diff_smooth = smooth_centered(np.diff(sig_smooth), smooth_order)

    # First-derivative peak: SDM lies before this point on the curve.
    search_d1 = diff_smooth[idx_start:idx_end]
    if len(search_d1) < 10:
        return float("nan")
    peaks_d1, props_d1 = find_peaks(search_d1, width=1, prominence=0.00001)
    if len(peaks_d1) == 0:
        return float("nan")
    best_d1 = int(np.argmax(search_d1[peaks_d1] * props_d1["widths"]))
    peak_loc_d1 = int(peaks_d1[best_d1] + idx_start)

    # Second derivative on the rising flank only.
    d2_smooth = smooth_centered(np.diff(diff_smooth), smooth_order)
    search_d2 = d2_smooth[idx_start:peak_loc_d1 + 1]
    if len(search_d2) < 10:
        return float("nan")

    peaks_d2, props_d2 = find_peaks(search_d2, width=1, prominence=1e-7)
    if len(peaks_d2) == 0:
        return float("nan")
    best_d2 = int(np.argmax(search_d2[peaks_d2] * props_d2["widths"]))
    sdm_idx = int(peaks_d2[best_d2] + idx_start)
    return float(time_min[sdm_idx])
