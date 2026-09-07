"""Per-chip artefact trim for the Data/Multi/ chips.

Background
----------
The Matthew_Multi titan branch returns Experiment objects whose
post-`idx_settled` traces still contain a short sharp downward dip at
the very start (firmware-settle leftover not removed by titan's
99.5%-thermal criterion).  Visible in `view_pixels.py` / main_DNA's
``all_wells_grid.html`` outputs: traces dip from ~510 -> ~420 in the
first ~0.5 min, then recover.

For classification, this artefact varies in length well-to-well and
chip-to-chip, so we need a uniform per-chip cut to align everything on
a common t=0.

Algorithm (chosen with user 2026-06-10)
---------------------------------------
For each well:

  1.  Smooth the active-pixel mean trace (centred MA, default span 10).
  2.  Compute the first derivative (np.diff).
  3.  Walk forward over [0, search_max_min] from idx_settled.  Find the
      FIRST index where the derivative is non-negative AND stays
      non-negative for `grace_samples` consecutive samples
      (= slope direction has flipped from negative to non-negative and
      is sustained, not a single-sample noise spike).
  4.  Guard against false positives: only honour the crossing if the
      smoothed signal actually dropped by more than `min_descent_v`
      before the crossing.  Otherwise return 0 (= no artefact present).

Chip cut = max(per-well cuts).  Wells without a detected artefact
contribute 0 to the max, so a single chip with a real artefact still
gets trimmed cleanly across all its wells.

Trimming itself is a thin shift of each well's `idx_start` /
`idx_settled` indices forward by the cut offset.  The underlying 3D
voltage arrays are untouched; `time_min`, `well_2d_bs_active_mean`,
etc. are all properties of these indices, so they recompute correctly.

NOTE: this module imports from the Matthew_Multi titan clone.  Import
``lacewing.quantification.multi_data_shared._titan_setup`` (or use a function from
``lacewing.quantification.multi_data_shared``) before importing this module if you are not
already inside the multi_data package.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Path setup so titan resolves to titan-signal-processing-multi.
# Importing the module installs the sys.path insertion as a side effect.
from . import _titan_setup  # noqa: F401


# A simple centred moving-average matching titan's smooth_centered.  We
# don't import it from titan because that module isn't reliably on the
# import path in all callers; the helper is 6 lines.
def _smooth_centered(signal: np.ndarray, span: int) -> np.ndarray:
    if span <= 1:
        return signal.astype(float, copy=True)
    n = len(signal)
    half = (span - 1) // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.mean(signal[lo:hi])
    return out


def _well_mean_trace(well) -> np.ndarray:
    """Active-pixel-averaged trace for one well, in raw-direction form.

    Uses `well_2d_nl_active_mean` (linearised, NOT baseline-subtracted)
    so the actual signal direction is preserved: the early-time
    artefact appears as a *downward* dip (not flipped by baseline
    subtraction at idx_settled).  This is what the user's screenshot
    shows and the negative-to-positive slope-flip detector relies on.
    """
    return np.asarray(well.well_2d_nl_active_mean, dtype=float)


@dataclass(frozen=True)
class TrimResult:
    """Outcome of running ``trim_chip_artefact`` on one Experiment."""
    chip_cut_idx: int             # samples from idx_settled to skip
    chip_cut_min: float           # same as above in minutes from idx_settled
    per_well_cut_idx: list[int]   # one value per well; 0 = no artefact
    per_well_cut_min: list[float]
    per_well_had_artefact: list[bool]


def _find_well_cut_idx(
    well,
    *,
    smooth_span: int = 10,
    search_max_min: float = 5.0,
    grace_samples: int = 3,
    min_descent_frac: float = 0.10,
    ascent_settle_frac: float = 0.30,
    ascent_post_window_min: float = 1.0,
) -> tuple[int, bool]:
    """Return (cut_idx_relative_to_idx_settled, had_artefact).

    Two-stage detection: the artefact has the shape
    drop -> bottom -> climb -> jump -> settle.  We want the cut AFTER
    the climb-and-jump, not at the bottom of the drop.

    Stage 1 (bottom of the drop): first index where the smoothed first
    derivative crosses negative -> non-negative and stays non-negative
    for ``grace_samples`` consecutive samples.  Guarded by
    ``min_descent_frac``: the pre-crossing descent must exceed this
    fraction of the search-window peak-to-peak range, otherwise we
    treat the well as having no artefact and return 0.

    Stage 2 (top of the upward jump): starting from the stage-1 index,
    walk forward and find the first index where the slope crosses
    positive -> non-positive and stays non-positive for
    ``grace_samples`` consecutive samples.  If no such crossing is
    found within ``search_max_min``, fall back to argmin(|slope|)
    over (stage-1, search_max_min] — the next turning point we can
    locate.
    """
    sig = _well_mean_trace(well)
    if sig.size < grace_samples + 2:
        return 0, False

    smoothed = _smooth_centered(sig, smooth_span)
    diff = np.diff(smoothed)        # length n-1, aligned to smoothed[1:]

    time_min = np.asarray(well.time_min, dtype=float)
    n_search = int(np.searchsorted(time_min, search_max_min))
    n_search = min(n_search, len(diff))
    if n_search < grace_samples + 1:
        return 0, False

    window_range = float(smoothed[:n_search].max()
                          - smoothed[:n_search].min())
    min_descent = min_descent_frac * max(window_range, 1e-9)

    # ---- Stage 1: bottom of the drop ----
    j1 = None
    saw_descent = False
    for j in range(n_search - grace_samples):
        if diff[j] < 0:
            saw_descent = True
            continue
        if not saw_descent:
            continue
        if all(diff[j + k] >= 0 for k in range(grace_samples)):
            pre = smoothed[: j + 1]
            descent = float(pre.max() - smoothed[j])
            if descent >= min_descent:
                j1 = j
            break

    if j1 is None:
        return 0, False

    # ---- Stage 2: end of the steep upward jump ----
    # The artefact's upward jump finishes within ~1 min of the bottom.
    # We find the peak slope in that window, then take the first index
    # past the peak where the slope drops below ``ascent_settle_frac``
    # of that peak — i.e. the steep climb has flattened out.
    dt_min = float(time_min[1] - time_min[0]) if len(time_min) > 1 else 1.0
    n_post = max(1, int(round(ascent_post_window_min / dt_min)))
    post_lo = j1 + 1
    post_hi = min(j1 + 1 + n_post, n_search)
    if post_hi - post_lo < grace_samples + 1:
        # Too little room left — fall back to using j1 itself.
        return j1 + 1, True

    post_slice = diff[post_lo:post_hi]
    s_max = float(post_slice.max())
    if s_max <= 0:
        # No actual ascent within the window — j1 is already the cut.
        return j1 + 1, True

    # Peak-slope location (absolute index in `diff`).
    j_peak = post_lo + int(np.argmax(post_slice))
    settle_thr = ascent_settle_frac * s_max

    j2 = None
    for j in range(j_peak + 1, post_hi - grace_samples + 1):
        # First index past the peak where the slope falls below
        # `settle_thr` and STAYS below for grace_samples consecutive
        # samples (guards against single-sample dips during the climb).
        if diff[j] >= settle_thr:
            continue
        if all(diff[j + k] < settle_thr
               for k in range(grace_samples)
               if (j + k) < len(diff)):
            j2 = j
            break

    if j2 is None:
        # The slope never fell below 10% of its peak within the window.
        # Use the end of the post-window as the cut — at worst this
        # over-trims by ASCENT_POST_WINDOW_MIN; at best it catches a
        # very long settle.
        j2 = post_hi - 1

    # cut_idx into time_min: diff[j] is the slope at time_min[j+1].
    return j2 + 1, True


def trim_chip_artefact(
    experiment,
    *,
    smooth_span: int = 10,
    search_max_min: float = 5.0,
    grace_samples: int = 3,
    min_descent_frac: float = 0.10,
    ascent_settle_frac: float = 0.30,
    ascent_post_window_min: float = 1.0,
    apply: bool = True,
) -> TrimResult:
    """Detect and (optionally) trim the chip's early artefact in place.

    Parameters
    ----------
    experiment : titan.Experiment
        Experiment object returned by ``titan_load_and_preprocessing``
        on the Matthew_Multi branch.
    smooth_span : int
        Centred-MA span used to smooth each well's trace before
        differentiating (default 10, matches titan/MATLAB).
    search_max_min : float
        Per-well search upper bound (default 5 min).  Past this we
        assume the well is into real amplification or noise, not
        artefact.
    grace_samples : int
        Number of consecutive non-negative diff samples required to
        confirm the slope flip (default 3, ~8s at the typical dt).
    min_descent_frac : float
        Magnitude guard: the descent before the crossing must exceed
        this fraction of the smoothed signal's [0, search_max_min]
        peak-to-peak range, otherwise the well is treated as having no
        artefact.
    apply : bool
        If True (default), push each well's ``idx_settled`` and
        ``idx_start`` forward by ``chip_cut_idx``.  If False, just
        return the detection result without mutating the Experiment
        (useful for inspection).

    Returns
    -------
    TrimResult
    """
    per_well_cut_idx: list[int] = []
    per_well_cut_min: list[float] = []
    per_well_had_artefact: list[bool] = []

    for well in experiment.wells_list:
        cut_idx, had = _find_well_cut_idx(
            well,
            smooth_span=smooth_span,
            search_max_min=search_max_min,
            grace_samples=grace_samples,
            min_descent_frac=min_descent_frac,
            ascent_settle_frac=ascent_settle_frac,
            ascent_post_window_min=ascent_post_window_min,
        )
        per_well_cut_idx.append(int(cut_idx))
        time_min = np.asarray(well.time_min, dtype=float)
        per_well_cut_min.append(float(time_min[cut_idx]) if cut_idx < len(time_min) else 0.0)
        per_well_had_artefact.append(bool(had))

    chip_cut_idx = int(max(per_well_cut_idx)) if per_well_cut_idx else 0
    # Translate chip_cut_idx (post-idx_settled offset) into minutes via
    # any well's time_min (they share the same time axis).
    if experiment.wells_list and chip_cut_idx < len(experiment.wells_list[0].time_min):
        chip_cut_min = float(experiment.wells_list[0].time_min[chip_cut_idx])
    else:
        chip_cut_min = 0.0

    if apply and chip_cut_idx > 0:
        for well in experiment.wells_list:
            # `idx_settled` and `idx_start` are absolute indices into
            # `time_npr` (the raw chip time vector).  Push both forward
            # by chip_cut_idx so that time_min's new zero lines up with
            # the post-artefact recovery point.
            well.idx_settled = int(well.idx_settled) + chip_cut_idx
            well.idx_start = max(int(well.idx_start), well.idx_settled)

    return TrimResult(
        chip_cut_idx=chip_cut_idx,
        chip_cut_min=chip_cut_min,
        per_well_cut_idx=per_well_cut_idx,
        per_well_cut_min=per_well_cut_min,
        per_well_had_artefact=per_well_had_artefact,
    )
