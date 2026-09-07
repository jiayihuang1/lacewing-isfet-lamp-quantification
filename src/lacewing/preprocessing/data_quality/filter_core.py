"""Preprocessing · MAD-ABCD Layer A/B/C/D filter core. [Cat A] Report §Data.

Pure filter math, importable from both the diagnostic
viewer (``build_filter_pipeline.py``) and the experiment cache builder.

The four layers of the chip-relative preprocessing filter:

- Layer A — amplitude floor.  Drop pixels whose dynamic range is below
  the NTC well's ``pct_amp``-th percentile.
- Layer B — shape floor.  Drop pixels whose net slope or trace min is
  below the NTC well's ``pct_shape``-th percentile.
- Layer C — kNN bad-neighbour count.  Drop pixels whose k nearest
  spatial neighbours (within the same well) are mostly A/B-bad.
- Layer D — individual disagreement with the local neighbourhood;
  L2 distance from the pixel trace to the median trace of its k
  spatial neighbours; drop above per-well percentile.

All thresholds are computed *per chip* from the chip's own NTC well or
from per-well percentiles - there are no hard-coded V values.

These functions are pure (no I/O, no globals).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------
# Robust statistics
# --------------------------------------------------------------------

def median_mad(x: np.ndarray) -> tuple[float, float]:
    """Median and (scaled) MAD of x; returns (median, mad).

    The scaling factor 1.4826 makes the MAD a consistent estimator of
    the standard deviation under a Gaussian distribution.
    """
    if len(x) == 0:
        return 0.0, 0.0
    med = float(np.median(x))
    mad = float(1.4826 * np.median(np.abs(x - med)))
    return med, mad


# --------------------------------------------------------------------
# Layer A and B
# --------------------------------------------------------------------

def compute_chip_thresholds(
    x_ntc: np.ndarray,
    pct_amp: float = 5.0,
    pct_shape: float = 5.0,
) -> dict[str, float]:
    """Compute NTC-referenced thresholds for one chip.

    Takes the (post-titan, baseline-anchored) traces of the **NTC
    pixels** of that chip only.  Caller is responsible for filtering
    the NTC pixel set first if e.g. Layer D was used to clean it.

    Returns thresholds for Layer A (dyn_range), Layer B-a (net_slope),
    and Layer B-b (trace_min), along with summary statistics of the
    NTC distribution for diagnostics.
    """
    if x_ntc.size == 0:
        raise ValueError("no NTC pixels supplied")

    dyn_range = x_ntc.max(axis=1) - x_ntc.min(axis=1)
    net_slope = x_ntc[:, -1] - x_ntc[:, 0]
    trace_min = x_ntc.min(axis=1)

    dr_med, dr_mad = median_mad(dyn_range)
    ns_med, ns_mad = median_mad(net_slope)
    tm_med, tm_mad = median_mad(trace_min)

    return {
        "n_ntc_pixels":             int(x_ntc.shape[0]),
        "pct_amp":                  float(pct_amp),
        "pct_shape":                float(pct_shape),
        "dyn_range_ntc_median":     dr_med,
        "dyn_range_ntc_mad":        dr_mad,
        "dyn_range_threshold":      float(np.percentile(dyn_range, pct_amp)),
        "net_slope_ntc_median":     ns_med,
        "net_slope_ntc_mad":        ns_mad,
        "net_slope_threshold":      float(np.percentile(net_slope, pct_shape)),
        "trace_min_ntc_median":     tm_med,
        "trace_min_ntc_mad":        tm_mad,
        "trace_min_threshold":      float(np.percentile(trace_min, pct_shape)),
    }


def apply_layers_ab(
    x_raw: np.ndarray,
    thresh: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(layerA_dropped, layerB_dropped)`` boolean masks for the
    full pixel set ``x_raw``.  Layer A drops are checked first; Layer
    B drops are then computed on the remaining pixels.  The two masks
    are disjoint by construction.
    """
    dyn_range = x_raw.max(axis=1) - x_raw.min(axis=1)
    net_slope = x_raw[:, -1] - x_raw[:, 0]
    trace_min = x_raw.min(axis=1)

    layerA = dyn_range < thresh["dyn_range_threshold"]
    layerB_raw = (net_slope < thresh["net_slope_threshold"]) | \
                 (trace_min < thresh["trace_min_threshold"])

    # Make B disjoint from A: B only "claims" pixels A hasn't already.
    layerB = layerB_raw & (~layerA)
    return layerA, layerB


# --------------------------------------------------------------------
# Spatial helpers shared by C and D
# --------------------------------------------------------------------

def _pairwise_squared_distance(pts: np.ndarray) -> np.ndarray:
    """Symmetric (n, n) pairwise squared distance for an (n, 2) array."""
    norms = np.sum(pts**2, axis=1)
    return norms[:, None] + norms[None, :] - 2 * (pts @ pts.T)


def _knn_indices(d2: np.ndarray, k: int) -> np.ndarray:
    """For an (n, n) squared-distance matrix, return (n, k) array of
    the k nearest neighbour indices for each row, *excluding self*."""
    n = d2.shape[0]
    n_k = min(k + 1, n)
    nn_idx = np.argpartition(d2, n_k - 1, axis=1)[:, :n_k]
    rows_arange = np.arange(n)[:, None]
    is_self = nn_idx == rows_arange
    neighbours_only = np.where(is_self, -1, nn_idx)
    neighbours_sorted = np.sort(neighbours_only, axis=1)
    return neighbours_sorted[:, -k:]


# --------------------------------------------------------------------
# Layer C
# --------------------------------------------------------------------

def apply_layer_c(
    rows: np.ndarray,
    cols: np.ndarray,
    well_id: np.ndarray,
    layerA: np.ndarray,
    layerB: np.ndarray,
    k_neighbours: int = 8,
    min_bad_frac: float = 0.5,
    eligible_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop pixels whose k spatial neighbours are mostly A/B-bad.

    ``eligible_mask`` (optional, len = N) lets the caller restrict
    which pixels are *candidates* for being dropped by this layer -
    e.g. pass ``well_id != NTC_WELL`` to skip the NTC well.  Pixels
    outside ``eligible_mask`` are never dropped here; the function
    still computes their bad-neighbour count (in ``bad_count``) for
    information.

    Returns ``(drop_mask, bad_count)``.
    """
    n_pix = len(rows)
    drop = np.zeros(n_pix, dtype=bool)
    bad_count = np.full(n_pix, -1, dtype=np.int32)
    layer_ab = layerA | layerB

    if eligible_mask is None:
        eligible_mask = np.ones(n_pix, dtype=bool)

    for w in sorted(set(int(v) for v in well_id)):
        w_idx = np.flatnonzero(well_id == w)
        if len(w_idx) < (k_neighbours + 1):
            continue
        pts = np.column_stack([rows[w_idx], cols[w_idx]]).astype(np.float32)
        local_ab = layer_ab[w_idx]

        d2 = _pairwise_squared_distance(pts)
        nn_k = _knn_indices(d2, k_neighbours)

        bad_neighbours = local_ab[nn_k].sum(axis=1)
        bad_count[w_idx] = bad_neighbours

        threshold = int(np.ceil(min_bad_frac * k_neighbours))
        local_eligible = eligible_mask[w_idx]
        local_drop = (bad_neighbours >= threshold) & ~local_ab & local_eligible
        drop[w_idx[local_drop]] = True

    return drop, bad_count


# --------------------------------------------------------------------
# Layer D
# --------------------------------------------------------------------

def apply_layer_d(
    x_raw: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    well_id: np.ndarray,
    excluded_mask: np.ndarray,
    metric: str = "median",
    k_neighbours: int = 8,
    pct_disagree: float = 95.0,
    eligible_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop pixels whose trace disagrees with their spatial neighbours.

    ``excluded_mask`` (len N): pixels already dropped by prior layers
    (typically A | B | C).  They're excluded from D scoring entirely.

    ``eligible_mask`` (optional, len N): if supplied, only pixels with
    True are candidates for being *dropped* by D.  Use this to skip
    the NTC well, e.g. ``well_id != NTC_WELL``.  Pixels outside the
    eligible set still get D-scored (so callers using D-on-NTC for
    threshold computation can read those scores), but their drop flag
    stays False.

    ``metric`` is one of {"mean", "median", "kth"}.
    """
    if metric not in ("mean", "median", "kth"):
        raise ValueError(f"unknown metric '{metric}'")

    n_pix = len(x_raw)
    drop = np.zeros(n_pix, dtype=bool)
    scores = np.full(n_pix, np.nan, dtype=np.float64)

    survive = ~excluded_mask
    if eligible_mask is None:
        eligible_mask = np.ones(n_pix, dtype=bool)

    for w in sorted(set(int(v) for v in well_id)):
        w_idx = np.flatnonzero((well_id == w) & survive)
        n_w = len(w_idx)
        if n_w < (k_neighbours + 1):
            continue
        pts = np.column_stack([rows[w_idx], cols[w_idx]]).astype(np.float32)
        traces = x_raw[w_idx]

        d2 = _pairwise_squared_distance(pts)
        nn_k = _knn_indices(d2, k_neighbours)

        if metric == "mean":
            ref = traces[nn_k].mean(axis=1)
            disagree = np.sqrt(np.sum((traces - ref) ** 2, axis=1))
        elif metric == "median":
            ref = np.median(traces[nn_k], axis=1)
            disagree = np.sqrt(np.sum((traces - ref) ** 2, axis=1))
        else:  # kth
            kth_idx = nn_k[:, -1]
            kth_trace = traces[kth_idx]
            disagree = np.sqrt(np.sum((traces - kth_trace) ** 2, axis=1))

        cut = float(np.percentile(disagree, pct_disagree))
        scores[w_idx] = disagree

        local_eligible = eligible_mask[w_idx]
        local_drop = (disagree > cut) & local_eligible
        drop[w_idx[local_drop]] = True

    return drop, scores


# --------------------------------------------------------------------
# Top-level: apply the requested layers for one chip in one call
# --------------------------------------------------------------------

def run_filter_pipeline_per_chip(
    x_raw: np.ndarray,
    well_id: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    ntc_well: int,
    layers: str,                     # "A" | "AB" | "ABC" | "ABCD"
    clean_ntc_first: bool,           # if True, apply D to NTC first to clean ref
    pct_amp: float = 5.0,
    pct_shape: float = 5.0,
    c_k: int = 8,
    c_min_bad_frac: float = 0.5,
    d_k: int = 8,
    d_pct: float = 95.0,
    d_metric: str = "median",
) -> tuple[np.ndarray, dict]:
    """Run the requested subset of layers on one chip's pixels.

    Returns ``(keep_mask, info)`` where ``keep_mask[i] == True``
    means pixel i survives the requested filter pipeline.

    Filtering policy (matching the experiment design):
      - Layers A, B, C *never* drop NTC pixels (eligible_mask
        excludes the NTC well).
      - Layer D *does* drop NTC pixels if ``layers`` includes "D".
      - If ``clean_ntc_first`` is True, Layer D is run on the NTC
        well *before* threshold computation; the NTC pixels D drops
        are excluded from the threshold's percentile distribution
        AND from the training cache (if layers includes "D" at all,
        which it always does when clean_ntc_first is True).

    ``info`` contains per-layer drop masks and thresholds for
    inspection / logging.
    """
    if layers not in ("A", "AB", "ABC", "ABCD"):
        raise ValueError(f"layers must be 'A', 'AB', 'ABC', or 'ABCD'; got {layers}")
    # ``clean_ntc_first`` may be True even when ``layers`` doesn't include D
    # on positive wells: this uses D *only* on NTC for threshold computation
    # (the resulting NTC drops are still removed from the training cache,
    # since they're genuinely broken pixels).

    ntc_well_mask = (well_id == ntc_well)
    positive_well_mask = ~ntc_well_mask
    n_pix = len(x_raw)

    info: dict = {"layers": layers, "clean_ntc_first": clean_ntc_first}

    # --- (Optional) Clean NTC pixels with Layer D before computing thresholds.
    ntc_d_drop = np.zeros(n_pix, dtype=bool)
    if clean_ntc_first:
        # Run D restricted to the NTC well only.
        ntc_d_drop, ntc_d_scores = apply_layer_d(
            x_raw, rows, cols, well_id,
            excluded_mask=~ntc_well_mask,        # NTC pixels are the "survivors"
            metric=d_metric,
            k_neighbours=d_k,
            pct_disagree=d_pct,
            eligible_mask=ntc_well_mask,
        )
        info["ntc_d_drop"] = ntc_d_drop
        info["ntc_d_scores"] = ntc_d_scores

    # --- Compute A/B thresholds from (optionally cleaned) NTC.
    ntc_pixels_for_thresh = x_raw[ntc_well_mask & ~ntc_d_drop]
    thresh = compute_chip_thresholds(
        ntc_pixels_for_thresh, pct_amp=pct_amp, pct_shape=pct_shape)
    info["thresholds"] = thresh

    # --- Layer A and B (positive wells only).
    if layers in ("A", "AB", "ABC", "ABCD"):
        layerA_all, layerB_all = apply_layers_ab(x_raw, thresh)
        # Restrict to positive wells: NTC pixels are never dropped by A/B.
        layerA = layerA_all & positive_well_mask
        layerB = layerB_all & positive_well_mask
        if layers == "A":
            layerB = np.zeros(n_pix, dtype=bool)
    else:
        layerA = np.zeros(n_pix, dtype=bool)
        layerB = np.zeros(n_pix, dtype=bool)
    info["layerA"] = layerA
    info["layerB"] = layerB

    # --- Layer C (positive wells only).
    if layers in ("ABC", "ABCD"):
        layerC, c_count = apply_layer_c(
            rows, cols, well_id, layerA, layerB,
            k_neighbours=c_k, min_bad_frac=c_min_bad_frac,
            eligible_mask=positive_well_mask,
        )
    else:
        layerC = np.zeros(n_pix, dtype=bool)
        c_count = np.full(n_pix, -1, dtype=np.int32)
    info["layerC"] = layerC
    info["c_count"] = c_count

    # --- Layer D (positive *and* NTC wells if D included).
    if layers == "ABCD":
        layerD, d_scores = apply_layer_d(
            x_raw, rows, cols, well_id,
            excluded_mask=(layerA | layerB | layerC),
            metric=d_metric,
            k_neighbours=d_k,
            pct_disagree=d_pct,
            eligible_mask=None,         # all wells eligible
        )
    else:
        layerD = np.zeros(n_pix, dtype=bool)
        d_scores = np.full(n_pix, np.nan, dtype=np.float64)
    info["layerD"] = layerD
    info["d_scores"] = d_scores

    # If we cleaned NTC up-front, those drops should also be removed
    # from the training cache (since they're genuinely broken pixels).
    drop_total = layerA | layerB | layerC | layerD | ntc_d_drop
    keep_mask = ~drop_total

    info["n_total"] = int(n_pix)
    info["n_kept"]  = int(keep_mask.sum())
    info["n_dropped"] = {
        "A":      int(layerA.sum()),
        "B":      int(layerB.sum()),
        "C":      int(layerC.sum()),
        "D":      int(layerD.sum()),
        "ntc_D":  int(ntc_d_drop.sum()),
        "total":  int(drop_total.sum()),
    }

    return keep_mask, info


# ====================================================================
# MAD-based threshold rule (exp8)
# ====================================================================
#
# Decoupled from the percentile path above.  Threshold form:
#   - Layer A: drop dyn_range < median - k * MAD (NTC pixels)
#   - Layer B: drop net_slope < median - k * MAD OR
#                   trace_min < median - k * MAD (NTC pixels)
#   - Layer C: unchanged from percentile path (no statistical threshold).
#   - Layer D: drop L2 disagreement > median + k * MAD (per well).
#
# `median` and `MAD` are computed from the same population the
# percentile rule uses: NTC pixels for A/B, per-well survivors for D.
# Single `k` parameter shared across A, B, D.
# --------------------------------------------------------------------


def compute_chip_thresholds_mad(
    x_ntc: np.ndarray,
    k: float,
) -> dict[str, float]:
    """MAD-rule analogue of ``compute_chip_thresholds``.

    For each per-pixel statistic (dyn_range, net_slope, trace_min) on
    this chip's NTC pixels, compute median and MAD-scaled spread, and
    set a lower-tail threshold at ``median - k * MAD``.
    """
    if x_ntc.size == 0:
        raise ValueError("no NTC pixels supplied")

    dyn_range = x_ntc.max(axis=1) - x_ntc.min(axis=1)
    net_slope = x_ntc[:, -1] - x_ntc[:, 0]
    trace_min = x_ntc.min(axis=1)

    dr_med, dr_mad = median_mad(dyn_range)
    ns_med, ns_mad = median_mad(net_slope)
    tm_med, tm_mad = median_mad(trace_min)

    return {
        "n_ntc_pixels":             int(x_ntc.shape[0]),
        "k":                        float(k),
        "dyn_range_ntc_median":     dr_med,
        "dyn_range_ntc_mad":        dr_mad,
        "dyn_range_threshold":      float(dr_med - k * dr_mad),
        "net_slope_ntc_median":     ns_med,
        "net_slope_ntc_mad":        ns_mad,
        "net_slope_threshold":      float(ns_med - k * ns_mad),
        "trace_min_ntc_median":     tm_med,
        "trace_min_ntc_mad":        tm_mad,
        "trace_min_threshold":      float(tm_med - k * tm_mad),
    }


def apply_layer_d_mad(
    x_raw: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    well_id: np.ndarray,
    excluded_mask: np.ndarray,
    k: float,
    metric: str = "median",
    k_neighbours: int = 8,
    eligible_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """MAD-rule analogue of ``apply_layer_d``.

    For each well separately, compute median and MAD of the L2
    disagreement scores over surviving pixels, then drop pixels whose
    disagreement exceeds ``median + k * MAD``.
    """
    if metric not in ("mean", "median", "kth"):
        raise ValueError(f"unknown metric '{metric}'")

    n_pix = len(x_raw)
    drop = np.zeros(n_pix, dtype=bool)
    scores = np.full(n_pix, np.nan, dtype=np.float64)

    survive = ~excluded_mask
    if eligible_mask is None:
        eligible_mask = np.ones(n_pix, dtype=bool)

    for w in sorted(set(int(v) for v in well_id)):
        w_idx = np.flatnonzero((well_id == w) & survive)
        n_w = len(w_idx)
        if n_w < (k_neighbours + 1):
            continue
        pts = np.column_stack([rows[w_idx], cols[w_idx]]).astype(np.float32)
        traces = x_raw[w_idx]

        d2 = _pairwise_squared_distance(pts)
        nn_k = _knn_indices(d2, k_neighbours)

        if metric == "mean":
            ref = traces[nn_k].mean(axis=1)
            disagree = np.sqrt(np.sum((traces - ref) ** 2, axis=1))
        elif metric == "median":
            ref = np.median(traces[nn_k], axis=1)
            disagree = np.sqrt(np.sum((traces - ref) ** 2, axis=1))
        else:  # kth
            kth_idx = nn_k[:, -1]
            kth_trace = traces[kth_idx]
            disagree = np.sqrt(np.sum((traces - kth_trace) ** 2, axis=1))

        med, mad = median_mad(disagree)
        cut = float(med + k * mad)
        scores[w_idx] = disagree

        local_eligible = eligible_mask[w_idx]
        local_drop = (disagree > cut) & local_eligible
        drop[w_idx[local_drop]] = True

    return drop, scores


def run_filter_pipeline_per_chip_mad(
    x_raw: np.ndarray,
    well_id: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    ntc_well: int,
    layers: str,                     # "A" | "AB" | "ABC" | "ABCD"
    clean_ntc_first: bool,
    k: float,
    c_k: int = 8,
    c_min_bad_frac: float = 0.5,
    d_k: int = 8,
    d_metric: str = "median",
) -> tuple[np.ndarray, dict]:
    """MAD-rule analogue of ``run_filter_pipeline_per_chip``.

    Same orchestration logic as the percentile pipeline, but Layers
    A, B, and D use ``median +- k * MAD`` thresholds in place of
    percentile cuts.  Layer C is unchanged (no statistical
    threshold).  Single ``k`` parameter shared across A, B, D.
    """
    if layers not in ("A", "AB", "ABC", "ABCD"):
        raise ValueError(f"layers must be 'A', 'AB', 'ABC', or 'ABCD'; got {layers}")

    ntc_well_mask = (well_id == ntc_well)
    positive_well_mask = ~ntc_well_mask
    n_pix = len(x_raw)

    info: dict = {"layers": layers, "clean_ntc_first": clean_ntc_first,
                  "k": float(k), "rule": "mad"}

    # --- (Optional) Clean NTC pixels with Layer D before computing thresholds.
    ntc_d_drop = np.zeros(n_pix, dtype=bool)
    if clean_ntc_first:
        ntc_d_drop, ntc_d_scores = apply_layer_d_mad(
            x_raw, rows, cols, well_id,
            excluded_mask=~ntc_well_mask,
            k=k,
            metric=d_metric,
            k_neighbours=d_k,
            eligible_mask=ntc_well_mask,
        )
        info["ntc_d_drop"] = ntc_d_drop
        info["ntc_d_scores"] = ntc_d_scores

    # --- Compute A/B thresholds from (optionally cleaned) NTC.
    ntc_pixels_for_thresh = x_raw[ntc_well_mask & ~ntc_d_drop]
    thresh = compute_chip_thresholds_mad(ntc_pixels_for_thresh, k=k)
    info["thresholds"] = thresh

    # --- Layer A and B (positive wells only).
    if layers in ("A", "AB", "ABC", "ABCD"):
        layerA_all, layerB_all = apply_layers_ab(x_raw, thresh)
        layerA = layerA_all & positive_well_mask
        layerB = layerB_all & positive_well_mask
        if layers == "A":
            layerB = np.zeros(n_pix, dtype=bool)
    else:
        layerA = np.zeros(n_pix, dtype=bool)
        layerB = np.zeros(n_pix, dtype=bool)
    info["layerA"] = layerA
    info["layerB"] = layerB

    # --- Layer C (positive wells only).  Unchanged from percentile pipeline.
    if layers in ("ABC", "ABCD"):
        layerC, c_count = apply_layer_c(
            rows, cols, well_id, layerA, layerB,
            k_neighbours=c_k, min_bad_frac=c_min_bad_frac,
            eligible_mask=positive_well_mask,
        )
    else:
        layerC = np.zeros(n_pix, dtype=bool)
        c_count = np.full(n_pix, -1, dtype=np.int32)
    info["layerC"] = layerC
    info["c_count"] = c_count

    # --- Layer D (positive and NTC wells if D included).
    if layers == "ABCD":
        layerD, d_scores = apply_layer_d_mad(
            x_raw, rows, cols, well_id,
            excluded_mask=(layerA | layerB | layerC),
            k=k,
            metric=d_metric,
            k_neighbours=d_k,
            eligible_mask=None,
        )
    else:
        layerD = np.zeros(n_pix, dtype=bool)
        d_scores = np.full(n_pix, np.nan, dtype=np.float64)
    info["layerD"] = layerD
    info["d_scores"] = d_scores

    drop_total = layerA | layerB | layerC | layerD | ntc_d_drop
    keep_mask = ~drop_total

    info["n_total"] = int(n_pix)
    info["n_kept"]  = int(keep_mask.sum())
    info["n_dropped"] = {
        "A":      int(layerA.sum()),
        "B":      int(layerB.sum()),
        "C":      int(layerC.sum()),
        "D":      int(layerD.sum()),
        "ntc_D":  int(ntc_d_drop.sum()),
        "total":  int(drop_total.sum()),
    }

    return keep_mask, info
