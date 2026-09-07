"""Multi-NTC-well analogues of run_filter_pipeline_per_chip{,_mad}.

The Final-chip pipelines in ``lacewing.preprocessing.data_quality.filter_core``
take a single ``ntc_well: int`` because the Final chips have exactly
one NTC well (well 5).  The Multi chips have 2-5 NTC wells, so we need
to derive A/B/D thresholds from a *pool* of NTC pixels — but layer C
(spatial k-NN) and layer D's per-well median trace must still iterate
over the real per-well groupings.

This module replicates the filter-core dispatchers with the single
change that ``ntc_well_mask`` is a boolean array (one entry per pixel)
instead of an integer well index.  Layer C / Layer D continue to be
called with the real ``well_id`` array so their per-well grouping is
unaffected.

All other parameters and behaviour match the upstream functions.
"""
from __future__ import annotations

import numpy as np

from lacewing.preprocessing.data_quality import filter_core


def run_filter_pipeline_per_chip_multi(
    x_raw: np.ndarray,
    well_id: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    ntc_well_mask: np.ndarray,           # NEW: pooled NTC pixel mask
    layers: str,
    clean_ntc_first: bool,
    pct_amp: float = 5.0,
    pct_shape: float = 5.0,
    c_k: int = 8,
    c_min_bad_frac: float = 0.5,
    d_k: int = 8,
    d_pct: float = 95.0,
    d_metric: str = "median",
) -> tuple[np.ndarray, dict]:
    """Multi-NTC analogue of ``filter_core.run_filter_pipeline_per_chip``."""
    if layers not in ("A", "AB", "ABC", "ABCD"):
        raise ValueError(f"layers must be 'A', 'AB', 'ABC', or 'ABCD'; got {layers}")

    ntc_well_mask = np.asarray(ntc_well_mask, dtype=bool)
    positive_well_mask = ~ntc_well_mask
    n_pix = len(x_raw)

    info: dict = {"layers": layers, "clean_ntc_first": clean_ntc_first,
                  "rule": "percentile"}

    ntc_d_drop = np.zeros(n_pix, dtype=bool)
    if clean_ntc_first:
        ntc_d_drop, ntc_d_scores = filter_core.apply_layer_d(
            x_raw, rows, cols, well_id,
            excluded_mask=~ntc_well_mask,
            metric=d_metric,
            k_neighbours=d_k,
            pct_disagree=d_pct,
            eligible_mask=ntc_well_mask,
        )
        info["ntc_d_drop"] = ntc_d_drop
        info["ntc_d_scores"] = ntc_d_scores

    ntc_pixels_for_thresh = x_raw[ntc_well_mask & ~ntc_d_drop]
    thresh = filter_core.compute_chip_thresholds(
        ntc_pixels_for_thresh, pct_amp=pct_amp, pct_shape=pct_shape)
    info["thresholds"] = thresh

    if layers in ("A", "AB", "ABC", "ABCD"):
        layerA_all, layerB_all = filter_core.apply_layers_ab(x_raw, thresh)
        layerA = layerA_all & positive_well_mask
        layerB = layerB_all & positive_well_mask
        if layers == "A":
            layerB = np.zeros(n_pix, dtype=bool)
    else:
        layerA = np.zeros(n_pix, dtype=bool)
        layerB = np.zeros(n_pix, dtype=bool)
    info["layerA"] = layerA
    info["layerB"] = layerB

    if layers in ("ABC", "ABCD"):
        layerC, c_count = filter_core.apply_layer_c(
            rows, cols, well_id, layerA, layerB,
            k_neighbours=c_k, min_bad_frac=c_min_bad_frac,
            eligible_mask=positive_well_mask,
        )
    else:
        layerC = np.zeros(n_pix, dtype=bool)
        c_count = np.full(n_pix, -1, dtype=np.int32)
    info["layerC"] = layerC
    info["c_count"] = c_count

    if layers == "ABCD":
        layerD, d_scores = filter_core.apply_layer_d(
            x_raw, rows, cols, well_id,
            excluded_mask=(layerA | layerB | layerC),
            metric=d_metric,
            k_neighbours=d_k,
            pct_disagree=d_pct,
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
    info["n_kept"] = int(keep_mask.sum())
    info["n_dropped"] = {
        "A":     int(layerA.sum()),
        "B":     int(layerB.sum()),
        "C":     int(layerC.sum()),
        "D":     int(layerD.sum()),
        "ntc_D": int(ntc_d_drop.sum()),
        "total": int(drop_total.sum()),
    }
    return keep_mask, info


def run_filter_pipeline_per_chip_mad_multi(
    x_raw: np.ndarray,
    well_id: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    ntc_well_mask: np.ndarray,
    layers: str,
    clean_ntc_first: bool,
    k: float,
    c_k: int = 8,
    c_min_bad_frac: float = 0.5,
    d_k: int = 8,
    d_metric: str = "median",
) -> tuple[np.ndarray, dict]:
    """Multi-NTC analogue of ``filter_core.run_filter_pipeline_per_chip_mad``."""
    if layers not in ("A", "AB", "ABC", "ABCD"):
        raise ValueError(f"layers must be 'A', 'AB', 'ABC', or 'ABCD'; got {layers}")

    ntc_well_mask = np.asarray(ntc_well_mask, dtype=bool)
    positive_well_mask = ~ntc_well_mask
    n_pix = len(x_raw)

    info: dict = {"layers": layers, "clean_ntc_first": clean_ntc_first,
                  "k": float(k), "rule": "mad"}

    ntc_d_drop = np.zeros(n_pix, dtype=bool)
    if clean_ntc_first:
        ntc_d_drop, ntc_d_scores = filter_core.apply_layer_d_mad(
            x_raw, rows, cols, well_id,
            excluded_mask=~ntc_well_mask,
            k=k,
            metric=d_metric,
            k_neighbours=d_k,
            eligible_mask=ntc_well_mask,
        )
        info["ntc_d_drop"] = ntc_d_drop
        info["ntc_d_scores"] = ntc_d_scores

    ntc_pixels_for_thresh = x_raw[ntc_well_mask & ~ntc_d_drop]
    thresh = filter_core.compute_chip_thresholds_mad(ntc_pixels_for_thresh, k=k)
    info["thresholds"] = thresh

    if layers in ("A", "AB", "ABC", "ABCD"):
        layerA_all, layerB_all = filter_core.apply_layers_ab(x_raw, thresh)
        layerA = layerA_all & positive_well_mask
        layerB = layerB_all & positive_well_mask
        if layers == "A":
            layerB = np.zeros(n_pix, dtype=bool)
    else:
        layerA = np.zeros(n_pix, dtype=bool)
        layerB = np.zeros(n_pix, dtype=bool)
    info["layerA"] = layerA
    info["layerB"] = layerB

    if layers in ("ABC", "ABCD"):
        layerC, c_count = filter_core.apply_layer_c(
            rows, cols, well_id, layerA, layerB,
            k_neighbours=c_k, min_bad_frac=c_min_bad_frac,
            eligible_mask=positive_well_mask,
        )
    else:
        layerC = np.zeros(n_pix, dtype=bool)
        c_count = np.full(n_pix, -1, dtype=np.int32)
    info["layerC"] = layerC
    info["c_count"] = c_count

    if layers == "ABCD":
        layerD, d_scores = filter_core.apply_layer_d_mad(
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
    info["n_kept"] = int(keep_mask.sum())
    info["n_dropped"] = {
        "A":     int(layerA.sum()),
        "B":     int(layerB.sum()),
        "C":     int(layerC.sum()),
        "D":     int(layerD.sum()),
        "ntc_D": int(ntc_d_drop.sum()),
        "total": int(drop_total.sum()),
    }
    return keep_mask, info
