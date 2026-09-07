"""Preprocessing · builds the MAD-ABCD chip-relative filter + spatA3 pipeline. [Cat A] Report §Data.

Build the per-chip NTC-referenced two-layer filter and export its
output for the HTML viewer.

Pipeline
--------
For each chip in the exp6 test set:

  1.  Load every pixel trace from predictions.npz (any one run will
      do; x_raw is identical across the 15 well-behaved runs).
  2.  Compute three per-pixel statistics:
        - dynamic_range   max(x) - min(x)
        - net_slope       x[-1] - x[0]                (signed)
        - trace_min       min(x)                      (signed)
  3.  Locate the chip's NTC well (well 5 per the project's labelling
      convention).  Compute robust statistics (median + MAD) of the
      three stats across that chip's NTC pixels.
  4.  Apply Layer A (amplitude): drop pixels whose dynamic_range is
      below ``ntc_dyn_range_median + k * ntc_dyn_range_mad``.
      (NTC pixels have small dynamic range; pixels with even less
      range are by definition not amplifying.)
  5.  Apply Layer B (shape): drop pixels whose net_slope < a
      "downward outlier" threshold, OR whose trace_min < a "drops too
      low" threshold.  Both thresholds are NTC_median - k * NTC_MAD.
  6.  Tag every pixel with its misclassification status across the
      15 well-behaved runs:
        always_correct       n_wrong == 0
        sometimes_wrong      1 <= n_wrong < 15
        always_wrong         n_wrong == 15
  7.  Export one JSON per chip plus an aggregate manifest with the
      filter thresholds and the recall/collateral numbers.

The exporter is read by ``filter_viewer.html`` which renders the
four steps side by side per (chip, well).

CPU-only.

Usage::

    python -m lacewing.preprocessing.data_quality.build_filter_pipeline
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from lacewing.classification import paths as cls_paths
from lacewing.preprocessing.data_quality.analyse_misclassification_stats import (
    collect_predictions,
    build_pixel_index,
    DEFAULT_EXCLUDED_MODELS,
)


THIS_DIR = Path(__file__).resolve().parent
EXPORT_DIR = THIS_DIR / "filter_data"
EXPORT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------
# Robust statistics
# --------------------------------------------------------------------

def median_mad(x: np.ndarray) -> tuple[float, float]:
    """Median and (scaled) MAD of x; returns (median, mad)."""
    if len(x) == 0:
        return 0.0, 0.0
    med = float(np.median(x))
    mad = float(1.4826 * np.median(np.abs(x - med)))    # ~std for Gaussian
    return med, mad


# --------------------------------------------------------------------
# Filter construction
# --------------------------------------------------------------------

def compute_chip_thresholds(
    x_raw: np.ndarray,
    well_id: np.ndarray,
    ntc_well: int,
    pct_amp: float,
    pct_shape: float,
    rule: str = "percentile",
    k_mad: float = 1.645,
) -> dict[str, float]:
    """Compute NTC-referenced thresholds for one chip.

    Two threshold rules:

      - ``rule="percentile"`` (default, original behaviour):
        Layer A drops pixels whose dyn_range falls below the
        ``pct_amp``-th percentile of NTC dyn_range; Layer B drops
        pixels whose net_slope or trace_min falls below the
        ``pct_shape``-th percentile of the corresponding NTC
        distribution.

      - ``rule="mad"``:
        Same statistics, but thresholds are set adaptively as
        ``median - k_mad * MAD``.  ``k_mad`` is shared across the
        three statistics.  On a clean Gaussian, ``k_mad=1.645``
        drops the bottom ~5 %.

    Returns the same dict shape in either case; the only differences
    are the threshold values and a ``"rule"`` / ``"k_mad"`` key.
    """
    if rule not in ("percentile", "mad"):
        raise ValueError(f"unknown rule '{rule}'; expected percentile|mad")

    ntc_mask = well_id == ntc_well
    if not ntc_mask.any():
        raise ValueError(f"no pixels with well_id == {ntc_well} on this chip")

    ntc = x_raw[ntc_mask]
    dyn_range = ntc.max(axis=1) - ntc.min(axis=1)
    net_slope = ntc[:, -1] - ntc[:, 0]
    trace_min = ntc.min(axis=1)

    dr_med, dr_mad = median_mad(dyn_range)
    ns_med, ns_mad = median_mad(net_slope)
    tm_med, tm_mad = median_mad(trace_min)

    if rule == "percentile":
        dr_thresh = float(np.percentile(dyn_range, pct_amp))
        ns_thresh = float(np.percentile(net_slope, pct_shape))
        tm_thresh = float(np.percentile(trace_min, pct_shape))
    else:  # mad
        dr_thresh = float(dr_med - k_mad * dr_mad)
        ns_thresh = float(ns_med - k_mad * ns_mad)
        tm_thresh = float(tm_med - k_mad * tm_mad)

    return {
        "n_ntc_pixels":             int(ntc_mask.sum()),
        "rule":                     rule,
        "k_mad":                    float(k_mad),
        "pct_amp":                  float(pct_amp),
        "pct_shape":                float(pct_shape),
        # Layer A: amplitude floor (drop below).
        "dyn_range_ntc_median":     dr_med,
        "dyn_range_ntc_mad":        dr_mad,
        "dyn_range_threshold":      dr_thresh,
        # Layer B-a: downward-slope floor (drop below).
        "net_slope_ntc_median":     ns_med,
        "net_slope_ntc_mad":        ns_mad,
        "net_slope_threshold":      ns_thresh,
        # Layer B-b: deep-min floor (drop below).
        "trace_min_ntc_median":     tm_med,
        "trace_min_ntc_mad":        tm_mad,
        "trace_min_threshold":      tm_thresh,
    }


def apply_layers(
    x_raw: np.ndarray,
    thresh: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (layerA_dropped, layerB_dropped) boolean masks.

    Layer A drops happen *first*; Layer B drops are then computed on
    the remaining pixels.  The two masks are disjoint by construction.
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
# Spatial helpers
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
# Layer C: kNN bad-neighbour count
# --------------------------------------------------------------------

def apply_layer_c(
    rows: np.ndarray,
    cols: np.ndarray,
    well_id: np.ndarray,
    layerA: np.ndarray,
    layerB: np.ndarray,
    k_neighbours: int = 8,
    min_bad_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop pixels whose k spatial neighbours are mostly A/B-bad.

    For each pixel, count how many of its ``k_neighbours`` nearest
    spatial neighbours within the same well were dropped by Layer A
    or B.  Drop the pixel if the fraction exceeds ``min_bad_frac``
    (default 0.5 = "majority of neighbours are bad").

    Only pixels surviving A and B are *candidates* for Layer C; a
    pixel already dropped by A or B is not re-tagged.
    """
    n_pix = len(rows)
    drop = np.zeros(n_pix, dtype=bool)
    bad_count = np.full(n_pix, -1, dtype=np.int32)

    layer_ab = layerA | layerB

    for w in sorted(set(int(v) for v in well_id)):
        w_idx = np.flatnonzero(well_id == w)
        n_w = len(w_idx)
        if n_w < (k_neighbours + 1):
            continue
        pts = np.column_stack([rows[w_idx], cols[w_idx]]).astype(np.float32)
        local_ab = layer_ab[w_idx]                          # (n_w,) bool

        d2 = _pairwise_squared_distance(pts)
        nn_k = _knn_indices(d2, k_neighbours)                # (n_w, k)

        # Count A+B drops among each pixel's k neighbours.
        bad_neighbours = local_ab[nn_k].sum(axis=1)          # (n_w,)
        bad_count[w_idx] = bad_neighbours

        # Threshold: drop a pixel if >= min_bad_frac of its
        # neighbours are bad, AND it itself wasn't already dropped by
        # A or B (we don't re-tag those).
        threshold = int(np.ceil(min_bad_frac * k_neighbours))
        local_drop = (bad_neighbours >= threshold) & ~local_ab
        drop[w_idx[local_drop]] = True

    return drop, bad_count


# --------------------------------------------------------------------
# Layer D: individual disagreement with the local spatial neighbourhood
# --------------------------------------------------------------------
#
# All three sub-strategies share the same shape:
#   1. For each surviving pixel (post-A/B/C), find its k nearest
#      spatial neighbours within the same well.
#   2. Compute a per-pixel disagreement score (semantics depending on
#      metric).
#   3. Drop pixels above the per-well percentile cutoff.
#
# Metrics:
#   - "mean":   L2 distance from pixel trace to the *mean* trace of
#               its k neighbours.  Sensitive to a single contaminating
#               neighbour (the wild-trace failure mode we observed).
#   - "median": L2 distance from pixel trace to the elementwise
#               *median* trace of its k neighbours.  Robust to a
#               single outlier neighbour.
#   - "kth":    L2 distance from pixel trace to the trace of its
#               *k-th nearest* neighbour (the farthest of the k).
#               LOF-style k-distance metric; a pixel is anomalous
#               only if all of its k neighbours are far away in
#               trace space, not just one.


def apply_layer_d(
    x_raw: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    well_id: np.ndarray,
    layerA: np.ndarray,
    layerB: np.ndarray,
    layerC: np.ndarray,
    metric: str,
    k_neighbours: int = 8,
    pct_disagree: float = 95.0,
    rule: str = "percentile",
    k_mad: float = 1.645,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one Layer D sub-strategy.

    Operates only on pixels that survived A, B, and C.  Returns
    ``(drop_mask, scores)``; ``scores[i] == NaN`` if pixel i didn't
    enter the layer.

    Two threshold rules (chosen by ``rule``):
      - ``"percentile"``: drop the top ``pct_disagree`` percent of
        disagreement scores per well (default behaviour).
      - ``"mad"``: drop scores above ``median + k_mad * MAD`` of the
        per-well disagreement distribution (upper-tail MAD rule).
    """
    if metric not in ("mean", "median", "kth"):
        raise ValueError(f"unknown metric '{metric}'")
    if rule not in ("percentile", "mad"):
        raise ValueError(f"unknown rule '{rule}'; expected percentile|mad")

    n_pix = len(x_raw)
    drop = np.zeros(n_pix, dtype=bool)
    scores = np.full(n_pix, np.nan, dtype=np.float64)
    survive = ~(layerA | layerB | layerC)

    for w in sorted(set(int(v) for v in well_id)):
        w_idx = np.flatnonzero((well_id == w) & survive)
        n_w = len(w_idx)
        if n_w < (k_neighbours + 1):
            continue
        pts = np.column_stack([rows[w_idx], cols[w_idx]]).astype(np.float32)
        traces = x_raw[w_idx]

        d2 = _pairwise_squared_distance(pts)
        nn_k = _knn_indices(d2, k_neighbours)                # (n_w, k)

        if metric == "mean":
            ref = traces[nn_k].mean(axis=1)                  # (n_w, T)
            disagree = np.sqrt(np.sum((traces - ref) ** 2, axis=1))
        elif metric == "median":
            ref = np.median(traces[nn_k], axis=1)
            disagree = np.sqrt(np.sum((traces - ref) ** 2, axis=1))
        else:  # "kth"
            kth_idx = nn_k[:, -1]
            kth_trace = traces[kth_idx]
            disagree = np.sqrt(np.sum((traces - kth_trace) ** 2, axis=1))

        if rule == "percentile":
            cut = float(np.percentile(disagree, pct_disagree))
        else:  # mad (upper tail)
            d_med, d_mad = median_mad(disagree)
            cut = float(d_med + k_mad * d_mad)
        scores[w_idx] = disagree
        drop[w_idx[disagree > cut]] = True

    return drop, scores


# --------------------------------------------------------------------
# Misclassification status
# --------------------------------------------------------------------

def compute_misclass_status(runs: list[dict]) -> np.ndarray:
    """Return one of {'always_correct', 'sometimes_wrong', 'always_wrong'}
    per pixel, computed across the 15 well-behaved runs.
    """
    n_pix = len(runs[0]["y_true"])
    n_wrong = np.zeros(n_pix, dtype=np.int32)
    for r in runs:
        n_wrong += (r["y_pred"] != r["y_true"]).astype(np.int32)
    n_runs = len(runs)

    out = np.full(n_pix, "sometimes_wrong", dtype=object)
    out[n_wrong == 0]      = "always_correct"
    out[n_wrong == n_runs] = "always_wrong"
    return out, n_wrong


# --------------------------------------------------------------------
# JSON export
# --------------------------------------------------------------------

def _safe_chip_name(chip: str) -> str:
    return chip.replace("/", "_").replace(",", "_")


def export_chip_json(
    chip: str,
    x_raw: np.ndarray,
    well_id: np.ndarray,
    pixel_id: np.ndarray,
    y_true: np.ndarray,
    status: np.ndarray,
    n_wrong: np.ndarray,
    layerA: np.ndarray,
    layerB: np.ndarray,
    layerC: np.ndarray,
    c_score: np.ndarray,
    layer_d_strategies: dict[str, np.ndarray],
    layer_d_scores: dict[str, np.ndarray],
    rows: np.ndarray,
    cols: np.ndarray,
    well_dims: dict[int, tuple[int, int]],
    thresholds: dict[str, float],
    out_dir: Path,
    downsample: int = 1,
) -> tuple[Path, dict]:
    """Write one JSON file containing all per-pixel data for the viewer.

    Each pixel records:
      - "layer_ab":  "A" | "B" | null   (whether A or B dropped it)
      - "c_dropped": bool                (whether Layer C dropped it)
      - "c_score":   int                 (bad-neighbour count: 0..k)
      - "d_dropped": {strategy -> bool}  (per-Layer-D-strategy)
      - "d_scores":  {strategy -> float} (per-Layer-D-strategy score)

    The viewer computes ``dropped_layer`` on the fly given the
    user-selected Layer D strategy:
        "A" if layer_ab == "A"
        "B" if layer_ab == "B"
        "C" if c_dropped
        "D" if d_dropped[selected_strategy]
        otherwise null (kept).
    """
    d_strategy_names = list(layer_d_strategies.keys())

    dyn_range = (x_raw.max(axis=1) - x_raw.min(axis=1)).tolist()
    net_slope = (x_raw[:, -1] - x_raw[:, 0]).tolist()
    trace_min = x_raw.min(axis=1).tolist()

    wells_data: dict[str, dict] = {}
    summary_rows = []
    for w in sorted(set(int(v) for v in well_id)):
        mask = well_id == w
        idx = np.flatnonzero(mask)
        n_total = int(mask.sum())
        n_A     = int(layerA[idx].sum())
        n_B     = int(layerB[idx].sum())
        n_C     = int(layerC[idx].sum())
        n_after_A = n_total - n_A
        n_after_B = n_after_A - n_B
        n_after_C = n_after_B - n_C

        # Per-D-strategy counts (independent: how many of this well's
        # pixels does each D sub-strategy drop?).
        n_D_per_strat = {
            name: int(mask_arr[idx].sum())
            for name, mask_arr in layer_d_strategies.items()
        }

        pixels = []
        for i in idx:
            if   layerA[i]: layer_ab = "A"
            elif layerB[i]: layer_ab = "B"
            else:           layer_ab = None
            trace = x_raw[i, ::downsample].tolist()

            cs = c_score[i]
            c_score_val = (None if (isinstance(cs, float) and cs != cs)
                           else (int(cs) if cs != -1 else None))

            d_dropped = {
                name: bool(mask_arr[i])
                for name, mask_arr in layer_d_strategies.items()
            }
            d_scores = {}
            for name, score_arr in layer_d_scores.items():
                v = score_arr[i]
                if v != v:
                    d_scores[name] = None
                else:
                    d_scores[name] = float(v)

            pixels.append({
                "pid":         int(pixel_id[i]),
                "trace":       [float(v) for v in trace],
                "row":         int(rows[i]),
                "col":         int(cols[i]),
                "dyn_range":   float(dyn_range[i]),
                "net_slope":   float(net_slope[i]),
                "trace_min":   float(trace_min[i]),
                "status":      str(status[i]),
                "n_wrong":     int(n_wrong[i]),
                "layer_ab":    layer_ab,
                "c_dropped":   bool(layerC[i]),
                "c_score":     c_score_val,
                "d_dropped":   d_dropped,
                "d_scores":    d_scores,
            })

        label = int(y_true[idx[0]]) if len(idx) else -1
        nrows, ncols = well_dims.get(w, (0, 0))
        wells_data[str(w)] = {
            "label":     label,
            "n_total":   n_total,
            "n_after_A": n_after_A,
            "n_after_B": n_after_B,
            "n_after_C": n_after_C,
            "n_D_per_strategy": n_D_per_strat,
            "nrows":     int(nrows),
            "ncols":     int(ncols),
            "pixels":    pixels,
        }
        summary_rows.append({
            "well":             w,
            "label":            label,
            "n_total":          n_total,
            "n_layerA_dropped": n_A,
            "n_after_A":        n_after_A,
            "n_layerB_dropped": n_B,
            "n_after_B":        n_after_B,
            "n_layerC_dropped": n_C,
            "n_after_C":        n_after_C,
            "n_layerD_dropped_per_strategy": n_D_per_strat,
        })

    payload = {
        "chip":              chip,
        "thresholds":        thresholds,
        "d_strategies":      d_strategy_names,
        "summary":           summary_rows,
        "wells":             wells_data,
        "trace_length_original": int(x_raw.shape[1]),
        "trace_length_exported": int(x_raw[0, ::downsample].shape[0]),
        "downsample":        int(downsample),
    }

    out_path = out_dir / f"{_safe_chip_name(chip)}.json"
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    return out_path, payload


# --------------------------------------------------------------------
# Recall / collateral table
# --------------------------------------------------------------------

def compute_recall_collateral(
    status: np.ndarray,
    layerA: np.ndarray,
    layerB: np.ndarray,
    layerC: np.ndarray,
    layer_d_strategies: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> dict:
    """How many of each (status x label) bucket gets dropped at each
    layer.  Layer C is a single layer (kNN bad-neighbour).  Layer D
    has several sub-strategies reported independently; for each
    sub-strategy we report what the *full* A+B+C+D pipeline catches.
    """
    out = {}
    statuses = ["always_correct", "sometimes_wrong", "always_wrong"]
    labels = [0, 1]
    for st in statuses:
        for lbl in labels:
            mask = (status == st) & (y_true == lbl)
            total = int(mask.sum())
            if total == 0:
                continue
            entry = {
                "total":     total,
                "dropped_A": int(layerA[mask].sum()),
                "dropped_B": int(layerB[mask].sum()),
                "dropped_C": int(layerC[mask].sum()),
            }
            for strat_name, strat_mask in layer_d_strategies.items():
                entry[f"dropped_D_{strat_name}"] = int(strat_mask[mask].sum())
                entry[f"dropped_total_{strat_name}"] = int(
                    (layerA[mask] | layerB[mask] |
                     layerC[mask] | strat_mask[mask]).sum()
                )
            out[f"{st}__y{lbl}"] = entry
    return out


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="exp6_final_to_sd")
    parser.add_argument("--rule", choices=["percentile", "mad"],
                        default="percentile",
                        help="threshold rule for Layers A, B, D: the "
                             "original percentile cut or the adaptive "
                             "MAD-based rule (default 'percentile')")
    parser.add_argument("--k-mad", type=float, default=1.645,
                        help="MAD multiplier shared across A/B/D when "
                             "rule='mad' (default 1.645, matches the 5th "
                             "percentile on a clean Gaussian)")
    parser.add_argument("--pct-amp", type=float, default=5.0,
                        help="percentile of NTC dyn_range used as the "
                             "Layer-A floor when rule='percentile' "
                             "(default 5)")
    parser.add_argument("--pct-shape", type=float, default=5.0,
                        help="percentile of NTC net_slope / trace_min "
                             "used as the Layer-B floor when "
                             "rule='percentile' (default 5)")
    # Layer C: kNN bad-neighbour count.
    parser.add_argument("--c-k", type=int, default=8,
                        help="kNN size for Layer C bad-neighbour count "
                             "(default 8)")
    parser.add_argument("--c-bad-frac", type=float, default=0.5,
                        help="min fraction of neighbours that must be "
                             "A/B-bad to drop the pixel (default 0.5)")
    # Layer D: individual disagreement.  Three metric sub-strategies
    # run in parallel; viewer selects which is visualised.
    parser.add_argument("--d-k", type=int, default=8,
                        help="kNN size for Layer D disagreement metrics "
                             "(default 8)")
    parser.add_argument("--d-pct", type=float, default=95.0,
                        help="per-well percentile cutoff for Layer D "
                             "when rule='percentile' (default 95)")
    parser.add_argument("--ntc-well", type=int,
                        default=cls_paths.NTC_WELL_INDEX,
                        help="well index of the NTC well per chip")
    parser.add_argument("--downsample", type=int, default=1,
                        help="time-axis stride to keep JSON small (default 1)")
    parser.add_argument("--out-dir", default=None,
                        help="override the output directory.  Default: "
                             "filter_data/ (percentile) or "
                             "filter_data_mad_k<value>/ (mad).")
    args = parser.parse_args()

    # Route output to a rule-aware subdirectory unless overridden.
    if args.out_dir is not None:
        export_dir = THIS_DIR / args.out_dir
    elif args.rule == "percentile":
        export_dir = THIS_DIR / "filter_data"
    else:  # mad
        k_tag = f"{args.k_mad}".replace(".", "p")
        export_dir = THIS_DIR / f"filter_data_mad_k{k_tag}"
    export_dir.mkdir(exist_ok=True)
    print(f"==> Output directory: {export_dir}")
    print(f"==> Rule: {args.rule}"
          + (f"  (k_mad = {args.k_mad})" if args.rule == "mad" else ""))

    exp_dir = cls_paths.experiment_dir(args.experiment)
    print(f"Reading runs under {exp_dir}/runs/")
    runs = collect_predictions(exp_dir, excluded_models=DEFAULT_EXCLUDED_MODELS)
    chip_id, well_id, pixel_id = build_pixel_index(runs)
    y_true  = runs[0]["y_true"]
    x_raw   = runs[0]["x_raw"]
    print(f"  {len(runs)} runs, {len(chip_id)} pixels, "
          f"{len(set(chip_id))} chips")

    status, n_wrong = compute_misclass_status(runs)

    # Lazy import here so spatial-coord extraction (which triggers
    # titan_load_and_preprocessing) only runs if we actually need it.
    from lacewing.preprocessing.data_quality.spatial_coords import (
        build_coords_for_chip,
    )

    d_strategy_names = ["mean", "median", "kth"]

    manifest_chips = []
    overall_recall: dict[str, dict] = defaultdict(dict)

    for chip in sorted(set(chip_id)):
        mask = chip_id == chip
        x_c = x_raw[mask]
        well_c = well_id[mask]
        pix_c = pixel_id[mask]
        yt_c = y_true[mask]
        st_c = status[mask]
        nw_c = n_wrong[mask]

        thresh = compute_chip_thresholds(
            x_c, well_c, args.ntc_well, args.pct_amp, args.pct_shape,
            rule=args.rule, k_mad=args.k_mad)
        layerA, layerB = apply_layers(x_c, thresh)

        # Spatial coordinates per pixel (reload chip via titan).
        print(f"  loading spatial coords for {chip}...")
        coords = build_coords_for_chip(chip)

        rows_c = np.empty(len(pix_c), dtype=np.int32)
        cols_c = np.empty(len(pix_c), dtype=np.int32)
        well_dims: dict[int, tuple[int, int]] = {}
        for w in sorted(set(int(v) for v in well_c)):
            w_mask = well_c == w
            pids = pix_c[w_mask].astype(np.int64)
            rows_c[w_mask] = coords[w]["rows"][pids]
            cols_c[w_mask] = coords[w]["cols"][pids]
            well_dims[w] = (coords[w]["nrows"], coords[w]["ncols"])

        # Layer C: kNN bad-neighbour count (single layer).
        layerC, c_score = apply_layer_c(
            rows_c, cols_c, well_c, layerA, layerB,
            k_neighbours=args.c_k,
            min_bad_frac=args.c_bad_frac,
        )

        # Layer D: three metric sub-strategies in parallel.
        layer_d_strategies: dict[str, np.ndarray] = {}
        layer_d_scores:     dict[str, np.ndarray] = {}
        for metric in d_strategy_names:
            drop, score = apply_layer_d(
                x_c, rows_c, cols_c, well_c,
                layerA, layerB, layerC,
                metric=metric,
                k_neighbours=args.d_k,
                pct_disagree=args.d_pct,
                rule=args.rule,
                k_mad=args.k_mad,
            )
            layer_d_strategies[metric] = drop
            layer_d_scores[metric] = score

        rec = compute_recall_collateral(
            st_c, layerA, layerB, layerC, layer_d_strategies, yt_c)
        for key, sub in rec.items():
            o = overall_recall[key]
            for k_, v in sub.items():
                o[k_] = o.get(k_, 0) + v

        out_path, payload = export_chip_json(
            chip, x_c, well_c, pix_c, yt_c, st_c, nw_c,
            layerA, layerB, layerC, c_score,
            layer_d_strategies, layer_d_scores,
            rows_c, cols_c, well_dims,
            thresh, export_dir, downsample=args.downsample,
        )
        manifest_chips.append({
            "chip":     chip,
            "json":     out_path.name,
            "n_pixels": int(mask.sum()),
            "n_wells":  len(payload["summary"]),
        })
        per_d_counts = ", ".join(
            f"{name}={int(mask_arr.sum())}"
            for name, mask_arr in layer_d_strategies.items()
        )
        print(f"  wrote {out_path.name}: {mask.sum()} pixels "
              f"-> A drops {int(layerA.sum())}, "
              f"B drops {int(layerB.sum())}, "
              f"C drops {int(layerC.sum())}, "
              f"D strategies: {per_d_counts}")

    # Manifest with overall numbers.
    manifest = {
        "pct_amp":            args.pct_amp,
        "pct_shape":          args.pct_shape,
        "d_strategies":       d_strategy_names,
        "layer_c_params":     {"k": args.c_k, "min_bad_frac": args.c_bad_frac},
        "layer_d_params":     {"k": args.d_k, "pct": args.d_pct},
        "ntc_well":           args.ntc_well,
        "experiment":         args.experiment,
        "chips":              manifest_chips,
        "overall_recall":     dict(overall_recall),
        "models_excluded":    list(DEFAULT_EXCLUDED_MODELS),
        "n_runs_used":        len(runs),
    }
    # Stash the rule + k_mad in the manifest so the viewer can label it.
    manifest["rule"] = args.rule
    manifest["k_mad"] = float(args.k_mad)
    (export_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"\nWrote {export_dir}/manifest.json")

    # ----------------------------------------------------------------
    # Headline recall / collateral table - one section per Layer D strategy
    # ----------------------------------------------------------------
    print(f"\n=== Recall / collateral (pct_amp={args.pct_amp}, "
          f"pct_shape={args.pct_shape}, both chips combined) ===")
    for strat in d_strategy_names:
        print(f"\n--- Layer D strategy: {strat} ---")
        print(f"{'bucket':<28} {'total':>7} {'A':>6} {'B':>6} "
              f"{'C':>6} {'D':>6} {'A+B+C+D':>9}")
        print("-" * 80)
        for key in ["always_wrong__y0", "always_wrong__y1",
                    "sometimes_wrong__y0", "sometimes_wrong__y1",
                    "always_correct__y0", "always_correct__y1"]:
            d = overall_recall.get(key)
            if d is None:
                continue
            print(f"{key:<28} {d['total']:>7} {d['dropped_A']:>6} "
                  f"{d['dropped_B']:>6} {d['dropped_C']:>6} "
                  f"{d[f'dropped_D_{strat}']:>6} "
                  f"{d[f'dropped_total_{strat}']:>9}")


if __name__ == "__main__":
    main()
