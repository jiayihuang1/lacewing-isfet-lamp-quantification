"""Cluster active pixels of each (chip, well) by three different
methods, for the manual labelling workflow.

Methods (user-confirmed 2026-06-23/24):

    A — trace_shape    K=8.  K-means on smoothed-and-normalised
                       per-pixel signal.  Pixels with similar
                       amplification shapes cluster together.
    B — per_pixel_ttp  K=8.  Per-pixel deployed-TTP method
                       (smooth=100, threshold-derivative); 1-D
                       k-means on the resulting TTP values.  NaN
                       pixels are dropped (the deployed method
                       can't extract a TTP from them).
    C — spatial        K=8.  K-means on (row, col) of the
                       surviving pixels.  Pure spatial blobs,
                       ~4 temperature pixels per cluster on
                       average.

Output: a JSON sidecar at
``Analysis/regression/data/manual_labels/cluster_data.json``.

JSON schema:
    {
      "schema": 2,
      "n_clusters": 8,
      ...
      "chips": {
        "<chip_id>": {
          "qlamp_well_ttp_min": {"0": 21.05, ...},
          "wells": {
            "<well_id>": {
              "well_nrows": ..., "well_ncols": ...,
              "n_pixels": ...,
              "rows": [...], "cols": [...],   # per surviving pixel
              "methods": {
                "trace_shape":   <method_block>,
                "per_pixel_ttp": <method_block>,
                "spatial":       <method_block>,
              }
            }
          }
        }
      }
    }

    <method_block> = {
      "cluster_id":    [int per surviving pixel] | "by_kept_idx",
      "kept_idx":      [int, ...]   # only used when some pixels dropped
                                    # (per_pixel_ttp); same length as cluster_id
      "cluster_mean":  {<cid>: [downsampled smoothed mean]},
      "cluster_size":  {<cid>: int},
      "cluster_traces":{<cid>: [{"trace":[...],"row":int,"col":int}]},
      "n_total":       int,         # pixels in this method's clustering input
      "n_dropped":     int,         # pixels excluded (only TTP-NaN drops, etc.)
    }

Run:
    python -m lacewing.quantification.regression_shared.data.build_clusters
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from lacewing.quantification.regression_shared.core import paths
from lacewing.quantification.regression_shared.data import labels as rlabels


# Defaults (user-confirmed 2026-06-23/24/25).
# K_VALUES drives both the JSON schema and the viewer dropdown.
K_VALUES    = (2, 4, 8)
SMOOTH_SPAN = 10                 # for the displayed-signal smoothing
DOWN_FACTOR = 4                  # downsample for transport
DT_SECONDS  = 4                  # frame period; existing cache convention
TRACES_PER_CLUSTER_MAX = 120     # subsample to keep JSON manageable

OUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data" / "manual_labels" / "cluster_data.json"
)


def _smooth_centered(signal: np.ndarray, span: int) -> np.ndarray:
    """Centred moving-average smoothing along the last axis."""
    if span <= 1:
        return signal.astype(np.float32, copy=True)
    half = (span - 1) // 2
    pad = np.pad(signal, ((0, 0),) * (signal.ndim - 1) + ((half, half),),
                  mode="edge")
    kernel = np.ones(2 * half + 1, dtype=np.float32) / (2 * half + 1)
    out = np.apply_along_axis(
        lambda v: np.convolve(v, kernel, mode="valid"), -1, pad
    )
    return out.astype(np.float32)


def _normalise_for_shape_clustering(traces: np.ndarray) -> np.ndarray:
    """Smooth, then per-trace z-score for pure-shape k-means features."""
    smooth = _smooth_centered(traces, SMOOTH_SPAN)
    mu = smooth.mean(axis=1, keepdims=True)
    sd = smooth.std(axis=1, keepdims=True) + 1e-6
    return (smooth - mu) / sd


def _per_pixel_ttp(traces: np.ndarray, t_axis_min: np.ndarray
                    ) -> np.ndarray:
    """Apply the deployed TTP-extraction method to each pixel trace.

    The deployed method internally smooths at span=100 (matches
    Lacewing_Readout_DNA_Quant.m), so we don't pre-smooth here.
    Returns an (N,) array of TTPs in minutes (NaN where extraction
    failed).
    """
    from lacewing.quantification.methods.ttp_threshold_derivative import (
        extract_ttp,
    )

    out = np.empty(len(traces), dtype=np.float64)
    for i, signal in enumerate(traces):
        try:
            ttp_min, _peak = extract_ttp(
                t_axis_min,
                signal.astype(np.float64),
                # Use the v1 7-min search-start floor for TTP — matches
                # the deployed convention, distinct from the v2 SDM/Cy0
                # 10-min floor.
                search_start_min=7.0,
            )
        except Exception:
            ttp_min = float("nan")
        out[i] = ttp_min
    return out


def _build_method_block(
    cluster_ids: np.ndarray,    # one per pixel in this method's input
    n_clusters: int,
    smoothed_traces: np.ndarray,
    rows: list[int], cols: list[int],
    t_idx_down: np.ndarray,
    rng: np.random.Generator,
    kept_idx: np.ndarray | None = None,
) -> dict:
    """Build the per-method JSON block for one K value.

    ``kept_idx`` is an array of indices into the parent (chip, well)
    pixel list when this method dropped some pixels (e.g. NaN-TTP for
    per_pixel_ttp).  When None, every pixel was included.
    """
    n_in = len(cluster_ids)
    cluster_means:  dict[str, list[float]] = {}
    cluster_sizes:  dict[str, int]         = {}
    cluster_traces: dict[str, list[dict]]  = {}

    for c in range(n_clusters):
        m = (cluster_ids == c)
        size = int(m.sum())
        cluster_sizes[str(c)] = size
        if size == 0:
            cluster_means[str(c)]  = []
            cluster_traces[str(c)] = []
            continue
        mean = smoothed_traces[m].mean(axis=0)[t_idx_down]
        cluster_means[str(c)] = [round(float(v), 4) for v in mean]
        idx_in_cluster = np.flatnonzero(m)
        if size > TRACES_PER_CLUSTER_MAX:
            keep = rng.choice(idx_in_cluster,
                              size=TRACES_PER_CLUSTER_MAX, replace=False)
            keep.sort()
        else:
            keep = idx_in_cluster
        trace_list: list[dict] = []
        for i in keep:
            parent_i = int(kept_idx[i]) if kept_idx is not None else int(i)
            trace_down = smoothed_traces[i][t_idx_down]
            trace_list.append({
                "trace": [round(float(v), 4) for v in trace_down],
                "row":   rows[parent_i],
                "col":   cols[parent_i],
            })
        cluster_traces[str(c)] = trace_list

    block: dict = {
        "k":              n_clusters,
        "cluster_mean":   cluster_means,
        "cluster_size":   cluster_sizes,
        "cluster_traces": cluster_traces,
        "n_total":        n_in,
        "cluster_id":     [int(c) for c in cluster_ids],
    }
    if kept_idx is not None:
        block["kept_idx"] = [int(i) for i in kept_idx]
    return block


def main() -> None:
    cache_path = paths.regression_cache_path("regress_all_filt_abcd_ntcRaw_madk1p5")
    if not cache_path.exists():
        raise SystemExit(
            f"Regression cache not found at {cache_path}. "
            f"Build it via lacewing.quantification.regression_shared.data.build_regression_cache first."
        )
    print(f"Loading cache: {cache_path}")
    cache = np.load(cache_path)
    X        = cache["X"].astype(np.float32, copy=False)
    chip_id  = np.asarray(cache["chip_id"])
    well_id  = np.asarray(cache["well_id"]).astype(np.int32)
    pixel_id = np.asarray(cache["pixel_id"]).astype(np.int32)
    split    = np.asarray(cache["split"]).astype(np.int8)

    # Train split only (the dose-response chips).
    keep = (split == 0)
    X        = X[keep]
    chip_id  = chip_id[keep]
    well_id  = well_id[keep]
    pixel_id = pixel_id[keep]

    train_chips = sorted(set(map(str, chip_id)))
    print(f"  train chips ({len(train_chips)}): {train_chips}")
    print(f"  total pixels = {len(X)}")

    n_T = X.shape[1]
    t_idx_down = np.arange(0, n_T, DOWN_FACTOR)
    t_axis_min_full = (np.arange(n_T) * DT_SECONDS / 60.0).astype(np.float64)
    t_axis_min_down = (t_idx_down * DT_SECONDS / 60.0).round(4).tolist()

    print("Resolving (row, col) per chip via spatial_coords...")
    from lacewing.preprocessing.data_quality.spatial_coords import (
        build_coords_for_chip,
    )

    chip_to_coords: dict[str, dict] = {}
    for ck in train_chips:
        try:
            chip_to_coords[ck] = build_coords_for_chip(ck)
        except Exception as e:
            print(f"  WARN  {ck}: spatial_coords failed -> {type(e).__name__}: {e}")
            chip_to_coords[ck] = {}

    out: dict = {
        "schema":         3,
        "k_values":       list(K_VALUES),
        "smooth_span":    SMOOTH_SPAN,
        "down_factor":    DOWN_FACTOR,
        "t_axis_min":     t_axis_min_down,
        "n_samples_per_mean": len(t_axis_min_down),
        "methods":        ["trace_shape", "per_pixel_ttp", "spatial"],
        "method_descriptions": {
            "trace_shape":
                "K-means on smoothed-and-normalised per-pixel trace shape.",
            "per_pixel_ttp":
                "1-D K-means on the deployed-TTP method's per-pixel TTP "
                "(smooth=100 internally; NaN pixels dropped).",
            "spatial":
                "K-means on (row, col) of surviving pixels.",
        },
        "chips":          {},
    }

    for ck in train_chips:
        chip_block: dict = {
            "qlamp_well_ttp_min": {},
            "wells":              {},
        }
        ttp_per_well = rlabels.qlamp_ttp_per_well(ck, n_wells=6)
        for w_idx in range(6):
            v = ttp_per_well[w_idx]
            if v == v:    # not NaN
                chip_block["qlamp_well_ttp_min"][str(w_idx)] = round(float(v), 3)

        coords = chip_to_coords.get(ck, {})
        well_mask = (chip_id == ck)
        wells_in_chip = sorted(set(well_id[well_mask].tolist()))

        print(f"\nChip {ck} ({int(well_mask.sum())} px, wells={wells_in_chip})")

        min_k = min(K_VALUES)
        for w in wells_in_chip:
            w_mask = well_mask & (well_id == w)
            traces  = X[w_mask]                       # (n, T)
            pix_ids = pixel_id[w_mask]
            n_w = len(traces)
            if n_w < min_k:
                print(f"    well {w}: only {n_w} pixels (< min K={min_k}); skipping")
                continue

            if w in coords:
                rows_w = coords[w]["rows"]
                cols_w = coords[w]["cols"]
                if pix_ids.max() >= len(rows_w):
                    print(f"    well {w}: pixel_id out of range "
                          f"({pix_ids.max()} >= {len(rows_w)}) — skipping")
                    continue
                rows = rows_w[pix_ids].astype(int).tolist()
                cols = cols_w[pix_ids].astype(int).tolist()
                well_nrows = int(coords[w]["nrows"])
                well_ncols = int(coords[w]["ncols"])
            else:
                rows = [0] * n_w
                cols = list(range(n_w))
                well_nrows = 1
                well_ncols = n_w

            # Shared one-time work (reused across K).
            smoothed = _smooth_centered(traces, SMOOTH_SPAN)
            feats_A  = _normalise_for_shape_clustering(traces)
            ttps     = _per_pixel_ttp(traces, t_axis_min_full)
            ok       = np.isfinite(ttps)
            n_ok     = int(ok.sum())
            n_drop   = int((~ok).sum())
            feats_C  = np.column_stack([rows, cols]).astype(np.float32)

            methods_block: dict[str, dict[str, dict]] = {
                "trace_shape":   {},
                "per_pixel_ttp": {},
                "spatial":       {},
            }

            shape_size_log: dict[int, list[int]] = {}
            spatial_size_log: dict[int, list[int]] = {}

            for K in K_VALUES:
                rng = np.random.default_rng(0)  # reset per-K for reproducibility

                # --- Method A: trace_shape ---
                if n_w >= K:
                    km_A = KMeans(n_clusters=K, random_state=0, n_init=10)
                    labels_A = km_A.fit_predict(feats_A)
                    block_A = _build_method_block(
                        labels_A, K, smoothed, rows, cols, t_idx_down, rng,
                        kept_idx=None,
                    )
                else:
                    block_A = {
                        "k": K, "cluster_mean": {}, "cluster_size": {},
                        "cluster_traces": {}, "n_total": n_w,
                        "cluster_id": [],
                        "skip_reason": f"only {n_w} pixels (< K={K})",
                    }
                methods_block["trace_shape"][str(K)] = block_A
                shape_size_log[K] = list(block_A.get("cluster_size", {}).values())

                # --- Method B: per_pixel_ttp ---
                if n_ok >= K:
                    feats_B = ttps[ok].reshape(-1, 1)
                    km_B = KMeans(n_clusters=K, random_state=0, n_init=10)
                    labels_B = km_B.fit_predict(feats_B)
                    # Re-order cluster ids by mean TTP so c0=earliest, c{K-1}=latest.
                    centres = km_B.cluster_centers_.flatten()
                    order = np.argsort(centres)
                    remap = {int(orig): int(new) for new, orig in enumerate(order)}
                    labels_B = np.array([remap[int(c)] for c in labels_B])
                    rng = np.random.default_rng(0)
                    block_B = _build_method_block(
                        labels_B, K, smoothed[ok], rows, cols, t_idx_down, rng,
                        kept_idx=np.flatnonzero(ok),
                    )
                    block_B["cluster_mean_ttp_min"] = {
                        str(c): round(float(ttps[ok][labels_B == c].mean()), 3)
                        if (labels_B == c).any() else None
                        for c in range(K)
                    }
                else:
                    block_B = {
                        "k": K, "cluster_mean": {}, "cluster_size": {},
                        "cluster_traces": {}, "n_total": n_ok,
                        "n_dropped": n_drop, "cluster_id": [], "kept_idx": [],
                        "cluster_mean_ttp_min": {},
                        "skip_reason":
                            f"only {n_ok} pixels survived TTP extraction (< K={K})",
                    }
                block_B["n_dropped"] = n_drop
                methods_block["per_pixel_ttp"][str(K)] = block_B

                # --- Method C: spatial ---
                if n_w >= K:
                    km_C = KMeans(n_clusters=K, random_state=0, n_init=10)
                    labels_C = km_C.fit_predict(feats_C)
                    rng = np.random.default_rng(0)
                    block_C = _build_method_block(
                        labels_C, K, smoothed, rows, cols, t_idx_down, rng,
                        kept_idx=None,
                    )
                else:
                    block_C = {
                        "k": K, "cluster_mean": {}, "cluster_size": {},
                        "cluster_traces": {}, "n_total": n_w,
                        "cluster_id": [],
                        "skip_reason": f"only {n_w} pixels (< K={K})",
                    }
                methods_block["spatial"][str(K)] = block_C
                spatial_size_log[K] = list(block_C.get("cluster_size", {}).values())

            chip_block["wells"][str(w)] = {
                "n_pixels":     n_w,
                "well_nrows":   well_nrows,
                "well_ncols":   well_ncols,
                "rows":         rows,
                "cols":         cols,
                "methods":      methods_block,
            }
            print(f"    well {w}: {n_w} px  ttp drop={n_drop}  "
                  f"shape K={dict(shape_size_log)}  "
                  f"spatial K={dict(spatial_size_log)}")

        out["chips"][ck] = chip_block

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nWrote {OUT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
