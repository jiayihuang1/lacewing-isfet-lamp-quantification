"""Add the 5 training-set chips to filter_data/ for the distribution viewer.

The main build_filter_pipeline.py exporter only writes the 2 held-out SD
chips because it reads exp6 predictions (which exist only for held-out
pixels).  For the distribution viewer we need the same per-pixel stats
(dyn_range / net_slope / trace_min, and the three Layer D disagreement
scores) for the 5 training chips too, so we can check whether the shape
of the per-well distributions is consistent across all 7 chips.

This script reuses the helpers in build_filter_pipeline.py but pulls raw
traces from the per-pixel cache (Analysis/classification/data/cache/
dataset_per_pixel.npz) instead of exp6 predictions, and skips fields
that don't apply to training pixels (misclassification status, drop
masks).

Output JSONs land in filter_data/ alongside the existing SD chip JSONs
and use the same schema, so the viewer needs no changes.  The manifest
is updated in-place so all 7 chips appear in the chip dropdown.

Usage:
    python -m lacewing.preprocessing.data_quality.export_training_chip_distributions
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lacewing.classification.core import paths as cls_paths
from lacewing.preprocessing.data_quality.build_filter_pipeline import (
    THIS_DIR,
    compute_chip_thresholds,
    apply_layers,
    apply_layer_c,
    apply_layer_d,
    _safe_chip_name,
)
from lacewing.preprocessing.data_quality.spatial_coords import (
    build_coords_for_chip,
)


EXPORT_DIR = THIS_DIR / "filter_data"
D_STRATEGIES = ["mean", "median", "kth"]
DEFAULTS = dict(
    pct_amp=5.0,
    pct_shape=5.0,
    c_k=8,
    c_bad_frac=0.5,
    d_k=8,
    d_pct=95.0,
    ntc_well=cls_paths.NTC_WELL_INDEX,
)


def _per_pixel_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        x.max(axis=1) - x.min(axis=1),
        x[:, -1] - x[:, 0],
        x.min(axis=1),
    )


def _export_chip(
    chip: str,
    x_c: np.ndarray,
    well_c: np.ndarray,
    pix_c: np.ndarray,
    yt_c: np.ndarray,
    out_dir: Path,
) -> dict:
    """Build the same JSON shape that build_filter_pipeline.export_chip_json
    would emit, but without misclassification fields."""
    thresh = compute_chip_thresholds(
        x_c, well_c, DEFAULTS["ntc_well"],
        DEFAULTS["pct_amp"], DEFAULTS["pct_shape"],
        rule="percentile",
    )
    layerA, layerB = apply_layers(x_c, thresh)

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

    layerC, c_score = apply_layer_c(
        rows_c, cols_c, well_c, layerA, layerB,
        k_neighbours=DEFAULTS["c_k"],
        min_bad_frac=DEFAULTS["c_bad_frac"],
    )

    layer_d_strategies: dict[str, np.ndarray] = {}
    layer_d_scores:     dict[str, np.ndarray] = {}
    for metric in D_STRATEGIES:
        drop, score = apply_layer_d(
            x_c, rows_c, cols_c, well_c,
            layerA, layerB, layerC,
            metric=metric,
            k_neighbours=DEFAULTS["d_k"],
            pct_disagree=DEFAULTS["d_pct"],
            rule="percentile",
        )
        layer_d_strategies[metric] = drop
        layer_d_scores[metric] = score

    dyn_range, net_slope, trace_min = _per_pixel_stats(x_c)
    n_pix = len(x_c)

    wells_data: dict[str, dict] = {}
    summary_rows = []
    for w in sorted(set(int(v) for v in well_c)):
        mask = well_c == w
        idx = np.flatnonzero(mask)
        n_total = int(mask.sum())
        n_A     = int(layerA[idx].sum())
        n_B     = int(layerB[idx].sum())
        n_C     = int(layerC[idx].sum())
        n_after_A = n_total - n_A
        n_after_B = n_after_A - n_B
        n_after_C = n_after_B - n_C
        n_D_per_strat = {
            name: int(mask_arr[idx].sum())
            for name, mask_arr in layer_d_strategies.items()
        }

        pixels = []
        for i in idx:
            if   layerA[i]: layer_ab = "A"
            elif layerB[i]: layer_ab = "B"
            else:           layer_ab = None
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
                d_scores[name] = None if v != v else float(v)
            pixels.append({
                "pid":         int(pix_c[i]),
                "trace":       [float(v) for v in x_c[i]],
                "row":         int(rows_c[i]),
                "col":         int(cols_c[i]),
                "dyn_range":   float(dyn_range[i]),
                "net_slope":   float(net_slope[i]),
                "trace_min":   float(trace_min[i]),
                # Training chips have no held-out predictions, so the
                # misclassification fields are stubbed out.  The
                # distribution viewer never reads them.
                "status":      "n/a",
                "n_wrong":     0,
                "layer_ab":    layer_ab,
                "c_dropped":   bool(layerC[i]),
                "c_score":     c_score_val,
                "d_dropped":   d_dropped,
                "d_scores":    d_scores,
            })

        label = int(yt_c[idx[0]]) if len(idx) else -1
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
        "chip":                  chip,
        "thresholds":            thresh,
        "d_strategies":          D_STRATEGIES,
        "summary":               summary_rows,
        "wells":                 wells_data,
        "trace_length_original": int(x_c.shape[1]),
        "trace_length_exported": int(x_c.shape[1]),
        "downsample":            1,
        "training_chip":         True,
    }
    out_path = out_dir / f"{_safe_chip_name(chip)}.json"
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    return {
        "chip":     chip,
        "json":     out_path.name,
        "n_pixels": int(n_pix),
        "n_wells":  len(summary_rows),
    }


def main() -> None:
    EXPORT_DIR.mkdir(exist_ok=True)
    cache_p = cls_paths.cache_path("dataset_per_pixel")
    if not cache_p.exists():
        raise SystemExit(
            f"missing per-pixel cache at {cache_p}.  Build it with "
            "`python -m lacewing.classification.data.build_dataset` first."
        )
    print(f"reading cache: {cache_p}")
    z = np.load(cache_p, allow_pickle=True)
    X        = z["X"]
    y        = z["y"]
    chip_id  = z["chip_id"]
    well_id  = z["well_id"]
    pixel_id = z["pixel_id"]

    training_chips = sorted(set(chip_id))
    print(f"training chips in cache: {training_chips}")

    # Update / extend the existing manifest in filter_data/ so the
    # viewer sees all 7 chips.
    manifest_path = EXPORT_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        existing_chip_names = {entry["chip"] for entry in manifest.get("chips", [])}
    else:
        manifest = {
            "pct_amp":        DEFAULTS["pct_amp"],
            "pct_shape":      DEFAULTS["pct_shape"],
            "d_strategies":   D_STRATEGIES,
            "layer_c_params": {"k": DEFAULTS["c_k"], "min_bad_frac": DEFAULTS["c_bad_frac"]},
            "layer_d_params": {"k": DEFAULTS["d_k"], "pct": DEFAULTS["d_pct"]},
            "ntc_well":       DEFAULTS["ntc_well"],
            "experiment":     None,
            "chips":          [],
            "overall_recall": {},
            "models_excluded": [],
            "n_runs_used":    0,
            "rule":           "percentile",
            "k_mad":          1.645,
        }
        existing_chip_names = set()

    new_entries = []
    for chip in training_chips:
        if chip in existing_chip_names:
            print(f"  {chip}: already in manifest, skipping")
            continue
        m = chip_id == chip
        x_c = X[m]
        well_c = well_id[m]
        pix_c = pixel_id[m]
        yt_c = y[m]
        print(f"  exporting {chip}: {int(m.sum())} pixels, "
              f"wells={sorted(set(int(v) for v in well_c))}")
        entry = _export_chip(chip, x_c, well_c, pix_c, yt_c, EXPORT_DIR)
        new_entries.append(entry)

    if new_entries:
        manifest["chips"].extend(new_entries)
        # Keep a stable ordering: SD chips first (they were exported by
        # the main pipeline), then training chips alphabetically by name.
        manifest["chips"].sort(key=lambda e: (not e.get("training_chip", False), e["chip"]))
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\nupdated manifest: {manifest_path} "
              f"({len(manifest['chips'])} chips total)")
    else:
        print("\nno new chips to export")


if __name__ == "__main__":
    main()
