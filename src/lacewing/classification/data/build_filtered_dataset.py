"""Build a per-pixel dataset cache with the four-layer chip-relative
filter applied per chip.

This is a filter-aware wrapper around ``build_dataset.build``: it
runs the standard build, then re-loads each chip via the spatial-
coordinate extractor in
``lacewing.preprocessing.data_quality.spatial_coords`` so it can
attach (row, col) coordinates to every pixel, applies the requested
subset of filter layers per chip, and writes a filtered cache.

Caller specifies:
  - ``scope``  ('final' or 'all')
  - ``layers`` ('A' / 'AB' / 'ABC' / 'ABCD')
  - ``clean_ntc_first`` (bool)
  - ``out_name`` cache stem (defaults to a sensible name based on
    layers + ntc treatment)

The resulting cache has the same schema as ``dataset_per_pixel.npz``
(``X``, ``y``, ``chip_id``, ``well_id``, ``pixel_id``) but contains
only the surviving (filter-passing) pixels.

Run::

    python -m lacewing.classification.data.build_filtered_dataset \\
        --scope final --layers ABCD --clean-ntc

CPU only.  Per-chip cost is dominated by titan load + linearisation
(roughly 10-20 s per chip on a laptop).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from lacewing.classification.core import paths
from lacewing.classification.data import build_dataset as bd
from lacewing.preprocessing.data_quality import filter_core
from lacewing.preprocessing.data_quality.spatial_coords import (
    build_coords_for_chip,
)


# --------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------

DEFAULT_PCT_AMP   = 5.0
DEFAULT_PCT_SHAPE = 5.0
DEFAULT_C_K       = 8
DEFAULT_C_BAD_FR  = 0.5
DEFAULT_D_K       = 8
DEFAULT_D_PCT     = 95.0
DEFAULT_D_METRIC  = "median"


# --------------------------------------------------------------------
# Per-chip processing: load via build_dataset, attach coords, filter
# --------------------------------------------------------------------

def _filter_one_chip(
    chip_path: Path,
    label_per_well: dict[int, int],
    layers: str,
    clean_ntc_first: bool,
    log_lines: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Returns (X_kept, y_kept, well_ids_kept, pixel_ids_kept) or None on failure."""
    # 1. Reuse the existing extraction (loose_mask + QC + onset trim).
    result = bd._process_chip(chip_path, label_per_well, log_lines)
    if result is None:
        return None
    X, y, well_ids, pix_ids = result

    # The chip-relative filter needs the NTC well as its reference.
    # Skip chips whose NTC well didn't survive titan + QC.
    if not (well_ids == paths.NTC_WELL_INDEX).any():
        msg = (f"    SKIP {chip_path.name}: no NTC pixels surviving QC "
               f"(well {paths.NTC_WELL_INDEX}); cannot compute "
               "chip-relative thresholds")
        print(msg); log_lines.append(msg)
        return None

    # 2. Spatial coords for the same pixel set.  build_coords_for_chip
    # reloads titan and recomputes the mask; we then index its
    # per-well arrays with pix_ids to align with our X rows.
    try:
        coords = build_coords_for_chip(chip_path.name)
    except (KeyError, FileNotFoundError, ValueError) as e:
        msg = (f"    SKIP {chip_path.name}: spatial coords unavailable "
               f"({type(e).__name__}: {e!s:.200})")
        print(msg); log_lines.append(msg)
        return None

    rows = np.empty(len(pix_ids), dtype=np.int32)
    cols = np.empty(len(pix_ids), dtype=np.int32)
    for w in sorted(set(int(v) for v in well_ids)):
        wm = well_ids == w
        if w not in coords:
            msg = (f"    SKIP {chip_path.name} well {w}: not in coords map.")
            print(msg); log_lines.append(msg)
            return None
        pids = pix_ids[wm].astype(np.int64)
        # pix_ids index into the per-well surviving-pixel list (same
        # construction used by both _process_chip and
        # build_coords_for_chip), so indexing is direct.
        if pids.max() >= len(coords[w]["rows"]):
            msg = (f"    SKIP {chip_path.name} well {w}: pix_id out of range "
                   f"({pids.max()} >= {len(coords[w]['rows'])})")
            print(msg); log_lines.append(msg)
            return None
        rows[wm] = coords[w]["rows"][pids]
        cols[wm] = coords[w]["cols"][pids]

    # 3. Apply the filter.
    keep_mask, info = filter_core.run_filter_pipeline_per_chip(
        X, well_ids, rows, cols,
        ntc_well=paths.NTC_WELL_INDEX,
        layers=layers,
        clean_ntc_first=clean_ntc_first,
        pct_amp=DEFAULT_PCT_AMP, pct_shape=DEFAULT_PCT_SHAPE,
        c_k=DEFAULT_C_K, c_min_bad_frac=DEFAULT_C_BAD_FR,
        d_k=DEFAULT_D_K, d_pct=DEFAULT_D_PCT, d_metric=DEFAULT_D_METRIC,
    )

    drop_info = info["n_dropped"]
    log_lines.append(
        f"    filter {layers}/ntc={'D' if clean_ntc_first else 'raw'}:"
        f" kept {info['n_kept']}/{info['n_total']}  "
        f"drops A={drop_info['A']} B={drop_info['B']} "
        f"C={drop_info['C']} D={drop_info['D']} "
        f"ntc_D={drop_info['ntc_D']}")

    return X[keep_mask], y[keep_mask], well_ids[keep_mask], pix_ids[keep_mask]


# --------------------------------------------------------------------
# Top-level build
# --------------------------------------------------------------------

def build_filtered(
    scope: str,
    layers: str,
    clean_ntc_first: bool,
    out_name: str,
) -> Path:
    log_lines = [
        f"# build_filtered_dataset.py run at {datetime.now().isoformat()}",
        f"# scope={scope}, layers={layers}, clean_ntc_first={clean_ntc_first}",
        f"# WINDOW={bd.WINDOW}, N_A_TYPE={bd.N_A_TYPE}, END_TIME_MIN={bd.END_TIME_MIN}",
        f"# filter defaults: pct_amp={DEFAULT_PCT_AMP}, pct_shape={DEFAULT_PCT_SHAPE}, "
        f"c_k={DEFAULT_C_K}, c_min_bad_frac={DEFAULT_C_BAD_FR}, "
        f"d_k={DEFAULT_D_K}, d_pct={DEFAULT_D_PCT}, d_metric={DEFAULT_D_METRIC}",
        "",
    ]

    all_X, all_y, all_chip, all_well, all_pix = [], [], [], [], []

    if scope == "final":
        chips_with_labels: list[tuple[Path, dict[int, int]]] = []
        for label_str, chip_path in paths.FINAL_CHIPS.items():
            log_lines.append(f"\n### {label_str}: {chip_path.name}")
            label_map = {w: 1 for w in paths.POSITIVE_WELL_INDICES}
            label_map[paths.NTC_WELL_INDEX] = 0
            chips_with_labels.append((chip_path, label_map))

    elif scope == "all":
        all_chips = bd._enumerate_all_chips()
        log_lines.append(f"## All-data scope: {len(all_chips)} chip folders")
        std_label_map = {w: 1 for w in paths.POSITIVE_WELL_INDICES}
        std_label_map[paths.NTC_WELL_INDEX] = 0
        chips_with_labels = [(p, std_label_map) for p in all_chips]

    else:
        raise ValueError(f"Unknown scope: {scope}")

    for chip_path, label_map in chips_with_labels:
        log_lines.append(f"\n### {chip_path.name}")
        out = _filter_one_chip(chip_path, label_map, layers,
                               clean_ntc_first, log_lines)
        if out is None:
            continue
        X, y, well_ids, pix_ids = out
        if X.shape[0] == 0:
            log_lines.append(f"    SKIP {chip_path.name}: no pixels survived filter")
            continue
        all_X.append(X); all_y.append(y)
        all_chip.append(np.full(X.shape[0], chip_path.name, dtype="<U64"))
        all_well.append(well_ids); all_pix.append(pix_ids)

    if not all_X:
        raise RuntimeError("No data survived the filter on any chip.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    chip = np.concatenate(all_chip, axis=0)
    well = np.concatenate(all_well, axis=0)
    pix = np.concatenate(all_pix, axis=0)

    n_chips = len(np.unique(chip))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    summary = (f"\n# TOTAL: {len(y)} pixels from {n_chips} chips "
               f"({n_pos} pos / {n_neg} neg)\n"
               f"# X.shape = {X.shape}  X.dtype = {X.dtype}\n")
    print(summary)
    log_lines.append(summary)

    out_path = paths.cache_path(out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
                        X=X, y=y, chip_id=chip, well_id=well, pixel_id=pix)
    log_path = out_path.parent / f"{out_name}_preprocessing_log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {log_path}")
    return out_path


# --------------------------------------------------------------------
# Naming convention for filtered caches
# --------------------------------------------------------------------

def default_out_name(scope: str, layers: str, clean_ntc_first: bool) -> str:
    ntc = "ntcD" if clean_ntc_first else "ntcRaw"
    return f"dataset_{scope}_filt_{layers.lower()}_{ntc}"


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["final", "all"], default="all",
                        help="which chips to include (default 'all')")
    parser.add_argument("--layers", choices=["A", "AB", "ABC", "ABCD"],
                        default="ABCD",
                        help="which filter layers to apply on positive wells")
    parser.add_argument("--clean-ntc", action="store_true",
                        help="run Layer D on NTC first and remove broken "
                             "NTC pixels from the cache (also affects A/B "
                             "thresholds since they're recomputed on the "
                             "cleaned NTC)")
    parser.add_argument("--out-name", default=None,
                        help="override the cache stem (default constructed "
                             "from layers and ntc treatment)")
    args = parser.parse_args()

    out_name = args.out_name or default_out_name(
        args.scope, args.layers, args.clean_ntc)
    build_filtered(
        scope=args.scope,
        layers=args.layers,
        clean_ntc_first=args.clean_ntc,
        out_name=out_name,
    )


if __name__ == "__main__":
    main()
