"""Build a spatially-smoothed classification cache from a MAD-filtered one.

Sweep A driver.  Reads an existing MAD-filtered classification cache
(e.g. ``dataset_all_filt_abcd_ntcRaw_madk1p5``), applies per-pixel
spatial smoothing (mean over Chebyshev-distance-``order`` 8-neighbour
active neighbours), and writes a new cache with the SAME schema.

Pipeline:

    raw -> linearise -> idx_active -> MAD ABCD -> SPATIAL SMOOTH (this)
        -> classifier

Cache name convention:

    <input_stem>_spatA<order>.npz

So for the locked classification baseline:

    dataset_all_filt_abcd_ntcRaw_madk1p5.npz  (input, order 0 baseline)
    dataset_all_filt_abcd_ntcRaw_madk1p5_spatA1.npz
    dataset_all_filt_abcd_ntcRaw_madk1p5_spatA2.npz
    dataset_all_filt_abcd_ntcRaw_madk1p5_spatA3.npz

``order=0`` is a pass-through copy so the sweep grid can include it
uniformly.

Run::

    python -m lacewing.classification.data.build_spatial_filtered_dataset \\
        --input dataset_all_filt_abcd_ntcRaw_madk1p5 --order 1
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from lacewing.classification.core import paths
from lacewing.preprocessing.spatial_filter.spatial_smooth import (
    smooth_pixel_traces,
)


def build(input_stem: str, order: int, out_stem: str | None = None) -> Path:
    src = paths.cache_path(input_stem)
    if not src.exists():
        raise FileNotFoundError(f"Source cache not found: {src}")
    print(f"Reading source cache: {src}")
    with np.load(src) as data:
        X = data["X"].astype(np.float32, copy=False)
        y = np.asarray(data["y"])
        chip_id = np.asarray(data["chip_id"])
        well_id = np.asarray(data["well_id"])
        pixel_id = np.asarray(data["pixel_id"])

    print(f"  X.shape = {X.shape}, dtype = {X.dtype}")
    print(f"  unique chips = {len(np.unique(chip_id))}")
    print(f"  pixel count  = {len(X)}")

    log_lines = [
        f"# build_spatial_filtered_dataset.py at {datetime.now().isoformat()}",
        f"# source     = {src}",
        f"# order      = {order}  (Chebyshev-distance, 8-neighbour, truncate at edge)",
        f"# n_pixels   = {len(X)}",
        f"# X.shape    = {X.shape}",
        "",
    ]

    X_out = smooth_pixel_traces(
        X=X, chip_id=chip_id, well_id=well_id, pixel_id=pixel_id,
        order=order, log_lines=log_lines,
    )
    assert X_out.shape == X.shape, "Sweep A must preserve pixel count + window."

    out_stem = out_stem or f"{input_stem}_spatA{order}"
    out_path = paths.cache_path(out_stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
                        X=X_out, y=y, chip_id=chip_id,
                        well_id=well_id, pixel_id=pixel_id)
    log_path = out_path.with_suffix("").with_suffix(".log")
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"Wrote {log_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="Source cache stem (e.g. dataset_all_filt_abcd_ntcRaw_madk1p5).")
    p.add_argument("--order", type=int, required=True,
                   help="Spatial order; 0 = identity passthrough, 1 = 3x3, 2 = 5x5, 3 = 7x7.")
    p.add_argument("--out", default=None,
                   help="Output cache stem (default: <input>_spatA<order>).")
    args = p.parse_args()
    build(args.input, args.order, args.out)


if __name__ == "__main__":
    main()
