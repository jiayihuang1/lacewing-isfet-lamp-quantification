"""Build a TILE-POOLED classification cache from a MAD-filtered one.

Sweep B driver.  Reads an existing MAD-filtered classification cache
(e.g. ``dataset_all_filt_abcd_ntcRaw_madk1p5``), pools each well's
active pixels into non-overlapping (2*order+1) x (2*order+1) tiles
(one super-pixel per tile, trace = mean of the active pixels in that
tile), and writes a new cache with the same schema.

Pipeline:

    raw -> linearise -> idx_active -> MAD ABCD -> TILE POOL (this)
        -> classifier

Pixel count SHRINKS by up to (2*order+1)^2 — supervisor's "we are
losing the number of pixels" interpretation.

Cache name convention:

    <input_stem>_spatB<order>.npz

So for the locked classification baseline:

    dataset_all_filt_abcd_ntcRaw_madk1p5_spatB1.npz   (3x3 tile)
    dataset_all_filt_abcd_ntcRaw_madk1p5_spatB2.npz   (5x5 tile)
    dataset_all_filt_abcd_ntcRaw_madk1p5_spatB3.npz   (7x7 tile)

``order=0`` would be an identity passthrough (same as the locked
baseline), so for Sweep B we only build orders 1..3.

Run::

    python -m lacewing.classification.data.build_spatial_pooled_dataset \\
        --input dataset_all_filt_abcd_ntcRaw_madk1p5 --order 1
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from lacewing.classification.core import paths
from lacewing.preprocessing.spatial_filter.spatial_smooth import (
    pool_into_tiles,
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
        f"# build_spatial_pooled_dataset.py at {datetime.now().isoformat()}",
        f"# source     = {src}",
        f"# order      = {order}  -> non-overlapping {2*order+1}x{2*order+1} tiles",
        f"# n_pixels_in = {len(X)}",
        "",
    ]

    X_o, y_o, chip_o, well_o, tid_o = pool_into_tiles(
        X=X, y=y, chip_id=chip_id, well_id=well_id, pixel_id=pixel_id,
        order=order, log_lines=log_lines,
    )

    out_stem = out_stem or f"{input_stem}_spatB{order}"
    out_path = paths.cache_path(out_stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
                        X=X_o, y=y_o, chip_id=chip_o,
                        well_id=well_o, pixel_id=tid_o)
    log_path = out_path.with_suffix("").with_suffix(".log")
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"  out X.shape = {X_o.shape}")
    print(f"  pixel count: {len(X)} -> {len(X_o)} "
          f"(ratio {len(X)/max(len(X_o),1):.2f}x)")
    print(f"Wrote {log_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="Source cache stem (e.g. dataset_all_filt_abcd_ntcRaw_madk1p5).")
    p.add_argument("--order", type=int, required=True,
                   help="Tile order; 1 = 3x3, 2 = 5x5, 3 = 7x7.")
    p.add_argument("--out", default=None,
                   help="Output cache stem (default: <input>_spatB<order>).")
    args = p.parse_args()
    build(args.input, args.order, args.out)


if __name__ == "__main__":
    main()
