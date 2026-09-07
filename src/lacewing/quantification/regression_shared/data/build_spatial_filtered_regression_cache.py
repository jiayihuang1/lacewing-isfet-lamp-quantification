"""Apply Layer E (Sweep A spatial smoothing) to the regression cache.

Mirror of lacewing.classification.data.build_spatial_filtered_dataset, but
operating on the regression cache (keeps y_ttp_min + split alongside X).

The RQ1 Week-11 sweep identified Sweep A order 3 (spatA3) as the
spatial-filtering winner, beating the locked MAD k=1.5 ABCD baseline by
+1.03 pp pixel accuracy.  We carry that preprocessing into the
quantification methods so the regression-side comparison stays
apples-to-apples with the new RQ1 SoTA.

Usage::

    python -m lacewing.quantification.regression_shared.data.build_spatial_filtered_regression_cache \\
        --input regress_all_filt_abcd_ntcRaw_madk1p5 --order 3

Output:
    Analysis/regression/data/cache/<input>_spatA<order>.npz
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np

from lacewing.quantification.regression_shared.core import paths
from lacewing.preprocessing.spatial_filter.spatial_smooth import (
    smooth_pixel_traces,
)


def build(input_stem: str, order: int, out_stem: str | None = None):
    src = paths.regression_cache_path(input_stem)
    if not src.exists():
        raise FileNotFoundError(f"Source regression cache not found: {src}")
    print(f"Reading source cache: {src}")
    with np.load(src) as data:
        X = data["X"].astype(np.float32, copy=False)
        y_ttp_min = np.asarray(data["y_ttp_min"])
        chip_id = np.asarray(data["chip_id"])
        well_id = np.asarray(data["well_id"])
        pixel_id = np.asarray(data["pixel_id"])
        split = np.asarray(data["split"])

    print(f"  X.shape = {X.shape}, dtype = {X.dtype}")
    print(f"  unique chips = {len(np.unique(chip_id))}")
    print(f"  pixel count  = {len(X)}")

    log_lines = [
        f"# build_spatial_filtered_regression_cache.py at {datetime.now().isoformat()}",
        f"# source     = {src}",
        f"# order      = {order}  (Sweep A: Chebyshev-distance, 8-neighbour, truncate at edge)",
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
    out_path = paths.regression_cache_path(out_stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X_out,
        y_ttp_min=y_ttp_min,
        chip_id=chip_id,
        well_id=well_id,
        pixel_id=pixel_id,
        split=split,
    )
    log_path = out_path.with_suffix("").with_suffix(".log")
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"Wrote {log_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True,
                   help="Source regression cache stem "
                        "(e.g. regress_all_filt_abcd_ntcRaw_madk1p5).")
    p.add_argument("--order", type=int, required=True,
                   help="Sweep A order; 0 = identity, 1 = 3x3, 2 = 5x5, 3 = 7x7.")
    p.add_argument("--out", default=None,
                   help="Output cache stem (default: <input>_spatA<order>).")
    args = p.parse_args()
    build(args.input, args.order, args.out)


if __name__ == "__main__":
    main()
