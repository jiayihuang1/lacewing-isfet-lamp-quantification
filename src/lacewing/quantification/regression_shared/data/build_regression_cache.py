"""Build the per-pixel TTP regression cache.

Reads the locked classification cache (MAD k=1.5 ABCD ntcRaw by
default) and relabels each surviving pixel with the well-mean qLAMP
TTP from lacewing.quantification.regression_shared.data.labels.  Drops pixels whose label
is NaN (e.g. PTC, NTC, non-train/test chips), so the output cache
contains exactly the pixels we can train or evaluate on.

Output schema (Analysis/regression/data/cache/<stem>.npz):
    X         (N, WINDOW)      float32 — per-pixel trace (unchanged from source)
    y_ttp_min (N,)             float32 — well-mean qLAMP TTP in minutes
    chip_id   (N,)             <U64    — chip folder name (unchanged)
    well_id   (N,)             uint8   — well index 0..3 (dose-response) or 0..4 (SD)
    pixel_id  (N,)             uint16  — pixel index within active set (unchanged)
    split     (N,)             uint8   — 0 = train (5 dose-response chips, wells 0-3)
                                         1 = test  (SD chip, wells 0-4 by dilution)

Run:
    python -m lacewing.quantification.regression_shared.data.build_regression_cache
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np

from lacewing.quantification.regression_shared.core import paths
from lacewing.quantification.regression_shared.data.labels import label_pixels


def build(source_stem: str = paths.LOCKED_CACHE_STEM,
          out_stem: str | None = None) -> tuple[int, int]:
    """Build the regression cache.

    Returns (n_train_pixels, n_test_pixels).
    """
    src = paths.classification_cache_path(source_stem)
    if not src.exists():
        raise FileNotFoundError(
            f"Source classification cache not found: {src}. "
            f"Build it first via `python -m lacewing.classification.data.build_filtered_dataset_mad --scope all --layers ABCD --k 1.5`."
        )

    print(f"Reading source cache: {src}")
    with np.load(src) as data:
        X = data["X"].astype(np.float32, copy=False)
        chip_id = data["chip_id"]
        well_id = data["well_id"]
        pixel_id = data["pixel_id"]

    print(f"  X.shape = {X.shape}, dtype = {X.dtype}")
    print(f"  unique chips = {len(np.unique(chip_id))}")

    y_ttp, split = label_pixels(chip_id, well_id)

    keep = ~np.isnan(y_ttp)
    n_dropped = int((~keep).sum())
    print(f"  Dropping {n_dropped} pixels with no TTP label "
          f"(NTC / PTC / other chips).")

    X        = X[keep]
    chip_id  = chip_id[keep]
    well_id  = well_id[keep]
    pixel_id = pixel_id[keep]
    y_ttp    = y_ttp[keep]
    split    = split[keep]

    n_train = int((split == 0).sum())
    n_test  = int((split == 1).sum())
    print(f"  After dropping: {len(X)} pixels.")
    print(f"  train pixels = {n_train}  (5 dose-response chips x wells 0-3)")
    print(f"  test  pixels = {n_test}   (SD chip x dilution wells)")

    # Per-chip summary so we can sanity-check well/label coverage.
    print("\n  Per-chip pixel count + label distribution:")
    for chip in sorted(np.unique(chip_id).tolist()):
        m = chip_id == chip
        wells = sorted(np.unique(well_id[m]).tolist())
        ttps = sorted({round(float(t), 3) for t in y_ttp[m]})
        sp = int(split[m][0])
        sp_str = {0: "train", 1: "test"}[sp]
        print(f"    {chip:<48}  {int(m.sum()):>6} px  split={sp_str:<5} "
              f"wells={wells}  ttps={ttps}")

    if out_stem is None:
        # Mirror the source stem but prefix with 'regress_' and strip
        # the leading 'dataset_' so the name is clean.
        out_stem = "regress_" + source_stem.replace("dataset_", "")
    out_path = paths.regression_cache_path(out_stem)
    np.savez_compressed(
        out_path,
        X=X,
        y_ttp_min=y_ttp,
        chip_id=chip_id,
        well_id=well_id,
        pixel_id=pixel_id,
        split=split,
    )
    log_lines = [
        f"# build_regression_cache.py at {datetime.now().isoformat()}",
        f"# source = {src}",
        f"# n_train = {n_train}",
        f"# n_test  = {n_test}",
        "",
        "# per-chip ttps and well coverage above; see build log.",
    ]
    log_path = out_path.with_suffix("").with_suffix(".log")
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"Wrote {log_path}")
    return n_train, n_test


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=paths.LOCKED_CACHE_STEM,
                   help="Source classification cache stem.")
    p.add_argument("--out", default=None,
                   help="Output regression cache stem.")
    args = p.parse_args()
    build(args.source, args.out)


if __name__ == "__main__":
    main()
