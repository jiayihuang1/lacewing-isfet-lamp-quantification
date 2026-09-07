"""B2a — build a per-pixel classifier cache for the UTI trial chips.

Runs the same multi-titan preprocessing pipeline used for the segmentation
cache (BS + MAD-ABCD QC k=1.5 + spatA3), on ALL 4 UTI chips and ALL wells
(including NC and no-amp wells), then writes a flat (N_pixels, 450) matrix
compatible with the exp6 COV classifier's expected input.

Per-well ground-truth amp+ / amp- labels are taken from manual_chip_ttps.json:
  y = 1 if well was labelled and NOT is_no_amp
  y = 0 if well was labelled and is_no_amp (NC / low-conc no-amp)
Wells not present in manual_chip_ttps.json are excluded (can't score).

Output:
    Analysis/quantification/chip_pipeline/output/uti_classifier_cache.npz
      X        (N, 450) float32   BS + QC + spatA3 per-pixel trace
      y        (N,)     int8      per-pixel label (broadcast per-well truth)
      chip_tag (N,)     <U40
      well_id  (N,)     int32
      row      (N,)     int32     chip row
      col      (N,)     int32     chip col

Usage:
    python -m lacewing.quantification.chip_pipeline.build_uti_classifier_cache
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lacewing.quantification.chip_pipeline import process_chip as pipe
from lacewing.quantification.chip_pipeline.process_all_uti_chips import CHIPS
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[3]
OUT_PATH = LACEWING_PKG_DIR / "quantification" / "chip_pipeline" / "output" / "uti_classifier_cache.npz"
MANUAL_LABELS = LACEWING_PKG_DIR / "quantification" / "chip_pipeline" / "output" / "manual_chip_ttps.json"


def _load_well_truth() -> dict[tuple[str, int], int]:
    """Return {(chip_tag, well_id) -> 0/1} from manual labels."""
    labels = json.loads(MANUAL_LABELS.read_text())["labels"]
    truth = {}
    for v in labels.values():
        truth[(v["chip_tag"], int(v["well_id"]))] = 0 if v.get("is_no_amp") else 1
    return truth


def main() -> None:
    well_truth = _load_well_truth()
    print(f"Loaded {len(well_truth)} labelled wells from manual_chip_ttps.json")

    all_X, all_y, all_chip, all_well, all_row, all_col = [], [], [], [], [], []

    for cfg in CHIPS:
        print(f"\n{'=' * 70}\n{cfg.chip_tag}\n{'=' * 70}")
        if not cfg.chip_dir.exists():
            print(f"  [SKIP] chip_dir missing")
            continue

        # Monkey-patch pipe globals for this chip.
        pipe.WELL_LABELS = cfg.well_labels
        pipe.WELL_LOG10 = cfg.well_log10
        pipe.NTC_WELLS = tuple(cfg.ntc_wells)
        pipe.TRIM_SEARCH_MAX_MIN = cfg.trim_search_max_min

        result = pipe.process_chip(cfg.chip_dir, save_per_pixel=True)

        for w in result.wells:
            key = (cfg.chip_tag, int(w.well))
            if key not in well_truth:
                continue
            if w.per_pixel_spat is None or len(w.per_pixel_spat) == 0:
                print(f"  well {w.well}: no per-pixel data, skip")
                continue

            X = w.per_pixel_spat.astype(np.float32, copy=False)
            n = X.shape[0]
            # Pad / truncate to 450 samples to match classifier input.
            T = X.shape[1]
            if T > 450:
                X = X[:, :450]
            elif T < 450:
                pad = np.zeros((n, 450 - T), dtype=np.float32)
                X = np.concatenate([X, pad], axis=1)

            y = np.full(n, well_truth[key], dtype=np.int8)
            all_X.append(X)
            all_y.append(y)
            all_chip.append(np.full(n, cfg.chip_tag, dtype="<U40"))
            all_well.append(np.full(n, int(w.well), dtype=np.int32))
            all_row.append(w.per_pixel_row.astype(np.int32, copy=False))
            all_col.append(w.per_pixel_col.astype(np.int32, copy=False))
            print(f"  well {w.well:2d}: {n:5d} pixels  y={well_truth[key]}  label={w.label}")

    if not all_X:
        raise RuntimeError("no pixels collected")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    chip_tag = np.concatenate(all_chip, axis=0)
    well_id = np.concatenate(all_well, axis=0)
    row = np.concatenate(all_row, axis=0)
    col = np.concatenate(all_col, axis=0)

    print(f"\n{'=' * 70}")
    print(f"cache: N={X.shape[0]} T={X.shape[1]} pos={(y == 1).sum()} neg={(y == 0).sum()}")
    print(f"  unique wells: {len(set(zip(chip_tag.tolist(), well_id.tolist())))}")
    print(f"  chip breakdown:")
    for c in sorted(np.unique(chip_tag)):
        m = chip_tag == c
        pos = int((y[m] == 1).sum())
        neg = int((y[m] == 0).sum())
        wells = len(set(zip(chip_tag[m].tolist(), well_id[m].tolist())))
        print(f"    {c}: {m.sum()} px ({pos} pos, {neg} neg) across {wells} wells")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        X=X, y=y, chip_tag=chip_tag, well_id=well_id, row=row, col=col,
    )
    print(f"\n[ok] wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
