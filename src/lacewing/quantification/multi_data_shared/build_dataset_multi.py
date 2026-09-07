"""Build the per-pixel cached dataset for the Data/Multi/ test chips.

Mirrors ``Analysis/classification/data/build_dataset.py`` exactly in
output schema and pixel-trace semantics, but loads via the
Matthew_Multi titan + the multi-Vref union-combine + the early-time
artefact trim:

    1. ``load_chip_combined`` -> Experiment with per-Vref pixels unioned
    2. ``trim_chip_artefact`` -> push idx_settled / idx_start past the
       early-time artefact (chip-uniform cut)
    3. For each well in the chip's label map, take per-pixel signals
       from the new t=0 (= post-trim onset), WINDOW samples, baseline-
       subtracted per pixel (start each at 0 from the new onset).

Two well-layout maps are baked in (see ``_labels.py``): Elena two-target
(NCS, NCC are NTC) and the P/N grid (under the pragmatic NTC fallback
pending supervisor clarification on whether N wells are NTC vs negative
samples).

Output: Analysis/multi_data/cache/<name>.npz with arrays

    X         (N_pixels, WINDOW)    float32   - per-pixel trace
    y         (N_pixels,)           uint8     - 1 = pos, 0 = NTC/negative
    chip_id   (N_pixels,)           <U64      - chip folder name
    well_id   (N_pixels,)           uint8     - well index 0..9
    pixel_id  (N_pixels,)           uint16    - index within active pixels

Run::

    python -m lacewing.quantification.multi_data_shared.build_dataset_multi
    python -m lacewing.quantification.multi_data_shared.build_dataset_multi --out my_cache
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

# Path setup so titan resolves to titan-signal-processing-multi.
from . import _titan_setup  # noqa: F401

from ._labels import N_WELLS_MULTI, label_per_well
from .load_combined import (DEFAULT_END_TIME_MIN, DEFAULT_N_A_TYPE,
                             load_chip_combined)
from .artefact_trim import trim_chip_artefact
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


# Window length copied from Analysis/classification/data/build_dataset.py.
WINDOW = 450

# NOTE: the Matthew_Multi titan's `well_2d` has odd units: absolute
# values are mV-like (first sample ~2350) but baseline-subtracted
# changes are already V-magnitude (~0.05 V swing during amplification),
# matching what the Final-chip pipeline produces.  Because every
# downstream consumer here baseline-subtracts the trace first, the
# scale already matches the training-time convention -- no extra
# rescale needed.  Don't introduce a /1000 here; we tried and the
# resulting cache had ~1e-5 amplitude which makes the trained models
# output noise.

# NOTE on QC: the existing Final-chip pipeline applies a QC mask in
# `build_dataset.py:_qc_mask` (finite samples + first-sample voltage in
# [0.5, 5.0] V + signal std > 1e-3).  Per user instruction 2026-06-10,
# QC is SKIPPED entirely on the Multi data — we rely only on the
# union-of-Vrefs active mask exposed by `load_chip_combined`, which is
# already the firmware-active mask AND-ed with titan's per-Vref
# gain+lin filter.  This is more permissive than the Final pipeline;
# the writeup should note that any per-pixel quality filtering on
# Multi happens only via the multi-Vref combine step.

PROJECT_ROOT = DATA_ROOT  # was: parents[2]
DATA_MULTI_ROOT = PROJECT_ROOT / "Data" / "Multi"
CACHE_DIR = Path(__file__).resolve().parent / "cache"


# ---------------------------------------------------------------------------
# Per-pixel extraction (mirrors build_dataset.py _extract_per_pixel_traces
# but without the QC mask, and with the mV->V rescale on well_2d).
# ---------------------------------------------------------------------------

def _extract_per_pixel_traces(well, loose_mask: np.ndarray,
                              window: int = WINDOW
                              ) -> tuple[np.ndarray, int, float]:
    """Same shape semantics as build_dataset.py _extract_per_pixel_traces.

    Differences vs the Final-chip pipeline:
      * `well_2d` is rescaled by 1/WELL_2D_MV_TO_V so values are in
        VOLTS (matching the training-time unit convention).
      * No QC mask.  We trust the union-of-Vrefs active mask from
        ``load_chip_combined`` (= firmware-active AND per-Vref gain+lin
        filter) as the only pixel-level quality gate.

    The trim step has already pushed `idx_settled` past the early
    artefact, so `time_min[0] == 0` is the trimmed onset.
    """
    full_all = np.asarray(well.well_2d, dtype=np.float32)
    keep = np.asarray(loose_mask, dtype=bool)
    full = full_all[:, keep]                                # (T, N_keep)

    time_min = np.asarray(well.time_min, dtype=np.float64)
    onset_idx = int(np.searchsorted(time_min, 0.0))
    full = full[onset_idx:, :]                              # (T_post, N)
    if full.shape[0] == 0:
        return (np.empty((0, window), dtype=np.float32), 0, 0.0)
    full = full - full[0:1, :]                              # baseline-subtract per pixel

    T_post = full.shape[0]
    if T_post >= window:
        X = full[:window, :].T                              # (N, window)
        n_pad = 0
        span_min = float(time_min[onset_idx + window - 1] -
                         time_min[onset_idx])
    else:
        n_pad = window - T_post
        pad = np.zeros((n_pad, full.shape[1]), dtype=np.float32)
        X = np.concatenate([full, pad], axis=0).T           # (N, window)
        span_min = float(time_min[-1] - time_min[onset_idx])
    return X, n_pad, span_min


# ---------------------------------------------------------------------------
# Per-chip driver
# ---------------------------------------------------------------------------

def _process_chip(chip_path: Path, log_lines: list[str]):
    """Return (X, y, well_ids, pix_ids, ntc_well_ids) or None."""
    print(f"  Loading {chip_path.name} ...")
    try:
        exp, diag = load_chip_combined(
            chip_path, n_wells=N_WELLS_MULTI,
            n_a_type=DEFAULT_N_A_TYPE,
            end_time_min=DEFAULT_END_TIME_MIN,
            print_status=False,
        )
    except Exception as e:
        msg = (f"    SKIP {chip_path.name}: titan/load_combined failed "
               f"({type(e).__name__}: {e!s:.200})")
        print(msg); log_lines.append(msg)
        return None

    try:
        trim = trim_chip_artefact(exp, apply=True)
    except Exception as e:
        msg = (f"    SKIP {chip_path.name}: trim_chip_artefact failed "
               f"({type(e).__name__}: {e!s:.200})")
        print(msg); log_lines.append(msg)
        return None
    log_lines.append(
        f"    n_refs={diag['n_refs']}, chip cut={trim.chip_cut_min:.3f} min, "
        f"per-well cut min={[round(x, 2) for x in trim.per_well_cut_min]}"
    )

    labels = label_per_well(chip_path.name)
    Xs, ys, well_ids, pix_ids = [], [], [], []
    for w_idx in sorted(labels.keys()):
        if w_idx >= len(exp.wells_list):
            log_lines.append(
                f"    WARNING {chip_path.name} well {w_idx}: out of range "
                f"({len(exp.wells_list)} wells), skipping")
            continue
        well = exp.wells_list[w_idx]

        # The union-of-Vref active mask lives on the well's idx_active
        # (set by load_chip_combined).  Use it as the loose mask.
        loose_mask = np.asarray(well.idx_active, dtype=bool)

        X, n_pad, span = _extract_per_pixel_traces(well, loose_mask)
        if X.shape[0] == 0:
            log_lines.append(
                f"    WARNING {chip_path.name} well {w_idx}: 0 pixels survive "
                f"loose-active + QC, skipping")
            continue
        label = labels[w_idx]
        log_lines.append(
            f"    well {w_idx} (label={label}): {X.shape[0]} pixels, "
            f"{X.shape[1]} samples, span {span:.2f} min, padded {n_pad}")

        Xs.append(X)
        ys.append(np.full(X.shape[0], label, dtype=np.uint8))
        well_ids.append(np.full(X.shape[0], w_idx, dtype=np.uint8))
        pix_ids.append(np.arange(X.shape[0], dtype=np.uint16))

    if not Xs:
        return None
    return (np.concatenate(Xs, axis=0),
            np.concatenate(ys, axis=0),
            np.concatenate(well_ids, axis=0),
            np.concatenate(pix_ids, axis=0))


# Chips to exclude from the enumeration.  Note the *substring*, not the
# full folder name -- matched via `in chip_path.name`.
EXCLUDED_CHIP_SUBSTRINGS: tuple[str, ...] = (
    # jh1125 2026-06-10: short trace (~349 frames) -> WINDOW=450 forces
    # ~26% of samples to be zero-padded.  Models trained on Final see
    # full 450 samples; padding makes the input mostly zeros over the
    # last ~7 min and predictions are not meaningful.  Skip for now.
    "manifold_test_05",
)


def _enumerate_chips() -> list[Path]:
    chips = sorted([p for p in DATA_MULTI_ROOT.iterdir()
                     if p.is_dir() and not p.name.startswith(".")])
    return [p for p in chips
            if not any(s in p.name for s in EXCLUDED_CHIP_SUBSTRINGS)]


# ---------------------------------------------------------------------------
# Build entrypoint
# ---------------------------------------------------------------------------

def build(out_name: str = "multi_raw") -> Path:
    log_lines = [
        f"# build_dataset_multi.py run at {datetime.now().isoformat()}",
        f"# WINDOW={WINDOW}, n_a_type={DEFAULT_N_A_TYPE}, "
        f"end_time_min={DEFAULT_END_TIME_MIN}",
        f"# data_root={DATA_MULTI_ROOT}",
        "",
    ]

    chips = _enumerate_chips()
    log_lines.append(f"## Found {len(chips)} chip folders")
    print(f"Found {len(chips)} chip folders under {DATA_MULTI_ROOT}")

    all_X, all_y, all_chip, all_well, all_pix = [], [], [], [], []
    for chip_path in chips:
        log_lines.append(f"\n### {chip_path.name}")
        out = _process_chip(chip_path, log_lines)
        if out is None:
            continue
        X, y, well_ids, pix_ids = out
        all_X.append(X); all_y.append(y)
        all_chip.append(np.full(X.shape[0], chip_path.name, dtype="<U64"))
        all_well.append(well_ids); all_pix.append(pix_ids)

    if not all_X:
        raise RuntimeError("No data survived loading on any Multi chip.")

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
    print(summary); log_lines.append(summary)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{out_name}.npz"
    np.savez_compressed(out_path,
                        X=X, y=y, chip_id=chip, well_id=well, pixel_id=pix)
    log_path = CACHE_DIR / f"{out_name}_log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {log_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="multi_raw",
                        help="Output cache stem (default: multi_raw)")
    args = parser.parse_args()
    build(out_name=args.out)


if __name__ == "__main__":
    main()
