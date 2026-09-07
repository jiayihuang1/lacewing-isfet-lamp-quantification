"""Build the per-pixel cached dataset used by every Paper 3 model.

Pipeline (per Paper 3 §IV.A, applied per chip then pooled):
    1. titan_load_and_preprocessing
    2. OVERRIDE titan's strict per-experiment active mask with the
       firmware-level mask + Paper-3-style QC (finite, V0 in [0.5, 5],
       std > 1e-3). Recovers ~600-3000 active pixels per well.
    3. For each well, take per-pixel signals from idx_start (time=0),
       450 samples, trapped-charge subtracted.

Two scopes supported:
    --scope final  (default)  Final 5 chips wells 0-4 = positive, well 5 = NTC,
                              + NTC wells from Others/ chips listed in inventory.
    --scope all               Every chip under DATA_ROOT that titan can load.
                              Wells 0-4 = positive, well 5 = NTC for all chips.
                              Chips marked classification='non_ntc' in the
                              inventory are still included (positives only).

Output: data/cache/<name>.npz (default name follows scope) with arrays
    X         (N_pixels, WINDOW)    float32   - per-pixel trace
    y         (N_pixels,)           uint8     - 1 = pos, 0 = neg
    chip_id   (N_pixels,)           <U64      - chip folder name
    well_id   (N_pixels,)           uint8     - well index 0..5
    pixel_id  (N_pixels,)           uint16    - index within active pixels

Run:
    python -m lacewing.classification.data.build_dataset                    # final
    python -m lacewing.classification.data.build_dataset --scope all        # all loadable chips
    python -m lacewing.classification.data.build_dataset --out my_cache     # explicit name
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lacewing.classification import paths  # noqa: E402
from lacewing.classification.data.ntc_inventory import load_ntc_chips  # noqa: E402

try:
    import titan.load_functions as titan_load  # noqa: E402
    import titan.preprocessing_functions as titan_preprocess  # noqa: E402
    from titan.load_and_preprocessing import titan_load_and_preprocessing  # noqa: E402
except ImportError:
    titan_load = titan_preprocess = None
    def titan_load_and_preprocessing(*args, **kwargs):
        raise RuntimeError(
            "titan-signal-processing not installed. Set LACEWING_TITAN_PATH "
            "or install the titan clone. Cache-building requires it; "
            "analysis of pre-built caches does not."
        )


WINDOW = 450                # Paper 3 §IV.A: first 450 samples after onset
N_A_TYPE = "v01"            # matches Analysis/quantification/quantification.py
END_TIME_MIN = 40
DEFAULT_START_TYPE = "temperature"
# titan start_type. "temperature" lets titan find idx_settled by spotting
# when the temperature plateau begins, which skips the reference-electrode
# pulse window. With start_type=None, idx_settled=0 and the 450-sample
# window includes the reference pulse - per-pixel-distinctive shapes that
# leak chip identity into the model. Discovered 2026-05-08: switching
# from None -> "temperature" drops cnn2d_spec random-split accuracy from
# 0.852 -> 0.702, exposing the prior number as inflated by reference-
# pulse leakage rather than amplification learning.
NROWS, NCOLS = 290, 204     # Lacewing chip dims (titan defaults)

# Loose-active QC thresholds (see module docstring + 2026-05-06 profiling)
V_FIRST_LO = 0.5            # first-sample voltage must exceed this
V_FIRST_HI = 5.0            # ... and stay below this
STD_MIN = 1e-3              # signal std (filters dead/zero pixels)


def _loose_active_mask_per_well(chip_path: Path) -> list[np.ndarray]:
    """Re-load the chip's firmware active mask and split per well.

    Returns one boolean mask per well (length = pixels in that well).
    This is BEFORE titan's strict per-experiment vrange/derivative
    filter; it's just the firmware's "responds to reference electrode"
    flag from find_active.bin.
    """
    readout = titan_load.find_most_recent_readout(chip_path)
    idx_active_file = titan_load.find_corresp_file(chip_path, readout,
                                                   "idx_active")
    idx_list = titan_load.binary_file_read(idx_active_file)
    idx_input = (np.array(idx_list[:NROWS * NCOLS])
                 .reshape(NROWS, NCOLS, order="F")
                 .reshape(NROWS * NCOLS, order="C"))
    fw_active = (idx_input == 400)
    return titan_preprocess.split_wells_idxactive(fw_active, paths.N_WELLS)


def _qc_mask(well_2d: np.ndarray) -> np.ndarray:
    """Drop pixels that are NaN/inf, dead, or saturated at the start."""
    finite = np.isfinite(well_2d).all(axis=0)
    v0 = well_2d[0, :]
    v_in_range = (v0 > V_FIRST_LO) & (v0 < V_FIRST_HI)
    has_signal = well_2d.std(axis=0) > STD_MIN
    return finite & v_in_range & has_signal


def _extract_per_pixel_traces(well, loose_mask: np.ndarray,
                              window: int = WINDOW
                              ) -> tuple[np.ndarray, int, float]:
    """Return (X, n_padded_samples, time_span_min).

    X shape: (n_active_pixels, window)
    Each row: pixel trace baseline-subtracted (start at 0), starting at
    titan's idx_start (amplification onset, time_min == 0).

    Uses `loose_mask` (firmware-active + QC) instead of well.idx_active.
    """
    # well_2d shape (T_settled_to_end, N_total_pixels), linearised.
    full_all = well.well_2d.astype(np.float32)
    # Apply loose active + QC mask (computed against post-settle frames)
    qc = _qc_mask(full_all)
    keep = loose_mask & qc
    full = full_all[:, keep]                            # (T, N_keep)

    time_min = well.time_min
    onset_idx = int(np.searchsorted(time_min, 0.0))
    full = full[onset_idx:, :]                          # (T_post, N)
    # Trapped-charge subtraction (Paper 3 §IV.A): start each pixel at 0
    # from the onset.
    full = full - full[0:1, :]

    T_post = full.shape[0]
    if T_post >= window:
        X = full[:window, :].T                          # (N, window)
        n_pad = 0
        span_min = float(time_min[onset_idx + window - 1] -
                         time_min[onset_idx])
    else:
        n_pad = window - T_post
        pad = np.zeros((n_pad, full.shape[1]), dtype=np.float32)
        X = np.concatenate([full, pad], axis=0).T       # (N, window)
        span_min = float(time_min[-1] - time_min[onset_idx])
    return X, n_pad, span_min


def _process_chip(chip_path: Path, label_per_well: dict[int, int],
                  log_lines: list[str], start_type=DEFAULT_START_TYPE):
    """Return (X (N,window), y (N,), well_ids (N,), pixel_ids (N,)).

    Returns None and logs the failure if titan_load_and_preprocessing
    raises (e.g. missing gain/log files in some Others/ chips).
    """
    print(f"  Loading {chip_path.name} ...")
    try:
        exp = titan_load_and_preprocessing(
            chip_path, n_wells=paths.N_WELLS,
            start_type=start_type, n_a_type=N_A_TYPE,
            end_time_min=END_TIME_MIN, print_status=False,
        )
    except (TypeError, FileNotFoundError, ValueError, IndexError, KeyError) as e:
        msg = (f"    SKIP {chip_path.name}: titan preprocessing failed "
               f"({type(e).__name__}: {e!s:.200})")
        print(msg)
        log_lines.append(msg)
        return None

    try:
        loose_per_well = _loose_active_mask_per_well(chip_path)
    except (TypeError, FileNotFoundError, ValueError) as e:
        msg = (f"    SKIP {chip_path.name}: loose active-mask reload failed "
               f"({type(e).__name__}: {e!s:.200})")
        print(msg)
        log_lines.append(msg)
        return None

    Xs, ys, well_ids, pix_ids = [], [], [], []
    for w_idx in sorted(label_per_well.keys()):
        if w_idx >= len(exp.wells_list):
            log_lines.append(
                f"    WARNING {chip_path.name} well {w_idx}: out of range "
                f"({len(exp.wells_list)} wells), skipping")
            continue
        well = exp.wells_list[w_idx]
        loose_mask = loose_per_well[w_idx]

        X, n_pad, span = _extract_per_pixel_traces(well, loose_mask)
        if X.shape[0] == 0:
            log_lines.append(
                f"    WARNING {chip_path.name} well {w_idx}: 0 pixels survive "
                f"loose-active + QC, skipping")
            continue
        label = label_per_well[w_idx]
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


def _enumerate_all_chips() -> list[Path]:
    """List every chip folder under DATA_ROOT that has a readout binary."""
    candidates: list[Path] = []
    for sub in sorted(paths.DATA_ROOT.iterdir()):
        if not sub.is_dir() or sub.name in {"Figs", "Lin"}:
            continue
        if sub.name.startswith("1e") or sub.name == "Others":
            for chip in sorted(p for p in sub.iterdir() if p.is_dir()):
                candidates.append(chip)
    return candidates


def build(scope: str = "final", out_name: str | None = None,
          start_type=DEFAULT_START_TYPE) -> None:
    log_lines = [f"# build_dataset.py run at {datetime.now().isoformat()}",
                 f"# scope={scope}, start_type={start_type!r}, WINDOW={WINDOW}, "
                 f"N_A_TYPE={N_A_TYPE}, END_TIME_MIN={END_TIME_MIN}", ""]

    all_X, all_y, all_chip, all_well, all_pix = [], [], [], [], []

    if scope == "final":
        if out_name is None:
            out_name = "dataset_per_pixel"

        print("Final 5 chips:")
        log_lines.append("## Final chips")
        for label_str, chip_path in paths.FINAL_CHIPS.items():
            log_lines.append(f"\n### {label_str}: {chip_path.name}")
            label_map = {w: 1 for w in paths.POSITIVE_WELL_INDICES}
            label_map[paths.NTC_WELL_INDEX] = 0
            result = _process_chip(chip_path, label_map, log_lines, start_type=start_type)
            if result is None:
                continue
            X, y, well_ids, pix_ids = result
            all_X.append(X); all_y.append(y)
            all_chip.append(np.full(X.shape[0], chip_path.name, dtype="<U64"))
            all_well.append(well_ids); all_pix.append(pix_ids)

        print("\nOthers/ NTC chips:")
        log_lines.append("\n## Others/ NTC chips")
        others = load_ntc_chips()
        if not others:
            msg = ("    NO Others/ NTC chips loaded - run "
                   "`python -m lacewing.classification.data.ntc_inventory` first.")
            print(msg); log_lines.append(msg)
        for chip_path, ntc_wells in others:
            log_lines.append(f"\n### NTC: {chip_path.name} wells={ntc_wells}")
            label_map = {w: 0 for w in ntc_wells}
            result = _process_chip(chip_path, label_map, log_lines, start_type=start_type)
            if result is None:
                continue
            X, y, well_ids, pix_ids = result
            all_X.append(X); all_y.append(y)
            all_chip.append(np.full(X.shape[0], chip_path.name, dtype="<U64"))
            all_well.append(well_ids); all_pix.append(pix_ids)

    elif scope == "all":
        if out_name is None:
            out_name = "dataset_all"

        all_chips = _enumerate_all_chips()
        print(f"Found {len(all_chips)} chip folders under {paths.DATA_ROOT}.")
        log_lines.append(f"## All-data scope: {len(all_chips)} chip folders")

        # Default labelling = standard 6-well layout (0-4 positive, 5 NTC).
        # Same convention used by ntc_inventory.csv defaults.
        std_label_map = {w: 1 for w in paths.POSITIVE_WELL_INDICES}
        std_label_map[paths.NTC_WELL_INDEX] = 0

        for chip_path in all_chips:
            log_lines.append(f"\n### {chip_path.name}")
            result = _process_chip(chip_path, std_label_map, log_lines, start_type=start_type)
            if result is None:
                continue
            X, y, well_ids, pix_ids = result
            all_X.append(X); all_y.append(y)
            all_chip.append(np.full(X.shape[0], chip_path.name, dtype="<U64"))
            all_well.append(well_ids); all_pix.append(pix_ids)

    else:
        raise ValueError(f"Unknown scope: {scope}. Choose 'final' or 'all'.")

    if not all_X:
        raise RuntimeError("No data was successfully loaded — check logs.")

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


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scope", choices=["final", "all"], default="final")
    p.add_argument("--out", default=None,
                   help="Cache name (default: 'dataset_per_pixel' for final, "
                        "'dataset_all' for all).")
    p.add_argument("--start-type", default=DEFAULT_START_TYPE,
                   help=f"titan start_type, default={DEFAULT_START_TYPE!r}. "
                        f"'temperature' skips the reference-pulse window; "
                        f"None treats sample 0 as the amplification onset "
                        f"(legacy, not recommended).")
    args = p.parse_args()
    # Allow --start-type none on the CLI to opt back into the legacy behaviour
    st = None if str(args.start_type).lower() == "none" else args.start_type
    build(scope=args.scope, out_name=args.out, start_type=st)
