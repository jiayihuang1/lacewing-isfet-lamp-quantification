"""Build the MAD-rule filtered per-pixel cache for the Multi chips.

Sibling of ``build_filtered_dataset_multi.py``; the only difference is
that A/B/D thresholds use ``median +- k * MAD`` instead of percentile
cuts.  Single ``k`` parameter shared across A/B/D.  Layer C is
unchanged.

Run::

    python -m lacewing.quantification.multi_data_shared.build_filtered_dataset_mad_multi \\
        --layers ABCD --k 1.5
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from . import _titan_setup  # noqa: F401

from ._labels import label_per_well, ntc_wells
from ._filter_multi import run_filter_pipeline_per_chip_mad_multi
from . import build_dataset_multi as bdm
from .build_filtered_dataset_multi import _derive_rows_cols_for_chip


# Defaults that match build_filtered_dataset_mad.py.
DEFAULT_C_K       = 8
DEFAULT_C_BAD_FR  = 0.5
DEFAULT_D_K       = 8
DEFAULT_D_METRIC  = "median"


def _filter_one_chip_mad(
    chip_path: Path,
    layers: str,
    clean_ntc_first: bool,
    k: float,
    log_lines: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    from .load_combined import load_chip_combined
    from .artefact_trim import trim_chip_artefact

    try:
        exp, _ = load_chip_combined(
            chip_path, n_wells=10,
            n_a_type=bdm.DEFAULT_N_A_TYPE,
            end_time_min=bdm.DEFAULT_END_TIME_MIN,
            print_status=False,
        )
        trim_chip_artefact(exp, apply=True)
    except Exception as e:
        msg = f"    SKIP {chip_path.name}: load/trim failed ({type(e).__name__}: {e!s:.200})"
        print(msg); log_lines.append(msg)
        return None

    labels = label_per_well(chip_path.name)
    ntcs = set(ntc_wells(chip_path.name))
    Xs, ys, well_ids, pix_ids = [], [], [], []
    for w_idx in sorted(labels.keys()):
        if w_idx >= len(exp.wells_list):
            continue
        well = exp.wells_list[w_idx]
        loose_mask = np.asarray(well.idx_active, dtype=bool)
        X, _, _ = bdm._extract_per_pixel_traces(well, loose_mask)
        if X.shape[0] == 0:
            continue
        Xs.append(X)
        ys.append(np.full(X.shape[0], labels[w_idx], dtype=np.uint8))
        well_ids.append(np.full(X.shape[0], w_idx, dtype=np.uint8))
        pix_ids.append(np.arange(X.shape[0], dtype=np.uint16))

    if not Xs:
        log_lines.append(f"    SKIP {chip_path.name}: no surviving pixels.")
        return None

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    well_id = np.concatenate(well_ids, axis=0).astype(np.int64)
    pix_id  = np.concatenate(pix_ids,  axis=0)

    ntc_mask = np.isin(well_id, sorted(ntcs))
    if not ntc_mask.any():
        log_lines.append(f"    SKIP {chip_path.name}: 0 NTC pixels.")
        return None

    try:
        rows, cols = _derive_rows_cols_for_chip(exp, well_id, pix_id)
    except RuntimeError as e:
        msg = f"    SKIP {chip_path.name}: row/col reconstruct failed ({e})"
        print(msg); log_lines.append(msg)
        return None

    keep_mask, info = run_filter_pipeline_per_chip_mad_multi(
        X, well_id, rows, cols,
        ntc_well_mask=ntc_mask,
        layers=layers,
        clean_ntc_first=clean_ntc_first,
        k=k,
        c_k=DEFAULT_C_K, c_min_bad_frac=DEFAULT_C_BAD_FR,
        d_k=DEFAULT_D_K, d_metric=DEFAULT_D_METRIC,
    )
    drop = info["n_dropped"]
    log_lines.append(
        f"    filter MAD k={k} {layers}/ntc={'D' if clean_ntc_first else 'raw'}:"
        f" kept {info['n_kept']}/{info['n_total']}  "
        f"drops A={drop['A']} B={drop['B']} C={drop['C']} "
        f"D={drop['D']} ntc_D={drop['ntc_D']}")

    return (X[keep_mask], y[keep_mask],
            well_id[keep_mask].astype(np.uint8),
            pix_id[keep_mask])


def build_filtered_mad_multi(layers: str, clean_ntc_first: bool,
                              k: float, out_name: str) -> Path:
    log_lines = [
        f"# build_filtered_dataset_mad_multi.py run at {datetime.now().isoformat()}",
        f"# layers={layers}, clean_ntc_first={clean_ntc_first}, k={k}",
        f"# WINDOW={bdm.WINDOW}, n_a_type={bdm.DEFAULT_N_A_TYPE}, "
        f"end_time_min={bdm.DEFAULT_END_TIME_MIN}",
        f"# rule=MAD; c_k={DEFAULT_C_K}, c_min_bad_frac={DEFAULT_C_BAD_FR}, "
        f"d_k={DEFAULT_D_K}, d_metric={DEFAULT_D_METRIC}",
        "",
    ]

    chips = bdm._enumerate_chips()
    log_lines.append(f"## Found {len(chips)} chip folders (excluded: {bdm.EXCLUDED_CHIP_SUBSTRINGS})")

    all_X, all_y, all_chip, all_well, all_pix = [], [], [], [], []
    for chip_path in chips:
        log_lines.append(f"\n### {chip_path.name}")
        out = _filter_one_chip_mad(chip_path, layers, clean_ntc_first, k, log_lines)
        if out is None:
            continue
        X, y, well_ids, pix_ids = out
        if X.shape[0] == 0:
            log_lines.append(f"    SKIP {chip_path.name}: 0 pixels surviving filter.")
            continue
        all_X.append(X); all_y.append(y)
        all_chip.append(np.full(X.shape[0], chip_path.name, dtype="<U64"))
        all_well.append(well_ids); all_pix.append(pix_ids)

    if not all_X:
        raise RuntimeError("No data survived the MAD filter on any Multi chip.")

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

    bdm.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = bdm.CACHE_DIR / f"{out_name}.npz"
    np.savez_compressed(out_path,
                        X=X, y=y, chip_id=chip, well_id=well, pixel_id=pix)
    log_path = bdm.CACHE_DIR / f"{out_name}_log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {log_path}")
    return out_path


def default_out_name_mad(layers: str, clean_ntc_first: bool, k: float) -> str:
    ntc = "ntcD" if clean_ntc_first else "ntcRaw"
    k_str = f"{k}".replace(".", "p")
    return f"multi_filt_{layers.lower()}_{ntc}_madk{k_str}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layers", choices=["A", "AB", "ABC", "ABCD"],
                        default="ABCD")
    parser.add_argument("--clean-ntc", action="store_true")
    parser.add_argument("--k", type=float, required=True,
                        help="MAD multiplier shared across A/B/D")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    name = args.out or default_out_name_mad(args.layers, args.clean_ntc, args.k)
    build_filtered_mad_multi(args.layers, args.clean_ntc, args.k, name)


if __name__ == "__main__":
    main()
