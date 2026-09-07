"""SDM + Cy0 landmark sweep.  Loads the 5 Final chips + the SD
multiplex chip via titan, runs the SDM and Cy0 extractors on every
well, writes a per-(chip,well) CSV with TTP/SDM/Cy0 columns, a
concentration-vs-landmark comparison plot, and per-chip overlay
plots showing each landmark's location on the well's trace.

The TTP baseline is always extracted with its canonical settings
(order-100 smoothing, 0.4-of-peak threshold).  The SDM and Cy0
extractors take their smoothing order and search-start floor from
runtime flags, so we can A/B different settings without overwriting
the previous run's results.

Outputs land in ``results/<out_dirname>/`` (default
``results/sdm_cy0/``).

Run::

    # v1 (default settings, what already lives in results/sdm_cy0/)
    python -m lacewing.quantification.runners.run_sdm_cy0

    # v2 with heavier smoothing and a later search-start for SDM/Cy0
    python -m lacewing.quantification.runners.run_sdm_cy0 \\
        --out-dirname sdm_cy0_v2_smooth100_start10 \\
        --smooth-order 100 --search-start-floor 10.0
"""
from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from lacewing.quantification.common import (
    CONC_KEYS,
    CONC_LOG,
    DATA_ROOT,
    EXPERIMENTS_EXTRA,
    EXPERIMENTS_FINAL,
    N_WELLS,
    PEAK_SEARCH_START_MIN,
    RESULTS_DIR,
    SD_LABEL_LOG,
    SD_WELL_LABELS,
    TTP_QLAMP,
    WELLS_FOR_QUANT,
    get_matlab_search_start,
    load_chip,
)
from lacewing.quantification.eval.schema import write_predictions, write_labels
from lacewing.quantification.methods.cy0 import CY0_SMOOTH_ORDER, extract_cy0
from lacewing.quantification.methods.pdf_regression import _log10_concentration_array
from lacewing.quantification.methods.sdm import extract_sdm
from lacewing.quantification.methods.ttp_threshold_derivative import extract_ttp
from lacewing.quantification.plotting.landmark_overlays import plot_chip_landmarks
from lacewing.quantification.regression_shared.data.labels import qlamp_ttp_per_well


def _qlamp_per_well(chip_key: str, n_wells: int) -> list[float]:
    """Per-well gold-standard qLAMP TTP for one chip.

    - On the 5 Final chips, wells in ``WELLS_FOR_QUANT`` share the
      chip's single concentration's qLAMP mean; wells outside that
      set get NaN (no clean per-well reference).
    - On the SD multiplex chip, every well's concentration comes
      from ``SD_WELL_LABELS[i]`` mapped through ``SD_LABEL_LOG``.
    - Any other chip: all NaN.
    """
    out = [float("nan")] * n_wells
    if chip_key in CONC_KEYS:
        conc_idx = CONC_KEYS.index(chip_key)
        chip_ref = float(TTP_QLAMP[conc_idx].mean())
        for w in WELLS_FOR_QUANT:
            if w < n_wells:
                out[w] = chip_ref
        return out
    if chip_key == "SD":
        for w in range(n_wells):
            if w >= len(SD_WELL_LABELS):
                continue
            label = SD_WELL_LABELS[w]
            if label not in SD_LABEL_LOG:
                continue
            conc_log = SD_LABEL_LOG[label]
            try:
                conc_idx = CONC_LOG.index(conc_log)
            except ValueError:
                continue
            out[w] = float(TTP_QLAMP[conc_idx].mean())
        return out
    return out


def run(
    out_dirname: str = "sdm_cy0",
    smooth_order_sdm_cy0: int = CY0_SMOOTH_ORDER,
    search_start_floor_sdm_cy0: float = PEAK_SEARCH_START_MIN,
) -> None:
    if not DATA_ROOT.exists():
        print(f"ERROR: Data directory not found: {DATA_ROOT}")
        sys.exit(1)

    out_dir = RESULTS_DIR / out_dirname
    per_well_dir = out_dir / "per_well_landmarks"
    out_dir.mkdir(exist_ok=True)
    per_well_dir.mkdir(exist_ok=True)

    print(f"=== Config: smooth_order_sdm_cy0={smooth_order_sdm_cy0}, "
          f"search_start_floor_sdm_cy0={search_start_floor_sdm_cy0:.1f} min ===")
    print(f"=== Outputs will land in: {out_dir} ===")

    print("\n=== Loading 5 Final-dataset chips + SD chip ===")
    chips: dict[str, object] = {}
    chip_meta: dict[str, dict] = {}
    for ck, ep in EXPERIMENTS_FINAL.items():
        exp = load_chip(ep, label=ck)
        if exp is not None:
            chips[ck] = exp
            chip_meta[ck] = {"path": ep, "kind": "final"}
    for ck, ep in EXPERIMENTS_EXTRA.items():
        exp = load_chip(ep, label=ck)
        if exp is not None:
            chips[ck] = exp
            chip_meta[ck] = {"path": ep, "kind": "extra"}

    if not chips:
        print("ERROR: no chips loaded; aborting.")
        sys.exit(1)

    # Per-(chip, well) landmark extraction.
    print("\n=== Per-well TTP / SDM / Cy0 ===")
    per_chip: dict[str, dict] = {}
    for ck, exp in chips.items():
        ttps_w, sdms_w, cy0s_w = [], [], []
        for well in exp.wells_list:
            t_safe_ttp = max(get_matlab_search_start(well),
                             PEAK_SEARCH_START_MIN)
            t_safe_sc = max(get_matlab_search_start(well),
                            search_start_floor_sdm_cy0)
            ttp_rel, _peak_abs = extract_ttp(
                well.time_min, well.well_2d_bs_active_mean,
                search_start_min=t_safe_ttp,
            )
            sdm_abs = extract_sdm(
                well.time_min, well.well_2d_bs_active_mean,
                search_start_min=t_safe_sc,
                smooth_order=smooth_order_sdm_cy0,
            )
            cy0_abs = extract_cy0(
                well.time_min, well.well_2d_bs_active_mean,
                search_start_min=t_safe_sc,
                smooth_order=smooth_order_sdm_cy0,
            )
            ttps_w.append(ttp_rel)
            sdms_w.append(sdm_abs)
            cy0s_w.append(cy0_abs)
        per_chip[ck] = {"ttps": ttps_w, "sdms": sdms_w, "cy0s": cy0s_w}
        print(f"  {ck} ({chip_meta[ck]['path'].name}):")
        for i_well in range(len(ttps_w)):
            def fmt(v: float) -> str:
                return f"{v:5.2f}" if not np.isnan(v) else "  NaN"
            print(f"    well {i_well}: "
                  f"TTP={fmt(ttps_w[i_well])}  "
                  f"SDM={fmt(sdms_w[i_well])}  "
                  f"Cy0={fmt(cy0s_w[i_well])}")

    # ---- Per-well landmark overlay plots (one figure per chip) ----
    print("\n=== Per-well landmark overlay plots ===")
    for ck, exp in chips.items():
        well_labels = SD_WELL_LABELS if ck == "SD" else None
        out_png = per_well_dir / f"{ck}_landmarks.png"
        n_wells_chip = len(exp.wells_list)
        qlamp_ref = _qlamp_per_well(ck, n_wells_chip)
        plot_chip_landmarks(
            exp, chip_label=ck,
            save_path=out_png,
            well_labels=well_labels,
            smooth_order_sdm_cy0=smooth_order_sdm_cy0,
            search_start_floor_sdm_cy0=search_start_floor_sdm_cy0,
            qlamp_ttp_per_well=qlamp_ref,
        )
        print(f"  Saved: {out_png}")

    # ---- Per-chip CSV ----
    csv_path = out_dir / "per_chip.csv"
    with open(csv_path, "w") as f:
        header = ["concentration", "chip_folder"]
        for tag in ("ttp", "sdm", "cy0"):
            header += [f"{tag}_well{w}" for w in range(N_WELLS)]
        f.write(",".join(header) + "\n")
        for ck in list(CONC_KEYS) + list(EXPERIMENTS_EXTRA.keys()):
            if ck not in per_chip:
                continue
            ttps = per_chip[ck]["ttps"]
            sdms = per_chip[ck]["sdms"]
            cy0s = per_chip[ck]["cy0s"]
            row = [ck, chip_meta[ck]["path"].name]
            row += [f"{v:.2f}" if not np.isnan(v) else "NaN" for v in ttps]
            row += [f"{v:.2f}" if not np.isnan(v) else "NaN" for v in sdms]
            row += [f"{v:.2f}" if not np.isnan(v) else "NaN" for v in cy0s]
            f.write(",".join(row) + "\n")
    print(f"\nWrote {csv_path}")

    # ---- Run-config sidecar (so the directory is self-documenting) ----
    cfg_path = out_dir / "config.txt"
    cfg_path.write_text(
        "SDM/Cy0 runner config:\n"
        f"  smooth_order_sdm_cy0         = {smooth_order_sdm_cy0}\n"
        f"  search_start_floor_sdm_cy0   = {search_start_floor_sdm_cy0:.1f} min\n"
        f"  TTP baseline always uses default settings (smooth=100, "
        f"floor={PEAK_SEARCH_START_MIN:.1f} min)\n"
    )
    print(f"Wrote {cfg_path}")

    # ---- Concentration-vs-landmark plot (Final chips only) ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    info_lines = []
    for tag, label, marker in [
        ("ttps", "TTP (baseline, 0.4-of-peak)", "o"),
        ("sdms", "SDM (2nd-derivative max)",   "s"),
        ("cy0s", "Cy0 (tangent intercept)",    "^"),
    ]:
        means = []
        for ck in CONC_KEYS:
            if ck not in per_chip:
                means.append(np.nan)
                continue
            vals = [per_chip[ck][tag][w] for w in WELLS_FOR_QUANT]
            means.append(float(np.nanmean(vals)))
        ax.plot(CONC_LOG, means, marker=marker, markersize=8,
                linewidth=1.6, label=label)

        valid = [(x, y) for x, y in zip(CONC_LOG, means) if not np.isnan(y)]
        if len(valid) >= 3:
            xs, ys = zip(*valid)
            r = float(np.corrcoef(xs, ys)[0, 1])
            slope = float(np.polyfit(xs, ys, 1)[0])
            tag_short = {"ttps": "TTP", "sdms": "SDM", "cy0s": "Cy0"}[tag]
            info_lines.append(
                f"{tag_short}: r={r:+.3f}, slope={slope:+.2f} min/decade")

    ax.set_xlabel(r"$\log_{10}$(concentration, copies/reaction)", fontsize=11)
    ax.set_ylabel("Landmark time (min from file start)", fontsize=11)
    ax.set_title(
        f"ISFET-LAMP onset landmarks vs concentration\n"
        f"(SDM/Cy0: smooth={smooth_order_sdm_cy0}, "
        f"start>={search_start_floor_sdm_cy0:.0f} min)",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if info_lines:
        ax.text(0.02, 0.02, "\n".join(info_lines),
                transform=ax.transAxes,
                fontsize=8, color="#444",
                ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="#bbb", alpha=0.9))

    fig.tight_layout()
    plot_path = out_dir / "comparison.png"
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {plot_path}")

    # ---- Canonical scoreboard schema (sdm/ and cy0/ subdirectories) ----
    # Expand per-well landmark times to per-pixel arrays.  Only wells with a
    # valid qLAMP TTP label are included (NaN wells / NTC skipped).
    sdm_pred_px: list[float] = []
    cy0_pred_px: list[float] = []
    true_ttp_px: list[float] = []
    chip_id_px: list[str] = []
    well_id_px: list[int] = []

    for ck, exp in chips.items():
        chip_folder = chip_meta[ck]["path"].name
        true_ttps = qlamp_ttp_per_well(chip_folder, n_wells=N_WELLS)
        sdms_w = per_chip[ck]["sdms"]
        cy0s_w = per_chip[ck]["cy0s"]
        for i_well, well in enumerate(exp.wells_list):
            if i_well >= len(true_ttps):
                continue
            if np.isnan(true_ttps[i_well]):
                continue
            # Include pixel even if the predicted landmark is NaN — the
            # scoreboard handles NaN predictions gracefully.
            n_px = int(np.array(well.well_2d_bs_active).shape[1])
            sdm_pred_px.extend([float(sdms_w[i_well])] * n_px)
            cy0_pred_px.extend([float(cy0s_w[i_well])] * n_px)
            true_ttp_px.extend([float(true_ttps[i_well])] * n_px)
            chip_id_px.extend([chip_folder] * n_px)
            well_id_px.extend([i_well] * n_px)

    if sdm_pred_px:
        chip_id_arr = np.array(chip_id_px, dtype="U160")
        well_id_arr = np.array(well_id_px, dtype=np.int32)
        true_arr = np.array(true_ttp_px, dtype=np.float32)
        log10_conc = _log10_concentration_array(chip_id_arr, well_id_arr)
        # Split = "test" iff pixel is from SD chip (matches labels.py + dataset.py convention).
        from lacewing.quantification.regression_shared.data.labels import SD_CHIP_FOLDER
        split_arr = np.where(
            chip_id_arr == SD_CHIP_FOLDER,
            np.array("test", dtype="U16"),
            np.array("train", dtype="U16"),
        ).astype("U16")

        # Shared labels file (same pixels for both methods)
        sdm_dir = out_dir / "sdm"
        cy0_dir = out_dir / "cy0"
        sdm_dir.mkdir(exist_ok=True)
        cy0_dir.mkdir(exist_ok=True)

        write_labels(
            sdm_dir / "labels.npz",
            ttp_true_min=true_arr,
            chip_id=chip_id_arr,
            well_id=well_id_arr,
            log10_concentration=log10_conc,
            split=split_arr,
        )
        write_labels(
            cy0_dir / "labels.npz",
            ttp_true_min=true_arr,
            chip_id=chip_id_arr,
            well_id=well_id_arr,
            log10_concentration=log10_conc,
            split=split_arr,
        )

        write_predictions(
            sdm_dir / "predictions.npz",
            ttp_pred_min=np.array(sdm_pred_px, dtype=np.float32),
        )
        write_predictions(
            cy0_dir / "predictions.npz",
            ttp_pred_min=np.array(cy0_pred_px, dtype=np.float32),
        )
        n_px = len(true_arr)
        print(f"\nWrote canonical schema: {sdm_dir}/predictions.npz ({n_px:,} pixels)")
        print(f"Wrote canonical schema: {sdm_dir}/labels.npz")
        print(f"Wrote canonical schema: {cy0_dir}/predictions.npz ({n_px:,} pixels)")
        print(f"Wrote canonical schema: {cy0_dir}/labels.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dirname", default="sdm_cy0",
        help="Subdirectory of results/ to write outputs into "
             "(default 'sdm_cy0').  Use a different name to keep "
             "previous runs.")
    parser.add_argument(
        "--smooth-order", type=int, default=CY0_SMOOTH_ORDER,
        help=f"Smoothing order for the SDM / Cy0 extractors "
             f"(default {CY0_SMOOTH_ORDER}, matching the deployed "
             f"threshold-derivative TTP and the plate-labelling "
             f"pipeline's Cy0).")
    parser.add_argument(
        "--search-start-floor", type=float, default=PEAK_SEARCH_START_MIN,
        help=f"Earliest minute at which SDM / Cy0 will start looking "
             f"for a landmark (default {PEAK_SEARCH_START_MIN:.1f}).  "
             f"TTP always uses its canonical floor.")
    args = parser.parse_args()

    run(
        out_dirname=args.out_dirname,
        smooth_order_sdm_cy0=args.smooth_order,
        search_start_floor_sdm_cy0=args.search_start_floor,
    )


if __name__ == "__main__":
    main()
