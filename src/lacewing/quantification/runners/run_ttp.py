"""Threshold-derivative TTP reproduction.  Loads the 5 Final chips +
the SD multiplex chip via titan, runs the deployed Lacewing TTP
extractor on every well, writes per-chip / cross-conc CSVs and the
slope + per-well-TTP + per-well-voltage diagnostics.

Outputs land in ``results/ttp/`` (and the SD multiplex panels in
``results/sd_verification/``).

Run::

    python -m lacewing.quantification.runners.run_ttp
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from lacewing.quantification.common import (
    CONC_KEYS,
    CONC_LOG,
    DATA_ROOT,
    EXP_LABELS,
    EXPERIMENTS_EXTRA,
    EXPERIMENTS_FINAL,
    MATLAB_COLORS,
    N_WELLS,
    PEAK_SEARCH_START_MIN,
    RESULTS_DIR,
    SD_WELL_LABELS,
    SUPERVISOR_AMPL_PEAKS_4REP,
    SUPERVISOR_AMPL_PEAKS_5REP,
    TTP_QLAMP,
    WELLS_FOR_QUANT,
    get_matlab_search_start,
    load_chip,
)
from lacewing.quantification.eval.schema import write_predictions, write_labels
from lacewing.quantification.methods.pdf_regression import _log10_concentration_array
from lacewing.quantification.methods.ttp_threshold_derivative import extract_ttp
from lacewing.quantification.plotting.sd_multiplex_panels import (
    plot_sd_multiplex_panels,
    plot_sd_ttp_vs_reaction,
)
from lacewing.quantification.regression_shared.data.labels import qlamp_ttp_per_well


TTP_DIR = RESULTS_DIR / "ttp"
SD_DIR = RESULTS_DIR / "sd_verification"
TTP_DIR.mkdir(exist_ok=True)
SD_DIR.mkdir(exist_ok=True)


def run() -> None:
    if not DATA_ROOT.exists():
        print(f"ERROR: Data directory not found: {DATA_ROOT}")
        sys.exit(1)

    print("=== Loading 5 Final-dataset chips with titan preprocessing ===")
    experiments: dict[str, object] = {}
    for ck, ep in EXPERIMENTS_FINAL.items():
        exp = load_chip(ep, label=ck)
        if exp is not None:
            experiments[ck] = exp

    if not experiments:
        print("ERROR: no experiments loaded; aborting.")
        sys.exit(1)

    print("\n=== Per-well TTP and amplification peak ===")
    per_chip: dict[str, dict] = {}
    for ck in CONC_KEYS:
        if ck not in experiments:
            continue
        exp = experiments[ck]
        ttps_w, peaks_w = [], []
        for i_well, well in enumerate(exp.wells_list):
            t_safe = max(get_matlab_search_start(well), PEAK_SEARCH_START_MIN)
            ttp_rel, peak_abs = extract_ttp(
                well.time_min, well.well_2d_bs_active_mean,
                search_start_min=t_safe,
            )
            ttps_w.append(ttp_rel)
            peaks_w.append(peak_abs)
        per_chip[ck] = {"ttps": ttps_w, "peaks": peaks_w,
                        "path": EXPERIMENTS_FINAL[ck]}
        print(f"  {ck} ({EXPERIMENTS_FINAL[ck].name}):")
        for i_well in range(N_WELLS):
            t = ttps_w[i_well]; p = peaks_w[i_well]
            t_s = f"{t:5.1f}" if not np.isnan(t) else "  NaN"
            p_s = f"{p:5.1f}" if not np.isnan(p) else "  NaN"
            mark = " *" if i_well in WELLS_FOR_QUANT else "  "
            print(f"    well {i_well}: TTP={t_s}  peak={p_s}{mark}")

    print("\n=== Per-chip mean over wells 0-3 vs supervisor's ampl_peaks ===")
    rows = []
    for i, ck in enumerate(CONC_KEYS):
        if ck not in per_chip:
            rows.append((ck, np.nan, np.nan, SUPERVISOR_AMPL_PEAKS_4REP[i],
                         SUPERVISOR_AMPL_PEAKS_5REP[i], np.nan, np.nan))
            continue
        ttps = per_chip[ck]["ttps"]
        peaks = per_chip[ck]["peaks"]
        ttp_mean = float(np.nanmean([ttps[w] for w in WELLS_FOR_QUANT]))
        peak_mean = float(np.nanmean([peaks[w] for w in WELLS_FOR_QUANT]))
        offset_4 = peak_mean - SUPERVISOR_AMPL_PEAKS_4REP[i]
        offset_5 = peak_mean - SUPERVISOR_AMPL_PEAKS_5REP[i]
        rows.append((ck, ttp_mean, peak_mean,
                     SUPERVISOR_AMPL_PEAKS_4REP[i], SUPERVISOR_AMPL_PEAKS_5REP[i],
                     offset_4, offset_5))

    print(f"  {'conc':<6} {'my_ttp':>8} {'my_peak':>9} {'sup_4rep':>10} "
          f"{'sup_5rep':>10} {'off(4)':>8} {'off(5)':>8}")
    for ck, mtt, mp_, s4, s5, o4, o5 in rows:
        fmt = lambda v: f"{v:>8.2f}" if not np.isnan(v) else f"{'NaN':>8}"
        off4_s = f"{o4:>+8.2f}" if not np.isnan(o4) else f"{'NaN':>8}"
        off5_s = f"{o5:>+8.2f}" if not np.isnan(o5) else f"{'NaN':>8}"
        print(f"  {ck:<6} {fmt(mtt)} {fmt(mp_)} {s4:>10.2f} {s5:>10.2f} "
              f"{off4_s} {off5_s}")

    # ---- Per-chip CSV ----
    perchip_path = TTP_DIR / "per_chip.csv"
    with open(perchip_path, "w") as f:
        f.write("concentration,chip_folder,"
                "ttp_well0,ttp_well1,ttp_well2,ttp_well3,ttp_well4,ttp_well5,"
                "peak_well0,peak_well1,peak_well2,peak_well3,peak_well4,peak_well5,"
                "ttp_mean_03,peak_mean_03\n")
        for ck in CONC_KEYS:
            if ck not in per_chip:
                continue
            ttps = per_chip[ck]["ttps"]
            peaks = per_chip[ck]["peaks"]
            ttp_m = float(np.nanmean([ttps[w] for w in WELLS_FOR_QUANT]))
            peak_m = float(np.nanmean([peaks[w] for w in WELLS_FOR_QUANT]))
            row = [ck, per_chip[ck]["path"].name]
            row += [f"{v:.2f}" if not np.isnan(v) else "NaN" for v in ttps]
            row += [f"{v:.2f}" if not np.isnan(v) else "NaN" for v in peaks]
            row += [f"{ttp_m:.2f}", f"{peak_m:.2f}"]
            f.write(",".join(row) + "\n")
    print(f"\n  Saved: {perchip_path}")

    # ---- Comparison-with-supervisor CSV ----
    cmp_path = TTP_DIR / "comparison_vs_supervisor.csv"
    with open(cmp_path, "w") as f:
        f.write("concentration,my_ttp_mean,my_peak_mean,"
                "sup_peak_4rep,sup_peak_5rep,offset_4rep,offset_5rep\n")
        for ck, mtt, mp_, s4, s5, o4, o5 in rows:
            f.write(f"{ck},{mtt:.2f},{mp_:.2f},{s4:.2f},{s5:.2f},"
                    f"{o4:+.2f},{o5:+.2f}\n")
    print(f"  Saved: {cmp_path}")

    # ---- Slope plot (peak vs concentration vs supervisor) ----
    x = np.array(CONC_LOG)
    my_peaks = np.array([r[2] for r in rows])
    valid = ~np.isnan(my_peaks)
    fig, ax = plt.subplots(figsize=(7, 5))
    if valid.sum() >= 2:
        slope_my = float(np.polyfit(x[valid], my_peaks[valid], 1)[0])
        ax.plot(x, my_peaks, "o-", color="C0",
                label=f"My peak (titan + MATLAB TTP, slope {slope_my:+.2f})")
    slope_4 = float(np.polyfit(x, SUPERVISOR_AMPL_PEAKS_4REP, 1)[0])
    slope_5 = float(np.polyfit(x, SUPERVISOR_AMPL_PEAKS_5REP, 1)[0])
    ax.plot(x, SUPERVISOR_AMPL_PEAKS_4REP, "s--", color="C1",
            label=f"Supervisor 4-rep (slope {slope_4:+.2f})")
    ax.plot(x, SUPERVISOR_AMPL_PEAKS_5REP, "^:", color="C2", alpha=0.6,
            label=f"Supervisor 5-rep (slope {slope_5:+.2f})")
    qlamp_mean = TTP_QLAMP.mean(axis=1)
    slope_q = float(np.polyfit(x, qlamp_mean, 1)[0])
    ax.plot(x, qlamp_mean, "d-.", color="C3", alpha=0.6,
            label=f"qLAMP gold-standard (slope {slope_q:+.2f})")
    ax.set_xticks(x); ax.set_xticklabels(EXP_LABELS)
    ax.set_xlabel("Concentration"); ax.set_ylabel("Amplification peak [min]")
    ax.set_title("Quantification slope: titan preprocessing + MATLAB TTP")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    plot_path = TTP_DIR / "slope.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    # ---- Per-well TTP scatter ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ttp_per_well_means = []
    for j, w in enumerate(WELLS_FOR_QUANT):
        ttps_for_well = np.array(
            [per_chip[ck]["ttps"][w] if ck in per_chip else np.nan
             for ck in CONC_KEYS])
        ax.plot(x, ttps_for_well, ".", color=MATLAB_COLORS[j], markersize=14,
                label=f"Well {w}")
    for i, ck in enumerate(CONC_KEYS):
        if ck not in per_chip:
            continue
        well_ttps = [per_chip[ck]["ttps"][w] for w in WELLS_FOR_QUANT]
        mean_ttp = float(np.nanmean(well_ttps))
        se_ttp = float(np.nanstd(well_ttps)
                       / np.sqrt(np.sum(~np.isnan(well_ttps))))
        ttp_per_well_means.append(mean_ttp)
        ax.errorbar(x[i], mean_ttp, yerr=se_ttp, fmt="k_", ms=18, lw=1.5,
                    capsize=5, zorder=5)
    means_arr = np.array(ttp_per_well_means)
    valid_means = ~np.isnan(means_arr)
    if valid_means.sum() >= 2:
        coeffs = np.polyfit(x[valid_means], means_arr[valid_means], 1)
        xf = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)
        ax.plot(xf, np.polyval(coeffs, xf), "k--", lw=1.2,
                label=f"Linear fit (slope {coeffs[0]:+.2f} min/decade)")
    ax.set_xticks(x); ax.set_xticklabels(EXP_LABELS)
    ax.set_xlim([x.min() - 0.5, x.max() + 0.5])
    ax.set_xlabel("Concentration"); ax.set_ylabel("TTP [min]")
    ax.set_title("Per-well TTP per concentration (wells 0-3)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    ttp_scatter_path = TTP_DIR / "per_well_scatter.png"
    plt.savefig(ttp_scatter_path, dpi=150)
    plt.close()
    print(f"  Saved: {ttp_scatter_path}")

    # ---- Per-chip voltage diagnostic plot ----
    fig, axes = plt.subplots(1, len(experiments), figsize=(4 * len(experiments), 4),
                             sharey=True)
    if len(experiments) == 1:
        axes = [axes]
    for ax, ck in zip(axes, CONC_KEYS):
        if ck not in experiments:
            continue
        exp = experiments[ck]
        avg_safe = np.mean([max(get_matlab_search_start(w), PEAK_SEARCH_START_MIN)
                            for w in exp.wells_list])
        for i_well, well in enumerate(exp.wells_list):
            ax.plot(well.time_min, well.well_2d_bs_active_mean * 1e3,
                    color=MATLAB_COLORS[i_well], linewidth=1.5,
                    label=f"well {i_well}")
        ax.axvline(avg_safe, color="r", linestyle="--", alpha=0.5)
        ax.set_title(f"{ck}")
        ax.set_xlabel("Time [min]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("V_well_avg * 1000 [mV]")
    axes[-1].legend(fontsize=8, loc="best")
    plt.tight_layout()
    voltage_path = TTP_DIR / "per_well_voltages.png"
    plt.savefig(voltage_path, dpi=150)
    plt.close()
    print(f"  Saved: {voltage_path}")

    # ---- Extra chips (SD multiplex) ----
    # Pre-initialise so the canonical schema block below can reference them
    # even if EXPERIMENTS_EXTRA is empty or no extras load successfully.
    extras: dict[str, object] = {}
    extra_results: dict[str, dict] = {}
    if EXPERIMENTS_EXTRA:
        print("\n=== Loading extra chips (not in concentration slope) ===")
        for ek, ep in EXPERIMENTS_EXTRA.items():
            ex = load_chip(ep, label=ek)
            if ex is not None:
                extras[ek] = ex

        if extras:
            print("\n=== Extra chips: per-well TTP and amplification peak ===")
            for ek, exp in extras.items():
                ttps_w, peaks_w = [], []
                for well in exp.wells_list:
                    t_dynamic = get_matlab_search_start(well)
                    t_safe = max(t_dynamic, PEAK_SEARCH_START_MIN)
                    ttp, peak = extract_ttp(
                        well.time_min, well.well_2d_bs_active_mean,
                        search_start_min=t_safe,
                    )
                    ttps_w.append(ttp); peaks_w.append(peak)
                extra_results[ek] = {"ttps": ttps_w, "peaks": peaks_w,
                                     "path": EXPERIMENTS_EXTRA[ek]}
                print(f"  {ek} ({EXPERIMENTS_EXTRA[ek].name}):")
                for i_well in range(N_WELLS):
                    t = ttps_w[i_well]; p = peaks_w[i_well]
                    t_s = f"{t:5.1f}" if not np.isnan(t) else "  NaN"
                    p_s = f"{p:5.1f}" if not np.isnan(p) else "  NaN"
                    mark = " *" if i_well in WELLS_FOR_QUANT else "  "
                    print(f"    well {i_well}: TTP={t_s}  peak={p_s}{mark}")
                ttp_m = float(np.nanmean([ttps_w[w] for w in WELLS_FOR_QUANT]))
                peak_m = float(np.nanmean([peaks_w[w] for w in WELLS_FOR_QUANT]))
                print(f"    mean wells 0-3: TTP={ttp_m:5.2f}  peak={peak_m:5.2f}")

            extra_csv = SD_DIR / "extra_per_chip.csv"
            with open(extra_csv, "w") as f:
                f.write("label,chip_folder,"
                        "ttp_well0,ttp_well1,ttp_well2,ttp_well3,ttp_well4,ttp_well5,"
                        "peak_well0,peak_well1,peak_well2,peak_well3,peak_well4,peak_well5,"
                        "ttp_mean_03,peak_mean_03\n")
                for ek in EXPERIMENTS_EXTRA:
                    if ek not in extra_results:
                        continue
                    ttps = extra_results[ek]["ttps"]
                    peaks = extra_results[ek]["peaks"]
                    ttp_m = float(np.nanmean([ttps[w] for w in WELLS_FOR_QUANT]))
                    peak_m = float(np.nanmean([peaks[w] for w in WELLS_FOR_QUANT]))
                    row = [ek, extra_results[ek]["path"].name]
                    row += [f"{v:.2f}" if not np.isnan(v) else "NaN" for v in ttps]
                    row += [f"{v:.2f}" if not np.isnan(v) else "NaN" for v in peaks]
                    row += [f"{ttp_m:.2f}", f"{peak_m:.2f}"]
                    f.write(",".join(row) + "\n")
            print(f"\n  Saved: {extra_csv}")

            # Per-well voltage plot for the extra chips
            fig, axes = plt.subplots(1, len(extras), figsize=(4 * len(extras), 4),
                                     sharey=True, squeeze=False)
            for ax, (ek, exp) in zip(axes[0], extras.items()):
                avg_safe = np.mean([max(get_matlab_search_start(w), PEAK_SEARCH_START_MIN)
                                    for w in exp.wells_list])
                for i_well, well in enumerate(exp.wells_list):
                    ax.plot(well.time_min, well.well_2d_bs_active_mean * 1e3,
                            color=MATLAB_COLORS[i_well], linewidth=1.5,
                            label=f"well {i_well}")
                ax.axvline(avg_safe, color="r", linestyle="--", alpha=0.5)
                ax.set_title(f"{ek} ({EXPERIMENTS_EXTRA[ek].name})", fontsize=9)
                ax.set_xlabel("Time [min]")
                ax.grid(alpha=0.3)
            axes[0][0].set_ylabel("V_well_avg * 1000 [mV]")
            axes[0][-1].legend(fontsize=8, loc="best")
            plt.tight_layout()
            extra_voltage_path = SD_DIR / "extra_per_well_voltages.png"
            plt.savefig(extra_voltage_path, dpi=150)
            plt.close()
            print(f"  Saved: {extra_voltage_path}")

            # SD multiplex slide reproduction
            if "SD" in extras:
                panels_path = SD_DIR / "sd_multiplex_panels.png"
                well_data_sd = plot_sd_multiplex_panels(
                    extras["SD"], SD_WELL_LABELS, panels_path)
                print(f"  Saved: {panels_path}")
                ttp_path = SD_DIR / "sd_ttp_vs_reaction.png"
                plot_sd_ttp_vs_reaction(well_data_sd, ttp_path)
                print(f"  Saved: {ttp_path}")

    # ---- Canonical scoreboard schema (predictions.npz + labels.npz) ----
    # Expand per-well TTP predictions to per-pixel arrays so the canonical
    # schema matches the per-pixel convention used by the ML methods.
    # Only wells with a valid qLAMP TTP label are included (NaN wells skipped).
    pred_ttp_px: list[float] = []
    true_ttp_px: list[float] = []
    chip_id_px: list[str] = []
    well_id_px: list[int] = []

    def _add_chip_pixels(chip_folder: str, exp_obj, ttps_w: list) -> None:
        """Append per-pixel arrays for one chip (all wells with valid labels)."""
        true_ttps = qlamp_ttp_per_well(chip_folder, n_wells=N_WELLS)
        for i_well, well in enumerate(exp_obj.wells_list):
            if i_well >= len(true_ttps):
                continue
            if np.isnan(true_ttps[i_well]) or np.isnan(ttps_w[i_well]):
                continue
            n_px = int(np.array(well.well_2d_bs_active).shape[1])
            pred_ttp_px.extend([float(ttps_w[i_well])] * n_px)
            true_ttp_px.extend([float(true_ttps[i_well])] * n_px)
            chip_id_px.extend([chip_folder] * n_px)
            well_id_px.extend([i_well] * n_px)

    # 1. Dose-response chips (5 Final chips) — already in per_chip + experiments
    for ck in CONC_KEYS:
        if ck not in per_chip or ck not in experiments:
            continue
        _add_chip_pixels(
            chip_folder=EXPERIMENTS_FINAL[ck].name,
            exp_obj=experiments[ck],
            ttps_w=per_chip[ck]["ttps"],
        )

    # 2. SD multiplex chip — already loaded into extras / extra_results above
    if "SD" in extras and "SD" in extra_results:
        _add_chip_pixels(
            chip_folder=EXPERIMENTS_EXTRA["SD"].name,
            exp_obj=extras["SD"],
            ttps_w=extra_results["SD"]["ttps"],
        )

    if pred_ttp_px:
        chip_id_arr = np.array(chip_id_px, dtype="U160")
        well_id_arr = np.array(well_id_px, dtype=np.int32)
        pred_arr = np.array(pred_ttp_px, dtype=np.float32)
        true_arr = np.array(true_ttp_px, dtype=np.float32)
        log10_conc = _log10_concentration_array(chip_id_arr, well_id_arr)
        # Split = "test" iff pixel is from SD chip (matches labels.py + dataset.py convention).
        from lacewing.quantification.regression_shared.data.labels import SD_CHIP_FOLDER
        split_arr = np.where(
            chip_id_arr == SD_CHIP_FOLDER,
            np.array("test", dtype="U16"),
            np.array("train", dtype="U16"),
        ).astype("U16")

        write_predictions(TTP_DIR / "predictions.npz",
                          ttp_pred_min=pred_arr)
        write_labels(
            TTP_DIR / "labels.npz",
            ttp_true_min=true_arr,
            chip_id=chip_id_arr,
            well_id=well_id_arr,
            log10_concentration=log10_conc,
            split=split_arr,
        )
        print(f"\n  Saved canonical schema: {TTP_DIR}/predictions.npz "
              f"({len(pred_arr):,} pixels)")
        print(f"  Saved canonical schema: {TTP_DIR}/labels.npz")

    print(f"\nAll outputs saved under: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
