"""Plotting · SD multiplex chip panels figure. [Cat A] Report figure builders.

SD multiplex chip plotting: 3-panel voltage + 10-smoothed diff +
100-smoothed diff figure, and the slide-style TTP-vs-reaction scatter.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lacewing.quantification.common import (
    MATLAB_COLORS,
    PEAK_SEARCH_START_MIN,
    SD_LABEL_LOG,
    compute_diff_signals_variable,
    find_diff_peak_time,
    get_matlab_search_start_absolute,
)
from lacewing.quantification.methods.ttp_threshold_derivative import extract_ttp


def plot_sd_multiplex_panels(exp, well_labels: list[str], save_path: Path):
    well_data = []
    for i, well in enumerate(exp.wells_list):
        t_abs = well.time_npr[well.idx_settled : well.idx_end] / 60.0
        v_abs_mV = well.well_2d_bs_active_mean * 1e3

        t_settle_abs = get_matlab_search_start_absolute(well)
        t_safe = max(t_settle_abs, t_abs[0] + PEAK_SEARCH_START_MIN)

        well_label = well_labels[i] if i < len(well_labels) else f"well {i}"

        if well_label == "NTC":
            ttp_abs, peak_abs, peak_10_abs = np.nan, np.nan, np.nan
        else:
            ttp_abs, peak_abs = extract_ttp(
                t_abs, well.well_2d_bs_active_mean,
                search_start_min=t_safe,
            )
            peak_10_abs = find_diff_peak_time(
                t_abs, well.well_2d_bs_active_mean,
                smooth_order=10, search_start_min=t_safe,
            )

        diff_10, t_diff_10 = compute_diff_signals_variable(
            t_abs, well.well_2d_bs_active_mean, 10)
        diff_100, t_diff_100 = compute_diff_signals_variable(
            t_abs, well.well_2d_bs_active_mean, 100)

        well_data.append(dict(
            time_abs=t_abs, V_mV=v_abs_mV,
            t_diff_10=t_diff_10, diff_10=diff_10,
            t_diff_100=t_diff_100, diff_100=diff_100,
            ttp_abs=ttp_abs,
            peak_abs=peak_abs,
            peak_10_abs=peak_10_abs,
            t_settle=t_settle_abs,
            label=well_label,
            color=MATLAB_COLORS[i % len(MATLAB_COLORS)],
        ))

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    avg_settle = float(np.mean([d["t_settle"] for d in well_data]))

    for d in well_data:
        axes[0].plot(d["time_abs"], d["V_mV"], color=d["color"], lw=1.6,
                     label=d["label"])
        axes[1].plot(d["t_diff_10"], d["diff_10"], color=d["color"],
                     lw=1.0, alpha=0.8)
        axes[2].plot(d["t_diff_100"], d["diff_100"], color=d["color"],
                     lw=1.6, label=d["label"])

    y_max_mid = max(np.nanmax(d["diff_10"]) for d in well_data)
    y_max_bot = max(np.nanmax(d["diff_100"]) for d in well_data)
    for d in well_data:
        if not np.isnan(d["peak_10_abs"]):
            axes[1].annotate(
                "", xy=(d["peak_10_abs"], y_max_mid * 0.95),
                xytext=(d["peak_10_abs"], y_max_mid * 1.1),
                arrowprops=dict(arrowstyle="->", color="k"))
        if not np.isnan(d["peak_abs"]):
            axes[2].annotate(
                "", xy=(d["peak_abs"], y_max_bot * 0.95),
                xytext=(d["peak_abs"], y_max_bot * 1.1),
                arrowprops=dict(arrowstyle="->", color="k"))

    axes[0].set(ylabel="Voltage (Lin) [mV]", title="Filtered outputs")
    axes[0].legend(loc="upper left", fontsize=9, ncol=2)
    axes[1].set(ylabel="Diff voltage [mV/s]",
                title="Amplification peaks (10-smoothed diff)")
    axes[2].set(ylabel="Diff voltage [mV/s]",
                title="Filtered (100-smoothed diff)", xlabel="Time [min]")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.axvline(avg_settle, color="grey", linestyle="--", alpha=0.6,
                   label="Temp settling (99.5%)" if ax is axes[0] else None)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return well_data


def plot_sd_ttp_vs_reaction(
    well_data: list[dict], save_path: Path,
    label_log: dict[str, int] | None = None,
) -> None:
    """Slide-style TTP-vs-reaction scatter using ABSOLUTE TTP values."""
    if label_log is None:
        label_log = SD_LABEL_LOG

    pts = [(d["label"], d["ttp_abs"]) for d in well_data
           if d["label"] in label_log and not np.isnan(d["ttp_abs"])]

    pts.sort(key=lambda p: label_log[p[0]])
    labels = [p[0] for p in pts]
    ttps = np.array([p[1] for p in pts])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, ttps, "ko", markersize=8)
    if len(pts) >= 2:
        coeffs = np.polyfit(x, ttps, 1)
        xf = np.linspace(x[0] - 0.3, x[-1] + 0.3, 100)
        ax.plot(xf, np.polyval(coeffs, xf), "k--", lw=1.0,
                label=f"Linear fit (slope {coeffs[0]:+.2f} min/step)")
        ax.legend(fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Reaction")
    ax.set_ylabel("TTP [min]")
    ax.set_title("SD chip — TTP per reaction (Absolute)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
