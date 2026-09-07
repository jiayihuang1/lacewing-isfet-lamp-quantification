"""Plotting · TTP landmark overlay figures. [Cat A] Report figure builders.

Per-well landmark-overlay plots: show TTP / SDM / Cy0 markers
directly on each well's per-well averaged trace.

Use this when you want to *see* where each method fires on the
actual curve.  Especially helpful for diagnosing why a landmark
disagrees with the others or with the supervisor's reference.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lacewing.quantification.common import (
    MATLAB_COLORS,
    N_WELLS,
    PEAK_SEARCH_START_MIN,
    get_matlab_search_start,
)
from lacewing.quantification.methods.cy0 import extract_cy0
from lacewing.quantification.methods.sdm import extract_sdm
from lacewing.quantification.methods.ttp_threshold_derivative import extract_ttp


LANDMARK_STYLES = {
    "TTP":  dict(color="#1f77b4", marker="o", label="TTP (0.4-of-peak)"),
    "SDM":  dict(color="#ff7f0e", marker="s", label="SDM (2nd-deriv max)"),
    "Cy0":  dict(color="#2ca02c", marker="^", label="Cy0 (tangent intercept)"),
}


def _landmark_value_on_signal(
    t_abs: np.ndarray, sig: np.ndarray, t_at: float,
) -> float:
    """Return the value of the smoothed-ish trace at t_at (linear interp)."""
    if np.isnan(t_at):
        return float("nan")
    return float(np.interp(t_at, t_abs, sig))


QLAMP_STYLE = dict(color="#d62728", linestyle="-.", linewidth=1.4, alpha=0.85)


def plot_chip_landmarks(
    exp,
    chip_label: str,
    save_path: Path,
    well_labels: list[str] | None = None,
    smooth_order_sdm_cy0: int | None = None,
    search_start_floor_sdm_cy0: float | None = None,
    qlamp_ttp_per_well: list[float] | None = None,
) -> None:
    """One figure per chip: a grid of subplots, one per well, each
    showing the well's averaged trace with TTP/SDM/Cy0 markers.

    ``smooth_order_sdm_cy0`` and ``search_start_floor_sdm_cy0`` let
    you override the smoothing order and search-start floor used
    *only* by the SDM / Cy0 extractors (TTP always uses its
    canonical settings, since it's the baseline).  Pass ``None`` to
    use the project defaults.

    ``qlamp_ttp_per_well`` is an optional list of length n_wells
    giving the gold-standard qLAMP TTP (minutes) to draw as a
    vertical reference line on each well's subplot.  Use ``NaN`` for
    wells with no qLAMP reference (e.g. NTC).  Pass ``None`` to
    omit the reference entirely.
    """
    n_wells = len(exp.wells_list)
    n_cols = min(3, n_wells)
    n_rows = int(np.ceil(n_wells / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.0 * n_cols, 3.6 * n_rows),
        sharex=False, sharey=False,
    )
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for i_well, well in enumerate(exp.wells_list):
        ax = axes_flat[i_well]
        t_safe_ttp = max(get_matlab_search_start(well), PEAK_SEARCH_START_MIN)
        floor_sc = (search_start_floor_sdm_cy0
                    if search_start_floor_sdm_cy0 is not None
                    else PEAK_SEARCH_START_MIN)
        t_safe_sc = max(get_matlab_search_start(well), floor_sc)

        ttp_val, _ = extract_ttp(
            well.time_min, well.well_2d_bs_active_mean,
            search_start_min=t_safe_ttp,
        )
        sdm_kwargs = dict(search_start_min=t_safe_sc)
        cy0_kwargs = dict(search_start_min=t_safe_sc)
        if smooth_order_sdm_cy0 is not None:
            sdm_kwargs["smooth_order"] = smooth_order_sdm_cy0
            cy0_kwargs["smooth_order"] = smooth_order_sdm_cy0
        sdm_val = extract_sdm(
            well.time_min, well.well_2d_bs_active_mean, **sdm_kwargs,
        )
        cy0_val = extract_cy0(
            well.time_min, well.well_2d_bs_active_mean, **cy0_kwargs,
        )

        # Plot the per-well averaged trace (mV) on the well's own
        # baseline-anchored time axis.
        t = well.time_min
        sig_mV = well.well_2d_bs_active_mean * 1e3
        ax.plot(t, sig_mV, color="#444", linewidth=1.4)
        ax.axvline(t_safe_ttp, color="grey", linestyle=":", linewidth=0.7,
                   alpha=0.6, label="TTP search start")
        if t_safe_sc != t_safe_ttp:
            ax.axvline(t_safe_sc, color="#aa6611", linestyle=":",
                       linewidth=0.7, alpha=0.6,
                       label="SDM/Cy0 search start")

        # qLAMP gold-standard reference (if supplied for this well).
        qlamp_val = (qlamp_ttp_per_well[i_well]
                     if qlamp_ttp_per_well is not None
                       and i_well < len(qlamp_ttp_per_well)
                     else float("nan"))
        if not np.isnan(qlamp_val):
            ax.axvline(qlamp_val, **QLAMP_STYLE,
                       label=f"qLAMP gold-std = {qlamp_val:.2f} min")

        # Overlay each landmark as a vertical line + a marker at the
        # landmark's time-on-trace y-value.
        for name, val in [("TTP", ttp_val), ("SDM", sdm_val), ("Cy0", cy0_val)]:
            if np.isnan(val):
                continue
            sty = LANDMARK_STYLES[name]
            y_on = _landmark_value_on_signal(t, sig_mV, val)
            ax.axvline(val, color=sty["color"], linestyle="--", linewidth=1.0,
                       alpha=0.8)
            delta = ""
            if not np.isnan(qlamp_val):
                delta = f"  ({val - qlamp_val:+.2f} vs qLAMP)"
            ax.plot([val], [y_on], marker=sty["marker"],
                    color=sty["color"], markersize=10, zorder=5,
                    label=f"{sty['label']} = {val:.2f} min{delta}")

        well_label = (well_labels[i_well] if well_labels and i_well < len(well_labels)
                      else f"well {i_well}")
        ax.set_title(f"{chip_label} — {well_label}", fontsize=10)
        ax.set_xlabel("Time [min]", fontsize=9)
        ax.set_ylabel("V_well_avg [mV]", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, frameon=True)

    # Hide unused subplots if any
    for j in range(n_wells, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
