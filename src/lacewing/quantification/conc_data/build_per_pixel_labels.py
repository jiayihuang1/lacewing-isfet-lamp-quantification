"""Data · KP per-pixel label construction. [Cat A] Report §NewData.

Build per-pixel TTP labels for the 5 Concentration Data Experiment chips.

Recipe (matches UTI plate_segmentation_cache but for the KP-conc data):

  Per well w:
    well_anchor = cy0_after_plate(well_mean_signal, plate_qLAMP_TTP)
                    - cy0 tangent-at-inflection restricted to a search window
                      around the plate anchor (so the search doesn't lock onto
                      the pre-amp thermal transient)
    Per pixel p in well w:
      pixel_shift[p] = argmax lag of cross-correlate(pixel_trace, well_mean_trace)
                        constrained to +/- MAX_LAG_MIN
      per_pixel_ttp[p] = well_anchor + pixel_shift[p] * dt

Output:
  Analysis/quantification/conc_data/output/per_pixel_labels.csv
    columns: chip_tag, well_idx, well_label, log10_conc, pixel_id,
             plate_ttp_min, well_anchor_min, shift_samples,
             shift_min, per_pixel_ttp_min

  Analysis/quantification/conc_data/output/per_pixel_shift_hist.png
    5-row figure (one per chip). Each row: 4 columns for the 4 concentrations.
    Histogram of per-pixel shifts (min) with per-conc summary stats.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate
from lacewing.quantification.conc_data.process_conc_chips import (  # noqa: E402
    _build_chip_configs,
    PLATE_FEATURES_JSON,
    _LOG10_TO_KP,
)
from lacewing.quantification.chip_pipeline import process_chip as pipe  # noqa: E402
from lacewing.quantification.systematic_ttp import compute_all_variants  # noqa: E402
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LAG_MIN = 5.0        # ±5 min lag window for xcorr (matches UTI)
XCORR_HALF_WINDOW_MIN = 8.0   # ±8 min around the well anchor for xcorr
                              # (isolates the rising flank so the correlation
                              # finds the per-pixel timing offset rather than
                              # being dominated by the flat pre-amp baseline)
SAMPLES_PER_MIN = 30     # 2s cycle time
MAX_LAG_SAMPLES = int(MAX_LAG_MIN * SAMPLES_PER_MIN)


def _plate_ttp_for_well(cfg, plate_per_conc, w_idx):
    """Return the per-well plate qLAMP TTP (in min) or NaN for PTC/NTC/unknown."""
    log10 = cfg.well_log10.get(w_idx)
    if log10 is None or not np.isfinite(log10):
        return float("nan")
    kp_label = _LOG10_TO_KP.get(float(log10))
    if kp_label is None:
        return float("nan")
    entry = plate_per_conc.get(kp_label)
    if entry is None:
        return float("nan")
    return float(entry.get("ttp_min"))


def _pixel_shift(pixel_trace, well_mean_trace, anchor_sample,
                 half_window_samples, max_lag_samples):
    """Integer sample lag that best aligns pixel to well-mean, within ±max_lag_samples.

    Restricts the correlation to a window around `anchor_sample` (typically
    the cy0_after_plate index) so the pre-amp baseline doesn't dominate the
    correlation. Positive shift → pixel amplifies LATER than the well mean.
    """
    n = len(pixel_trace)
    lo = max(0, anchor_sample - half_window_samples)
    hi = min(n, anchor_sample + half_window_samples)
    if hi - lo < 5:
        return 0
    p = np.nan_to_num(pixel_trace[lo:hi].astype(np.float64), nan=0.0)
    m = np.nan_to_num(well_mean_trace[lo:hi].astype(np.float64), nan=0.0)
    p_std, m_std = float(p.std()), float(m.std())
    if p_std < 1e-9 or m_std < 1e-9:
        return 0
    p = (p - p.mean()) / p_std
    m = (m - m.mean()) / m_std
    xcorr = correlate(p, m, mode="full")
    lags = np.arange(-(len(m) - 1), len(p))
    valid = np.abs(lags) <= max_lag_samples
    if not valid.any():
        return 0
    return int(lags[valid][np.argmax(xcorr[valid])])


def process_chip(cfg):
    """Return list of per-pixel-label rows for one chip."""
    print(f"\n{'=' * 70}\nProcessing {cfg.chip_tag}\n{'=' * 70}")
    if not cfg.chip_dir.exists():
        print(f"  [SKIP] chip_dir not found: {cfg.chip_dir}")
        return []

    plate_data = json.loads(PLATE_FEATURES_JSON.read_text())
    plate_per_conc = plate_data[cfg.plate_key]["per_conc_kp"]

    old_labels = pipe.WELL_LABELS
    old_log10  = pipe.WELL_LOG10
    old_ntc    = pipe.NTC_WELLS
    old_trim   = pipe.TRIM_SEARCH_MAX_MIN
    try:
        pipe.WELL_LABELS = cfg.well_labels
        pipe.WELL_LOG10  = cfg.well_log10
        pipe.NTC_WELLS   = tuple(cfg.ntc_wells)
        pipe.TRIM_SEARCH_MAX_MIN = cfg.trim_search_max_min

        result = pipe.process_chip(cfg.chip_dir, save_per_pixel=True)

    finally:
        pipe.WELL_LABELS = old_labels
        pipe.WELL_LOG10  = old_log10
        pipe.NTC_WELLS   = old_ntc
        pipe.TRIM_SEARCH_MAX_MIN = old_trim

    rows = []
    for w in result.wells:
        plate_ttp = _plate_ttp_for_well(cfg, plate_per_conc, w.well)
        if not np.isfinite(plate_ttp):
            print(f"  well {w.well} ({w.label}): no plate TTP → skip labelling")
            continue
        if w.per_pixel_qc is None or len(w.per_pixel_qc) == 0:
            print(f"  well {w.well} ({w.label}): no per-pixel data → skip")
            continue

        # Well-mean anchor via cy0_after_plate.
        variants = compute_all_variants(
            w.mean_after_spat, w.time_min, plate_ttp_min=plate_ttp,
        )
        well_anchor = variants["cy0_after_plate"].ttp_min
        if not np.isfinite(well_anchor):
            print(f"  well {w.well} ({w.label}): cy0_after_plate = NaN → skip")
            continue

        # Well-mean of the QC-filtered pixels (pre-spatA3) is the alignment target.
        well_mean_trace = w.per_pixel_qc.mean(axis=0)
        n_pix = w.per_pixel_qc.shape[0]
        dt = float(w.time_min[1] - w.time_min[0])  # ≈ 1/SAMPLES_PER_MIN
        max_lag_samples = int(MAX_LAG_MIN / dt)
        half_window_samples = int(XCORR_HALF_WINDOW_MIN / dt)
        anchor_sample = int(np.searchsorted(w.time_min, well_anchor))

        shifts = np.empty(n_pix, dtype=np.int32)
        for i in range(n_pix):
            shifts[i] = _pixel_shift(
                w.per_pixel_qc[i], well_mean_trace,
                anchor_sample, half_window_samples, max_lag_samples,
            )

        print(f"  well {w.well} ({w.label:>7}): n_pix={n_pix:>4}  "
              f"plate={plate_ttp:.3f}  cy0_after_plate={well_anchor:.3f}  "
              f"shift_mean={shifts.mean() * dt:+.3f}min  "
              f"shift_std={shifts.std() * dt:.3f}min  "
              f"shift_range=[{shifts.min() * dt:+.2f},{shifts.max() * dt:+.2f}]min")

        log10 = cfg.well_log10.get(w.well, float("nan"))
        for pid, sh in enumerate(shifts):
            rows.append({
                "chip_tag":        cfg.chip_tag,
                "chip_label":      cfg.chip_label,
                "well_idx":        w.well,
                "well_label":      w.label,
                "log10_conc":      log10 if np.isfinite(log10) else None,
                "pixel_id":        pid,
                "plate_ttp_min":   plate_ttp,
                "well_anchor_min": well_anchor,
                "shift_samples":   int(sh),
                "shift_min":       float(sh) * dt,
                "per_pixel_ttp_min": float(well_anchor + sh * dt),
            })
    return rows


def plot_shift_histograms(df: pd.DataFrame, out_path: Path):
    """One row per chip, one column per concentration; hist of shift_min."""
    chip_tags = sorted(df["chip_tag"].unique())
    concs = sorted(df["log10_conc"].dropna().unique())
    n_rows, n_cols = len(chip_tags), len(concs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.4 * n_rows),
                             sharex=True, sharey=False, squeeze=False)
    for r, ct in enumerate(chip_tags):
        for c, conc in enumerate(concs):
            ax = axes[r][c]
            sub = df[(df["chip_tag"] == ct) & (df["log10_conc"] == conc)]
            if len(sub) == 0:
                ax.set_visible(False)
                continue
            vals = sub["shift_min"].values
            ax.hist(vals, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
            ax.axvline(0, color="#888", lw=0.8, ls="--")
            ax.axvline(vals.mean(), color="#c81c1c", lw=1.1,
                       label=f"mean={vals.mean():+.2f}")
            ax.set_title(f"{ct[:16]} · 10^{int(conc)}  (n={len(vals)}, "
                         f"std={vals.std():.2f})", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.tick_params(labelsize=7)
            if r == n_rows - 1:
                ax.set_xlabel("pixel shift (min)", fontsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.suptitle(
        f"Per-pixel shift vs well-mean via cross-correlation  (max lag ±{MAX_LAG_MIN} min)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_path}")


def main():
    all_rows = []
    for cfg in _build_chip_configs():
        all_rows.extend(process_chip(cfg))
    df = pd.DataFrame(all_rows)
    csv_out = OUT_DIR / "per_pixel_labels.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nwrote {csv_out}  ({len(df):,} pixel rows across "
          f"{df['chip_tag'].nunique()} chips)")

    # Summary
    print("\n=== per-chip / per-conc summary ===")
    summary = df.groupby(["chip_tag", "log10_conc"]).agg(
        n_pix=("pixel_id", "count"),
        plate_ttp=("plate_ttp_min", "first"),
        anchor=("well_anchor_min", "first"),
        shift_mean=("shift_min", "mean"),
        shift_std=("shift_min", "std"),
        shift_min=("shift_min", "min"),
        shift_max=("shift_min", "max"),
    ).round(3)
    print(summary.to_string())

    # Note: wells at the same concentration on the same chip may have different
    # cy0_after_plate anchors (they're different wells), so 'anchor' shown per
    # (chip, conc) is FIRST-well-only. See CSV for the full per-well breakdown.

    plot_shift_histograms(df, OUT_DIR / "per_pixel_shift_hist.png")


if __name__ == "__main__":
    main()
