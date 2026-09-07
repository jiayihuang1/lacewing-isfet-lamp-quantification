"""Data · chip-processing pipeline (used by both UTI and KP data). [Cat A] Report §Data (chip preprocessing).

Process one UTI-trial chip end-to-end with the rule-based TTP methods.

Chip: Data/UTI Trial Data/26-07-28 E. Coli Serial Dilution Synthetic/
        D20260727_E00_C00_F4500KHz_U_E_coli_03/

Well layout (from user photo, 10-well chip):
    Row 1: well 0 = 1e5,  well 1 = 1e5
    Row 2: well 2 = 1e4,  well 3 = 1e4
    Row 3: well 4 = 5e3,  well 5 = 5e3
    Row 4: well 6 = 1e2,  well 7 = 1e2
    Row 5: well 8 = NTC,  well 9 = NTC

Pipeline (matches training-time preprocessing for the classifier /
quantifier caches — cache stem `dataset_all_filt_abcd_ntcRaw_madk1p5_spatA3`):

    1. Load via multi-titan (n_wells=10).
    2. Artefact trim (per-chip max of per-well cuts; shifts idx_settled).
    3. Extract per-pixel traces (baseline-subtracted, 450-sample window).
    4. Pixel-level QC: MAD-A/B/C/D (k=1.5) with NTC-pooled thresholds.
    5. Spatial smoothing spatA3 (order=3 → up to 48 neighbours pooled).
    6. Well-mean signal per well.
    7. Rule-based TTP: ttp_threshold_derivative, cy0, sdm.

Emits JSON summary + all-stage per-well curves to
    Analysis/quantification/chip_pipeline/output/e_coli_03_stages.json

Run:
    python -m lacewing.quantification.chip_pipeline.process_chip
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# CRITICAL: pin the multi-titan clone BEFORE any titan import.
# lacewing.quantification.multi_data_shared._titan_setup inserts Code/titan-signal-processing-multi
# at the front of sys.path. Must run before load_chip_combined / trim /
# _extract_per_pixel_traces (which all `import titan.*`).
# We do NOT import lacewing.quantification.common in this module because
# common.py unconditionally installs the OLD titan-signal-processing on
# sys.path in its module body. TTP extractors are inlined below to avoid it.
# ---------------------------------------------------------------------------
from lacewing.quantification.multi_data_shared import _titan_setup  # noqa: F401

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from lacewing.quantification.multi_data_shared.load_combined import load_chip_combined
from lacewing.quantification.multi_data_shared.artefact_trim import trim_chip_artefact
from lacewing.quantification.multi_data_shared.build_dataset_multi import _extract_per_pixel_traces
from lacewing.quantification.multi_data_shared.build_filtered_dataset_multi import (
    _derive_rows_cols_for_chip,
)
from lacewing.quantification.multi_data_shared._filter_multi import (
    run_filter_pipeline_per_chip_mad_multi,
)
from lacewing.preprocessing.spatial_filter.spatial_smooth import _smooth_one_well
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


# ---------------------------------------------------------------------------
# Chip + well-layout config
# ---------------------------------------------------------------------------

PROJECT_ROOT = DATA_ROOT  # was: parents[3]
CHIP_PATH = (
    PROJECT_ROOT
    / "Data"
    / "UTI Trial Data"
    / "26-07-28 E. Coli Serial Dilution Synthetic"
    / "D20260727_E00_C00_F4500KHz_U_E_coli_03"
)
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# Well index → concentration string (for labels / plots).
WELL_LABELS = {
    0: "1e5 (L)", 1: "1e5 (R)",
    2: "1e4 (L)", 3: "1e4 (R)",
    4: "5e3 (L)", 5: "5e3 (R)",
    6: "1e2 (L)", 7: "1e2 (R)",
    8: "NTC (L)", 9: "NTC (R)",
}
# Well index → log10 concentration (for scatter/dose-response later).
# NTC wells → NaN.
WELL_LOG10 = {
    0: 5.0, 1: 5.0, 2: 4.0, 3: 4.0, 4: np.log10(5e3),
    5: np.log10(5e3), 6: 2.0, 7: 2.0, 8: float("nan"), 9: float("nan"),
}
NTC_WELLS = (8, 9)

# Optional override for the artefact-trim `search_max_min` parameter (default
# lives in trim_chip_artefact = 5.0 min). Monkey-patched by
# process_all_uti_chips per ChipConfig.trim_search_max_min. None → use the
# trim algorithm's own default.
TRIM_SEARCH_MAX_MIN: float | None = None

# LightCycler 96 ground-truth mapping (from Protocol 26-07-27.docx page 2).
# The LC96 plate carries triplicates of the SAME serial-dilution the chip
# receives, but at 10× lower concentrations per rung (LC96 E5 = 4.17e4 copies
# vs chip 1e5). Absolute Cq/TTP is not directly comparable, but the ORDERING
# and slope should agree.
#
# LC96 pos → chip well index it corresponds to (E-notation on the LC96,
# concentration on the chip):
#   B2, C2, D2 = E5 (LC96 4.17e4) → chip 1e5 → wells 0, 1
#   B3, C3, D3 = E4 (LC96 4.17e3) → chip 1e4 → wells 2, 3
#   B4, C4, D4 = E3 (LC96 4.17e2) → chip 5e3 → wells 4, 5   (supervisor: E3 loading unreliable)
#   B5, C5, D5 = E2 (LC96 4.17e1) → chip 1e2 → wells 6, 7   (supervisor: E2 loading unreliable)
#   F2, F3, F4 = NC                → chip NTC → wells 8, 9
#
# The mapping file lives in the same folder as the chip data. When present,
# the payload gains an "lc96" block per well with the triplicate Cq values.
LC96_EXPORT = (
    PROJECT_ROOT
    / "Data"
    / "UTI Trial Data"
    / "26-07-28 E. Coli Serial Dilution Synthetic"
    / "EP_26-06-27_Chip_LNCaP - Abs Quant.txt"
)
LC96_WELL_MAP = {
    0: ["B2", "C2", "D2"], 1: ["B2", "C2", "D2"],
    2: ["B3", "C3", "D3"], 3: ["B3", "C3", "D3"],
    4: ["B4", "C4", "D4"], 5: ["B4", "C4", "D4"],
    6: ["B5", "C5", "D5"], 7: ["B5", "C5", "D5"],
    8: ["F2", "F3", "F4"], 9: ["F2", "F3", "F4"],
}
# Supervisor flagged the low-concentration rows (LC96 E2 and E3) as unreliable
# — the LC96 replicate Cq scatter is inconsistent with the dilution ladder,
# and the chip's 5e3/1e2 wells show the same wet-loading artefact.
LC96_UNRELIABLE_WELLS = {4, 5, 6, 7}

# Filter/smoothing config — matches the training cache
# `dataset_all_filt_abcd_ntcRaw_madk1p5_spatA3`.
MAD_K = 1.5
FILTER_LAYERS = "ABCD"
CLEAN_NTC_FIRST = False   # matches the ntcRaw suffix in the cache stem
SPAT_ORDER = 3

# Rule-based TTP config — matches Analysis/quantification/common.py.
# SMOOTH_ORDER=10 is the plate-processing pipeline's legacy MATLAB constant
# and is retained for the plot-side derivative rendering at the bottom of
# this file. All three chip-TTP extractors (threshold_derivative, cy0, sdm)
# use TTP_SMOOTH_ORDER=100 to match the deployed MATLAB Lacewing readout
# and the Analysis/quantification/methods/ defaults.
SMOOTH_ORDER = 10
TTP_SMOOTH_ORDER = 100
SEARCH_END_MIN = 35.0
THRESHOLD_FRAC = 0.4
CY0_BASELINE_WINDOW_MIN = 1.0

# Titan preprocessing.
N_WELLS = 10
# n_a_type=v06 matches lacewing.quantification.multi_data_shared.load_combined.DEFAULT_N_A_TYPE
# — the value used for every Data/Multi/ chip we built training caches on.
# titan's v05 and v06 share the same parser branch (load_functions.py:72)
# so both "accept" the file, but v06 is what the training pipeline uses.
N_A_TYPE = "v06"
END_TIME_MIN = 40


# ---------------------------------------------------------------------------
# Signal-processing helpers (MATLAB-parity smooth + rule extractors).
# Inlined from lacewing.quantification.common + methods.{ttp_threshold_derivative,
# cy0, sdm} to avoid importing common.py (which pulls in the OLD titan).
# ---------------------------------------------------------------------------

def smooth_centered(signal: np.ndarray, span: int) -> np.ndarray:
    """Centered moving average, MATLAB-style (even spans → span-1)."""
    if span <= 1:
        return signal.copy()
    n = len(signal)
    half = (span - 1) // 2
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.mean(signal[lo:hi])
    return out


def get_matlab_search_start(well) -> float:
    """99.5% thermal-settling point (per-well), minutes from t=0."""
    temp_1d = well.well_temp_mean_NEW
    time_1d = well.time_min
    steady_mean = float(np.mean(temp_1d[len(temp_1d) // 2:]))
    target = 0.995 * steady_mean
    idx_hit = np.where(temp_1d >= target)[0]
    return float(time_1d[idx_hit[0]]) if idx_hit.size > 0 else 0.0


def extract_ttp_threshold(
    time_min: np.ndarray, signal: np.ndarray,
    search_start_min: float, search_end_min: float = SEARCH_END_MIN,
    smooth_order: int = TTP_SMOOTH_ORDER, threshold_frac: float = THRESHOLD_FRAC,
) -> tuple[float, float]:
    """MATLAB baseline: 0.4-threshold on smoothed 1st derivative.

    Returns (ttp_min, peak_min); either may be NaN if no peak found.
    """
    idx_start = int(np.searchsorted(time_min, search_start_min))
    idx_end = int(np.searchsorted(time_min, search_end_min))

    diff_raw = np.diff(signal)
    sig_smooth = smooth_centered(signal, smooth_order)
    diff_smooth = smooth_centered(np.diff(sig_smooth), smooth_order)

    search_slice = diff_smooth[idx_start:idx_end]
    if len(search_slice) < 10:
        return float("nan"), float("nan")

    peaks, props = find_peaks(search_slice, width=1, prominence=0.00001)
    if len(peaks) == 0:
        return float("nan"), float("nan")

    best = int(np.argmax(search_slice[peaks] * props["widths"]))
    peak_loc = int(peaks[best] + idx_start)
    peak_min = float(time_min[peak_loc])

    # Take-off (0.4-threshold) crossing — MATCH THE MATLAB PORT EXACTLY:
    # baseline = MIN of the smoothed derivative in [idx_start, peak];
    # threshold formula uses (peak_val - min); the crossing test then runs on
    # the RAW derivative (not the smoothed one) inside the same window and
    # returns the LAST sample below threshold.
    #
    # A previous transcription used mean-of-smoothed as the baseline, walked
    # backwards on the smoothed derivative, and returned the first sample
    # below threshold. That produced TTPs ~3 min earlier than canonical
    # across every well of this chip — see 2026-07-28 verify vs
    # lacewing.quantification.methods.ttp_threshold_derivative.extract_ttp.
    region_smooth = diff_smooth[idx_start:peak_loc + 1]
    if len(region_smooth) < 2:
        return float("nan"), peak_min
    thr_val = float(region_smooth.min()
                    + threshold_frac * (diff_smooth[peak_loc] - region_smooth.min()))

    below = np.where(diff_raw[idx_start:peak_loc + 1] < thr_val)[0]
    if len(below) == 0:
        return float("nan"), peak_min
    return float(time_min[idx_start + below[-1]]), peak_min


def extract_cy0(
    time_min: np.ndarray, signal: np.ndarray,
    search_start_min: float, search_end_min: float = SEARCH_END_MIN,
    smooth_order: int = TTP_SMOOTH_ORDER,
    baseline_window_min: float = CY0_BASELINE_WINDOW_MIN,
) -> float:
    """Cy0 tangent-at-inflection intercept (min from t=0)."""
    idx_start = int(np.searchsorted(time_min, search_start_min))
    idx_end = int(np.searchsorted(time_min, search_end_min))

    sig_smooth = smooth_centered(signal, smooth_order)
    diff_smooth = smooth_centered(np.diff(sig_smooth), smooth_order)

    search = diff_smooth[idx_start:idx_end]
    if len(search) < 10:
        return float("nan")
    peaks, props = find_peaks(search, width=1, prominence=0.00001)
    if len(peaks) == 0:
        return float("nan")
    best = int(np.argmax(search[peaks] * props["widths"]))
    peak_loc = int(peaks[best] + idx_start)

    # Baseline = median of smoothed signal over the 1 min before search_start.
    baseline_end_idx = idx_start
    baseline_start_idx = int(np.searchsorted(
        time_min, search_start_min - baseline_window_min))
    if baseline_end_idx - baseline_start_idx < 3:
        return float("nan")
    y_base = float(np.median(sig_smooth[baseline_start_idx:baseline_end_idx]))

    # Tangent at peak-of-derivative: y = m*(t - t0) + y0 where m is the
    # slope IN PER-MIN UNITS, not per-sample. diff_smooth[k] is a per-sample
    # delta (numpy.diff), so we divide by dt to convert to per-min.
    dt = float(time_min[1] - time_min[0])
    if dt <= 0:
        return float("nan")
    m = float(diff_smooth[peak_loc]) / dt
    if not np.isfinite(m) or abs(m) < 1e-6:
        return float("nan")
    t0 = float(time_min[peak_loc])
    y0 = float(sig_smooth[peak_loc])
    # Solve y_base = m*(t - t0) + y0 → t = t0 + (y_base - y0)/m.
    return t0 + (y_base - y0) / m


def extract_sdm(
    time_min: np.ndarray, signal: np.ndarray,
    search_start_min: float, search_end_min: float = SEARCH_END_MIN,
    smooth_order: int = TTP_SMOOTH_ORDER,
) -> float:
    """Second-derivative maximum (start of exponential phase)."""
    idx_start = int(np.searchsorted(time_min, search_start_min))
    idx_end = int(np.searchsorted(time_min, search_end_min))

    sig_smooth = smooth_centered(signal, smooth_order)
    diff_smooth = smooth_centered(np.diff(sig_smooth), smooth_order)

    search_d1 = diff_smooth[idx_start:idx_end]
    if len(search_d1) < 10:
        return float("nan")
    peaks_d1, props_d1 = find_peaks(search_d1, width=1, prominence=0.00001)
    if len(peaks_d1) == 0:
        return float("nan")
    best_d1 = int(np.argmax(search_d1[peaks_d1] * props_d1["widths"]))
    peak_d1_loc = int(peaks_d1[best_d1] + idx_start)

    # SDM = argmax of smoothed 2nd derivative on the RISING flank (up to
    # and INCLUDING the d1 peak). Uses find_peaks + best-by-(peak×width)
    # to be robust to noise — a plain argmax picks the loudest noise
    # spike, which is why the pre-fix version returned near-identical
    # values (~10.37 min) across most wells: it was latching onto a
    # recurring noise sample rather than a real curvature peak.
    d2_smooth = smooth_centered(np.diff(diff_smooth), smooth_order)
    search_d2 = d2_smooth[idx_start:peak_d1_loc + 1]
    if len(search_d2) < 10:
        return float("nan")
    peaks_d2, props_d2 = find_peaks(search_d2, width=1, prominence=1e-7)
    if len(peaks_d2) == 0:
        return float("nan")
    best_d2 = int(np.argmax(search_d2[peaks_d2] * props_d2["widths"]))
    sdm_idx = int(peaks_d2[best_d2] + idx_start)
    return float(time_min[sdm_idx])


# ---------------------------------------------------------------------------
# Per-well spatial smoothing driver (specialised for this single chip so we
# don't need spatial_coords.build_coords_for_chip, which is wired to the old
# titan clone).
# ---------------------------------------------------------------------------

def spatA_smooth_chip(
    X: np.ndarray,               # (N_keep, T)
    well_id: np.ndarray,          # (N_keep,)
    rows: np.ndarray,             # (N_keep,) well-local row
    cols: np.ndarray,             # (N_keep,) well-local col
    order: int,
) -> np.ndarray:
    """Spatial mean smoothing over Chebyshev-order-N 8-neighbours, per well."""
    X_out = np.empty_like(X)
    for w in sorted(set(int(v) for v in well_id)):
        mask = well_id == w
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            continue
        X_well = X[idxs]
        r_well = rows[idxs]
        c_well = cols[idxs]
        X_out[idxs] = _smooth_one_well(X_well, r_well, c_well, order=order)
    return X_out


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class WellStages:
    """Per-well signal at every stage of the pipeline.

    Well-mean arrays are 1-D (well-mean over surviving pixels at that stage).
    Optional per-pixel arrays hold every kept pixel's spatA3-smoothed trace.
    Time axis common to all stages: `time_min` (length = T = 450).
    """
    well: int
    label: str
    log10_conc: float
    time_min: np.ndarray
    n_active_raw: int         # after loose-active mask (pre-QC)
    n_kept: int               # after MAD-ABCD
    search_start_min: float
    mean_bs_raw: np.ndarray               # step 3 — per-pixel BS mean
    mean_after_qc: np.ndarray             # step 4 — mean of QC-surviving pixels
    mean_after_spat: np.ndarray           # step 5 — mean of spatA3-smoothed pixels
    smoothed_signal: np.ndarray           # centred MA(order=10) of mean_after_spat
    diff_smooth: np.ndarray               # 1st derivative of smoothed_signal
    diff2_smooth: np.ndarray              # 2nd derivative
    ttp_threshold: float                  # min from t=0 (may be NaN)
    ttp_threshold_peak: float             # 1st-deriv peak time (context)
    ttp_cy0: float                        # Cy0 intercept (min from t=0)
    ttp_sdm: float                        # SDM (min from t=0)
    # OPTIONAL per-pixel data — populated when the caller sets
    # `save_per_pixel=True` in process_chip. Shape: (n_kept, T)
    per_pixel_qc: np.ndarray | None = None        # BS + QC filtered, NO spatA3 (preserves per-pixel timing)
    per_pixel_spat: np.ndarray | None = None      # BS + QC + spatA3 (matches training-time input)
    per_pixel_row: np.ndarray | None = None       # (n_kept,) — chip row of each pixel
    per_pixel_col: np.ndarray | None = None       # (n_kept,) — chip col of each pixel


@dataclass
class ChipResult:
    chip_name: str
    chip_cut_min: float
    per_well_cut_min: list[float]
    per_well_had_artefact: list[bool]
    filter_info: dict = field(default_factory=dict)
    wells: list[WellStages] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_chip(chip_path: Path, save_per_pixel: bool = False) -> ChipResult:
    print(f"Loading {chip_path.name}")
    exp, diag = load_chip_combined(
        chip_path, n_wells=N_WELLS, n_a_type=N_A_TYPE,
        end_time_min=END_TIME_MIN, print_status=False,
    )
    print(f"  n_refs = {diag['n_refs']}")

    if TRIM_SEARCH_MAX_MIN is not None:
        trim = trim_chip_artefact(exp, apply=True, search_max_min=TRIM_SEARCH_MAX_MIN)
    else:
        trim = trim_chip_artefact(exp, apply=True)
    print(f"  Artefact trim: chip cut = {trim.chip_cut_min:.3f} min")
    print(f"  Per-well cuts (min): {[round(x, 2) for x in trim.per_well_cut_min]}")

    # ---- Per-pixel extraction (step 3) ----
    Xs, well_ids, pix_ids = [], [], []
    per_well_active: dict[int, int] = {}
    for w_idx in sorted(WELL_LABELS.keys()):
        if w_idx >= len(exp.wells_list):
            print(f"    WARN well {w_idx} out of range ({len(exp.wells_list)} wells)")
            continue
        well = exp.wells_list[w_idx]
        loose_mask = np.asarray(well.idx_active, dtype=bool)
        X, _, _ = _extract_per_pixel_traces(well, loose_mask)
        per_well_active[w_idx] = int(X.shape[0])
        if X.shape[0] == 0:
            continue
        Xs.append(X)
        well_ids.append(np.full(X.shape[0], w_idx, dtype=np.uint8))
        pix_ids.append(np.arange(X.shape[0], dtype=np.uint16))
    print(f"  Active pixels per well: {per_well_active}")

    X_all = np.concatenate(Xs, axis=0)
    well_all = np.concatenate(well_ids, axis=0).astype(np.int64)
    pix_all = np.concatenate(pix_ids, axis=0)

    # ---- (row, col) reconstruction (needed for spatA + layer C/D) ----
    rows, cols = _derive_rows_cols_for_chip(exp, well_all, pix_all)

    # ---- Step 4: MAD-A/B/C/D pixel QC ----
    ntc_mask = np.isin(well_all, list(NTC_WELLS))
    keep_mask, info = run_filter_pipeline_per_chip_mad_multi(
        X_all, well_all, rows, cols,
        ntc_well_mask=ntc_mask,
        layers=FILTER_LAYERS,
        clean_ntc_first=CLEAN_NTC_FIRST,
        k=MAD_K,
    )
    drop = info["n_dropped"]
    print(f"  MAD-ABCD (k={MAD_K}) kept {info['n_kept']}/{info['n_total']}; "
          f"drop A={drop['A']} B={drop['B']} C={drop['C']} D={drop['D']}")

    X_qc = X_all[keep_mask]
    well_qc = well_all[keep_mask]
    rows_qc = rows[keep_mask]
    cols_qc = cols[keep_mask]

    # ---- Step 5: spatA3 spatial smoothing on surviving pixels ----
    X_spat = spatA_smooth_chip(X_qc, well_qc, rows_qc, cols_qc, order=SPAT_ORDER)
    print(f"  spatA{SPAT_ORDER} applied")

    # ---- Steps 6-7 per well ----
    result = ChipResult(
        chip_name=chip_path.name,
        chip_cut_min=trim.chip_cut_min,
        per_well_cut_min=list(trim.per_well_cut_min),
        per_well_had_artefact=list(trim.per_well_had_artefact),
        filter_info={
            "layers": FILTER_LAYERS, "k": MAD_K,
            "n_total": int(info["n_total"]), "n_kept": int(info["n_kept"]),
            "n_dropped": {k: int(v) for k, v in drop.items()},
            "spat_order": SPAT_ORDER,
        },
    )

    # Time axis (same for every well after trim).
    time_min_ref = np.asarray(exp.wells_list[0].time_min, dtype=np.float64)
    onset_idx = int(np.searchsorted(time_min_ref, 0.0))
    T = X_all.shape[1]
    time_min_window = time_min_ref[onset_idx:onset_idx + T]
    if len(time_min_window) < T:
        # Pad if the trace was too short (matches _extract_per_pixel_traces's
        # zero-padding logic — time axis then extends beyond real samples).
        pad = np.linspace(
            float(time_min_window[-1]) + 1e-3,
            float(time_min_window[-1]) + 0.001 * (T - len(time_min_window)),
            T - len(time_min_window),
        )
        time_min_window = np.concatenate([time_min_window, pad])

    for w_idx in sorted(WELL_LABELS.keys()):
        if w_idx not in per_well_active:
            continue
        # Well-mean at each stage.
        mask_raw = well_all == w_idx
        mask_qc = well_qc == w_idx
        if mask_raw.sum() == 0:
            continue
        mean_raw = X_all[mask_raw].mean(axis=0)
        mean_qc = (X_qc[mask_qc].mean(axis=0) if mask_qc.any()
                   else np.full(T, np.nan))
        mean_spat = (X_spat[mask_qc].mean(axis=0) if mask_qc.any()
                     else np.full(T, np.nan))

        # Search start = per-well thermal-settle.
        well = exp.wells_list[w_idx]
        search_start = get_matlab_search_start(well)

        # Rule extractors on the well-mean spatA-smoothed signal.
        sig = mean_spat
        if np.all(np.isnan(sig)):
            ttp_thr, ttp_peak, ttp_cy0_v, ttp_sdm_v = (float("nan"),) * 4
            smoothed = np.full(T, np.nan)
            diff_s = np.full(T - 1, np.nan)
            diff2_s = np.full(T - 2, np.nan)
        else:
            ttp_thr, ttp_peak = extract_ttp_threshold(time_min_window, sig, search_start)
            ttp_cy0_v = extract_cy0(time_min_window, sig, search_start)
            ttp_sdm_v = extract_sdm(time_min_window, sig, search_start)
            smoothed = smooth_centered(sig, SMOOTH_ORDER)
            diff_s = smooth_centered(np.diff(smoothed), SMOOTH_ORDER)
            diff2_s = smooth_centered(np.diff(diff_s), SMOOTH_ORDER)

        # Optional per-pixel arrays for downstream segmentation labelling.
        # NOTE: we save the PRE-spatA3 (QC-only) per-pixel trace so cross-correlation
        # registration can pick up per-pixel timing variance. spatA3 averages each
        # pixel with up to 48 neighbours, which erases the timing variance we need
        # for per-pixel alignment. The training-time pipeline uses spat-smoothed
        # signals for input, but for LABEL construction we align on raw QC signals.
        if save_per_pixel and mask_qc.any():
            per_pixel_qc_w = X_qc[well_qc == w_idx].astype(np.float32, copy=False)
            per_pixel_row_w = rows_qc[well_qc == w_idx].astype(np.int32, copy=False)
            per_pixel_col_w = cols_qc[well_qc == w_idx].astype(np.int32, copy=False)
            # Also keep the spatA3 version for the model INPUT (labels come from raw).
            per_pixel_spat_w = X_spat[well_qc == w_idx].astype(np.float32, copy=False)
        else:
            per_pixel_qc_w = None
            per_pixel_spat_w = None
            per_pixel_row_w = None
            per_pixel_col_w = None

        result.wells.append(WellStages(
            well=w_idx,
            label=WELL_LABELS[w_idx],
            log10_conc=WELL_LOG10[w_idx],
            time_min=time_min_window,
            n_active_raw=int(mask_raw.sum()),
            n_kept=int(mask_qc.sum()),
            search_start_min=search_start,
            mean_bs_raw=mean_raw,
            mean_after_qc=mean_qc,
            mean_after_spat=mean_spat,
            smoothed_signal=smoothed,
            diff_smooth=diff_s,
            diff2_smooth=diff2_s,
            ttp_threshold=ttp_thr,
            ttp_threshold_peak=ttp_peak,
            ttp_cy0=ttp_cy0_v,
            ttp_sdm=ttp_sdm_v,
            per_pixel_qc=per_pixel_qc_w,
            per_pixel_spat=per_pixel_spat_w,
            per_pixel_row=per_pixel_row_w,
            per_pixel_col=per_pixel_col_w,
        ))
        print(f"    well {w_idx:>2} {WELL_LABELS[w_idx]:<10} "
              f"pixels raw={result.wells[-1].n_active_raw:>4} "
              f"kept={result.wells[-1].n_kept:>4} "
              f"search_start={search_start:.2f} min | "
              f"TTP thr={ttp_thr:.2f} cy0={ttp_cy0_v:.2f} sdm={ttp_sdm_v:.2f}")

    return result


# ---------------------------------------------------------------------------
# Serialisation for the HTML viewer
# ---------------------------------------------------------------------------

def _decimate(arr: np.ndarray, n: int) -> np.ndarray:
    """Uniform decimation to n points."""
    if arr is None or len(arr) <= n:
        return arr
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return arr[idx]


def _parse_lc96_cq_by_position(path: Path) -> dict[str, dict]:
    """Parse the LightCycler 96 'Abs Quant' TSV into {position: {cq, call}}.

    Returns e.g. {"B2": {"cq": 8.81, "call": "Positive"}, ...}.
    Negative / invalid wells get cq=None but keep their call for context.
    """
    import csv
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pos = (row.get("Position") or "").strip()
            if not pos:
                continue
            cq_raw = (row.get("Cq") or "").strip()
            try:
                cq = float(cq_raw)
            except ValueError:
                cq = None
            rows[pos] = {
                "cq": cq,
                "call": (row.get("Call") or "").strip(),
                "sample_name": (row.get("Sample Name") or "").strip(),
            }
    return rows


def _lc96_block_for_well(well_idx: int, cq_by_pos: dict[str, dict]) -> dict | None:
    """Build the per-well LC96 comparison block: triplicate Cq + mean."""
    positions = LC96_WELL_MAP.get(well_idx)
    if positions is None:
        return None
    reps = []
    for pos in positions:
        entry = cq_by_pos.get(pos)
        if entry is None:
            reps.append({"position": pos, "cq": None, "call": "Missing"})
        else:
            reps.append({"position": pos,
                         "cq": entry["cq"], "call": entry["call"]})
    finite_cqs = [r["cq"] for r in reps if r["cq"] is not None]
    return {
        "positions": positions,
        "replicates": reps,
        "cq_mean": (sum(finite_cqs) / len(finite_cqs)) if finite_cqs else None,
        "n_positive": len(finite_cqs),
        "n_replicates": len(reps),
        "unreliable": well_idx in LC96_UNRELIABLE_WELLS,
    }


def result_to_payload(result: ChipResult, decimate_to: int = 300) -> dict:
    """Emit a JSON-serialisable payload the HTML viewer consumes."""
    def _to_list(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return [None if not np.isfinite(v) else float(v) for v in x.tolist()]
        return x

    lc96_by_pos = _parse_lc96_cq_by_position(LC96_EXPORT) if LC96_EXPORT.exists() else {}

    wells = []
    for w in result.wells:
        wells.append({
            "well": w.well,
            "label": w.label,
            "log10_conc": (None if not np.isfinite(w.log10_conc) else w.log10_conc),
            "n_active_raw": w.n_active_raw,
            "n_kept": w.n_kept,
            "search_start_min": w.search_start_min,
            "time_min":       _to_list(_decimate(w.time_min, decimate_to)),
            "time_min_diff":  _to_list(_decimate(w.time_min[:-1], decimate_to)),
            "time_min_diff2": _to_list(_decimate(w.time_min[:-2], decimate_to)),
            "mean_bs_raw":     _to_list(_decimate(w.mean_bs_raw, decimate_to)),
            "mean_after_qc":   _to_list(_decimate(w.mean_after_qc, decimate_to)),
            "mean_after_spat": _to_list(_decimate(w.mean_after_spat, decimate_to)),
            "smoothed_signal": _to_list(_decimate(w.smoothed_signal, decimate_to)),
            "diff_smooth":     _to_list(_decimate(w.diff_smooth, decimate_to)),
            "diff2_smooth":    _to_list(_decimate(w.diff2_smooth, decimate_to)),
            "ttp": {
                "threshold_derivative": (None if not np.isfinite(w.ttp_threshold)
                                          else w.ttp_threshold),
                "threshold_deriv_peak": (None if not np.isfinite(w.ttp_threshold_peak)
                                          else w.ttp_threshold_peak),
                "cy0": None if not np.isfinite(w.ttp_cy0) else w.ttp_cy0,
                "sdm": None if not np.isfinite(w.ttp_sdm) else w.ttp_sdm,
            },
            "lc96": _lc96_block_for_well(w.well, lc96_by_pos) if lc96_by_pos else None,
        })

    return {
        "chip_name": result.chip_name,
        "chip_cut_min": result.chip_cut_min,
        "per_well_cut_min": [
            (None if not np.isfinite(v) else float(v))
            for v in result.per_well_cut_min
        ],
        "per_well_had_artefact": [bool(v) for v in result.per_well_had_artefact],
        "filter_info": result.filter_info,
        "wells": wells,
        "config": {
            "n_wells": N_WELLS,
            "n_a_type": N_A_TYPE,
            "end_time_min": END_TIME_MIN,
            "mad_k": MAD_K,
            "filter_layers": FILTER_LAYERS,
            "spat_order": SPAT_ORDER,
            "smooth_order": SMOOTH_ORDER,
            "search_end_min": SEARCH_END_MIN,
            "threshold_frac": THRESHOLD_FRAC,
        },
    }


def main() -> None:
    if not CHIP_PATH.exists():
        raise FileNotFoundError(f"Chip path not found: {CHIP_PATH}")
    result = process_chip(CHIP_PATH)
    payload = result_to_payload(result)

    out_json = OUT_DIR / "e_coli_03_stages.json"
    out_json.write_text(json.dumps(payload, indent=None))
    print(f"[ok] wrote {out_json}")

    # Compact per-well TTP summary to stdout + JSON side-car.
    print("\n=== TTP summary (min from t=0) ===")
    print(f"{'well':>4}  {'label':<10}  {'thr':>6}  {'peak':>6}  {'cy0':>6}  {'sdm':>6}  "
          f"{'n_raw':>5}  {'n_kept':>5}")
    summary_rows = []
    for w in result.wells:
        row = {
            "well": w.well, "label": w.label,
            "log10_conc": None if not np.isfinite(w.log10_conc) else w.log10_conc,
            "n_active_raw": w.n_active_raw, "n_kept": w.n_kept,
            "search_start_min": w.search_start_min,
            "ttp_threshold": None if not np.isfinite(w.ttp_threshold) else w.ttp_threshold,
            "ttp_threshold_peak": None if not np.isfinite(w.ttp_threshold_peak) else w.ttp_threshold_peak,
            "ttp_cy0": None if not np.isfinite(w.ttp_cy0) else w.ttp_cy0,
            "ttp_sdm": None if not np.isfinite(w.ttp_sdm) else w.ttp_sdm,
        }
        summary_rows.append(row)
        print(f"{w.well:>4}  {w.label:<10}  "
              f"{w.ttp_threshold:>6.2f}  {w.ttp_threshold_peak:>6.2f}  "
              f"{w.ttp_cy0:>6.2f}  {w.ttp_sdm:>6.2f}  "
              f"{w.n_active_raw:>5}  {w.n_kept:>5}")

    (OUT_DIR / "e_coli_03_ttp_summary.json").write_text(
        json.dumps(summary_rows, indent=2)
    )


if __name__ == "__main__":
    main()
