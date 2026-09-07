"""Shared infrastructure for every quantification method.

Holds the bits that don't change between method families: project
paths, chip lists, search-window defaults, the MATLAB-style centred
moving-average helper, and the titan-preprocessed chip loader.

Method-specific extractors live under ``methods/``; runner scripts
live under ``runners/``.  Both pull their config and the loader from
here.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Path setup so titan is importable regardless of cwd. Missing titan is a
# soft failure: import still succeeds, but any function that actually needs
# titan (e.g. titan_load_and_preprocessing) will fail on call. Analysis of
# pre-built caches works without titan.
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
QUANT_DIR = THIS_FILE.parent
# src/lacewing/quantification/common.py -> repo root is parents[3]
_REPO_ROOT_DEFAULT = QUANT_DIR.parents[2]

_env_titan = os.environ.get("LACEWING_TITAN_PATH")
_env_data = os.environ.get("LACEWING_DATA_ROOT")

# PROJECT_ROOT is the anchor for data lookups (DATA_ROOT below). Override
# with LACEWING_DATA_ROOT if the raw Data/ tree lives elsewhere; otherwise
# it defaults to the repo root (one level above src/).
PROJECT_ROOT: Path = Path(_env_data) if _env_data else _REPO_ROOT_DEFAULT

_titan_found = False
if _env_titan and Path(_env_titan).exists():
    sys.path.insert(0, str(_env_titan))
    _titan_found = True
else:
    for root in [PROJECT_ROOT, _REPO_ROOT_DEFAULT]:
        titan_pkg = root / "Code" / "titan-signal-processing"
        if titan_pkg.exists():
            sys.path.insert(0, str(titan_pkg))
            _titan_found = True
            break
        titan_pkg = root / "titan-signal-processing"
        if titan_pkg.exists():
            sys.path.insert(0, str(titan_pkg))
            _titan_found = True
            break

if _titan_found:
    from titan.load_and_preprocessing import titan_load_and_preprocessing  # noqa: E402
else:
    warnings.warn(
        "titan-signal-processing not found on disk. Raw-chip processing and "
        "cache-building will fail; analysis of pre-built caches still works. "
        "Set LACEWING_TITAN_PATH to enable full functionality.",
        stacklevel=2,
    )
    def titan_load_and_preprocessing(*args, **kwargs):
        raise RuntimeError(
            "titan-signal-processing not installed. Set LACEWING_TITAN_PATH."
        )


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

DATA_ROOT = PROJECT_ROOT / "Data" / "24_CoV_Quantification"
RESULTS_DIR = QUANT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Chip lists
# ---------------------------------------------------------------------------

# Final dataset selection (Lacewing_Files_Final.mat - 1 chip per concentration).
EXPERIMENTS_FINAL: dict[str, Path] = {
    "1e5": DATA_ROOT / "1e5" / "D20240821_E01_C07_F4500KHz_U_1e5",
    "1e6": DATA_ROOT / "1e6" / "D20240821_E01_C09_F4500KHz_U_1e6",
    "1e7": DATA_ROOT / "1e7" / "D20240820_E02_C03_F4500KHz_U_1e7",
    "1e8": DATA_ROOT / "1e8" / "D20240822_E02_C00_F4500KHz_U_1e8",
    "1e9": DATA_ROOT / "1e9" / "D20240822_E02_C00_F4500KHz_U_1e9",
}

# Extra chips outside the dose-response series.
EXPERIMENTS_EXTRA: dict[str, Path] = {
    "SD": DATA_ROOT / "Others" / "D20240719_E03_C44_F4500KHz_U_COV_SD",
}

CONC_KEYS = ["1e5", "1e6", "1e7", "1e8", "1e9"]
CONC_LOG = [5, 6, 7, 8, 9]
EXP_LABELS = ["A", "B", "C", "D", "E"]
WELLS_FOR_QUANT = [0, 1, 2, 3]            # MATLAB ampl_peaks(:, 1:4) in 0-indexed

# SD multiplex chip well-to-reaction mapping.
SD_WELL_LABELS = ["E", "D", "C", "B", "A", "NTC"]
SD_LABEL_LOG = {"A": 5, "B": 6, "C": 7, "D": 8, "E": 9}


# ---------------------------------------------------------------------------
# titan preprocessing settings
# ---------------------------------------------------------------------------

N_WELLS = 6
N_A_TYPE = "v01"
END_TIME_MIN = 40


# ---------------------------------------------------------------------------
# Default landmark-extraction parameters (shared by every method)
# ---------------------------------------------------------------------------

PEAK_SEARCH_START_MIN = 7.0
SEARCH_END_MIN = 35.0           # MATLAB uses x_end_diff = N - 6
SMOOTH_ORDER = 10               # MATLAB n_filter_order=10
THRESHOLD_FRAC = 0.4            # MATLAB 0.4-threshold for the baseline TTP rule


# ---------------------------------------------------------------------------
# Supervisor reference values (for the baseline-TTP comparison)
# ---------------------------------------------------------------------------

# From Lacewing_CoV_Quantification.m and Lacewing_Quantification_Results.xlsx.
# Order matches CONC_KEYS = [1e5, 1e6, 1e7, 1e8, 1e9].
SUPERVISOR_AMPL_PEAKS_4REP = np.array([25.90, 23.50, 20.52, 15.26, 14.78])
SUPERVISOR_AMPL_PEAKS_5REP = np.array([23.72, 21.59, 19.37, 15.06, 14.92])

# qLAMP gold-standard reference (Lacewing_Readout_DNA_Quant.m line 803).
TTP_QLAMP = np.array([
    [20.01, 18.52, 18.61, 19.41, 18.84, 22.30, 29.83, 20.85],   # 1e5
    [17.34, 17.46, 17.53, 18.64, 16.93, 16.97, 15.57, 15.55],   # 1e6
    [14.62, 14.55, 14.56, 14.94, 15.46, 15.38, 14.83, 15.56],   # 1e7
    [11.53, 11.55, 11.52, 11.62,  9.62,  9.68,  9.46,  9.82],   # 1e8
    [ 9.49,  9.55,  9.42,  9.54,  7.22,  7.24,  7.93,  8.23],   # 1e9
])

MATLAB_COLORS = [
    (0.0000, 0.4470, 0.7410),  # blue
    (0.8500, 0.3250, 0.0980),  # red
    (0.9290, 0.6940, 0.1250),  # yellow
    (0.4940, 0.1840, 0.5560),  # purple
    (0.4660, 0.6740, 0.1880),  # green
    (0.3010, 0.7450, 0.9330),  # cyan
]


# ---------------------------------------------------------------------------
# MATLAB-style smoothing helper
# ---------------------------------------------------------------------------

def smooth_centered(signal: np.ndarray, span: int) -> np.ndarray:
    """Centered moving average matching MATLAB's smooth(signal, span).

    MATLAB silently rounds even spans down to the nearest odd integer
    (so smooth(x, 100) effectively uses a 99-sample window).  This
    implementation matches that behaviour: ``half = (span-1)//2``
    gives a window of ``2*half + 1`` samples, with truncated symmetric
    handling at the array boundaries.
    """
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


# ---------------------------------------------------------------------------
# Per-well search-start computation (99.5% thermal-settling point)
# ---------------------------------------------------------------------------

def get_matlab_search_start(well) -> float:
    """Dynamic 99.5% thermal settling point in minutes-from-stable-window."""
    temp_1d = well.well_temp_mean_NEW
    time_1d = well.time_min

    steady_state_mean = np.mean(temp_1d[len(temp_1d)//2:])
    target_temp = 0.995 * steady_state_mean
    idx_hit = np.where(temp_1d >= target_temp)[0]

    return float(time_1d[idx_hit[0]]) if idx_hit.size > 0 else 0.0


def get_matlab_search_start_absolute(well) -> float:
    """99.5% thermal settling point in absolute minutes from file start."""
    temp_1d = well.well_temp_mean_NEW
    time_abs = well.time_npr[well.idx_settled : well.idx_end] / 60.0

    steady_state_mean = np.mean(temp_1d[len(temp_1d)//2:])
    target_temp = 0.995 * steady_state_mean
    idx_hit = np.where(temp_1d >= target_temp)[0]

    return float(time_abs[idx_hit[0]]) if idx_hit.size > 0 else float(time_abs[0])


# ---------------------------------------------------------------------------
# Generic derivative-peak finder (used by SD-multiplex plotting)
# ---------------------------------------------------------------------------

def compute_diff_signals_variable(
    time_min: np.ndarray, signal: np.ndarray, order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (diff_smooth_mV_per_s, t_diff_min) for a specific smoothing order."""
    dt_sec = float(time_min[1] - time_min[0]) * 60.0
    scale = 1e3 / dt_sec
    sig_smooth = smooth_centered(signal, order)
    diff_smooth = smooth_centered(np.diff(sig_smooth), order) * scale
    t_diff = time_min[1:1 + len(diff_smooth)]
    return diff_smooth, t_diff


def find_diff_peak_time(
    time_min: np.ndarray, signal: np.ndarray,
    smooth_order: int,
    search_start_min: float,
    search_end_min: float = SEARCH_END_MIN,
) -> float:
    """Time of the largest derivative peak (smoothed at ``smooth_order``)
    inside ``[search_start_min, search_end_min]``.  NaN if no peak."""
    idx_start = int(np.searchsorted(time_min, search_start_min))
    idx_end = int(np.searchsorted(time_min, search_end_min))
    sig_smooth = smooth_centered(signal, smooth_order)
    diff_smooth = smooth_centered(np.diff(sig_smooth), smooth_order)

    if idx_end > len(diff_smooth):
        idx_end = len(diff_smooth)
    if idx_end - idx_start < 10:
        return float("nan")
    search_slice = diff_smooth[idx_start:idx_end]
    peaks, props = find_peaks(search_slice, width=1, prominence=0.00001)
    if len(peaks) == 0:
        return float("nan")
    best = int(np.argmax(search_slice[peaks] * props["widths"]))
    peak_loc = int(peaks[best] + idx_start)
    return float(time_min[peak_loc])


# ---------------------------------------------------------------------------
# Chip loader
# ---------------------------------------------------------------------------

def load_chip(exp_path: Path, label: str = ""):
    """titan_load_and_preprocessing wrapper with a friendly error message."""
    if not exp_path.exists():
        print(f"  MISSING folder: {exp_path}")
        return None
    print(f"  Loading {label}: {exp_path.name}", flush=True)
    try:
        return titan_load_and_preprocessing(
            exp_path,
            n_wells=N_WELLS,
            start_type="temperature",
            n_a_type=N_A_TYPE,
            end_time_min=END_TIME_MIN,
            print_status=False,
        )
    except (TypeError, Exception) as e:
        print(f"    FAILED ({type(e).__name__}): {e}")
        return None
