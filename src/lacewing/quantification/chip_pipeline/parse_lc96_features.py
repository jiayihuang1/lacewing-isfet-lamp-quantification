"""Parse Roche LightCycler 96 .lc96p files → per-well plate features.

For each well with amplification data, extracts:
  * baseline_mean, baseline_std   (cycles 1-5)
  * Cq_start                       — first cycle where fluor > baseline_mean + 3*baseline_std
  * Cq_takeoff                     — argmax of first derivative on the smoothed fluor curve
  * Cq_plateau                     — cycle where smoothed 1st deriv drops below 5% of its max, after Cq_takeoff
  * plateau_fluor                  — mean of last 5 cycles

All boundaries are in CYCLES (integer or fractional after interpolation).
Convert cycles → minutes via cycles × 0.5  (LC96 LAMP runs 30-second cycles).

The extraction uses only geometric features of the smoothed fluorescence
curve — no absolute thresholds or fitted models are used to place the
CRITICAL boundaries (Cq_takeoff, Cq_plateau). Cq_start uses a 3σ threshold
above baseline, which is a well-established qPCR convention.

Usage
-----
    python -m lacewing.quantification.chip_pipeline.parse_lc96_features
    → writes plate_features.json into each date folder.
"""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[3]
DATA_ROOT = PROJECT_ROOT / "Data"

# ---------------------------------------------------------------------------
# LC96 files we need to parse
# ---------------------------------------------------------------------------

LC96_FILES: list[tuple[str, Path]] = [
    # tag, path
    ("26-06-24_NC",
     DATA_ROOT / "UTI Trial Data" / "26-06-24 Negative Control Test" / "EP_26-06-24_UTI_NC.lc96p"),
    ("26-06-30_KP",
     DATA_ROOT / "UTI Trial Data" / "26-06-30 E. Coli Serial Dilution Synthetic" / "EP_26-06-30_KP_SD.lc96p"),
    ("26-06-30_EC",
     DATA_ROOT / "UTI Trial Data" / "26-06-30 E. Coli Serial Dilution Synthetic" / "EP_26-06-30_EC_SD.lc96p"),
    ("26-07-10_EC",
     DATA_ROOT / "UTI Trial Data" / "26-07-10 E. Coli Serial Dilution Synthetic" / "EP_26-07_10_EC_SD.lc96p"),
    ("26-07-28_EC",
     DATA_ROOT / "UTI Trial Data" / "26-07-28 E. Coli Serial Dilution Synthetic" / "EP_26-07-28_ECSD.lc96p"),
]

CYCLE_TIME_MIN = 0.5  # 30-second LAMP cycles on LC96


# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------

def _load_rdml(lc96p_path: Path) -> str:
    with zipfile.ZipFile(lc96p_path) as z:
        return z.read("rdml_data.xml").decode("utf-8", errors="replace")


def _rid_to_position(rid: int) -> str:
    """LC96 uses 1-indexed react ids across an 8×12 plate.
    react id 1 = A1, id 12 = A12, id 13 = B1, ..., id 96 = H12.
    """
    row = chr(ord("A") + (rid - 1) // 12)
    col = ((rid - 1) % 12) + 1
    return f"{row}{col}"


def _parse_all_wells(rdml: str) -> dict[str, dict]:
    """Return {position: {cycles: np.ndarray, fluor: np.ndarray, temp: np.ndarray}}.

    Only wells with amplification data (<adp> entries) are included.
    """
    wells: dict[str, dict] = {}
    for m in re.finditer(r'<react id="(\d+)"[^>]*>(.*?)</react>', rdml, re.DOTALL):
        rid = int(m.group(1))
        body = m.group(2)
        # Extract every <adp> block
        adps = re.findall(
            r"<adp>\s*<cyc>(\d+)</cyc>\s*<tmp>([\d\.eE+-]+)</tmp>"
            r"\s*<fluor>([\d\.eE+-]+)</fluor>\s*</adp>",
            body,
        )
        if not adps:
            continue
        cyc = np.array([int(c) for c, _, _ in adps], dtype=int)
        tmp = np.array([float(t) for _, t, _ in adps], dtype=float)
        fluor = np.array([float(f) for _, _, f in adps], dtype=float)
        pos = _rid_to_position(rid)
        wells[pos] = {"react_id": rid, "cycles": cyc, "temp": tmp, "fluor": fluor}
    return wells


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class WellFeatures:
    position: str
    n_cycles: int
    baseline_mean: float
    baseline_std: float
    plateau_fluor: float
    # Cq features — all in units of CYCLES (float, fractional via linear interp)
    Cq_start: float | None       # first cycle where fluor exceeds baseline + 3σ
    Cq_takeoff: float | None     # argmax of first derivative on smoothed fluor
    Cq_plateau: float | None     # cycle where 1st deriv drops below 5% of its max (after takeoff)
    max_slope: float             # peak value of the smoothed first derivative
    # Provenance
    smoothing_window: int
    baseline_window: tuple[int, int]  # (start_cyc, end_cyc) inclusive
    is_positive: bool             # True if Cq_takeoff was found


def _smooth_savgol(y: np.ndarray, window: int) -> np.ndarray:
    """Savitzky–Golay smoothing, order 3. Window must be odd."""
    if window % 2 == 0:
        window += 1
    if len(y) < window:
        window = max(3, len(y) - (len(y) + 1) % 2)
    if window < 3:
        return y.copy()
    return savgol_filter(y, window_length=window, polyorder=min(3, window - 1))


def _first_cross(x: np.ndarray, threshold: float) -> float | None:
    """Linearly interpolated cycle at which x first exceeds threshold.
    Returns None if never crosses. x is indexed 0..N-1 → cycles 1..N.
    """
    above = x > threshold
    if not above.any():
        return None
    idx = int(np.argmax(above))  # first True
    if idx == 0:
        # Already above at cycle 1 → no meaningful crossing.
        return 1.0
    # Linear interp between idx-1 and idx
    y0, y1 = x[idx - 1], x[idx]
    if y1 == y0:
        return float(idx)  # degenerate
    frac = (threshold - y0) / (y1 - y0)
    return float((idx - 1) + frac) + 1.0  # +1 to convert 0-idx → 1-idx cycle


def _last_above(x: np.ndarray, threshold: float, start_idx: int) -> float | None:
    """Last (highest) index at or after start_idx where x > threshold.
    Returns cycle number (1-indexed, fractional via interp with x[idx+1] if available).
    """
    region = x[start_idx:]
    below = np.where(region <= threshold)[0]
    if len(below) == 0:
        # Never drops below in the observation window
        return float(len(x))  # last cycle
    # First drop-below is at start_idx + below[0]
    drop_idx = start_idx + int(below[0])
    if drop_idx == start_idx:
        return None  # never was above
    # Linear interp between drop_idx-1 and drop_idx
    y0, y1 = x[drop_idx - 1], x[drop_idx]
    if y1 == y0:
        return float(drop_idx)
    frac = (y0 - threshold) / (y0 - y1)
    return float((drop_idx - 1) + frac) + 1.0


def extract_features(
    fluor: np.ndarray,
    baseline_window: tuple[int, int] = (1, 5),
    smoothing_window: int = 9,
    start_threshold_sigmas: float = 3.0,
    plateau_slope_frac: float = 0.05,
) -> WellFeatures:
    """Compute per-well features from a single fluorescence curve.

    Parameters
    ----------
    fluor:
        1D array, length = n_cycles. Assumed indexed as cycle 1..N in position 0..N-1.
    baseline_window:
        (start_cyc, end_cyc) inclusive, 1-indexed. Cycles used for baseline stats.
    smoothing_window:
        Savitzky–Golay window (odd int). Applied before derivative-based features.
    start_threshold_sigmas:
        Multiplier on baseline std for Cq_start crossing test.
    plateau_slope_frac:
        Fraction of max slope below which we call the curve "plateaued". Applied to
        the smoothed first derivative.
    """
    n = len(fluor)

    # Baseline stats
    b_start = max(baseline_window[0] - 1, 0)   # convert 1-idx → 0-idx
    b_end   = min(baseline_window[1], n)
    baseline_slice = fluor[b_start:b_end]
    baseline_mean = float(baseline_slice.mean())
    baseline_std  = float(baseline_slice.std(ddof=0))
    # If truly flat, add tiny epsilon so 3σ threshold isn't 0
    if baseline_std < 1e-9:
        baseline_std = 1e-9

    # Plateau fluor — mean of last 5 cycles
    plateau_fluor = float(fluor[-min(5, n):].mean())

    # Smoothed fluor + first derivative
    fluor_smooth = _smooth_savgol(fluor, smoothing_window)
    d1 = np.gradient(fluor_smooth)

    # Cq_start (only meaningful if fluor rises noticeably above baseline noise)
    Cq_start = _first_cross(fluor_smooth, baseline_mean + start_threshold_sigmas * baseline_std)

    # Cq_takeoff — argmax of first derivative (in cycles 1..N)
    # Only trust this if d1's peak is at least 5x baseline noise-scale of d1 itself
    d1_baseline_noise = float(np.std(d1[b_start:b_end], ddof=0))
    d1_baseline_noise = max(d1_baseline_noise, 1e-9)
    argmax_idx = int(np.argmax(d1))
    max_slope = float(d1[argmax_idx])

    # Consider well "positive" if max slope is meaningfully above baseline derivative noise
    is_positive = bool(max_slope > 5.0 * d1_baseline_noise) and (Cq_start is not None) and (Cq_start < n - 5)

    Cq_takeoff = float(argmax_idx + 1) if is_positive else None

    # Cq_plateau — first cycle after Cq_takeoff where d1 drops below `plateau_slope_frac * max_slope`
    if is_positive and Cq_takeoff is not None:
        plateau_thr = plateau_slope_frac * max_slope
        # Search from Cq_takeoff (0-idx = argmax_idx) onward
        Cq_plateau = _last_above(d1, plateau_thr, argmax_idx)
    else:
        Cq_plateau = None

    return WellFeatures(
        position="",  # filled by caller
        n_cycles=n,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        plateau_fluor=plateau_fluor,
        Cq_start=Cq_start,
        Cq_takeoff=Cq_takeoff,
        Cq_plateau=Cq_plateau,
        max_slope=max_slope,
        smoothing_window=smoothing_window,
        baseline_window=baseline_window,
        is_positive=is_positive,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process_lc96(lc96p_path: Path) -> dict:
    """Parse one .lc96p file into a per-well features dict."""
    rdml = _load_rdml(lc96p_path)
    wells = _parse_all_wells(rdml)
    out: dict[str, dict] = {}
    for pos, w in wells.items():
        feats = extract_features(w["fluor"])
        feats.position = pos
        # Build the JSON-serialisable record
        d = asdict(feats)
        # Add convenient minutes fields
        d["Cq_start_min"] = (d["Cq_start"] * CYCLE_TIME_MIN) if d["Cq_start"] is not None else None
        d["Cq_takeoff_min"] = (d["Cq_takeoff"] * CYCLE_TIME_MIN) if d["Cq_takeoff"] is not None else None
        d["Cq_plateau_min"] = (d["Cq_plateau"] * CYCLE_TIME_MIN) if d["Cq_plateau"] is not None else None
        # Raw curve too — useful for downstream plots
        d["cycles"] = w["cycles"].tolist()
        d["temp"] = w["temp"].tolist()
        d["fluor"] = w["fluor"].tolist()
        out[pos] = d
    return {
        "source_file": str(lc96p_path.relative_to(PROJECT_ROOT)),
        "cycle_time_min": CYCLE_TIME_MIN,
        "wells": out,
    }


def main() -> None:
    for tag, path in LC96_FILES:
        if not path.exists():
            print(f"[skip] {tag}: {path} not found")
            continue
        print(f"[proc] {tag}: {path.name}")
        result = process_lc96(path)

        n_wells = len(result["wells"])
        n_positive = sum(1 for w in result["wells"].values() if w["is_positive"])
        print(f"       parsed {n_wells} wells; {n_positive} positive (Cq_takeoff found)")

        out_path = path.parent / "plate_features.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  ->   {out_path}")


if __name__ == "__main__":
    main()
