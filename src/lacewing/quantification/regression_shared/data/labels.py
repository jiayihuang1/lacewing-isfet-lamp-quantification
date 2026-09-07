"""Per-pixel qLAMP TTP label assignment.

Pulls TTP_QLAMP from lacewing.quantification.common (the matrix-style
fluorometer qLAMP reference, 5 concentrations x 8 replicate columns).

Convention (mirrors Analysis/quantification/runners/run_sdm_cy0.py
_qlamp_per_well, restricted to the wells we want to use for regression):

  * Dose-response chips (1e5..1e9): wells 0-3 (the four amplification
    replicates, see Week 8 PowerPoint: 'E1-R1, E1-R2, E2-R1, E2-R2')
    share the chip's qLAMP mean over the 8 replicate columns.  PTC
    (well 4) and NTC (well 5) get NaN and are excluded.

  * SD multiplex chip: wells 0-4 each get the qLAMP mean of their
    own dilution (via SD_WELL_LABELS / SD_LABEL_LOG).  Well 5 (NTC)
    gets NaN.
"""
from __future__ import annotations

import numpy as np

# Constants below are mirrored from Analysis/quantification/common.py
# (we don't import from there because that module pulls in titan, which
# isn't needed for relabelling and complicates the HPC import path).
# Update both copies if these ever change.

# Order matches CONC_KEYS = [1e5, 1e6, 1e7, 1e8, 1e9].
# qLAMP gold-standard reference (Lacewing_Readout_DNA_Quant.m line 803).
TTP_QLAMP: np.ndarray = np.array([
    [20.01, 18.52, 18.61, 19.41, 18.84, 22.30, 29.83, 20.85],   # 1e5
    [17.34, 17.46, 17.53, 18.64, 16.93, 16.97, 15.57, 15.55],   # 1e6
    [14.62, 14.55, 14.56, 14.94, 15.46, 15.38, 14.83, 15.56],   # 1e7
    [11.53, 11.55, 11.52, 11.62,  9.62,  9.68,  9.46,  9.82],   # 1e8
    [ 9.49,  9.55,  9.42,  9.54,  7.22,  7.24,  7.93,  8.23],   # 1e9
])
CONC_KEYS: list[str] = ["1e5", "1e6", "1e7", "1e8", "1e9"]
CONC_LOG:  list[int] = [5, 6, 7, 8, 9]

# SD multiplex chip well-to-reaction mapping.
SD_WELL_LABELS: list[str] = ["E", "D", "C", "B", "A", "NTC"]
SD_LABEL_LOG:   dict[str, int] = {"A": 5, "B": 6, "C": 7, "D": 8, "E": 9}


# Wells used for regression on the dose-response chips: the four
# amplification replicates only.  PTC + NTC are out of scope.
DOSE_RESPONSE_WELLS = (0, 1, 2, 3)

# SD chip name as it appears in chip_id arrays (the data folder name).
SD_CHIP_FOLDER = "D20240719_E03_C44_F4500KHz_U_COV_SD"

# Dose-response chip folder names.  Keyed by CONC_KEYS for easy lookup.
DOSE_RESPONSE_CHIP_FOLDERS: dict[str, str] = {
    "1e5": "D20240821_E01_C07_F4500KHz_U_1e5",
    "1e6": "D20240821_E01_C09_F4500KHz_U_1e6",
    "1e7": "D20240820_E02_C03_F4500KHz_U_1e7",
    "1e8": "D20240822_E02_C00_F4500KHz_U_1e8",
    "1e9": "D20240822_E02_C00_F4500KHz_U_1e9",
}
FOLDER_TO_CONC_KEY: dict[str, str] = {
    v: k for k, v in DOSE_RESPONSE_CHIP_FOLDERS.items()
}


def qlamp_ttp_per_well(chip_folder: str, n_wells: int) -> list[float]:
    """Return a length-n_wells list of TTPs (or NaN) for one chip.

    For dose-response chips: wells DOSE_RESPONSE_WELLS get the chip's
    qLAMP mean; others get NaN.

    For the SD chip: each well's TTP is the qLAMP mean of the matching
    dilution; NTC + out-of-range wells get NaN.

    For any other chip: all NaN.
    """
    out = [float("nan")] * n_wells

    if chip_folder in FOLDER_TO_CONC_KEY:
        conc_key = FOLDER_TO_CONC_KEY[chip_folder]
        conc_idx = CONC_KEYS.index(conc_key)
        chip_ref = float(TTP_QLAMP[conc_idx].mean())
        for w in DOSE_RESPONSE_WELLS:
            if w < n_wells:
                out[w] = chip_ref
        return out

    if chip_folder == SD_CHIP_FOLDER:
        for w in range(n_wells):
            if w >= len(SD_WELL_LABELS):
                continue
            label = SD_WELL_LABELS[w]
            if label not in SD_LABEL_LOG:
                # Well marked 'NTC' or similar: no TTP.
                continue
            conc_log = SD_LABEL_LOG[label]
            try:
                conc_idx = CONC_LOG.index(conc_log)
            except ValueError:
                continue
            out[w] = float(TTP_QLAMP[conc_idx].mean())
        return out

    return out


def split_for_chip(chip_folder: str) -> str:
    """Return 'train', 'test', or 'skip' for one chip folder."""
    if chip_folder in FOLDER_TO_CONC_KEY:
        return "train"
    if chip_folder == SD_CHIP_FOLDER:
        return "test"
    return "skip"


def label_pixels(
    chip_ids: np.ndarray,
    well_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised relabelling.

    Inputs
        chip_ids  (N,)  <U64 chip folder names
        well_ids  (N,)  uint8 well indices

    Returns
        y_ttp_min  (N,)  float32, NaN for pixels without a known TTP
        split_id   (N,)  uint8, 0 = train, 1 = test, 2 = skip
    """
    chip_ids = np.asarray(chip_ids)
    well_ids = np.asarray(well_ids).astype(np.int32)
    n = len(chip_ids)

    y = np.full(n, np.nan, dtype=np.float32)
    split = np.full(n, 2, dtype=np.uint8)  # 2 = skip

    # Cache per-chip TTP arrays so we don't recompute for every pixel.
    unique_chips = np.unique(chip_ids)
    per_chip_ttp: dict[str, list[float]] = {}
    per_chip_split: dict[str, str] = {}
    for chip in unique_chips:
        chip = str(chip)
        sp = split_for_chip(chip)
        per_chip_split[chip] = sp
        if sp == "skip":
            continue
        # Use a generous n_wells; we look up by integer index anyway.
        per_chip_ttp[chip] = qlamp_ttp_per_well(chip, n_wells=10)

    for chip in unique_chips:
        chip_s = str(chip)
        mask = chip_ids == chip
        sp = per_chip_split[chip_s]
        if sp == "skip":
            continue
        ttps = per_chip_ttp[chip_s]
        well_ids_chip = well_ids[mask]
        # Vectorised TTP lookup
        ttp_lookup = np.array(ttps + [np.nan], dtype=np.float32)
        # Anything out-of-bounds well_id maps to the trailing NaN.
        clipped = np.where(well_ids_chip < len(ttps), well_ids_chip, len(ttps))
        y_chip = ttp_lookup[clipped]
        y[mask] = y_chip
        split[mask] = 0 if sp == "train" else 1

    return y, split
