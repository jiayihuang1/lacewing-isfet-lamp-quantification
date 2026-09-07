"""Recover per-pixel (row, col) coordinates for the exp6 test set.

The exp6 failure dumps record ``pixel_id`` as the index within the
post-filter surviving pixels of each well (the same indexing
``build_dataset.py`` produces).  To draw spatial maps we need to
invert that: for each (chip, well, pixel_id) triple, recover the
(row, col) on the chip's pixel grid.

This module reloads each chip via titan, replays the loose-active +
QC mask used by ``build_dataset._extract_per_pixel_traces``, and
returns the (row, col) corresponding to each surviving pixel.

Loading a chip via titan is the slow step (tens of seconds per chip)
so the resulting coordinate map is cached as a small npz alongside
the filter JSONs.

CPU only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lacewing.classification.core import paths as cls_paths

# titan + build_dataset are not import-safe at module level (they need
# extra sys.path setup that paths.py does).  Importing paths above pulls
# them in automatically.
try:
    import titan.load_functions as titan_load           # noqa: E402
    from titan.load_and_preprocessing import (          # noqa: E402
        titan_load_and_preprocessing,
    )
    import titan.preprocessing_functions as titan_preprocess  # noqa: E402
except ImportError:
    titan_load = titan_preprocess = None
    def titan_load_and_preprocessing(*args, **kwargs):
        raise RuntimeError(
            "titan-signal-processing not installed. Set LACEWING_TITAN_PATH."
        )

from lacewing.classification.data import build_dataset    # noqa: E402


# --------------------------------------------------------------------
# Build per-well coordinate lookup that mirrors build_dataset's mask
# --------------------------------------------------------------------

@dataclass
class WellSpatial:
    well_nrows: int
    well_ncols: int
    # For each surviving pixel (post loose-mask + QC), the index into
    # the flat 1D (well_nrows * well_ncols) array.  pixel_id from the
    # failure-dump indexes into this array.
    surviving_flat_idx: np.ndarray  # shape (n_survive,) int


def _wells_spatial_for_chip(chip_path: Path) -> dict[int, WellSpatial]:
    """Return WellSpatial for every well on the chip.

    Reproduces the exact mask sequence used by
    ``build_dataset._extract_per_pixel_traces``:
        loose firmware active mask, then per-frame QC mask, both
        applied to the full pixel array of the well.
    """
    exp = titan_load_and_preprocessing(
        chip_path,
        n_wells=cls_paths.N_WELLS,
        start_type=build_dataset.DEFAULT_START_TYPE,
        n_a_type=build_dataset.N_A_TYPE,
        end_time_min=build_dataset.END_TIME_MIN,
        print_status=False,
    )
    loose_per_well = build_dataset._loose_active_mask_per_well(chip_path)

    out: dict[int, WellSpatial] = {}
    for w_idx, well in enumerate(exp.wells_list):
        full_all = well.well_2d.astype(np.float32)
        qc       = build_dataset._qc_mask(full_all)
        keep     = loose_per_well[w_idx] & qc
        survive  = np.flatnonzero(keep)
        out[w_idx] = WellSpatial(
            well_nrows=int(well.well_nrows),
            well_ncols=int(well.well_ncols),
            surviving_flat_idx=survive,
        )
    return out


def find_chip_dir(chip_name: str) -> Path:
    """Locate a chip directory under the data root.

    The exp6 chip_id is just the directory name; we search under
    Data/24_CoV_Quantification/ for it.
    """
    data_root = cls_paths.DATA_ROOT
    for p in data_root.rglob(chip_name):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"could not locate chip directory {chip_name} "
                            f"under {data_root}")


# --------------------------------------------------------------------
# Public entry point: build pixel_id -> (row, col) for one chip
# --------------------------------------------------------------------

def build_coords_for_chip(chip_name: str) -> dict[int, dict]:
    """Return ``{well_id: {"nrows": int, "ncols": int,
                            "rows": np.ndarray, "cols": np.ndarray}}``.

    ``rows[i]`` and ``cols[i]`` are the (row, col) for pixel_id == i
    (the index into the post-filter surviving pixels of that well).
    """
    chip_path = find_chip_dir(chip_name)
    spatial = _wells_spatial_for_chip(chip_path)
    out: dict[int, dict] = {}
    for w_idx, sp in spatial.items():
        flat = sp.surviving_flat_idx
        rows = (flat // sp.well_ncols).astype(np.int32)
        cols = (flat %  sp.well_ncols).astype(np.int32)
        out[w_idx] = {
            "nrows": sp.well_nrows,
            "ncols": sp.well_ncols,
            "rows":  rows,
            "cols":  cols,
        }
    return out
