"""Union-combine per-Vref pixels into a single Experiment.

Background
----------
On the Matthew_Multi titan branch, ``titan_load_and_preprocessing``
linearises every Vref slice but returns an Experiment built from only
ONE slice (``ref_idx=-1`` by default).  On chips with multiple Vrefs
(e.g. Elena), the active-pixel populations are **disjoint** across
Vrefs — pixels in-range at one Vref are out-of-range at another — so
using a single slice means throwing away potentially-useful pixels.

This module provides ``load_chip_combined`` which loads all Vref slices
and unions them into a single Experiment whose per-well linearised
arrays are stitched from the per-Vref linearised arrays: each pixel's
trace comes from whichever Vref it was active in.

Pure additive wrapper.  Does not modify titan or main_DNA.

User-confirmed design decisions (2026-06-10):
  - disjoint active-pixel sets across Vrefs (verified on Elena: 0 overlap)
  - "combine" = union the active masks; per-pixel trace from its home Vref
  - time axes match across Vrefs (same timestamps), so concatenation is
    along the pixel axis only, not time
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Path setup so titan resolves to titan-signal-processing-multi.
from . import _titan_setup  # noqa: F401

# Bare imports because Matthew_Multi titan uses bare-style imports internally.
try:
    from titan.load_and_preprocessing import titan_load_and_preprocessing  # noqa: E402
    from titan import load_functions as titan_load                          # noqa: E402
except ImportError:
    titan_load = None
    def titan_load_and_preprocessing(*args, **kwargs):
        raise RuntimeError(
            "titan-signal-processing-multi not installed. Set "
            "LACEWING_TITAN_MULTI_PATH or clone the Matthew_Multi branch."
        )


# Default settings for the Multi/ chips (firmware v5p0).
DEFAULT_N_WELLS = 10
DEFAULT_N_A_TYPE = "v06"
DEFAULT_END_TIME_MIN = 40


def _peek_n_refs(chip_path: Path) -> int:
    """How many Vref slices does this chip have?

    Reads the vref sweep file directly so we don't have to run titan
    end-to-end just to find out.  Returns 1 if no vref sweep file is
    present (legacy single-Vref chip).
    """
    try:
        n_refs, _ = titan_load.load_vref_sweep(chip_path)
        return int(n_refs) if n_refs else 1
    except Exception:
        return 1


def load_chip_combined(
    chip_path: Path,
    *,
    n_wells: int = DEFAULT_N_WELLS,
    n_a_type: str = DEFAULT_N_A_TYPE,
    end_time_min: int = DEFAULT_END_TIME_MIN,
    print_status: bool = False,
):
    """Load a Multi/ chip and union its Vref slices into one Experiment.

    For single-Vref chips this is a one-shot ``titan_load_and_preprocessing``
    call (no-op combine).  For multi-Vref chips we load each Vref slice
    separately, then mutate the last-loaded Experiment so that, per well:

      * ``idx_active`` is the union of per-Vref active masks
      * ``well_3d_lin`` is stitched: each active pixel's trace comes
        from its home Vref's linearised array

    Returns the combined Experiment and a small dict of diagnostics.
    """
    chip_path = Path(chip_path)
    n_refs = _peek_n_refs(chip_path)

    if n_refs <= 1:
        exp = titan_load_and_preprocessing(
            chip_path,
            n_wells=n_wells,
            n_a_type=n_a_type,
            end_time_min=end_time_min,
            print_status=print_status,
            ref_idx=-1,
        )
        diag = {
            "chip_path":  chip_path,
            "n_refs":     1,
            "per_ref_active_per_well": [
                [int(w.idx_active.sum()) for w in exp.wells_list]
            ],
            "combined_active_per_well":
                [int(w.idx_active.sum()) for w in exp.wells_list],
            "overlap_per_well": [0] * n_wells,  # only one ref, no overlap concept
        }
        return exp, diag

    # ---- Multi-Vref path ----
    per_ref_exps = []
    for ref_i in range(n_refs):
        exp_i = titan_load_and_preprocessing(
            chip_path,
            n_wells=n_wells,
            n_a_type=n_a_type,
            end_time_min=end_time_min,
            print_status=print_status,
            ref_idx=ref_i,
        )
        per_ref_exps.append(exp_i)

    # Use the last Experiment as the combined target; mutate its wells.
    combined = per_ref_exps[-1]

    per_ref_active_per_well: list[list[int]] = [
        [int(w.idx_active.sum()) for w in e.wells_list]
        for e in per_ref_exps
    ]
    combined_active_per_well: list[int] = []
    overlap_per_well: list[int] = []

    for i_well in range(n_wells):
        # Active masks across refs for this well (length = pixels in well).
        masks = [np.asarray(e.wells_list[i_well].idx_active, dtype=bool)
                 for e in per_ref_exps]
        union_mask = np.zeros_like(masks[0])
        for m in masks:
            union_mask |= m
        combined_active_per_well.append(int(union_mask.sum()))

        # Overlap = pixels active in more than one ref.
        any_overlap = np.zeros_like(masks[0])
        seen_once = np.zeros_like(masks[0])
        for m in masks:
            any_overlap |= seen_once & m
            seen_once |= m
        overlap_per_well.append(int(any_overlap.sum()))

        # Stitch per-pixel traces: take the trace from the first ref the
        # pixel was active in.  (For disjoint masks this is just "the ref
        # it was active in"; for overlapping masks we pick the first.)
        target_well = combined.wells_list[i_well]
        # well_3d_lin shape: (well_nrows, well_ncols, n_time).  Flatten
        # the spatial dims to match idx_active's flat layout (length
        # well_nrows * well_ncols, "C" order — see split_wells_idxactive).
        rows = target_well.well_nrows
        cols = target_well.well_ncols
        n_time = target_well.well_3d_lin.shape[2]
        # Default fill = last-loaded ref's existing array (preserves any
        # pixel only active there; matches the un-combined behaviour for
        # those pixels).
        stitched_2d = target_well.well_3d_lin.reshape(
            rows * cols, n_time, order="C"
        ).astype(np.float32, copy=True)

        # For every other ref, write that ref's trace for the pixels it
        # owns *exclusively at this point in the iteration*.
        assigned = masks[-1].copy()   # pixels already taken by last ref
        for ref_i in range(n_refs - 1):
            ref_mask = masks[ref_i]
            # Pixels active in this ref AND not already assigned.
            take = ref_mask & ~assigned
            if not take.any():
                continue
            src_2d = per_ref_exps[ref_i].wells_list[i_well].well_3d_lin.reshape(
                rows * cols, n_time, order="C"
            )
            stitched_2d[take, :] = src_2d[take, :].astype(np.float32, copy=False)
            assigned |= take

        # Write back into the well's 3D array.
        target_well.well_3d_lin = stitched_2d.reshape(
            rows, cols, n_time, order="C"
        )
        # Update active mask to the union.
        target_well.idx_active = union_mask

    diag = {
        "chip_path":              chip_path,
        "n_refs":                 n_refs,
        "per_ref_active_per_well": per_ref_active_per_well,
        "combined_active_per_well": combined_active_per_well,
        "overlap_per_well":       overlap_per_well,
    }
    return combined, diag
