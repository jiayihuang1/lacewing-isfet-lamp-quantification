"""Per-chip well-to-label mappings for the Data/Multi/ chips.

Two layouts (per supervisor's chip diagrams + user-confirmed 2026-06-10):

  * Elena chip (D20260320_..._Elena_steap_cv): two-target chip
        row 1: [S,  C]    wells 0, 1
        row 2: [C,  S]    wells 2, 3
        row 3: [S,  C]    wells 4, 5
        row 4: [C,  S]    wells 6, 7
        row 5: [NCS, NCC] wells 8, 9
    -> wells 0-7 = positive (any amplification), wells 8-9 = NTC

  * P/N chips (all other 5 chips): single-target P/N grid
        row 1: [P, N]    wells 0, 1
        row 2: [N, P]    wells 2, 3
        row 3: [P, N]    wells 4, 5
        row 4: [N, P]    wells 6, 7
        row 5: [P, N]    wells 8, 9
    -> wells 0, 3, 4, 7, 8 = positive
    -> wells 1, 2, 5, 6, 9 = NTC under the pragmatic fallback (could be
                              "negative sample" rather than NTC; supervisor
                              to confirm)

Wells used as the **filter reference** are the NTC wells.  Currently
both layouts pool all of a chip's NTC wells when computing filter
thresholds.
"""
from __future__ import annotations


# Chip-name substring -> layout key.  We match against the substring
# (the data folder name format is D<date>_..._U_<exp_label>).
_LAYOUT_BY_SUBSTRING: list[tuple[str, str]] = [
    ("Elena_steap_cv", "elena_two_target"),
]
_DEFAULT_LAYOUT = "pn"


def chip_layout(chip_name: str) -> str:
    """Return the layout key for a chip folder name."""
    for substr, key in _LAYOUT_BY_SUBSTRING:
        if substr in chip_name:
            return key
    return _DEFAULT_LAYOUT


# Layout key -> {well_idx: label} for label_per_well used downstream.
# label: 1 = positive (any amplification), 0 = NTC / negative class.
_LABELS_BY_LAYOUT: dict[str, dict[int, int]] = {
    "elena_two_target": {
        0: 1, 1: 1, 2: 1, 3: 1,   # S/C amplification wells
        4: 1, 5: 1, 6: 1, 7: 1,
        8: 0, 9: 0,               # NCS, NCC
    },
    "pn": {
        0: 1, 1: 0,   # row 1: P, N
        2: 0, 3: 1,   # row 2: N, P
        4: 1, 5: 0,   # row 3: P, N
        6: 0, 7: 1,   # row 4: N, P
        8: 1, 9: 0,   # row 5: P, N
    },
}


# Layout key -> tuple of NTC well indices (used to derive filter thresholds).
_NTC_WELLS_BY_LAYOUT: dict[str, tuple[int, ...]] = {
    "elena_two_target": (8, 9),
    "pn":                (1, 2, 5, 6, 9),
}


def label_per_well(chip_name: str) -> dict[int, int]:
    return dict(_LABELS_BY_LAYOUT[chip_layout(chip_name)])


def positive_wells(chip_name: str) -> tuple[int, ...]:
    return tuple(w for w, l in label_per_well(chip_name).items() if l == 1)


def ntc_wells(chip_name: str) -> tuple[int, ...]:
    return tuple(_NTC_WELLS_BY_LAYOUT[chip_layout(chip_name)])


N_WELLS_MULTI = 10
