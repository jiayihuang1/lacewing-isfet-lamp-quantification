"""Inventory NTC wells for chips under Data/24_CoV_Quantification/Others/.

Important context discovered 2026-05-06: lacewing_log.txt is purely
hardware status (firmware version, voltage references, active-pixel
counts). It contains NO biological sample annotations. Per-well sample
identity (NTC vs positive) lives in the Run Tracker Excel sheet
(~/OneDrive/Master Data Folder/Run Tracker.xlsx) which is not in this
repo, OR in the folder name (e.g. _SD = SARS-CoV-2 Single Dilution
multiplex, where well 5 = NTC by convention).

Strategy
--------
1. List every chip folder under Others/.
2. Default 'ntc_well_indices' to '5' (the user's stated convention:
   well 6 = NTC for the Final chips, same layout assumed elsewhere).
3. Default 'classification' to 'mixed' (5 positive + 1 NTC) for any
   6-well chip whose name does NOT contain explicit hints to the
   contrary, 'unknown' otherwise.
4. Note any folder-name hints that might warrant a different mapping.
5. WRITE OUT THE CSV AND ASK THE USER TO REVIEW BEFORE
   build_dataset.py runs. The user can edit ntc_well_indices to add
   or remove well indices, or set classification to 'non_ntc' to drop
   a chip entirely.

CSV columns
-----------
chip_name           - folder basename
chip_path           - absolute path
classification      - mixed | ntc_only | non_ntc | unknown
ntc_well_indices    - comma-separated 0..5 well indices (NTC samples)
folder_name_hints   - any keywords spotted in the folder name
notes               - free-text for the user
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lacewing.classification import paths  # noqa: E402


# Folder-name keywords that suggest a non-NTC chip (e.g. multiplexes,
# concentration tests, mutation spectrums). Presence is a hint, not a verdict.
NON_NTC_HINTS = [
    "1e5", "1e6", "1e7", "1e8", "1e9",      # concentration-only chips
    "BRAF_MT", "Beads", "Bead", "VP",       # experiment-specific
    "Pilot", "Alter",                       # pilot runs
]
# Folder-name keywords that suggest a chip dedicated to NTC.
NTC_HINTS = ["NTC", "negative", "blank"]


def _scan_folder_name(name: str) -> list[str]:
    hints = []
    n_lower = name.lower()
    for k in NON_NTC_HINTS + NTC_HINTS:
        if k.lower() in n_lower:
            hints.append(k)
    return hints


def inventory(others_dir: Path = paths.OTHERS_DIR,
              out_csv: Path = paths.NTC_INVENTORY_CSV) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    chip_dirs = sorted(p for p in others_dir.iterdir() if p.is_dir())
    print(f"Inventorying {len(chip_dirs)} chip folders under {others_dir} ...")

    rows = []
    for chip in chip_dirs:
        hints = _scan_folder_name(chip.name)

        ntc_hint = any(h in NTC_HINTS for h in hints)
        non_ntc_hint = any(h in NON_NTC_HINTS for h in hints)

        if ntc_hint:
            classification = "ntc_only"
            ntc_wells = "0,1,2,3,4,5"
            note = "Folder name suggests NTC-only chip - all 6 wells set as NTC."
        elif non_ntc_hint:
            # Chip has a positive-experiment hint, but if it follows the
            # standard 6-well layout (5 pos + 1 NTC) well 5 is still NTC.
            classification = "mixed"
            ntc_wells = "5"
            note = ("Default: 6-well layout, well 5 = NTC. "
                    "Edit if this chip uses a different layout.")
        else:
            classification = "mixed"
            ntc_wells = "5"
            note = "No folder hints; defaulting to standard layout."

        rows.append(dict(
            chip_name=chip.name,
            chip_path=str(chip),
            classification=classification,
            ntc_well_indices=ntc_wells,
            folder_name_hints=";".join(hints),
            notes=note,
        ))

    fieldnames = ["chip_name", "chip_path", "classification",
                  "ntc_well_indices", "folder_name_hints", "notes"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1

    print(f"\nWrote {out_csv}")
    print("Default classification counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print("\n" + "=" * 70)
    print("ACTION REQUIRED: review ntc_inventory.csv before build_dataset.py.")
    print("  - For chips whose well 5 is NOT NTC, set classification=non_ntc")
    print("    (or hand-edit ntc_well_indices).")
    print("  - For dedicated NTC chips, the inventory may have set well 5 only;")
    print("    edit ntc_well_indices to include all NTC wells (e.g. 0,1,2,3,4,5).")
    print("  - The Run Tracker spreadsheet (OneDrive) is the authoritative source.")
    print("=" * 70)


def load_ntc_chips(csv_path: Path = paths.NTC_INVENTORY_CSV
                   ) -> list[tuple[Path, list[int]]]:
    """Read inventory CSV; return [(chip_path, ntc_well_indices), ...] for
    chips classified as ntc_only or mixed with at least one NTC well."""
    if not csv_path.exists():
        return []
    out = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["classification"] in ("ntc_only", "mixed"):
                wells = [int(w) for w in row["ntc_well_indices"].split(",")
                         if w.strip()]
                if wells:
                    out.append((Path(row["chip_path"]), wells))
    return out


if __name__ == "__main__":
    inventory()
