"""Parse Book.xlsx (LC96 vendor export for the new Concentration Data Experiment)
into a plate_features.json-shaped dict that matches the UTI trial's convention.

For UTI we parse the raw .lc96p and compute Cq_takeoff ourselves; for the new
dataset the raw .lc96p was NOT delivered — we only have Book.xlsx which carries
the LC96 software's auto-computed Cq. We therefore adopt the vendor Cq as our
plate TTP, and flag this difference in the viewer caption.

Book.xlsx columns of interest:
  - Replicate Group  → LC96 plate position (A1..H12), used as the join key
  - Position         → sample-name label (e.g. Kp_3, NTC)
  - Cq               → vendor Cq (cycles). Convert × 0.5 for minutes (30-sec cycles).
  - Call             → Positive/Negative
"""
from __future__ import annotations

import json
from pathlib import Path

import openpyxl
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[3]
DATA_ROOT = PROJECT_ROOT / "Data" / "Concentration Data Experiment"
BOOK_XLSX = DATA_ROOT / "Book.xlsx"

# LC96 LAMP protocol: 30-second cycles → 0.5 min per cycle.
CYCLE_TIME_MIN = 0.5

OUT_JSON = DATA_ROOT / "plate_features.json"


def _f(x) -> float | None:
    """Best-effort parse of an Excel cell into a float, or None."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_book_xlsx(path: Path = BOOK_XLSX) -> dict:
    """Parse Book.xlsx into two dicts, one per chip.

    Interpretation (per supervisor guidance 2026-08-14):
      - Rows 2-11 of Book.xlsx (first 10 data rows) = Chip 1 LC96 wells.
      - Rows 12-21 (next 10 rows) = Chip 2 LC96 wells.
      Each chip's 10 rows collapse into ~5 LC96 positions (multiple replicates per pos).
    """
    if not path.exists():
        raise FileNotFoundError(path)
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Sheet1"]
    header = [str(c).strip() if c is not None else "" for c in next(ws.iter_rows(values_only=True))]

    def col(name: str) -> int:
        for i, h in enumerate(header):
            if h == name:
                return i
        raise KeyError(f"column '{name}' not found in {path.name}. Have: {header}")

    ci_pos       = col("Position")
    ci_sample    = col("Sample Name")
    ci_gene      = col("Gene Name")
    ci_cq        = col("Cq")
    ci_call      = col("Call")
    ci_rep_group = col("Replicate Group")
    ci_cq_mean   = col("Cq Mean")

    all_rows = list(ws.iter_rows(min_row=2, values_only=True))
    # Drop fully-empty rows so the first-10 / next-10 split lands on real data.
    all_rows = [r for r in all_rows if not (r is None or all(c is None for c in r))]

    chips: dict[str, dict[str, dict]] = {"chip1": {}, "chip2": {}}
    for idx, row in enumerate(all_rows):
        chip_key = "chip1" if idx < 10 else "chip2"
        wells = chips[chip_key]

        rep_group = row[ci_rep_group]
        if rep_group is None:
            continue
        lc96_pos = str(rep_group).strip()
        if not lc96_pos:
            continue

        cq          = _f(row[ci_cq])
        cq_mean     = _f(row[ci_cq_mean])
        call_raw    = str(row[ci_call]).strip() if row[ci_call] is not None else ""
        is_positive = call_raw.lower() == "positive" and cq is not None
        sample_name = str(row[ci_sample]).strip() if row[ci_sample] is not None else ""
        gene_name   = str(row[ci_gene]).strip() if row[ci_gene] is not None else ""
        position    = str(row[ci_pos]).strip() if row[ci_pos] is not None else ""

        entry = wells.setdefault(lc96_pos, {
            "lc96_position": lc96_pos,
            "sample_label": position,
            "sample_name": sample_name,
            "gene_name": gene_name,
            "replicates": [],
        })
        entry["replicates"].append({
            "cq": cq,
            "cq_mean": cq_mean,
            "call": call_raw,
            "is_positive": bool(is_positive),
        })

    # Aggregate per well: adopt vendor Cq_mean if present, else mean of finite reps.
    for chip_key, wells in chips.items():
        for pos, entry in wells.items():
            reps = entry["replicates"]
            finite = [r["cq"] for r in reps if r["cq"] is not None]
            n_pos  = sum(1 for r in reps if r["is_positive"])
            cq_from_reps = (sum(finite) / len(finite)) if finite else None
            cq_mean_reported = next((r["cq_mean"] for r in reps if r["cq_mean"] is not None), None)
            chosen_cq = cq_mean_reported if cq_mean_reported is not None else cq_from_reps

            entry["n_replicates"] = len(reps)
            entry["n_positive"] = n_pos
            entry["Cq_takeoff"] = chosen_cq
            entry["Cq_takeoff_min"] = (chosen_cq * CYCLE_TIME_MIN) if chosen_cq is not None else None
            entry["is_positive"] = bool(chosen_cq is not None and n_pos > 0)
            entry["Cq_start"] = None
            entry["Cq_start_min"] = None
            entry["Cq_plateau"] = None
            entry["Cq_plateau_min"] = None

    payload = {
        "source": str(path.relative_to(PROJECT_ROOT)),
        "cycle_time_min": CYCLE_TIME_MIN,
        "notes": (
            "Cq_takeoff is the LC96 vendor auto-Cq (from Book.xlsx). "
            "Raw .lc96p not delivered for this dataset → we cannot recompute "
            "Cq_takeoff via derivative-argmax like we do for UTI. "
            "Book.xlsx interpretation: first 10 rows → chip1, next 10 rows → chip2 "
            "(per supervisor guidance 2026-08-14)."
        ),
        "chips": {ck: {"wells": w} for ck, w in chips.items()},
    }
    return payload


def main() -> None:
    payload = parse_book_xlsx()
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[ok] wrote {OUT_JSON}  ({len(payload['wells'])} LC96 positions)")
    for pos in sorted(payload["wells"].keys()):
        w = payload["wells"][pos]
        cq_str  = "NaN" if w["Cq_takeoff"] is None else f"{w['Cq_takeoff']:.2f}"
        ttp_str = "NaN" if w["Cq_takeoff_min"] is None else f"{w['Cq_takeoff_min']:.2f} min"
        print(f"  {pos}: sample={w['sample_label']:6s} gene={w['gene_name']:8s} "
              f"Cq={cq_str:>6s} TTP={ttp_str:>10s} "
              f"n_pos={w['n_positive']}/{w['n_replicates']}")


if __name__ == "__main__":
    main()
