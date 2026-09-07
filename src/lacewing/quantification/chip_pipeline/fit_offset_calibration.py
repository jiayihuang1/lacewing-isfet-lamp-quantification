"""Fit the chip↔plate TTP offset calibration curve.

Model: offset(min) = a * log10_conc + b
    where offset = chip_TTP - plate_TTP.

Primary data source: COVID Lacewing_Quantification_Results.xlsx, Sheet2.
  → 5 concentrations (1e9..1e5) × 4 chip extractions (=20 paired points).
  → Chip TTPs already computed; no rule extractor invoked here.

The COVID and UTI runs share:
  - Chip: Lacewing ISFET-LAMP, same well volume range
  - Plate: LightCycler 96 with SYBR Green + qPCR conventions
  - Cycle time: 30 s
The only known difference is the primer set. Primer efficiency affects the
per-decade slope. We fit on COVID, use as the initial UTI calibration, and
add UTI-derived points later once processed chips give us paired TTPs.

Usage
-----
    python -m lacewing.quantification.chip_pipeline.fit_offset_calibration
    → writes offset_calibration.json + diagnostic plot to the module folder.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[3]
COVID_XLSX = PROJECT_ROOT / "Data" / "24_CoV_Quantification" / "Lacewing_Quantification_Results.xlsx"

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "offset_calibration.json"
OUT_PNG  = OUT_DIR / "offset_calibration.png"


# ---------------------------------------------------------------------------
# COVID Sheet2 loader
# ---------------------------------------------------------------------------

def load_covid_paired_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (log10_conc, chip_ttp, plate_ttp) arrays from COVID Sheet2.

    Sheet2 layout (from earlier inspection):
      row 0: (blank) | TTP Lacewing [min] | ... | (blank) | (blank) | (blank) | TTP plate [min]
      row 1: Concentration [1eX] | 9 | 8 | 7 | 6 | 5 | ... | Conc | 9 | 8 | 7 | 6 | 5
      row 2: Extractions | v | v | v | v | v |  | | Extractions | pv | pv | pv | pv | pv
      rows 3-5: 3 more chip extraction rows (plate has only 1 row in current file)
      row 6: PTC | ...
    """
    wb = openpyxl.load_workbook(COVID_XLSX, data_only=True)
    ws = wb["Sheet2"]

    concs = np.array([9, 8, 7, 6, 5], dtype=int)

    # Chip TTPs: rows 2-5 (1-indexed), cols 2-6 (B..F)
    chip = np.array([
        [ws.cell(row=r, column=c).value for c in range(2, 7)]
        for r in range(3, 7)  # openpyxl is 1-indexed; row 3 == "Extractions" row (data row 2)
    ], dtype=float)

    # Plate TTPs: row 2 only (Extractions), cols 10-14 (J..N)
    plate = np.array([ws.cell(row=3, column=c).value for c in range(10, 15)], dtype=float)

    # Build long-form arrays: (n_reps * n_conc,)
    n_reps, n_conc = chip.shape
    log10_conc_long = np.tile(concs, n_reps).astype(float)
    chip_long = chip.reshape(-1)
    plate_long = np.tile(plate, n_reps)
    return log10_conc_long, chip_long, plate_long


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    a: float
    b: float
    r2: float
    residual_std: float
    n_points: int
    per_conc_mean_offset: dict[str, float]
    per_conc_std_offset: dict[str, float]
    source: str
    notes: str


def fit_calibration(
    log10_conc: np.ndarray,
    chip_ttp: np.ndarray,
    plate_ttp: np.ndarray,
    exclude_concs: tuple[int, ...] = (),
    source: str = "COVID Sheet2",
    notes: str = "",
) -> Calibration:
    """Fit offset = a * log10_conc + b (least squares)."""
    keep = ~np.isnan(chip_ttp) & ~np.isnan(plate_ttp)
    for c in exclude_concs:
        keep &= (log10_conc != c)
    lc = log10_conc[keep]
    offset = chip_ttp[keep] - plate_ttp[keep]

    # OLS fit
    a, b = np.polyfit(lc, offset, 1)
    pred = a * lc + b
    ss_res = float(np.sum((offset - pred) ** 2))
    ss_tot = float(np.sum((offset - offset.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    residual_std = float(np.std(offset - pred, ddof=0))

    # Per-concentration summary
    per_mean = {}
    per_std = {}
    for c in sorted(set(lc.tolist()), reverse=True):
        mask = lc == c
        vals = offset[mask]
        per_mean[str(int(c))] = float(vals.mean())
        per_std[str(int(c))] = float(vals.std(ddof=0))

    return Calibration(
        a=float(a),
        b=float(b),
        r2=r2,
        residual_std=residual_std,
        n_points=int(keep.sum()),
        per_conc_mean_offset=per_mean,
        per_conc_std_offset=per_std,
        source=source,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------

def plot_calibration(
    log10_conc: np.ndarray,
    chip_ttp: np.ndarray,
    plate_ttp: np.ndarray,
    cal_full: Calibration,
    cal_clean: Calibration,
    exclude_concs: tuple[int, ...],
    out_png: Path,
) -> None:
    offset = chip_ttp - plate_ttp
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: raw chip/plate TTP scatter with y=x reference
    ax1.scatter(plate_ttp, chip_ttp, s=45, alpha=0.7,
                c=["#b30000" if c in exclude_concs else "#1f3a6b" for c in log10_conc])
    lo = min(plate_ttp.min(), chip_ttp.min()) - 1
    hi = max(plate_ttp.max(), chip_ttp.max()) + 1
    ax1.plot([lo, hi], [lo, hi], "--", color="#888", linewidth=1, label="y = x")
    ax1.set_xlabel("plate TTP (min)")
    ax1.set_ylabel("chip TTP (min)")
    ax1.set_title(f"COVID chip vs plate TTP (n={len(chip_ttp)}) — red = excluded",
                  fontsize=11, color="#1f3a6b")
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.legend(loc="lower right", fontsize=9)

    # Panel 2: offset vs log10_conc with both fits
    for c, o in zip(log10_conc, offset):
        colour = "#b30000" if c in exclude_concs else "#1f3a6b"
        ax2.scatter(c, o, s=45, alpha=0.7, color=colour)

    lc_grid = np.linspace(log10_conc.min() - 0.5, log10_conc.max() + 0.5, 50)
    ax2.plot(lc_grid, cal_full.a * lc_grid + cal_full.b, ":",
             color="#888", label=f"full fit (n={cal_full.n_points}): "
                                  f"offset = {cal_full.a:+.2f}·log10 + {cal_full.b:+.2f}")
    ax2.plot(lc_grid, cal_clean.a * lc_grid + cal_clean.b, "-",
             color="#2ca02c", linewidth=2,
             label=f"clean fit (n={cal_clean.n_points}): "
                   f"offset = {cal_clean.a:+.2f}·log10 + {cal_clean.b:+.2f}\n"
                   f"r² = {cal_clean.r2:.3f}, res_std = {cal_clean.residual_std:.2f} min")

    ax2.axhline(0, color="#888", linewidth=0.5, alpha=0.5)
    ax2.set_xlabel("log10 concentration (copies/rxn)")
    ax2.set_ylabel("offset = chip_TTP − plate_TTP (min)")
    ax2.set_title("Offset vs concentration + linear fit",
                  fontsize=11, color="#1f3a6b")
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("Chip↔plate TTP offset calibration (COVID data)",
                 fontsize=12, color="#1f3a6b", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    log10_conc, chip_ttp, plate_ttp = load_covid_paired_data()

    print(f"Loaded {len(chip_ttp)} paired points from COVID Sheet2")
    print(f"  log10 concentrations: {sorted(set(log10_conc.tolist()), reverse=True)}")
    print(f"  chip TTP range: [{chip_ttp.min():.2f}, {chip_ttp.max():.2f}] min")
    print(f"  plate TTP range: [{plate_ttp.min():.2f}, {plate_ttp.max():.2f}] min")

    # Full fit
    cal_full = fit_calibration(
        log10_conc, chip_ttp, plate_ttp, exclude_concs=(),
        source="COVID Lacewing_Quantification_Results.xlsx Sheet2 — all points",
        notes="Includes 1e7 which has an anomalous plate value (18.47 vs expected ~14)."
    )
    print(f"\nFull fit (n={cal_full.n_points}):")
    print(f"  a={cal_full.a:.3f} min/decade  b={cal_full.b:.3f} min")
    print(f"  r²={cal_full.r2:.3f}   residual std={cal_full.residual_std:.2f} min")

    # Clean fit — exclude 1e7 (plate value looks anomalous)
    cal_clean = fit_calibration(
        log10_conc, chip_ttp, plate_ttp, exclude_concs=(7,),
        source="COVID Lacewing_Quantification_Results.xlsx Sheet2 — excl 1e7",
        notes="1e7 plate TTP (18.47) is inconsistent with the concentration ladder — excluded."
    )
    print(f"\nClean fit (n={cal_clean.n_points}, excluded log10=7):")
    print(f"  a={cal_clean.a:.3f} min/decade  b={cal_clean.b:.3f} min")
    print(f"  r²={cal_clean.r2:.3f}   residual std={cal_clean.residual_std:.2f} min")

    # Per-concentration offsets — helps see the trend
    print("\nPer-concentration mean offsets (all data):")
    for c in sorted(cal_full.per_conc_mean_offset.keys(), reverse=True):
        mean = cal_full.per_conc_mean_offset[c]
        std = cal_full.per_conc_std_offset[c]
        print(f"  log10={c}   mean={mean:+.2f}  std={std:.2f}")

    # Save JSON
    payload = {
        "primary_calibration": asdict(cal_clean),
        "full_data_calibration": asdict(cal_full),
        "cycle_time_min": 0.5,
        "formula": "offset_min = a * log10_conc + b   (offset = chip_TTP - plate_TTP)",
        "usage": ("To convert plate TTP → expected chip TTP: "
                  "chip_TTP_expected = plate_TTP + (a * log10_conc + b)"),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\n[ok] wrote {OUT_JSON}")

    # Diagnostic plot
    plot_calibration(log10_conc, chip_ttp, plate_ttp,
                     cal_full=cal_full, cal_clean=cal_clean,
                     exclude_concs=(7,), out_png=OUT_PNG)
    print(f"[ok] wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
