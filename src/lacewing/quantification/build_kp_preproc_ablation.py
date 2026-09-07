"""Build · KP 3-state preprocessing ablation figure + table data. [Cat A] Report §NewData.

KP preprocessing ablation on computing baselines.

For each of the 3 computing extractors (threshold-derivative TTP, Cy0, SDM):
  - Compute per-well predictions on 3 preprocessing states:
      1. Deployed: baseline-subtraction only (mean_raw)
      2. +MAD:     deployed + MAD-ABCD pixel-quality filter (mean_qc)
      3. +spatA3:  deployed + MAD-ABCD + order-3 spatial averaging (mean_spat)
  - Score each state's MAE against the per-well plate takeoff (LC96 anchor)
  - Pool across all 5 KP chips, amp-positive wells only

Requires the per-well mean_raw / mean_qc / mean_spat signals per state.
process_conc_chips.py only saves mean_after_spat by default, so we rerun
the pipeline in-process and capture all three signal stages.

Output:
    Analysis/quantification/results/kp_preproc_ablation/summary.csv
    Analysis/quantification/results/kp_preproc_ablation/per_well.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lacewing.quantification.conc_data.process_conc_chips import (
    _build_chip_configs,
    PLATE_FEATURES_JSON,
    _LOG10_TO_KP,
)
from lacewing.quantification.chip_pipeline import process_chip as pipe
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


OUT_DIR = LACEWING_PKG_DIR / "quantification" / "results" / "kp_preproc_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The 3 computing extractors (using the pipeline's local copies to keep
# smoothing consistent with the RQ2 chapter numbers).
EXTRACTORS = ("ttp", "cy0", "sdm")


def extract_all(sig: np.ndarray, time_min: np.ndarray, search_start: float) -> dict[str, float]:
    """Run all three extractors on one signal, return dict of name -> ttp_min."""
    out: dict[str, float] = {}
    if np.all(np.isnan(sig)):
        return {k: float("nan") for k in EXTRACTORS}
    try:
        ttp, _peak = pipe.extract_ttp_threshold(time_min, sig, search_start)
        out["ttp"] = float(ttp)
    except Exception:
        out["ttp"] = float("nan")
    try:
        out["cy0"] = float(pipe.extract_cy0(time_min, sig, search_start))
    except Exception:
        out["cy0"] = float("nan")
    try:
        out["sdm"] = float(pipe.extract_sdm(time_min, sig, search_start))
    except Exception:
        out["sdm"] = float("nan")
    return out


def _plate_ttp_for_well(cfg, plate_per_conc, w_idx: int):
    log10 = cfg.well_log10.get(w_idx)
    if log10 is None or not np.isfinite(log10):
        return None
    kp_label = _LOG10_TO_KP.get(float(log10))
    if kp_label is None:
        return None
    entry = plate_per_conc.get(kp_label)
    if entry is None:
        return None
    return float(entry.get("ttp_min"))


def process_chip_with_ablation(cfg) -> list[dict]:
    """Run the full pipeline on one chip, capture all 3 preprocessing states
    per well, and score each state's extractor predictions against plate TTP.
    """
    plate_data = json.loads(PLATE_FEATURES_JSON.read_text())
    plate_per_conc = plate_data[cfg.plate_key]["per_conc_kp"]

    # Push chip config into pipeline globals (same pattern as
    # build_regression_cache.py).
    old = (pipe.WELL_LABELS, pipe.WELL_LOG10, pipe.NTC_WELLS, pipe.TRIM_SEARCH_MAX_MIN)
    try:
        pipe.WELL_LABELS = cfg.well_labels
        pipe.WELL_LOG10 = cfg.well_log10
        pipe.NTC_WELLS = tuple(cfg.ntc_wells)
        pipe.TRIM_SEARCH_MAX_MIN = cfg.trim_search_max_min
        result = pipe.process_chip(cfg.chip_dir, save_per_pixel=True)
    finally:
        pipe.WELL_LABELS, pipe.WELL_LOG10, pipe.NTC_WELLS, pipe.TRIM_SEARCH_MAX_MIN = old

    rows: list[dict] = []
    for w in result.wells:
        plate_ttp = _plate_ttp_for_well(cfg, plate_per_conc, w.well)
        if plate_ttp is None:
            continue  # skip PTC/NTC
        # w exposes mean_raw / mean_qc / mean_after_spat (see process_chip.py).
        # And w.time_min for the time axis.
        # w.search_start_min for per-well search start.
        for state, sig_attr in (
            ("deployed", "mean_bs_raw"),
            ("+MAD", "mean_after_qc"),
            ("+spatA3", "mean_after_spat"),
        ):
            sig = getattr(w, sig_attr, None)
            if sig is None:
                continue
            ext = extract_all(np.asarray(sig, dtype=np.float64),
                              np.asarray(w.time_min, dtype=np.float64),
                              float(w.search_start_min))
            for method, pred in ext.items():
                rows.append({
                    "chip": cfg.chip_tag,
                    "well": w.well,
                    "log10_conc": w.log10_conc,
                    "plate_ttp": plate_ttp,
                    "state": state,
                    "method": method,
                    "pred": pred,
                    "err": (pred - plate_ttp) if np.isfinite(pred) else float("nan"),
                })
    return rows


def main():
    all_rows = []
    for cfg in _build_chip_configs():
        if not cfg.chip_dir.exists():
            print(f"[skip] {cfg.chip_tag} — chip_dir missing")
            continue
        print(f"\n=== {cfg.chip_tag} ===")
        rows = process_chip_with_ablation(cfg)
        all_rows.extend(rows)
        print(f"  {len(rows)} (well, state, method) rows added")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "per_well.csv", index=False)

    # Summary: mean per-well MAE per (state, method), pooled across all chips
    summary = (
        df.dropna(subset=["err"])
        .assign(abs_err=lambda d: d["err"].abs())
        .groupby(["state", "method"])
        .agg(mae=("abs_err", "mean"), n=("abs_err", "size"))
        .reset_index()
    )
    print("\n=== Pooled per-well MAE against plate TTP (all 5 KP chips) ===")
    pivot = summary.pivot(index="method", columns="state", values="mae").round(3)
    # Reorder columns to match the report's ablation table (deployed → +MAD → +spatA3)
    pivot = pivot[["deployed", "+MAD", "+spatA3"]]
    print(pivot.to_string())

    n_pivot = summary.pivot(index="method", columns="state", values="n").astype(int)
    print("\n=== Well count per (state, method) ===")
    print(n_pivot.to_string())

    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    pivot.to_csv(OUT_DIR / "mae_table.csv")
    print(f"\nSaved {OUT_DIR / 'summary.csv'}")
    print(f"Saved {OUT_DIR / 'mae_table.csv'}")


if __name__ == "__main__":
    main()
