"""SD-only dose-response evaluation for the regression sweep.

For each saved regression run, compute per-well predicted TTP on the
held-out SD chip and the Pearson r / slope vs log10(concentration)
across the 5 dilution wells.

Output: Analysis/regression/results/sd_dose_response_summary.json
    {
      "rule_based": {
          "TTP":   {"r": ..., "r2": ..., "slope": ...},
          "SDM":   {...},
          "Cy0":   {...},
      },
      "cells": {
          "<model>__<loss>": {
              "model": ..., "loss": ...,
              "best3_r2": ..., "best3_slope": ...,
              "best_seed_r2": ..., "best_seed_slope": ...,
              "seeds": [{seed, r, r2, slope, val_mae}, ...],
          },
      },
    }

Run::
    python -m lacewing.quantification.regression_shared.eval_sd_dose_response
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from lacewing.quantification.regression_shared.data.labels import SD_LABEL_LOG, SD_WELL_LABELS


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
OUT_PATH = RESULTS_DIR / "sd_dose_response_summary.json"

# Rule-based per-well SD TTPs (from Analysis/quantification/results).
# Wells 0..4 = E, D, C, B, A (log10 conc = 9, 8, 7, 6, 5).
RULE_BASED_PATH = (HERE.parent / "quantification" / "results"
                   / "sdm_cy0_v2_smooth100_start10" / "per_chip.csv")


def _well_log_conc() -> dict[int, float]:
    """Well index -> log10(conc) for the SD chip."""
    out = {}
    for w, label in enumerate(SD_WELL_LABELS):
        if label in SD_LABEL_LOG:
            out[w] = float(SD_LABEL_LOG[label])
    return out


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(xs, ys))
    sxx = sum((xi - mx) ** 2 for xi in xs)
    syy = sum((yi - my) ** 2 for yi in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / (math.sqrt(sxx) * math.sqrt(syy))


def _slope(xs: list[float], ys: list[float]) -> float:
    """OLS slope of ys on xs (min/decade when xs = log10 conc)."""
    if len(xs) < 2:
        return float("nan")
    a = np.array(xs); b = np.array(ys)
    return float(np.polyfit(a, b, 1)[0])


def _final_val_mae(rd: Path) -> float | None:
    p = rd / "metrics.csv"
    if not p.exists():
        return None
    last = None
    with p.open() as fh:
        for row in csv.DictReader(fh):
            last = row
    if not last:
        return None
    try:
        return float(last["va_mae_norm"])
    except (KeyError, ValueError):
        return None


def _per_well_pred(rd: Path) -> dict[int, float]:
    """Return {well_id: mean predicted TTP minutes} on the SD chip."""
    p = rd / "predictions.npz"
    if not p.exists():
        return {}
    d = np.load(p, allow_pickle=True)
    well = d["well_id"]
    pred = d["y_pred_min"]
    out: dict[int, float] = {}
    for w in sorted(set(int(x) for x in well.tolist())):
        m = (well == w)
        if m.sum():
            out[w] = float(pred[m].mean())
    return out


def _eval_run(rd: Path, well_to_logconc: dict[int, float]) -> dict | None:
    pw = _per_well_pred(rd)
    if not pw:
        return None
    xs, ys = [], []
    for w, lc in sorted(well_to_logconc.items()):
        if w in pw:
            xs.append(lc); ys.append(pw[w])
    if len(xs) < 2:
        return None
    r = _pearson_r(xs, ys)
    sl = _slope(xs, ys)
    return {
        "r": r, "r2": r * r, "slope": sl,
        "per_well_pred": {str(w): float(pw[w]) for w in sorted(pw)},
        "val_mae": _final_val_mae(rd),
    }


def _rule_based() -> dict[str, dict[str, float]]:
    """Parse the SD row from the v2 per-chip CSV."""
    with RULE_BASED_PATH.open() as fh:
        rdr = csv.DictReader(fh)
        sd_row = None
        for row in rdr:
            if row["concentration"] == "SD":
                sd_row = row; break
    if sd_row is None:
        raise FileNotFoundError(f"No SD row in {RULE_BASED_PATH}")
    well_to_logconc = _well_log_conc()
    out: dict[str, dict[str, float]] = {}
    for method, prefix in [("TTP", "ttp"), ("SDM", "sdm"), ("Cy0", "cy0")]:
        xs, ys = [], []
        for w in sorted(well_to_logconc):
            key = f"{prefix}_well{w}"
            v = sd_row.get(key, "")
            try:
                fv = float(v)
            except ValueError:
                continue
            if math.isnan(fv):
                continue
            xs.append(well_to_logconc[w]); ys.append(fv)
        r = _pearson_r(xs, ys)
        out[method] = {
            "r": r, "r2": r * r, "slope": _slope(xs, ys),
            "n_wells": len(xs),
        }
    return out


def main() -> None:
    well_to_logconc = _well_log_conc()
    print(f"SD well -> log10(conc): {well_to_logconc}")

    rule = _rule_based()
    print("Rule-based SD-only:")
    for k, v in rule.items():
        print(f"  {k}:  r={v['r']:+.4f}  r2={v['r2']:.4f}  "
              f"slope={v['slope']:+.3f} min/decade  n={v['n_wells']}")

    # Walk all reg_<model>_<loss> dirs.
    cells: dict[str, dict] = {}
    cell_dirs = sorted(d for d in RESULTS_DIR.iterdir()
                       if d.is_dir() and d.name.startswith("reg_"))
    for cd in cell_dirs:
        runs_dir = cd / "runs"
        if not runs_dir.exists():
            continue
        # Parse "reg_<model>_<loss>".  Loss is the trailing token.
        rest = cd.name[len("reg_"):]
        if rest.endswith("_huber"):
            model = rest[:-len("_huber")]; loss = "huber"
        elif rest.endswith("_mse"):
            model = rest[:-len("_mse")]; loss = "mse"
        else:
            print(f"Skipping {cd.name}: cannot parse loss suffix")
            continue

        seed_rows: list[dict] = []
        for rd in sorted(runs_dir.iterdir()):
            if not rd.is_dir():
                continue
            try:
                seed = int(rd.name.rsplit("seed", 1)[1])
            except (ValueError, IndexError):
                continue
            ev = _eval_run(rd, well_to_logconc)
            if ev is None:
                continue
            ev["seed"] = seed
            ev["run_dir"] = rd.name
            seed_rows.append(ev)

        if not seed_rows:
            continue

        # best-3 by lowest val MAE (matches viewer convention).
        with_val = [r for r in seed_rows if r.get("val_mae") is not None]
        picked = (sorted(with_val, key=lambda r: (r["val_mae"], r["seed"]))[:3]
                  if len(with_val) >= 3 else seed_rows)
        best3_r2 = float(np.mean([r["r2"] for r in picked]))
        best3_slope = float(np.mean([r["slope"] for r in picked]))

        best_seed = max(seed_rows, key=lambda r: r["r2"])

        cells[f"{model}__{loss}"] = {
            "model": model, "loss": loss,
            "n_seeds": len(seed_rows),
            "best3_r2": best3_r2,
            "best3_slope": best3_slope,
            "best_seed_r2": best_seed["r2"],
            "best_seed_slope": best_seed["slope"],
            "best_seed_seed": best_seed["seed"],
            "best_seed_per_well_pred": best_seed["per_well_pred"],
            "seeds": [
                {"seed": r["seed"], "r": r["r"], "r2": r["r2"],
                 "slope": r["slope"], "val_mae": r["val_mae"],
                 "in_best3": r in picked}
                for r in seed_rows
            ],
        }

    # Sort by best3_r2 for at-a-glance leaderboard.
    sorted_cells = dict(sorted(cells.items(),
                               key=lambda kv: -kv[1]["best3_r2"]))
    out = {
        "metric": "Pearson r squared between per-well predicted TTP "
                  "and log10(concentration) on the SD held-out chip "
                  "(n=5 wells).",
        "well_log_conc": {str(k): v for k, v in well_to_logconc.items()},
        "rule_based": rule,
        "cells": sorted_cells,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")
    print(f"Top 5 (model, loss) by best-3 r2:")
    for k, v in list(sorted_cells.items())[:5]:
        print(f"  {k:<30} best3 r2={v['best3_r2']:.4f}  "
              f"slope={v['best3_slope']:+.3f}  (n_seeds={v['n_seeds']})")


if __name__ == "__main__":
    main()
