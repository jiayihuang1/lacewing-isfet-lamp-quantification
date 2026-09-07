"""Dose-response evaluation: inference on the 5 dose-response chips.

For each saved regression run, load the best checkpoint and run
inference on the dose-response chips (which were in the training set).
Compute per-chip mean predicted TTP and the Pearson r / slope vs
log10(concentration) across the 5 chips.

This is the apples-to-apples comparison with rule-based TTP / SDM / Cy0
methods, which compute the same statistic on the same 5 chips
(deterministic rules, no train/test split).

The regression numbers are IN-SAMPLE (chips were in training).  This
is the same evaluation regime the rule-based methods use, so the
comparison is fair on the metric itself; the caveat is just that
neither side is a held-out test in this comparison.

Output: Analysis/regression/results/dose_response_summary.json
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
              "seeds": [...],
          },
      },
    }

Run::
    python -m lacewing.quantification.regression_shared.eval_dose_response
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from lacewing.classification import models
from lacewing.quantification.regression_shared.core import paths
from lacewing.quantification.regression_shared.data.labels import (
    CONC_LOG, DOSE_RESPONSE_CHIP_FOLDERS, FOLDER_TO_CONC_KEY, CONC_KEYS,
)


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
OUT_PATH = RESULTS_DIR / "dose_response_summary.json"
CACHE_STEM = "regress_all_filt_abcd_ntcRaw_madk1p5"
RULE_BASED_CSV = (HERE.parent / "quantification" / "results"
                  / "sdm_cy0_v2_smooth100_start10" / "per_chip.csv")


def _pearson_r(xs, ys) -> float:
    if len(xs) < 2: return float("nan")
    n = len(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(xs, ys))
    sxx = sum((xi - mx) ** 2 for xi in xs)
    syy = sum((yi - my) ** 2 for yi in ys)
    if sxx == 0 or syy == 0: return float("nan")
    return sxy / (math.sqrt(sxx) * math.sqrt(syy))


def _slope(xs, ys) -> float:
    if len(xs) < 2: return float("nan")
    return float(np.polyfit(np.array(xs), np.array(ys), 1)[0])


def _final_val_mae(rd: Path) -> float | None:
    p = rd / "metrics.csv"
    if not p.exists(): return None
    last = None
    with p.open() as fh:
        for row in csv.DictReader(fh): last = row
    if not last: return None
    try: return float(last["va_mae_norm"])
    except (KeyError, ValueError): return None


def _load_config(rd: Path) -> dict:
    """Parse the simple `key: <json>` config.yaml emitted by train.py."""
    p = rd / "config.yaml"
    out: dict = {}
    if not p.exists(): return out
    for line in p.read_text().splitlines():
        if not line.strip() or ":" not in line: continue
        k, _, v = line.partition(":")
        try: out[k.strip()] = json.loads(v.strip())
        except json.JSONDecodeError: out[k.strip()] = v.strip()
    return out


# Cache the per-pixel dose-response data once.
_CACHE: dict | None = None


def _dose_response_pixels() -> dict:
    """Return X (N, T), chip_id (N,) restricted to the 5 dose-response chips.

    Pulled from the regression cache (split==0 = the dose-response
    training pixels)."""
    global _CACHE
    if _CACHE is not None: return _CACHE
    cache_path = paths.regression_cache_path(CACHE_STEM)
    with np.load(cache_path) as data:
        X = data["X"].astype(np.float32, copy=False)
        chip = np.asarray(data["chip_id"])
        split = np.asarray(data["split"])
    mask = (split == 0)
    X = X[mask]; chip = chip[mask]
    print(f"  Dose-response pixels: {len(X)}  "
          f"chips: {sorted(set(chip.tolist()))}")
    _CACHE = {"X": X, "chip": chip}
    return _CACHE


@torch.no_grad()
def _predict_per_chip(rd: Path, device: str = "cpu") -> dict[str, float] | None:
    """Run inference on the dose-response chips and return per-chip mean TTP."""
    ckpt_path = rd / "checkpoints" / "best.pt"
    if not ckpt_path.exists(): return None
    cfg = _load_config(rd)
    model_name = cfg.get("model")
    if not model_name: return None
    y_mean = float(cfg.get("y_mean_train", 0.0))
    y_std  = float(cfg.get("y_std_train",  1.0))
    normalise = bool(cfg.get("normalise_y", True))

    pixels = _dose_response_pixels()
    X = pixels["X"]; chip = pixels["chip"]
    input_shape = (1, X.shape[1])
    model = models.build(model_name, input_shape).to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    except Exception as e:
        print(f"  [skip] {rd.name}: load failed ({e})")
        return None
    model.eval()

    preds = []
    BATCH = 1024
    for i in range(0, len(X), BATCH):
        batch = torch.from_numpy(X[i:i + BATCH][:, None, :]).to(device)
        p = model(batch).squeeze(-1).cpu().numpy()
        preds.append(p)
    preds = np.concatenate(preds)
    if normalise:
        preds = preds * y_std + y_mean

    # Per-chip mean.
    per_chip: dict[str, float] = {}
    for c in sorted(set(chip.tolist())):
        m = (chip == c)
        per_chip[str(c)] = float(preds[m].mean())
    return per_chip


def _eval_run(rd: Path, device: str) -> dict | None:
    per_chip = _predict_per_chip(rd, device=device)
    if per_chip is None: return None
    # Build (log10 conc, predicted TTP) pairs for the 5 chips.
    xs, ys = [], []
    for chip, pred in per_chip.items():
        if chip not in FOLDER_TO_CONC_KEY: continue
        conc_key = FOLDER_TO_CONC_KEY[chip]
        lc = float(CONC_LOG[CONC_KEYS.index(conc_key)])
        xs.append(lc); ys.append(pred)
    if len(xs) < 2: return None
    # Sort by lc so the per-chip dict reads naturally.
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
    r = _pearson_r(xs, ys)
    return {
        "r": r, "r2": r * r, "slope": _slope(xs, ys),
        "per_chip_pred": per_chip,
        "log_concs": xs, "pred_ttps": ys,
        "val_mae": _final_val_mae(rd),
    }


def _rule_based() -> dict[str, dict[str, float]]:
    """Compute rule-based per-chip metrics from the same 5 dose-response chips.

    Uses the chip-mean of wells 0..3 for each method, per chip's row in
    the v2 per-chip CSV.  Mirrors the rule-based reports' published
    Pearson r / slope numbers (computed on the dose-response chips).
    """
    rows: dict[str, dict[str, float]] = {}
    with RULE_BASED_CSV.open() as fh:
        for row in csv.DictReader(fh):
            rows[row["concentration"]] = row
    out: dict[str, dict[str, float]] = {}
    for method, prefix in [("TTP", "ttp"), ("SDM", "sdm"), ("Cy0", "cy0")]:
        xs, ys = [], []
        for conc_key, lc in zip(CONC_KEYS, CONC_LOG):
            row = rows.get(conc_key)
            if row is None: continue
            vals = []
            for w in (0, 1, 2, 3):
                v = row.get(f"{prefix}_well{w}", "")
                try: fv = float(v)
                except ValueError: continue
                if not math.isnan(fv): vals.append(fv)
            if vals:
                xs.append(float(lc)); ys.append(float(np.mean(vals)))
        r = _pearson_r(xs, ys)
        out[method] = {
            "r": r, "r2": r * r, "slope": _slope(xs, ys),
            "n_chips": len(xs),
            "log_concs": xs, "chip_ttps": ys,
        }
    return out


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    rule = _rule_based()
    print("Rule-based dose-response (per-chip mean of wells 0..3, 5 chips):")
    for k, v in rule.items():
        print(f"  {k}:  r={v['r']:+.4f}  r2={v['r2']:.4f}  "
              f"slope={v['slope']:+.3f} min/decade  n={v['n_chips']}")

    cells: dict[str, dict] = {}
    cell_dirs = sorted(d for d in RESULTS_DIR.iterdir()
                       if d.is_dir() and d.name.startswith("reg_"))
    n_cells = len(cell_dirs)
    for ci, cd in enumerate(cell_dirs):
        runs_dir = cd / "runs"
        if not runs_dir.exists(): continue
        rest = cd.name[len("reg_"):]
        if rest.endswith("_huber"): model = rest[:-len("_huber")]; loss = "huber"
        elif rest.endswith("_mse"): model = rest[:-len("_mse")]; loss = "mse"
        else: continue

        print(f"\n[{ci+1}/{n_cells}] {cd.name}")
        seed_rows: list[dict] = []
        for rd in sorted(runs_dir.iterdir()):
            if not rd.is_dir(): continue
            try: seed = int(rd.name.rsplit("seed", 1)[1])
            except (ValueError, IndexError): continue
            ev = _eval_run(rd, device=device)
            if ev is None: continue
            ev["seed"] = seed
            ev["run_dir"] = rd.name
            seed_rows.append(ev)
            print(f"    seed {seed:>2}: r={ev['r']:+.4f}  "
                  f"slope={ev['slope']:+.3f}  val_mae={ev['val_mae']}")

        if not seed_rows: continue
        with_val = [r for r in seed_rows if r.get("val_mae") is not None]
        picked = (sorted(with_val, key=lambda r: (r["val_mae"], r["seed"]))[:3]
                  if len(with_val) >= 3 else seed_rows)
        best3_r2 = float(np.mean([r["r2"] for r in picked]))
        best3_slope = float(np.mean([r["slope"] for r in picked]))
        best_seed = max(seed_rows, key=lambda r: r["r2"])

        cells[f"{model}__{loss}"] = {
            "model": model, "loss": loss,
            "n_seeds": len(seed_rows),
            "best3_r2": best3_r2, "best3_slope": best3_slope,
            "best_seed_r2": best_seed["r2"],
            "best_seed_slope": best_seed["slope"],
            "best_seed_seed": best_seed["seed"],
            "best_seed_per_chip_pred": best_seed["per_chip_pred"],
            "best_seed_log_concs": best_seed["log_concs"],
            "best_seed_pred_ttps": best_seed["pred_ttps"],
            "seeds": [
                {"seed": r["seed"], "r": r["r"], "r2": r["r2"],
                 "slope": r["slope"], "val_mae": r["val_mae"],
                 "in_best3": r in picked}
                for r in seed_rows
            ],
        }

    sorted_cells = dict(sorted(cells.items(),
                               key=lambda kv: -kv[1]["best3_r2"]))
    out = {
        "metric": ("Pearson r squared between per-chip mean predicted TTP "
                   "and log10(concentration), n=5 dose-response chips.  "
                   "Both regression and rule-based methods are IN-SAMPLE "
                   "(regression: chips were in training; rule-based: "
                   "deterministic rules with no train/test split)."),
        "rule_based": rule,
        "cells": sorted_cells,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")
    print(f"\nTop 5 (model, loss) by best-3 r2:")
    for k, v in list(sorted_cells.items())[:5]:
        print(f"  {k:<30} best3 r2={v['best3_r2']:.4f}  "
              f"slope={v['best3_slope']:+.3f}  (n_seeds={v['n_seeds']})")


if __name__ == "__main__":
    main()
