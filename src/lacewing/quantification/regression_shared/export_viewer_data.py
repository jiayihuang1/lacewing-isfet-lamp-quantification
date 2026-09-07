"""Export a single JSON sidecar for the regression results HTML viewer.

The viewer needs three things per (model, loss) cell:
  1. Headline metrics (best-3 mean: pixel MAE, pixel R^2, well MAE,
     well R^2, picked seeds).
  2. The best run's per-well predictions (y_true, y_pred per well).
  3. A sample of per-pixel traces from the best run, with each pixel's
     X trace (450 floats, downsampled if needed) and its prediction.

We cap per-cell pixel sampling so the JSON stays small enough to
serve from disk (browsers choke past ~30 MB).  500 pixels per well x
5 wells x 24 cells x 450 samples = ~270 MB of floats -- way too much.
We:
    * downsample X by factor DOWN (default 4 -> 113 samples per pixel)
    * cap pixels per well to N_PER_WELL (default 80)
    * round floats to 3 dp for compactness in JSON
"""
from __future__ import annotations

import csv
import json
import statistics as stats
from pathlib import Path

import numpy as np
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[2]
ROOT         = LACEWING_PKG_DIR / "regression" / "results"
CACHE_PATH   = LACEWING_PKG_DIR / "regression" / "data" / "cache" / "regress_all_filt_abcd_ntcRaw_madk1p5.npz"
OUT_PATH     = LACEWING_PKG_DIR / "regression" / "viewer" / "regression_viewer_data.json"

MODELS = ["ann", "cnn1d", "fcn", "resnet", "inception",
          "transformer", "transformer_patch",
          "cnn_transformer_par", "cnn_transformer_seq",
          "gru", "cnn_gru_par", "cnn_gru_seq"]
LOSSES = ["huber", "mse"]

# Compression knobs.
DOWN        = 4              # keep every 4th time sample
N_PER_WELL  = 80             # pixels per well per cell


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


def _pearson_r2(xs: list[float], ys: list[float]) -> float:
    """Squared Pearson correlation, the rule-based-method convention.

    Matches what the SDM / TTP / Cy0 reports compute (Pearson r,
    squared) — measures TREND between the n=5 per-well predicted
    means and the n=5 qLAMP ground-truth values.  Differs from the
    standard R² (1 − SS_res/SS_tot) when the model's predictions are
    on a different scale or intercept than the ground truth.
    """
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
    r = sxy / (sxx ** 0.5 * syy ** 0.5)
    return r * r


def _collect_runs(exp_dir: Path) -> list[dict]:
    runs = []
    rd_pool = sorted((exp_dir / "runs").iterdir()) if (exp_dir / "runs").exists() else []
    for rd in rd_pool:
        if not rd.is_dir():
            continue
        jp = rd / "test_metrics.json"
        if not jp.exists():
            continue
        m = json.loads(jp.read_text())
        seed = int(rd.name.rsplit("seed", 1)[1])

        # Compute the apples-to-apples per-well Pearson² from this
        # run's per_well payload.  This matches what the SDM/TTP/Cy0
        # reports compute (Pearson r between n=5 per-well predictions
        # and n=5 qLAMP ground truths, squared).
        per_well = m.get("per_well", [])
        xs_pw = [float(pw["y_true_min"]) for pw in per_well]
        ys_pw = [float(pw["y_pred_min"]) for pw in per_well]
        m.setdefault("well", {})["pearson_r2"] = _pearson_r2(xs_pw, ys_pw)

        runs.append({
            "seed":     seed,
            "rd":       rd,
            "metrics":  m,
            "val_mae":  _final_val_mae(rd),
        })
    return runs


def main() -> None:
    print(f"Loading cache: {CACHE_PATH}")
    cache = np.load(CACHE_PATH)
    X_all       = cache["X"].astype(np.float32, copy=False)
    chip_all    = np.asarray(cache["chip_id"])
    well_all    = np.asarray(cache["well_id"])
    pixel_all   = np.asarray(cache["pixel_id"])
    split_all   = np.asarray(cache["split"])
    y_ttp_all   = np.asarray(cache["y_ttp_min"])

    # The viewer focuses on the held-out test split (split == 1 = SD).
    test_mask = (split_all == 1)
    X_te      = X_all[test_mask]
    chip_te   = chip_all[test_mask]
    well_te   = well_all[test_mask]
    pixel_te  = pixel_all[test_mask]
    y_te      = y_ttp_all[test_mask]

    print(f"Test pixels in cache: {len(X_te)}")
    print(f"  chips:  {sorted(set(map(str, chip_te)))}")
    print(f"  wells:  {sorted(set(map(int, well_te)))}")

    # Downsample X for transport.
    n_T = X_te.shape[1]
    t_idx = np.arange(0, n_T, DOWN)
    X_te_down = X_te[:, t_idx]
    # Each frame is 4 s, so DOWN=4 corresponds to a 16 s effective spacing.
    # Just emit indices as "minutes-ish" via an absolute time axis (the
    # cache stores X with sample index, not real minutes; for visual
    # purposes a 0..N grid is fine but we expose the sample indices too).
    t_axis = (t_idx * 4 / 60).tolist()   # 4 s sample period -> minutes

    # Build a per-(chip, well) pixel index map for quick sampling.
    keys = np.array([f"{c}::{w}" for c, w in zip(chip_te, well_te)])
    unique_keys = sorted(set(keys.tolist()))

    rng = np.random.default_rng(0)
    sample_idx_by_key: dict[str, np.ndarray] = {}
    for k in unique_keys:
        m = (keys == k)
        idx = np.flatnonzero(m)
        if len(idx) > N_PER_WELL:
            idx = rng.choice(idx, size=N_PER_WELL, replace=False)
            idx.sort()
        sample_idx_by_key[k] = idx

    # Pre-emit per-key sampled traces (shared across all cells).
    sampled_traces: dict[str, dict] = {}
    for k in unique_keys:
        idx = sample_idx_by_key[k]
        traces = X_te_down[idx]                # (n, T_down)
        sampled_traces[k] = {
            "n":        int(len(idx)),
            "pixel_id": pixel_te[idx].tolist(),
            "traces":   [[round(float(v), 4) for v in row] for row in traces],
            # Per-pixel true TTP (constant across the well, but include it
            # so the viewer doesn't need to look up per-well metadata).
            "y_true":   round(float(y_te[idx[0]]), 3),
            "chip_id":  str(chip_te[idx[0]]),
            "well_id":  int(well_te[idx[0]]),
        }

    # Per-cell: pick best-3 seeds by val MAE; record headline metrics
    # AND the per-pixel predictions of the best seed for the sampled set.
    cells = {}
    for mdl in MODELS:
        for loss in LOSSES:
            exp = ROOT / f"reg_{mdl}_{loss}"
            runs = _collect_runs(exp)
            if not runs:
                continue
            valid = [r for r in runs if r["val_mae"] is not None]
            picked = sorted(valid, key=lambda r: r["val_mae"])[:3] if len(valid) >= 3 else runs
            best   = picked[0] if picked else runs[0]

            # Mean across best-3 seeds.
            def _safe_mean(values):
                clean = [v for v in values if v == v]   # drop NaN
                return stats.mean(clean) if clean else float("nan")

            b3 = {
                "pixel_mae_min": _safe_mean([r["metrics"]["pixel"]["mae_min"] for r in picked]),
                "pixel_rmse_min": _safe_mean([r["metrics"]["pixel"]["rmse_min"] for r in picked]),
                "pixel_r2":       _safe_mean([r["metrics"]["pixel"]["r2"] for r in picked]),
                "well_mae_min":   _safe_mean([r["metrics"]["well"]["mae_min"] for r in picked]),
                "well_r2":        _safe_mean([r["metrics"]["well"]["r2"] for r in picked]),
                # Apples-to-apples with the rule-based methods
                # (TTP / SDM / Cy0): Pearson² across the 5 SD-test
                # dilution wells.
                "well_pearson_r2": _safe_mean(
                    [r["metrics"]["well"]["pearson_r2"] for r in picked]),
                "baseline_constant_mean_mae": _safe_mean(
                    [r["metrics"]["pixel"]["mae_baseline_constant_mean"] for r in picked]),
                "n_seeds_picked": len(picked),
                "picked_seeds": [r["seed"] for r in picked],
            }

            # Load best seed's per-pixel predictions and align to our
            # sampled indices via (chip, well, pixel_id) -> y_pred lookup.
            preds_npz = best["rd"] / "predictions.npz"
            pp = np.load(preds_npz)
            pp_keys = np.array([f"{c}::{w}::{p}" for c, w, p in
                                 zip(pp["chip_id"], pp["well_id"], pp["pixel_id"])])
            pred_by_key = dict(zip(pp_keys.tolist(),
                                    pp["y_pred_min"].astype(float).tolist()))

            preds_sampled: dict[str, list] = {}
            for k in unique_keys:
                idx = sample_idx_by_key[k]
                chip_w = sampled_traces[k]["chip_id"]
                well_w = sampled_traces[k]["well_id"]
                pix_ids = sampled_traces[k]["pixel_id"]
                preds = []
                for px in pix_ids:
                    full_key = f"{chip_w}::{well_w}::{px}"
                    preds.append(round(pred_by_key.get(full_key, float("nan")), 3))
                preds_sampled[k] = preds

            # Per-well aggregated y_pred (mean of all test pixels in that
            # well, computed from the FULL predictions cache so it
            # matches what test_metrics.json reports).
            per_well_pred: dict[str, float] = {}
            per_well_true: dict[str, float] = {}
            mask_keys = np.array([f"{c}::{w}" for c, w in
                                   zip(pp["chip_id"], pp["well_id"])])
            for k in unique_keys:
                m_ = (mask_keys == k)
                if m_.any():
                    per_well_pred[k] = round(float(pp["y_pred_min"][m_].mean()), 3)
                    per_well_true[k] = round(float(pp["y_true_min"][m_][0]), 3)

            # Best seed's per-well Pearson² (computed from the FULL
            # predictions cache, so it matches what's shown in the
            # per-well chart).
            xs_seed = [per_well_true[k]  for k in unique_keys if per_well_true.get(k) is not None]
            ys_seed = [per_well_pred[k]  for k in unique_keys if per_well_pred.get(k) is not None]
            best_seed_well_pearson_r2 = _pearson_r2(xs_seed, ys_seed)

            cells[f"{mdl}__{loss}"] = {
                "model":            mdl,
                "loss":             loss,
                "best_seed":        best["seed"],
                "best_seed_val_mae_norm": best["val_mae"],
                "best_seed_well_pearson_r2": best_seed_well_pearson_r2,
                "best3":            b3,
                "preds_sampled":    preds_sampled,
                "per_well_pred":    per_well_pred,
                "per_well_true":    per_well_true,
            }

    payload = {
        "schema":       1,
        "t_axis_min":   t_axis,
        "down_factor":  DOWN,
        "n_samples_per_trace": int(X_te_down.shape[1]),
        "sampled_traces": sampled_traces,
        "unique_keys":  unique_keys,
        "models":       MODELS,
        "losses":       LOSSES,
        "cells":        cells,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nWrote {OUT_PATH} ({size_mb:.1f} MB)")
    print(f"  cells: {len(cells)}")
    print(f"  unique (chip, well) keys: {len(unique_keys)}")
    print(f"  traces per key: <= {N_PER_WELL}, downsampled by {DOWN}")


if __name__ == "__main__":
    main()
