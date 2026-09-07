"""F1.1 — per-timepoint XGBoost classifier for TTP extraction.

Implements the supervisor's 2026-06-26 framing: treat each timepoint of a
pixel trace as a sample, label it pre/post amplification using the
per-well qLAMP TTP as the hard boundary, train XGBoost on a small
hand-crafted feature vector per (pixel, t).  At test time, score every
timepoint and derive TTP per pixel from the first timepoint where
P(post-amp | t) > 0.5.

Pipeline
--------
1. Load regression cache (default: ``regress_all_filt_abcd_ntcRaw_madk1p5_spatA3``;
   the locked MAD k=1.5 ABCD + Layer E Sweep A order 3 preprocessing).
2. Train / val / test split via lacewing.quantification.regression_shared.data.dataset.make_split
   (1 chip held as val).
3. Per-(pixel, t) features:
     [s(t), s'(t), s''(t), rolling_mean_50(t), rolling_std_50(t), t_elapsed]
   Rolling stats are causal (only past samples).
4. Per-(pixel, t) binary label:
     1 if t * SAMPLES_PER_MIN >= TTP_qLAMP[pixel],  else 0
5. Train XGBoost (hist tree method, early stopping on val log-loss).
6. Predict per-timepoint probabilities on test pixels, derive
   TTP_pred = first t where p > 0.5 (NaN if never crossed).
7. Aggregate per (chip, well), compute pixel / well MAE + R^2 + the same
   per-well payload as lacewing.quantification.regression_shared.core.train.

Outputs (under Analysis/quantification/methods/results/<experiment>/seed<N>/):
    config.json                — exact run config
    xgb_model.json             — saved XGBoost model
    test_metrics.json          — same schema as the regression baselines
    predictions.npz            — per-test-pixel TTP_pred (minutes)
    feature_importance.csv     — XGBoost gain per feature

Usage
-----
    python -m lacewing.quantification.methods.per_timepoint_xgb \\
        --seed 0

    # Override preprocessing (e.g. to compare with non-spatial cache):
    python -m lacewing.quantification.methods.per_timepoint_xgb \\
        --seed 0 --cache regress_all_filt_abcd_ntcRaw_madk1p5
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from lacewing.quantification.regression_shared.core import paths as reg_paths
from lacewing.quantification.regression_shared.data import dataset as reg_ds


# 30-minute window over 450 samples = 4 s per sample = 15 samples per minute.
SAMPLES_PER_MIN = 15
N_SAMPLES = 450
ROLLING_WINDOW = 50         # ~3.3 minutes of context for rolling mean/std
BASELINE_WINDOW = 30        # first 2 minutes used as the per-pixel baseline


# Feature names in the order build_features() emits them.  Keeping this in
# one place so feature_importance.csv stays in sync.
FEATURE_NAMES = [
    # ----- Original 6 features (the survey recipe) ------------------------
    "s",                # 0  signal value at t
    "ds",               # 1  first derivative
    "d2s",              # 2  second derivative
    "rolling_mean_50",  # 3  causal rolling mean (50 samples ≈ 3.3 min)
    "rolling_std_50",   # 4  causal rolling std
    "t_elapsed",        # 5  t / (T - 1)
    # ----- Stage 1 additions (richer trajectory signals) ------------------
    "rise_above_baseline",     # 6  s(t) - baseline_mean
    "z_against_baseline",      # 7  (s(t) - baseline_mean) / (baseline_std + eps)
    "slope_over_50",           # 8  (s(t) - s(t-49)) / 50   (smoothed slope)
]

# All paths live under Analysis/quantification/methods/.
HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results"
FEATURE_CACHE_DIR = HERE / "cache"
FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _rolling_mean_std_causal(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling mean + std over the last ``window`` samples.

    Inputs
        x       (N, T) float32
        window  int     window length in samples

    Returns
        means, stds  each (N, T) float32

    For t < window the window is truncated (uses just the available samples
    from t=0).  This is the same convention used by the supervisor's
    feature-engineering framing — partial windows at the start, not NaN.
    """
    assert x.ndim == 2
    n, T = x.shape
    # Cumulative sums for vectorised rolling mean/var.
    csum = np.cumsum(x, axis=1, dtype=np.float64)
    csum2 = np.cumsum(x.astype(np.float64) ** 2, axis=1)
    # Sums over (t-window+1 .. t) using prefix sums.
    t_idx = np.arange(T)
    lo = np.maximum(0, t_idx - window + 1)
    counts = (t_idx - lo + 1).astype(np.float64)           # (T,)
    # csum[..., t] - csum[..., lo-1], handle lo==0 by taking csum[..., t]
    # directly when lo==0.
    csum_lo = np.where(lo > 0, csum[:, np.clip(lo - 1, 0, T - 1)], 0.0)
    csum2_lo = np.where(lo > 0, csum2[:, np.clip(lo - 1, 0, T - 1)], 0.0)
    sums = csum - csum_lo
    sums2 = csum2 - csum2_lo
    means = sums / counts
    # Var = E[x^2] - (E[x])^2 in the window; clip to >=0 for numerical safety.
    variances = np.maximum(0.0, sums2 / counts - means ** 2)
    stds = np.sqrt(variances)
    return means.astype(np.float32), stds.astype(np.float32)


def build_features(X: np.ndarray) -> np.ndarray:
    """Build the per-(pixel, t) feature tensor (Stage 1: 9 features).

    Input
        X    (N_pixels, T)  float32  raw pixel traces

    Returns
        feats  (N_pixels * T, len(FEATURE_NAMES))  float32  rows in
        (pixel, t) order.  See FEATURE_NAMES for the column meanings.
    """
    n, T = X.shape
    s = X.astype(np.float32, copy=False)
    d1 = np.gradient(s, axis=1).astype(np.float32)
    d2 = np.gradient(d1, axis=1).astype(np.float32)
    rmean, rstd = _rolling_mean_std_causal(s, window=ROLLING_WINDOW)
    t_elapsed = (np.arange(T, dtype=np.float32) / (T - 1)).reshape(1, T)
    t_elapsed = np.broadcast_to(t_elapsed, (n, T))

    # Per-pixel baseline statistics from the FIRST BASELINE_WINDOW samples
    # only (~2 min).  Causal: never peeks at future samples.  Each pixel
    # gets its own baseline, propagated identically across all timesteps.
    baseline_mean = s[:, :BASELINE_WINDOW].mean(axis=1, keepdims=True)   # (N, 1)
    baseline_std = s[:, :BASELINE_WINDOW].std(axis=1, keepdims=True)     # (N, 1)
    rise_above_baseline = s - baseline_mean                              # (N, T)
    z_against_baseline = rise_above_baseline / (baseline_std + 1e-8)     # (N, T)

    # Smoothed slope: change over the last ROLLING_WINDOW samples.  Uses
    # s shifted by (window - 1); for t < window-1 we fall back to using
    # s[0] as the reference (so early samples report cumulative rise).
    lag = ROLLING_WINDOW - 1
    s_lagged = np.empty_like(s)
    s_lagged[:, :lag] = s[:, :1]                  # broadcast s[0] for the warm-up
    s_lagged[:, lag:] = s[:, :T - lag]
    slope_over_50 = (s - s_lagged) / float(ROLLING_WINDOW)

    n_feat = len(FEATURE_NAMES)
    feats = np.empty((n, T, n_feat), dtype=np.float32)
    feats[..., 0] = s
    feats[..., 1] = d1
    feats[..., 2] = d2
    feats[..., 3] = rmean
    feats[..., 4] = rstd
    feats[..., 5] = t_elapsed
    feats[..., 6] = rise_above_baseline
    feats[..., 7] = z_against_baseline
    feats[..., 8] = slope_over_50
    return feats.reshape(n * T, n_feat)


def build_labels(y_ttp_min: np.ndarray, T: int = N_SAMPLES) -> np.ndarray:
    """Per-(pixel, t) binary labels: 1 iff t >= TTP_qLAMP * SAMPLES_PER_MIN.

    Input
        y_ttp_min  (N_pixels,)  float32  qLAMP TTP per pixel (minutes)

    Returns
        labels  (N_pixels * T,)  uint8
    """
    n = len(y_ttp_min)
    ttp_idx = (y_ttp_min * SAMPLES_PER_MIN).astype(np.float32)   # fractional
    t_grid = np.arange(T, dtype=np.float32)[None, :]              # (1, T)
    lbl = (t_grid >= ttp_idx[:, None]).astype(np.uint8)
    return lbl.reshape(n * T)


# Bumped whenever build_features() changes its column layout.  Old caches
# with a different version stem will simply be ignored and rebuilt.
FEATURE_VERSION = "v2_stage1"


def _features_cache_path(cache_stem: str, split_kind: str) -> Path:
    return (FEATURE_CACHE_DIR /
            f"feats_{FEATURE_VERSION}_{cache_stem}_{split_kind}.npz")


def get_or_build_features(X: np.ndarray, cache_stem: str, split_kind: str
                           ) -> np.ndarray:
    """Cache features keyed by (regression cache stem, split kind)."""
    cache_path = _features_cache_path(cache_stem, split_kind)
    if cache_path.exists():
        with np.load(cache_path) as data:
            return data["feats"].astype(np.float32, copy=False)
    feats = build_features(X)
    np.savez_compressed(cache_path, feats=feats)
    return feats


# ---------------------------------------------------------------------------
# Per-pixel TTP from per-(pixel, t) probabilities
# ---------------------------------------------------------------------------

def derive_ttp_per_pixel(p_post_flat: np.ndarray, n_pixels: int,
                          T: int = N_SAMPLES, threshold: float = 0.5
                          ) -> np.ndarray:
    """First-crossing rule.

    Input
        p_post_flat  (n_pixels * T,) float32  P(post-amp | (pixel, t))
        n_pixels     int

    Returns
        ttp_pred_min (n_pixels,) float32, NaN where p never crosses threshold.
    """
    p = p_post_flat.reshape(n_pixels, T)
    crossed = p > threshold
    has_any = crossed.any(axis=1)
    first_idx = np.where(has_any, crossed.argmax(axis=1), -1)
    ttp_pred_min = np.where(
        has_any,
        first_idx.astype(np.float32) / float(SAMPLES_PER_MIN),
        np.float32(np.nan),
    )
    return ttp_pred_min


# ---------------------------------------------------------------------------
# Metrics + reporting (in MINUTES, matching the regression baseline schema)
# ---------------------------------------------------------------------------

def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE / RMSE / R^2 / constant-mean baseline.  Drops NaN predictions."""
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    n_total = len(y_true)
    n_used = int(mask.sum())
    y_true_m = y_true[mask]
    y_pred_m = y_pred[mask]
    err = y_pred_m - y_true_m
    mae = float(np.mean(np.abs(err))) if n_used else float("nan")
    rmse = float(np.sqrt(np.mean(err ** 2))) if n_used else float("nan")
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true_m - y_true_m.mean()) ** 2)) if n_used else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae_baseline = (float(np.mean(np.abs(y_true_m - y_true_m.mean())))
                    if n_used else float("nan"))
    return {
        "mae_min":  mae,
        "rmse_min": rmse,
        "r2":       r2,
        "mae_baseline_constant_mean": mae_baseline,
        "n":        n_used,
        "n_total":  n_total,
        "n_nan_pred": n_total - n_used,
    }


def _per_well_predictions(pred_min: np.ndarray, chip: np.ndarray,
                          well: np.ndarray) -> dict[str, float]:
    """Per-well mean of per-pixel predictions, ignoring NaN."""
    keys = np.array([f"{c}::{w}" for c, w in zip(chip, well)])
    out: dict[str, list[float]] = {}
    for k, p in zip(keys, pred_min):
        if not np.isnan(p):
            out.setdefault(k, []).append(float(p))
    return {k: float(np.mean(v)) if v else float("nan")
            for k, v in out.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULTS = {
    "cache":         "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3",
    "experiment":    "f1p1_xgb_spatA3_stage1",
    "val_n_chips":   1,
    "n_estimators":  200,
    "max_depth":     6,
    "learning_rate": 0.1,
    "n_threads":     0,         # 0 = let XGBoost auto-pick
    "early_stopping_rounds": 20,
    "threshold":     0.5,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default=DEFAULTS["cache"])
    p.add_argument("--experiment", default=DEFAULTS["experiment"])
    p.add_argument("--val-n-chips", type=int, default=DEFAULTS["val_n_chips"])
    p.add_argument("--n-estimators", type=int, default=DEFAULTS["n_estimators"])
    p.add_argument("--max-depth", type=int, default=DEFAULTS["max_depth"])
    p.add_argument("--learning-rate", type=float,
                   default=DEFAULTS["learning_rate"])
    p.add_argument("--n-threads", type=int, default=DEFAULTS["n_threads"])
    p.add_argument("--early-stopping-rounds", type=int,
                   default=DEFAULTS["early_stopping_rounds"])
    p.add_argument("--threshold", type=float, default=DEFAULTS["threshold"])
    p.add_argument("--tag", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Quick smoke test: subsample to 200 train pixels, "
                        "n_estimators=20, no feature caching.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    import xgboost as xgb       # imported here so the module loads without it

    # ----- seeding -----
    from lacewing.classification.core.seeding import seed_everything
    seed_everything(args.seed)

    # ----- load + split -----
    print(f"Loading cache: {args.cache}")
    arr = reg_ds.make_split(args.cache, seed=args.seed,
                             val_n_chips=args.val_n_chips,
                             normalise_y=False)   # keep TTP in minutes
    print(f"  n_tr_pixels={len(arr.X_tr)}  "
          f"n_va_pixels={len(arr.X_va)}  n_te_pixels={len(arr.X_te)}")
    print(f"  val chips:  {sorted(set(arr.chip_va.tolist()))}")
    print(f"  test chips: {sorted(set(arr.chip_te.tolist()))}")
    print(f"  train TTPs (unique): "
          f"{sorted(np.unique(arr.y_tr).round(2).tolist())}")

    if args.smoke:
        print("\nSMOKE TEST: subsampling train to 200, val to 100.")
        rng = np.random.default_rng(args.seed)
        tr_keep = rng.choice(len(arr.X_tr), 200, replace=False)
        va_keep = rng.choice(len(arr.X_va), 100, replace=False)
        arr.X_tr = arr.X_tr[tr_keep]; arr.y_tr = arr.y_tr[tr_keep]
        arr.X_va = arr.X_va[va_keep]; arr.y_va = arr.y_va[va_keep]
        arr.chip_tr = arr.chip_tr[tr_keep]; arr.well_tr = arr.well_tr[tr_keep]
        arr.chip_va = arr.chip_va[va_keep]; arr.well_va = arr.well_va[va_keep]

    # ----- build features + labels -----
    print("\nBuilding features (per-pixel feature cache may take ~30 s if cold)...")
    t0 = time.time()
    if args.smoke:
        feats_tr = build_features(arr.X_tr)
        feats_va = build_features(arr.X_va)
        feats_te = build_features(arr.X_te)
    else:
        feats_tr = get_or_build_features(arr.X_tr, args.cache, "tr")
        feats_va = get_or_build_features(arr.X_va, args.cache, "va")
        feats_te = get_or_build_features(arr.X_te, args.cache, "te")
    print(f"  feat shapes: tr={feats_tr.shape}  va={feats_va.shape}  "
          f"te={feats_te.shape}   ({time.time()-t0:.1f} s)")

    print("Building per-timepoint binary labels...")
    lbl_tr = build_labels(arr.y_tr)
    lbl_va = build_labels(arr.y_va)
    lbl_te = build_labels(arr.y_te)
    print(f"  label fractions positive (post-amp): "
          f"tr={lbl_tr.mean():.3f}  va={lbl_va.mean():.3f}  "
          f"te={lbl_te.mean():.3f}")

    # ----- train -----
    print("\nTraining XGBoost...")
    clf_args = dict(
        n_estimators=20 if args.smoke else args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=args.n_threads if args.n_threads else None,
        random_state=args.seed,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    print(f"  XGBoost args: {clf_args}")
    clf = xgb.XGBClassifier(**clf_args)
    t1 = time.time()
    clf.fit(feats_tr, lbl_tr,
            eval_set=[(feats_va, lbl_va)],
            verbose=False)
    train_time_s = time.time() - t1
    best_iter = int(getattr(clf, "best_iteration", clf.n_estimators - 1))
    best_score = float(getattr(clf, "best_score", float("nan")))
    print(f"  trained in {train_time_s/60:.1f} min  "
          f"(best_iteration={best_iter}, best_val_logloss={best_score:.4f})")

    # ----- predict + derive TTP -----
    print("Predicting on test set...")
    p_te = clf.predict_proba(feats_te)[:, 1]
    ttp_pred_min = derive_ttp_per_pixel(p_te, n_pixels=len(arr.X_te),
                                         T=N_SAMPLES,
                                         threshold=args.threshold)
    n_nan = int(np.isnan(ttp_pred_min).sum())
    print(f"  test pixels: {len(arr.X_te)}; "
          f"NaN predictions (never crossed P>{args.threshold}): {n_nan}")

    # ----- metrics -----
    pixel_metrics = _regression_metrics(arr.y_te, ttp_pred_min)
    well_pred_min = _per_well_predictions(ttp_pred_min, arr.chip_te, arr.well_te)
    keys = [f"{c}::{w}" for c, w in zip(arr.chip_te, arr.well_te)]
    well_true: dict[str, float] = {}
    for k, yv in zip(keys, arr.y_te):
        well_true[k] = float(yv)
    well_keys = sorted(set(keys))
    yw_true = np.array([well_true[k]
                        for k in well_keys
                        if not np.isnan(well_pred_min.get(k, np.nan))])
    yw_pred = np.array([well_pred_min[k]
                        for k in well_keys
                        if not np.isnan(well_pred_min.get(k, np.nan))])
    well_metrics = _regression_metrics(yw_true, yw_pred)

    # ----- write outputs -----
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = RESULTS_ROOT / args.experiment / f"seed{args.seed}{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "experiment":    args.experiment,
        "method":        "F1.1 per-timepoint XGBoost classifier",
        "cache":         args.cache,
        "seed":          args.seed,
        "val_n_chips":   args.val_n_chips,
        "samples_per_min": SAMPLES_PER_MIN,
        "n_samples":     N_SAMPLES,
        "rolling_window": ROLLING_WINDOW,
        "threshold":     args.threshold,
        "xgb":           clf_args,
        "best_iteration": best_iter,
        "best_val_logloss": best_score,
        "train_time_s":  float(train_time_s),
        "train_time_min": float(train_time_s / 60.0),
        "started_at":    datetime.now().isoformat(),
        "smoke":         bool(args.smoke),
    }
    (run_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    test_payload = {
        "pixel": pixel_metrics,
        "well":  well_metrics,
        "per_well": [
            {"chip_id": k.split("::")[0],
             "well_id": int(k.split("::")[1]),
             "y_true_min": float(well_true[k]),
             "y_pred_min": float(well_pred_min.get(k, float("nan")))}
            for k in well_keys
        ],
        "best_iteration":    best_iter,
        "best_val_logloss":  best_score,
        "train_time_s":      float(train_time_s),
        "threshold":         args.threshold,
        "n_test_pixels_nan": n_nan,
    }
    (run_dir / "test_metrics.json").write_text(
        json.dumps(test_payload, indent=2), encoding="utf-8")

    np.savez_compressed(
        run_dir / "predictions.npz",
        y_true_min=arr.y_te.astype(np.float32),
        y_pred_min=ttp_pred_min.astype(np.float32),
        chip_id=arr.chip_te, well_id=arr.well_te, pixel_id=arr.pixel_te,
    )

    # Feature importance
    importance = clf.get_booster().get_score(importance_type="gain")
    with (run_dir / "feature_importance.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature_idx", "feature_name", "gain"])
        for i, name in enumerate(FEATURE_NAMES):
            w.writerow([i, name, importance.get(f"f{i}", 0.0)])

    clf.save_model(str(run_dir / "xgb_model.json"))

    print(f"\nWrote {run_dir}/")
    print(f"  Test pixel MAE = {pixel_metrics['mae_min']:.3f} min  "
          f"(baseline constant-mean = "
          f"{pixel_metrics['mae_baseline_constant_mean']:.3f} min)")
    print(f"  Test well  MAE = {well_metrics['mae_min']:.3f} min  "
          f"(n_wells = {well_metrics['n']})")
    print(f"  R^2 (pixel) = {pixel_metrics['r2']:.3f}    "
          f"R^2 (well) = {well_metrics['r2']:.3f}")
    print(f"  training time = {train_time_s/60:.1f} min")


if __name__ == "__main__":
    main()
