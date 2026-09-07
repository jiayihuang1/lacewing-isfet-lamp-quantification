"""RQ2 — whole-curve XGBoost regressor (Hennigs 2025 stage 2).

Whole 450-sample pixel trace as fixed-length feature vector -> scalar
predicted TTP.  Mirror of the neural regressors in
``Analysis/regression/core/train.py`` in terms of splits, metrics, and
JSON schema, so this cell is directly comparable to the 12 neural
cells of the reg_sweep on the spatA3 cache.

Design notes
------------
- Input row = one pixel trace, 450 features.  Same shape and content as
  the neural regressors receive; XGBoost just treats the 450 as tabular
  features rather than a sequence.
- Split: reuses lacewing.quantification.regression_shared.data.dataset.make_split with the
  spatA3 cache stem (val_n_chips=1 chip held out from the 5 dose-response
  chips; test = SD chip).
- Loss:  XGBoost's built-in squared-error objective (analog of MSE) or
  pseudo-Huber (analog of HuberLoss).  Chosen by --loss.
- No label normalisation: targets stay in raw minutes so eval MAE is in
  minutes directly.

Usage
-----
    python -m lacewing.quantification.regression_shared.methods.xgboost_wholecurve \\
        --seed 0 --loss mse

Outputs
-------
Analysis/regression/results/reg_xgb_wholecurve_<loss>/runs/<run_id>/:
    config.json
    test_metrics.json      -- pixel + well + per_well payload, matching
                              the neural test_metrics.json
    predictions.npz        -- y_true_min, y_pred_min, ids
    feature_importance.csv -- gain per timestep
    xgb_model.json
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime

import numpy as np

from lacewing.classification.core.seeding import seed_everything
from lacewing.quantification.regression_shared.core import paths as reg_paths
from lacewing.quantification.regression_shared.data import dataset as reg_ds


N_SAMPLES = 450


DEFAULTS = {
    "cache":         "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3",
    "experiment_fmt": "reg_xgb_wholecurve_{loss}",
    "loss":          "mse",       # "mse" or "huber" (pseudo-Huber)
    "val_n_chips":   1,
    "n_estimators":  400,
    "max_depth":     6,
    "learning_rate": 0.05,
    "n_threads":     0,
    "early_stopping_rounds": 30,
    "huber_slope":   1.0,          # pseudo-Huber delta parameter (in minutes)
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default=DEFAULTS["cache"])
    p.add_argument("--experiment", default=None,
                   help="Defaults to reg_xgb_wholecurve_<loss>.")
    p.add_argument("--loss", choices=["mse", "huber"], default=DEFAULTS["loss"])
    p.add_argument("--huber-slope", type=float,
                   default=DEFAULTS["huber_slope"],
                   help="Delta for pseudo-Huber (in minutes). Default 1 min.")
    p.add_argument("--val-n-chips", type=int, default=DEFAULTS["val_n_chips"])
    p.add_argument("--n-estimators", type=int, default=DEFAULTS["n_estimators"])
    p.add_argument("--max-depth", type=int, default=DEFAULTS["max_depth"])
    p.add_argument("--learning-rate", type=float,
                   default=DEFAULTS["learning_rate"])
    p.add_argument("--n-threads", type=int, default=DEFAULTS["n_threads"])
    p.add_argument("--early-stopping-rounds", type=int,
                   default=DEFAULTS["early_stopping_rounds"])
    p.add_argument("--tag", default=None)
    return p.parse_args()


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae_baseline = float(np.mean(np.abs(y_true - y_true.mean())))
    return {
        "mae_min":  mae,
        "rmse_min": rmse,
        "r2":       r2,
        "mae_baseline_constant_mean": mae_baseline,
        "n":        int(len(y_true)),
    }


def _per_well_predictions(pred_min: np.ndarray, chip: np.ndarray,
                           well: np.ndarray) -> dict[str, float]:
    keys = np.array([f"{c}::{w}" for c, w in zip(chip, well)])
    out: dict[str, list[float]] = {}
    for k, p in zip(keys, pred_min):
        out.setdefault(k, []).append(float(p))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main() -> None:
    args = _parse_args()
    import xgboost as xgb

    seed_everything(args.seed)

    experiment = args.experiment or DEFAULTS["experiment_fmt"].format(
        loss=args.loss)

    # ----- load + split -----
    print(f"Loading cache: {args.cache}")
    arr = reg_ds.make_split(args.cache, seed=args.seed,
                             val_n_chips=args.val_n_chips,
                             normalise_y=False)
    print(f"  n_tr={len(arr.X_tr)}  n_va={len(arr.X_va)}  n_te={len(arr.X_te)}")
    print(f"  val chips:  {sorted(set(arr.chip_va.tolist()))}")
    print(f"  test chips: {sorted(set(arr.chip_te.tolist()))}")

    # ----- train -----
    if args.loss == "mse":
        objective = "reg:squarederror"
    else:
        # XGBoost's pseudo-Huber objective; delta controls the linear-region
        # transition (in minutes since we don't normalise y).
        objective = "reg:pseudohubererror"

    print(f"\nTraining XGBoost regressor (loss={args.loss})...")
    reg_args = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        tree_method="hist",
        objective=objective,
        eval_metric="mae",
        n_jobs=args.n_threads if args.n_threads else None,
        random_state=args.seed,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    if args.loss == "huber":
        reg_args["huber_slope"] = args.huber_slope
    print(f"  XGBoost args: {reg_args}")
    reg = xgb.XGBRegressor(**reg_args)
    t0 = time.time()
    reg.fit(arr.X_tr, arr.y_tr,
            eval_set=[(arr.X_va, arr.y_va)],
            verbose=False)
    train_time = time.time() - t0
    best_iter = int(getattr(reg, "best_iteration", args.n_estimators - 1))
    best_score = float(getattr(reg, "best_score", float("nan")))
    print(f"  trained in {train_time/60:.1f} min  "
          f"(best_iteration={best_iter}, best_val_MAE={best_score:.4f})")

    # ----- predict + metrics -----
    preds_min = reg.predict(arr.X_te).astype(np.float32)
    pixel_metrics = _regression_metrics(arr.y_te, preds_min)
    well_pred_min = _per_well_predictions(preds_min, arr.chip_te, arr.well_te)
    keys = [f"{c}::{w}" for c, w in zip(arr.chip_te, arr.well_te)]
    well_true = {}
    for k, yv in zip(keys, arr.y_te):
        well_true[k] = float(yv)
    well_keys = sorted(set(keys))
    yw_true = np.array([well_true[k] for k in well_keys])
    yw_pred = np.array([well_pred_min[k] for k in well_keys])
    well_metrics = _regression_metrics(yw_true, yw_pred)

    print(f"\nTest pixel MAE = {pixel_metrics['mae_min']:.3f} min  "
          f"(constant-mean baseline = "
          f"{pixel_metrics['mae_baseline_constant_mean']:.3f})")
    print(f"Test well  MAE = {well_metrics['mae_min']:.3f} min")
    print(f"R^2 pixel = {pixel_metrics['r2']:.3f}    "
          f"R^2 well  = {well_metrics['r2']:.3f}")

    # ----- run dir -----
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_id = f"{ts}_xgb_wholecurve_{args.loss}_seed{args.seed}{tag}"
    run_dir = reg_paths.experiment_dir(experiment) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "experiment":       experiment,
        "method":           "RQ2 whole-curve XGBoost scalar regressor",
        "cache":            args.cache,
        "loss":             args.loss,
        "huber_slope":      args.huber_slope if args.loss == "huber" else None,
        "seed":             args.seed,
        "val_n_chips":      args.val_n_chips,
        "xgb":              reg_args,
        "best_iteration":   best_iter,
        "best_val_mae":     best_score,
        "train_time_s":     float(train_time),
        "started_at":       datetime.now().isoformat(),
        "n_train":          int(len(arr.X_tr)),
        "n_val":            int(len(arr.X_va)),
        "n_test":           int(len(arr.X_te)),
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
             "y_pred_min": float(well_pred_min[k])}
            for k in well_keys
        ],
        "best_iteration":   best_iter,
        "best_val_mae":     best_score,
        "train_time_s":     float(train_time),
    }
    (run_dir / "test_metrics.json").write_text(
        json.dumps(test_payload, indent=2), encoding="utf-8")

    np.savez_compressed(
        run_dir / "predictions.npz",
        y_true_min=arr.y_te.astype(np.float32),
        y_pred_min=preds_min,
        chip_id=arr.chip_te, well_id=arr.well_te, pixel_id=arr.pixel_te,
    )

    importance = reg.get_booster().get_score(importance_type="gain")
    with (run_dir / "feature_importance.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature_idx", "timestep", "gain"])
        for i in range(N_SAMPLES):
            w.writerow([i, i, importance.get(f"f{i}", 0.0)])

    reg.save_model(str(run_dir / "xgb_model.json"))

    print(f"\nWrote {run_dir}/")


if __name__ == "__main__":
    main()
