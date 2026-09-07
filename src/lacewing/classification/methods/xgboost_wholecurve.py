"""RQ1 — whole-curve XGBoost classifier (Hennigs 2025 stage 1).

Mirror of the neural-classifier training loop in
``Analysis/classification/core/train.py``, but the model is an XGBoost
binary classifier that takes the entire 450-sample pixel trace as a
fixed-length feature vector (one row = one pixel).

This lets us cleanly slot XGBoost into the same spatA3 comparison as the
12 neural architectures already trained by the Week 11 sweep, with
matching test-set splits and matching test_metrics.json schema.

Design notes
------------
- Input row = one pixel trace, 450 features.  No hand-engineered
  features, no per-timestep reframing.  This is what Hennigs et al.
  2025 did (they used all 40 fluorescence values per qPCR curve; we
  use all 450 samples per ISFET trace).
- Split: same `sd_test` split as the neural sweep — train + val from
  the 5 Final chips (chip-stratified 85/15 per chip), test = SD + PnG_Bead.
- Metrics: reuses lacewing.classification.core.evaluate for both
  pixel_metrics and well_metrics so the JSON schema matches the neural
  runs.

Usage
-----
    python -m lacewing.classification.methods.xgboost_wholecurve \\
        --seed 0 \\
        --cache dataset_all_filt_abcd_ntcRaw_madk1p5_spatA3

Outputs
-------
Analysis/classification/results/spatA3_xgb_wholecurve/runs/<run_id>/:
    config.json
    test_metrics.json      -- same schema as neural runs
    predictions.npz        -- y_true, y_score, y_pred, chip_id, well_id, pixel_id
    feature_importance.csv -- XGBoost gain per timestep-feature (450 rows)
    xgb_model.json
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from lacewing.classification.core import evaluate as eval_mod
from lacewing.classification.core import paths
from lacewing.classification.core.seeding import seed_everything
from lacewing.classification.data import dataset as clf_ds


N_SAMPLES = 450

RESULTS_ROOT = paths.CLASSIFICATION_RESULTS


DEFAULTS = {
    "cache":         "dataset_all_filt_abcd_ntcRaw_madk1p5_spatA3",
    "experiment":    "spatA3_xgb_wholecurve",
    "split":         "sd_test",
    "fold":          0,       # sd_test default = SD + PnG_Bead
    "n_estimators":  400,
    "max_depth":     6,
    "learning_rate": 0.05,
    "n_threads":     0,       # 0 = XGBoost auto-pick
    "early_stopping_rounds": 30,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default=DEFAULTS["cache"])
    p.add_argument("--experiment", default=DEFAULTS["experiment"])
    p.add_argument("--split", default=DEFAULTS["split"],
                   choices=["sd_test", "random", "chip"])
    p.add_argument("--fold", default=str(DEFAULTS["fold"]),
                   help="For sd_test: comma-separated test chip folder "
                        "names.  Default = SD + PnG_Bead.")
    p.add_argument("--n-estimators", type=int, default=DEFAULTS["n_estimators"])
    p.add_argument("--max-depth", type=int, default=DEFAULTS["max_depth"])
    p.add_argument("--learning-rate", type=float,
                   default=DEFAULTS["learning_rate"])
    p.add_argument("--n-threads", type=int, default=DEFAULTS["n_threads"])
    p.add_argument("--early-stopping-rounds", type=int,
                   default=DEFAULTS["early_stopping_rounds"])
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    import xgboost as xgb

    seed_everything(args.seed)

    # ----- load + split (matches neural train.py) -----
    print(f"Loading cache: {args.cache}")
    cache_path = paths.cache_path(args.cache)
    arr = clf_ds.make_split(features="raw", split=args.split,
                             fold=args.fold, seed=args.seed,
                             cache_path=cache_path)
    print(f"  n_tr={len(arr.X_tr)}  n_va={len(arr.X_va)}  n_te={len(arr.X_te)}")
    print(f"  tr class balance: {int(arr.y_tr.sum())}/{len(arr.y_tr)} positive")
    print(f"  te class balance: {int(arr.y_te.sum())}/{len(arr.y_te)} positive")

    # ----- flatten (N, 1, T) -> (N, T) for tabular XGBoost -----
    X_tr = arr.X_tr.reshape(len(arr.X_tr), -1).astype(np.float32)
    X_va = arr.X_va.reshape(len(arr.X_va), -1).astype(np.float32)
    X_te = arr.X_te.reshape(len(arr.X_te), -1).astype(np.float32)
    assert X_tr.shape[1] == N_SAMPLES, (
        f"Expected {N_SAMPLES} timesteps, got {X_tr.shape[1]}")

    # ----- train -----
    print("\nTraining XGBoost binary classifier...")
    clf_args = dict(
        n_estimators=args.n_estimators,
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
    t0 = time.time()
    clf.fit(X_tr, arr.y_tr,
            eval_set=[(X_va, arr.y_va)],
            verbose=False)
    train_time = time.time() - t0
    best_iter = int(getattr(clf, "best_iteration", args.n_estimators - 1))
    best_score = float(getattr(clf, "best_score", float("nan")))
    print(f"  trained in {train_time/60:.1f} min  "
          f"(best_iteration={best_iter}, best_val_logloss={best_score:.4f})")

    # ----- predict + metrics -----
    y_score = clf.predict_proba(X_te)[:, 1].astype(np.float32)
    y_pred = (y_score >= 0.5).astype(np.uint8)
    pix = eval_mod.pixel_metrics(arr.y_te, y_score, threshold=0.5)
    well = eval_mod.well_metrics(arr.y_te, y_score, arr.chip_te,
                                  arr.well_te, threshold=0.5)
    print(f"\nTest pixel: {asdict(pix)}")
    print(f"Test well : {well}")

    # ----- run dir -----
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    # Match the neural-run naming convention: include the fold as a suffix.
    fold_str = str(args.fold)[:80]  # trim long comma-lists
    run_id = f"{ts}_xgb_wholecurve_{args.split}_fold{fold_str}_seed{args.seed}{tag}"
    run_dir = RESULTS_ROOT / args.experiment / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ----- outputs -----
    cfg = {
        "experiment":     args.experiment,
        "method":         "RQ1 whole-curve XGBoost binary classifier",
        "cache":          args.cache,
        "split":          args.split,
        "fold":           args.fold,
        "seed":           args.seed,
        "xgb":            clf_args,
        "best_iteration": best_iter,
        "best_val_logloss": best_score,
        "train_time_s":   float(train_time),
        "started_at":     datetime.now().isoformat(),
        "n_train":        int(len(arr.X_tr)),
        "n_val":          int(len(arr.X_va)),
        "n_test":         int(len(arr.X_te)),
    }
    (run_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    test_metrics = dict(
        pixel=asdict(pix),
        well=well,
        train_time_sec=float(train_time),
        n_parameters=int(clf.get_booster().num_boosted_rounds()),
        decision_threshold=0.5,
    )
    (run_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8")

    np.savez_compressed(
        run_dir / "predictions.npz",
        y_true=arr.y_te.astype(np.uint8),
        y_score=y_score,
        y_pred=y_pred,
        chip_id=arr.chip_te, well_id=arr.well_te, pixel_id=arr.pixel_te,
    )

    # Feature importance across the 450 timesteps.
    importance = clf.get_booster().get_score(importance_type="gain")
    with (run_dir / "feature_importance.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature_idx", "timestep", "gain"])
        for i in range(N_SAMPLES):
            w.writerow([i, i, importance.get(f"f{i}", 0.0)])

    clf.save_model(str(run_dir / "xgb_model.json"))

    print(f"\nWrote {run_dir}/")


if __name__ == "__main__":
    main()
