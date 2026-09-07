"""RQ1 · generic classifier sweep runner. [Cat A] Report §RQ1.

Run a grid of (model, split, fold, seed) configs in sequence.

Each cell shells out to `train.py` so a single failure does not abort
the rest of the sweep. Results land in results/runs/ + the aggregate
results/per_method_metrics.csv.

Edit MODELS / SEEDS / CHIP_FOLDS below to control the grid.

Usage:
    python -m lacewing.classification.run_sweep              # full default sweep
    python -m lacewing.classification.run_sweep --quick      # 1 seed, 1 model
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from ..core import paths


MODELS = [
    "ann", "cnn1d", "fcn", "resnet", "inception",
    "autoencoder", "cnn2d_spec",
]
# Default model list for the new --split chipkfold experiment. Drops
# cnn2d_spec per supervisor Week-4 decision (results not strong; we keep
# source + prior experiments intact, just stop running it in new sweeps).
MODELS_CHIPKFOLD_DEFAULT = [
    "ann", "cnn1d", "fcn", "resnet", "inception", "autoencoder",
]
SEEDS = [0, 1, 2]                  # Paper 3 reports a single seed; we do 3 for variance
CHIP_FOLDS = ["1e5", "1e6", "1e7", "1e8", "1e9"]
# Default fold indices for --split chipkfold (must match KFOLD_K in data/dataset.py)
CHIPKFOLD_FOLDS = ["0", "1", "2", "3", "4"]


def _run(cmd: list[str]) -> int:
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    return subprocess.call(cmd)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true",
                   help="Just run cnn2d_spec with seed=0, random split.")
    p.add_argument("--skip-chip-fold", action="store_true",
                   help="Skip the chip-fold CV runs (slow).")
    p.add_argument("--models", nargs="+", default=None,
                   help="Override MODELS list.")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Override SEEDS list.")
    p.add_argument("--folds", nargs="+", default=None,
                   help="Override CHIP_FOLDS list (only used with chip split).")
    p.add_argument("--experiment", default=paths.DEFAULT_EXPERIMENT,
                   help="Experiment subfolder under results/.")
    p.add_argument("--cache", default="dataset_per_pixel",
                   help="Dataset cache name (under data/cache/<name>.npz).")
    p.add_argument("--split", default=None,
                   help="Force a single split type ('random', 'chip', or "
                        "'chipkfold'). If set, only that split is run "
                        "(otherwise: random + every chip-fold unless "
                        "--skip-chip-fold).")
    args = p.parse_args()

    py = sys.executable
    base = [py, "-m", "lacewing.classification.train",
            "--experiment", args.experiment, "--cache", args.cache]

    if args.quick:
        _run(base + ["--model", "cnn2d_spec", "--split", "random", "--seed", "0"])
        return

    # If the user picked --split chipkfold, swap to the chipkfold defaults
    # (smaller model list without cnn2d_spec, fold indices 0..k-1).
    if args.split == "chipkfold":
        models = args.models or MODELS_CHIPKFOLD_DEFAULT
        seeds = args.seeds or SEEDS
        folds = args.folds or CHIPKFOLD_FOLDS
    else:
        models = args.models or MODELS
        seeds = args.seeds or SEEDS
        folds = args.folds or CHIP_FOLDS

    do_random = (args.split is None or args.split == "random")
    do_chipfold = (args.split is None or args.split == "chip") and not args.skip_chip_fold
    do_chipkfold = (args.split == "chipkfold")

    if do_chipkfold:
        n_runs = len(models) * len(seeds) * len(folds)
        print(f"Sweep (chipkfold): experiment={args.experiment} cache={args.cache}")
        print(f"  {len(models)} models x {len(seeds)} seeds x "
              f"{len(folds)} folds = {n_runs} runs")
    else:
        n_runs = len(models) * len(seeds) * (
            (1 if do_random else 0) + (len(folds) if do_chipfold else 0))
        print(f"Sweep: experiment={args.experiment} cache={args.cache}")
        print(f"  {len(models)} models x {len(seeds)} seeds x "
              f"({1 if do_random else 0} random + "
              f"{len(folds) if do_chipfold else 0} chip folds) = {n_runs} runs")

    t0 = time.time()
    failures = []
    for model in models:
        for seed in seeds:
            if do_chipkfold:
                for fold in folds:
                    rc = _run(base + ["--model", model, "--split", "chipkfold",
                                      "--fold", fold, "--seed", str(seed)])
                    if rc != 0:
                        failures.append((model, "chipkfold", fold, seed))
                continue

            if do_random:
                rc = _run(base + ["--model", model, "--split", "random",
                                  "--seed", str(seed)])
                if rc != 0:
                    failures.append((model, "random", "0", seed))

            if not do_chipfold:
                continue
            for fold in folds:
                rc = _run(base + ["--model", model, "--split", "chip",
                                  "--fold", fold, "--seed", str(seed)])
                if rc != 0:
                    failures.append((model, "chip", fold, seed))

    elapsed = time.time() - t0
    print(f"\nSweep complete in {elapsed/60:.1f} min")
    if failures:
        print(f"Failures ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
    print(f"\nAggregate metrics: {paths.metrics_csv(args.experiment)}")
    print(f"Run dirs:          {paths.runs_dir(args.experiment)}")


if __name__ == "__main__":
    main()
