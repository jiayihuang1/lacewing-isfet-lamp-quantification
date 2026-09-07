"""RQ1 · CoV classifier catalogue on the 5-training / 2-SD-test protocol. [Cat A] Report §RQ1 Tab. 5.X.

exp6: train on the 5 Final chips, test on 2 supervisor-selected SD chips.

This is the Week-5 supervisor request — a fixed external test set rather
than a CV split. Each Final chip contributes ~85% pixels to train and
~15% to val (chip-stratified), and the held-out SD chips are evaluated
as a single external test set.

Defaults:
    test chips = D20240719_E03_C44_F4500KHz_U_COV_SD,
                 D20240814_E00_C00_F4500KHz_U_PnG_Bead
    models     = all 7 (ann, cnn1d, fcn, resnet, inception,
                       autoencoder, cnn2d_spec)
    seeds      = 0, 1, 2
    cache      = dataset_all  (built by `python -m lacewing.classification
                              .data.build_dataset --scope all`)

Run locally:
    python -m lacewing.classification.run_exp6
For HPC, see jobs/exp6_sweep.pbs (array job, one element per model/seed).

Single-job invocation (used by the PBS array script):
    python -m lacewing.classification.run_exp6 --model cnn2d_spec --seed 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from ..core import paths


EXPERIMENT = "exp6_final_to_sd"
CACHE = "dataset_all"

MODELS = [
    "ann", "cnn1d", "fcn", "resnet", "inception",
    "autoencoder", "cnn2d_spec",
]
SEEDS = [0, 1, 2]
TEST_CHIPS = ",".join([
    "D20240719_E03_C44_F4500KHz_U_COV_SD",
    "D20240814_E00_C00_F4500KHz_U_PnG_Bead",
])


def _run(cmd: list[str]) -> int:
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    return subprocess.call(cmd)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None,
                   help="If given, run only this model (used by the PBS "
                        "array job to dispatch one cell at a time).")
    p.add_argument("--seed", type=int, default=None,
                   help="If given, run only this seed.")
    p.add_argument("--test-chips", default=TEST_CHIPS,
                   help="Comma-separated test-chip folder names.")
    p.add_argument("--device", default=None,
                   help="cpu / cuda. Passed through to train.py; defaults "
                        "to auto-detect inside train.py.")
    p.add_argument("--epochs", type=int, default=40)
    args = p.parse_args()

    models = [args.model] if args.model else MODELS
    seeds = [args.seed] if args.seed is not None else SEEDS

    py = sys.executable
    base = [py, "-m", "lacewing.classification.core.train",
            "--experiment", EXPERIMENT,
            "--cache", CACHE,
            "--split", "sd_test",
            "--fold", args.test_chips,
            "--epochs", str(args.epochs)]
    if args.device:
        base += ["--device", args.device]

    n_runs = len(models) * len(seeds)
    print(f"exp6: {len(models)} models x {len(seeds)} seeds = {n_runs} runs")
    print(f"  test chips: {args.test_chips}")
    print(f"  experiment: {EXPERIMENT}  cache: {CACHE}")

    t0 = time.time()
    failures: list[tuple[str, int]] = []
    for model in models:
        for seed in seeds:
            rc = _run(base + ["--model", model, "--seed", str(seed)])
            if rc != 0:
                failures.append((model, seed))

    elapsed = time.time() - t0
    print(f"\nexp6 sweep finished in {elapsed/60:.1f} min")
    if failures:
        print(f"Failures ({len(failures)}):")
        for f in failures:
            print(f"  model={f[0]} seed={f[1]}")
    print(f"\nAggregate metrics: {paths.metrics_csv(EXPERIMENT)}")
    print(f"Run dirs:          {paths.runs_dir(EXPERIMENT)}")


if __name__ == "__main__":
    main()
