"""RQ1 · architecture × MAD-k sweep on CoV. [Cat A] Report §RQ1.

exp9: model-choice strand of RQ1.

Benchmark transformer-family architectures against the existing
ANN/CNN1D/FCN/ResNet/Inception lineup, under three preprocessing
regimes (matching the exp6/exp7/exp8 framing):

    "unfiltered"        no preprocessing filter           (= exp6 setting)
    "pct_abcd_ntcRaw"   percentile-rule ABCD ntcRaw       (= exp7 winner)
    "madk1.5_abcd_ntcRaw"  MAD k=1.5 ABCD ntcRaw          (= exp8 winner)

Models (4 new architectures, all torch / SGD compatible):
    transformer            vanilla TS-transformer (linear-embed each sample,
                           sinusoidal PE, 3 encoder layers, mean-pool)
    transformer_patch      patch-based (P=15) -> 30 tokens, otherwise same
    cnn_transformer_par    parallel two-stream: CNN1D body + vanilla
                           transformer, features concatenated then FC head
    cnn_transformer_seq    sequential: CNN1D stem (no flatten) -> 32-dim
                           token sequence (T'=111) -> transformer encoder

Each (preprocessing, model) cell is the experiment-id used to name
the results directory: ``exp9_<model>_<preproc>``.

Usage::

    # Local single-cell run (one model, one seed, one preprocessing):
    python -m lacewing.classification.experiments.run_exp9 \\
        --preproc madk1.5_abcd_ntcRaw --model transformer --seed 0

    # PBS array dispatch: index into the (preproc, model, seed) grid.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass

from ..core import paths


# --------------------------------------------------------------------
# Preprocessing regimes -> cache file stems
# --------------------------------------------------------------------
#
# Each entry: id (used in experiment name) -> cache stem (under
# Analysis/classification/data/cache/).  The unfiltered regime uses
# `dataset_all`, the same cache exp6 trained on.

PREPROCS: dict[str, str] = {
    "unfiltered":          "dataset_all",
    "pct_abcd_ntcRaw":     "dataset_all_filt_abcd_ntcRaw",
    "madk1.5_abcd_ntcRaw": "dataset_all_filt_abcd_ntcRaw_madk1p5",
}


# --------------------------------------------------------------------
# Experiment grid
# --------------------------------------------------------------------

@dataclass(frozen=True)
class ExpConfig:
    name:       str   # experiment id, e.g. "exp9_transformer_madk1.5_abcd_ntcRaw"
    model:      str   # model name in models.REGISTRY
    preproc:    str   # preprocessing regime key
    cache_stem: str   # cache file stem


MODELS = [
    "transformer",
    "transformer_patch",
    "cnn_transformer_par",
    "cnn_transformer_seq",
    "gru",
    "cnn_gru_par",
    "cnn_gru_seq",
]

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

TEST_CHIPS = ",".join([
    "D20240719_E03_C44_F4500KHz_U_COV_SD",
    "D20240814_E00_C00_F4500KHz_U_PnG_Bead",
])


def _exp_name(model: str, preproc: str) -> str:
    return f"exp9_{model}_{preproc}"


EXPERIMENTS: list[ExpConfig] = [
    ExpConfig(
        name=_exp_name(m, p),
        model=m,
        preproc=p,
        cache_stem=PREPROCS[p],
    )
    for p in PREPROCS
    for m in MODELS
]
EXP_BY_NAME = {e.name: e for e in EXPERIMENTS}


# --------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------

def _run(cmd: list[str]) -> int:
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    return subprocess.call(cmd)


def run_one(exp: ExpConfig, seed: int, *,
            epochs: int, device: str | None, test_chips: str) -> int:
    py = sys.executable
    cmd = [
        py, "-m", "lacewing.classification.core.train",
        "--experiment", exp.name,
        "--cache",      exp.cache_stem,
        "--split",      "sd_test",
        "--fold",       test_chips,
        "--epochs",     str(epochs),
        "--model",      exp.model,
        "--seed",       str(seed),
    ]
    if device:
        cmd += ["--device", device]
    return _run(cmd)


def run_sweep(experiments: list[ExpConfig], seeds: list[int], *,
              epochs: int, device: str | None, test_chips: str) -> None:
    n = len(experiments) * len(seeds)
    print(f"exp9: {len(experiments)} (preproc, model) cells x "
          f"{len(seeds)} seeds = {n} runs")
    t0 = time.time()
    failures: list[tuple[str, int]] = []
    for exp in experiments:
        for seed in seeds:
            rc = run_one(exp, seed,
                         epochs=epochs, device=device, test_chips=test_chips)
            if rc != 0:
                failures.append((exp.name, seed))
    elapsed = time.time() - t0
    print(f"\nexp9 sweep finished in {elapsed/60:.1f} min")
    if failures:
        print(f"Failures ({len(failures)}):")
        for f in failures:
            print(f"  exp={f[0]}  seed={f[1]}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment",
                   choices=[e.name for e in EXPERIMENTS],
                   default=None,
                   help="Pin to one (preproc, model) cell.")
    p.add_argument("--preproc",
                   choices=list(PREPROCS),
                   default=None,
                   help="Pin to one preprocessing regime.")
    p.add_argument("--model",
                   choices=MODELS,
                   default=None,
                   help="Pin to one model.")
    p.add_argument("--seed", type=int, default=None,
                   help="Pin to one seed (used by PBS array index).")
    p.add_argument("--test-chips", default=TEST_CHIPS)
    p.add_argument("--device", default=None,
                   help="cpu / cuda. Passed through to train.")
    p.add_argument("--epochs", type=int, default=40)
    args = p.parse_args()

    if args.experiment:
        experiments = [EXP_BY_NAME[args.experiment]]
    else:
        experiments = EXPERIMENTS
        if args.preproc:
            experiments = [e for e in experiments if e.preproc == args.preproc]
        if args.model:
            experiments = [e for e in experiments if e.model == args.model]

    seeds = [args.seed] if args.seed is not None else SEEDS

    # Sanity: refuse if a required cache is missing.
    missing = [e.cache_stem for e in experiments
               if not paths.cache_path(e.cache_stem).exists()]
    if missing:
        print("Missing caches (build them first):", file=sys.stderr)
        for s in sorted(set(missing)):
            print(f"  {s}.npz", file=sys.stderr)
        sys.exit(2)

    run_sweep(experiments, seeds,
              epochs=args.epochs, device=args.device,
              test_chips=args.test_chips)


if __name__ == "__main__":
    main()
