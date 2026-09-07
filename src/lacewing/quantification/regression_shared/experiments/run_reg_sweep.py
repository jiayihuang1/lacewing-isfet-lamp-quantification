"""Per-cell dispatcher for the regression architecture sweep.

A "cell" is (model, loss, seed); the PBS array script maps an array
index onto one cell and runs this script with the right arguments.

Cells:
    12 models x 2 losses x 10 seeds = 240 cells.

Each cell's outputs land under
    Analysis/regression/results/reg_<model>_<loss>/runs/<run_id>/

The PBS script (jobs/reg_sweep.pbs) computes the (model, loss, seed)
triple from PBS_ARRAY_INDEX and invokes this module.

Run a single cell locally::

    python -m lacewing.quantification.regression_shared.experiments.run_reg_sweep \\
        --model cnn_transformer_seq --loss huber --seed 0

Loop everything serially (slow!)::

    python -m lacewing.quantification.regression_shared.experiments.run_reg_sweep --all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass


MODELS: list[str] = [
    "ann", "cnn1d", "fcn", "resnet", "inception",
    "transformer", "transformer_patch",
    "cnn_transformer_par", "cnn_transformer_seq",
    "gru", "cnn_gru_par", "cnn_gru_seq",
]
LOSSES: list[str] = ["huber", "mse"]
SEEDS: list[int] = list(range(10))

# Locked preprocessing.  Cache built by build_regression_cache.py from
# the MAD k=1.5 ABCD ntcRaw classification cache.
CACHE_STEM = "regress_all_filt_abcd_ntcRaw_madk1p5"


@dataclass(frozen=True)
class Cell:
    model: str
    loss:  str
    seed:  int

    @property
    def experiment(self) -> str:
        # One results dir per (model, loss); seeds land as separate
        # runs inside it.  Same convention as classification sweeps.
        return f"reg_{self.model}_{self.loss}"


def all_cells() -> list[Cell]:
    out: list[Cell] = []
    for m in MODELS:
        for l in LOSSES:
            for s in SEEDS:
                out.append(Cell(m, l, s))
    return out


def cell_for_index(idx: int) -> Cell:
    """Map PBS_ARRAY_INDEX -> Cell.

    Order: model fast-cycle, then loss, then seed (so consecutive cells
    are different seeds of the same (model, loss) — easy to monitor a
    single architecture's spread).
    """
    cells = all_cells()
    if not (0 <= idx < len(cells)):
        raise IndexError(f"index {idx} out of range (have {len(cells)} cells)")
    return cells[idx]


def run_one(cell: Cell, *, device: str | None,
            cache: str = CACHE_STEM,
            experiment_suffix: str = "",
            extra_train_args: list[str] | None = None) -> int:
    experiment = cell.experiment + experiment_suffix
    cmd = [
        sys.executable, "-m", "lacewing.quantification.regression_shared.core.train",
        "--model", cell.model,
        "--loss",  cell.loss,
        "--seed",  str(cell.seed),
        "--cache", cache,
        "--experiment", experiment,
    ]
    if device:
        cmd += ["--device", device]
    if extra_train_args:
        cmd += list(extra_train_args)
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    return subprocess.call(cmd)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, default=None)
    p.add_argument("--loss",  choices=LOSSES, default=None)
    p.add_argument("--seed",  type=int,        default=None)
    p.add_argument("--all",   action="store_true",
                   help="Iterate every cell sequentially.")
    p.add_argument("--device", default=None)
    p.add_argument("--cache",  default=CACHE_STEM)
    p.add_argument("--experiment-suffix", default="",
                   help="Appended to the experiment dir name.  Use e.g. "
                        "'_spatA3' to keep spatA3 results separate from "
                        "the original non-spatial sweep.")
    args, extra = p.parse_known_args()

    if args.all:
        cells = all_cells()
    elif args.model and args.loss and args.seed is not None:
        cells = [Cell(args.model, args.loss, args.seed)]
    else:
        p.error("Provide --all or all of (--model, --loss, --seed).")
        return

    t0 = time.time()
    failures: list[tuple[str, int]] = []
    for c in cells:
        rc = run_one(c, device=args.device, cache=args.cache,
                      experiment_suffix=args.experiment_suffix,
                      extra_train_args=extra)
        if rc != 0:
            failures.append((c.experiment + args.experiment_suffix, c.seed))
    elapsed = time.time() - t0
    print(f"\nFinished {len(cells)} cells in {elapsed/60:.1f} min")
    if failures:
        print(f"Failures ({len(failures)}):")
        for exp, seed in failures:
            print(f"  {exp}  seed={seed}")


if __name__ == "__main__":
    main()
