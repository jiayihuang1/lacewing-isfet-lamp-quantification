"""Sweep B — non-overlapping tile pooling on top of MAD k=1.5 ABCD ntcRaw.

Supervisor's interpretation: trade pixel count for noise reduction.
Each non-overlapping (2k+1)x(2k+1) tile becomes ONE super-pixel whose
trace is the mean of its active constituent pixels.  Pixel count
shrinks by up to (2k+1)^2.

Pipeline:

    raw -> linearise -> idx_active -> MAD ABCD -> TILE POOL (here)
        -> classifier

Grid (3 orders x 5 architectures x 10 seeds = 150 cells):

  * orders: {1, 2, 3}
        - order 0 = identity = the locked MAD baseline (already in
          the exp8/exp9 result tables, not re-run here).
        - order 1 = 3x3 tile
        - order 2 = 5x5 tile
        - order 3 = 7x7 tile

  * architectures: ann, cnn1d, fcn, resnet, inception (same lineup as
    Sweep A so the result bars line up).

  * seeds: 0..9.

Each cell's cache: ``dataset_all_filt_abcd_ntcRaw_madk1p5_spatB<order>``,
built by ``lacewing.classification.data.build_spatial_pooled_dataset``.

Usage::

    python -m lacewing.classification.experiments.run_spatB \\
        --order 1 --model cnn_transformer_seq --seed 0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass

from ..core import paths


LOCKED_INPUT_STEM = "dataset_all_filt_abcd_ntcRaw_madk1p5"


def cache_stem_for_order(order: int) -> str:
    return f"{LOCKED_INPUT_STEM}_spatB{order}"


ORDERS = (1, 2, 3)

# Phase 1 (already run): the 5 incumbents — ann, cnn1d, fcn, resnet, inception.
# Phase 2 (new): the 7 transformer / GRU family architectures (mirrors exp9).
MODELS = (
    "ann", "cnn1d", "fcn", "resnet", "inception",
    "transformer", "transformer_patch",
    "cnn_transformer_par", "cnn_transformer_seq",
    "gru", "cnn_gru_par", "cnn_gru_seq",
)

SEEDS = tuple(range(10))

TEST_CHIPS = ",".join([
    "D20240719_E03_C44_F4500KHz_U_COV_SD",
    "D20240814_E00_C00_F4500KHz_U_PnG_Bead",
])


@dataclass(frozen=True)
class Cell:
    order: int
    model: str
    seed:  int

    @property
    def experiment(self) -> str:
        return f"spatB{self.order}_{self.model}"

    @property
    def cache_stem(self) -> str:
        return cache_stem_for_order(self.order)


def all_cells() -> list[Cell]:
    return [Cell(o, m, s) for o in ORDERS for m in MODELS for s in SEEDS]


def cell_for_index(idx: int) -> Cell:
    cells = all_cells()
    if not (0 <= idx < len(cells)):
        raise IndexError(f"index {idx} out of range (have {len(cells)} cells)")
    return cells[idx]


def _run(cmd: list[str]) -> int:
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    return subprocess.call(cmd)


def run_one(cell: Cell, *,
            epochs: int, device: str | None, test_chips: str) -> int:
    py = sys.executable
    cmd = [
        py, "-m", "lacewing.classification.core.train",
        "--experiment", cell.experiment,
        "--cache",      cell.cache_stem,
        "--split",      "sd_test",
        "--fold",       test_chips,
        "--epochs",     str(epochs),
        "--model",      cell.model,
        "--seed",       str(cell.seed),
    ]
    if device:
        cmd += ["--device", device]
    return _run(cmd)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--order", type=int, choices=list(ORDERS), default=None)
    p.add_argument("--model", choices=list(MODELS), default=None)
    p.add_argument("--seed",  type=int, default=None)
    p.add_argument("--all",   action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--test-chips", default=TEST_CHIPS)
    args = p.parse_args()

    if args.all:
        cells = all_cells()
    elif (args.order is not None and args.model is not None
          and args.seed is not None):
        cells = [Cell(args.order, args.model, args.seed)]
    else:
        p.error("Provide --all or all of (--order, --model, --seed).")
        return

    missing = sorted({c.cache_stem for c in cells
                      if not paths.cache_path(c.cache_stem).exists()})
    if missing:
        print("Missing caches (build them first via "
              "build_spatial_pooled_dataset):", file=sys.stderr)
        for s in missing:
            print(f"  {s}.npz", file=sys.stderr)
        sys.exit(2)

    print(f"spatB: {len(cells)} cells to run.")
    t0 = time.time()
    failures: list[tuple[str, int]] = []
    for c in cells:
        rc = run_one(c,
                     epochs=args.epochs, device=args.device,
                     test_chips=args.test_chips)
        if rc != 0:
            failures.append((c.experiment, c.seed))
    elapsed = time.time() - t0
    print(f"\nspatB finished in {elapsed/60:.1f} min")
    if failures:
        print(f"Failures ({len(failures)}):")
        for exp, seed in failures:
            print(f"  exp={exp}  seed={seed}")


if __name__ == "__main__":
    main()
