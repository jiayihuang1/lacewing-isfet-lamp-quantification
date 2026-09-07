"""Sweep A — per-pixel spatial smoothing on top of MAD k=1.5 ABCD ntcRaw.

Hypothesis: LAMP amplification is spatially correlated, so smoothing
each pixel's trace with its Chebyshev-distance-k 8-neighbour active
neighbours should suppress sensor noise and improve classification
accuracy.  Pixel count is unchanged (per-pixel granularity preserved).

Pipeline:

    raw -> linearise -> idx_active -> MAD ABCD -> SPATIAL SMOOTH (here)
        -> classifier

Grid (4 orders x 5 architectures x 10 seeds = 200 cells):

  * orders: {0, 1, 2, 3}
        - order 0 = identity passthrough (= the existing MAD baseline)
        - order 1 = 3x3 box minus centre, up to 8 active neighbours
        - order 2 = 5x5 box minus centre, up to 24 active neighbours
        - order 3 = 7x7 box minus centre, up to 48 active neighbours

  * architectures: ann, cnn1d, fcn, resnet, inception (the 5 incumbents
    used in every exp6/7/8/9 result table — keeps the new bars slot-
    compatible with the existing comparison chart).

  * seeds: 0..9.

Each cell's cache: ``dataset_all_filt_abcd_ntcRaw_madk1p5_spatA<order>``.
Built by ``lacewing.classification.data.build_spatial_filtered_dataset``
once per order before the sweep starts (PBS prereq).

Each cell uses the existing sd_test split (held-out chips = the
COV_SD + PnG_Bead chip), matching the exp6/7/8/9 protocol.

Usage::

    # Local single-cell run:
    python -m lacewing.classification.experiments.run_spatA \\
        --order 1 --model cnn_transformer_seq --seed 0

    # PBS array dispatch (200 cells): see jobs/spatA_sweep.pbs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass

from ..core import paths


# --------------------------------------------------------------------
# Order -> cache stem
# --------------------------------------------------------------------

LOCKED_INPUT_STEM = "dataset_all_filt_abcd_ntcRaw_madk1p5"


def cache_stem_for_order(order: int) -> str:
    return f"{LOCKED_INPUT_STEM}_spatA{order}"


ORDERS = (0, 1, 2, 3)

# Phase 1 (already run): the 5 incumbents — ann, cnn1d, fcn, resnet, inception.
# Phase 2 (new): the 7 transformer / GRU family architectures (mirrors exp9).
# Both phases combined here so the dispatcher accepts any valid architecture.
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


# --------------------------------------------------------------------
# Cell
# --------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    order: int
    model: str
    seed:  int

    @property
    def experiment(self) -> str:
        return f"spatA{self.order}_{self.model}"

    @property
    def cache_stem(self) -> str:
        return cache_stem_for_order(self.order)


def all_cells() -> list[Cell]:
    return [Cell(o, m, s) for o in ORDERS for m in MODELS for s in SEEDS]


def cell_for_index(idx: int) -> Cell:
    """Map PBS_ARRAY_INDEX -> Cell.  Order: model fast, then seed, then order."""
    cells = all_cells()
    if not (0 <= idx < len(cells)):
        raise IndexError(f"index {idx} out of range (have {len(cells)} cells)")
    return cells[idx]


# --------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------

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
    p.add_argument("--all",   action="store_true",
                   help="Iterate every cell sequentially (slow!).")
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

    # Sanity: refuse if a required cache is missing.
    missing = sorted({c.cache_stem for c in cells
                      if not paths.cache_path(c.cache_stem).exists()})
    if missing:
        print("Missing caches (build them first via "
              "build_spatial_filtered_dataset):", file=sys.stderr)
        for s in missing:
            print(f"  {s}.npz", file=sys.stderr)
        sys.exit(2)

    print(f"spatA: {len(cells)} cells to run.")
    t0 = time.time()
    failures: list[tuple[str, int]] = []
    for c in cells:
        rc = run_one(c,
                     epochs=args.epochs, device=args.device,
                     test_chips=args.test_chips)
        if rc != 0:
            failures.append((c.experiment, c.seed))
    elapsed = time.time() - t0
    print(f"\nspatA finished in {elapsed/60:.1f} min")
    if failures:
        print(f"Failures ({len(failures)}):")
        for exp, seed in failures:
            print(f"  exp={exp}  seed={seed}")


if __name__ == "__main__":
    main()
