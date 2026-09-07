"""exp7: layer-by-layer preprocessing-filter ablation on the
exp6-style external-chip test.

Same setup as exp6 (train on 5 Final CoV chips, test on 2 held-out
SD chips), but with the chip-relative four-layer preprocessing
filter applied to *both* training and test caches.  Each experiment
turns on a different subset of layers / NTC treatment so we can
read off the marginal contribution of each layer.

Experiments (8 total + the unfiltered exp6 baseline)
-----------------------------------------------------
    exp7_A_ntcRaw     positive wells: Layer A only.    NTC: untouched.
    exp7_A_ntcD       positive wells: Layer A only.    NTC: D-cleaned.
    exp7_AB_ntcRaw    positive wells: Layers A+B.      NTC: untouched.
    exp7_AB_ntcD      positive wells: Layers A+B.      NTC: D-cleaned.
    exp7_ABC_ntcRaw   positive wells: Layers A+B+C.    NTC: untouched.
    exp7_ABC_ntcD     positive wells: Layers A+B+C.    NTC: D-cleaned.
    exp7_ABCD_ntcRaw  positive wells: Layers A+B+C+D.  NTC: untouched.
    exp7_ABCD_ntcD    positive wells: Layers A+B+C+D.  NTC: D-cleaned.

"NTC: D-cleaned" means Layer D is applied to the NTC well first to
drop genuinely broken NTC pixels.  Those dropped pixels are excluded
from both A/B threshold computation (so the chip-relative thresholds
are slightly tighter) and from the training cache.

"NTC: untouched" means NTC pixels go through unchanged - Layer D is
applied only to the positive wells (if D is among the active layers).

Models
------
Only the 5 well-behaved time-domain architectures
(``ann, cnn1d, fcn, resnet, inception``).  cnn2d_spec and
autoencoder are excluded - cnn2d_spec has seed-1 collapse and
autoencoder has the polarity flip under protocol shift.

Usage::

    # Build all 8 caches:
    python -m lacewing.classification.experiments.run_exp7 --build-all-caches

    # Run the full sweep locally:
    python -m lacewing.classification.experiments.run_exp7

    # Run a single (experiment, model, seed) cell (used by PBS):
    python -m lacewing.classification.experiments.run_exp7 \\
        --experiment exp7_ABCD_ntcD --model ann --seed 0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass

from ..core import paths


# --------------------------------------------------------------------
# Experiment configuration table
# --------------------------------------------------------------------

@dataclass(frozen=True)
class ExpConfig:
    """One experiment cell."""
    name:            str           # experiment id (used as results subdir)
    layers:          str           # 'A' / 'AB' / 'ABC' / 'ABCD'
    clean_ntc_first: bool          # whether to D-clean NTC before A/B thresholds
    cache_stem:      str           # cache file stem (under data/cache/)


def _cache_stem(layers: str, clean_ntc: bool, scope: str = "all") -> str:
    ntc = "ntcD" if clean_ntc else "ntcRaw"
    return f"dataset_{scope}_filt_{layers.lower()}_{ntc}"


def _exp_name(layers: str, clean_ntc: bool) -> str:
    return f"exp7_{layers}_{'ntcD' if clean_ntc else 'ntcRaw'}"


EXPERIMENTS: list[ExpConfig] = [
    ExpConfig(_exp_name(L, c), L, c, _cache_stem(L, c))
    for L in ("A", "AB", "ABC", "ABCD")
    for c in (False, True)
]
EXP_BY_NAME = {e.name: e for e in EXPERIMENTS}


MODELS = ["ann", "cnn1d", "fcn", "resnet", "inception"]
SEEDS  = [0, 1, 2]

TEST_CHIPS = ",".join([
    "D20240719_E03_C44_F4500KHz_U_COV_SD",
    "D20240814_E00_C00_F4500KHz_U_PnG_Bead",
])


# --------------------------------------------------------------------
# Cache building (one-off, CPU only)
# --------------------------------------------------------------------

def build_all_caches(scope: str = "all") -> None:
    """Build the 8 filtered caches the experiments consume.

    Cheap-ish: each cache requires loading every chip once via titan
    (~10-20 s/chip).  Caches are independent so this could be
    parallelised but is fine to run sequentially.
    """
    from lacewing.classification.data import build_filtered_dataset as bf
    for L in ("A", "AB", "ABC", "ABCD"):
        for c in (False, True):
            stem = _cache_stem(L, c, scope=scope)
            out  = paths.cache_path(stem)
            if out.exists():
                print(f"[skip] {stem}.npz already exists.")
                continue
            print(f"\n[build] {stem}  (layers={L}, clean_ntc={c})")
            bf.build_filtered(
                scope=scope, layers=L,
                clean_ntc_first=c, out_name=stem,
            )


# --------------------------------------------------------------------
# Training dispatch (single cell or full sweep)
# --------------------------------------------------------------------

def _run(cmd: list[str]) -> int:
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    return subprocess.call(cmd)


def run_one(exp: ExpConfig, model: str, seed: int, *,
            epochs: int, device: str | None,
            test_chips: str) -> int:
    """Train one (experiment, model, seed) cell."""
    py = sys.executable
    cmd = [
        py, "-m", "lacewing.classification.core.train",
        "--experiment", exp.name,
        "--cache",      exp.cache_stem,
        "--split",      "sd_test",
        "--fold",       test_chips,
        "--epochs",     str(epochs),
        "--model",      model,
        "--seed",       str(seed),
    ]
    if device:
        cmd += ["--device", device]
    return _run(cmd)


def run_sweep(experiments: list[ExpConfig], models: list[str],
              seeds: list[int], *, epochs: int, device: str | None,
              test_chips: str) -> None:
    n_runs = len(experiments) * len(models) * len(seeds)
    print(f"exp7: {len(experiments)} experiments x {len(models)} models "
          f"x {len(seeds)} seeds = {n_runs} runs")

    t0 = time.time()
    failures: list[tuple[str, str, int]] = []
    for exp in experiments:
        for model in models:
            for seed in seeds:
                rc = run_one(exp, model, seed,
                             epochs=epochs, device=device,
                             test_chips=test_chips)
                if rc != 0:
                    failures.append((exp.name, model, seed))

    elapsed = time.time() - t0
    print(f"\nexp7 sweep finished in {elapsed/60:.1f} min")
    if failures:
        print(f"Failures ({len(failures)}):")
        for f in failures:
            print(f"  exp={f[0]}  model={f[1]}  seed={f[2]}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment",
                   choices=[e.name for e in EXPERIMENTS],
                   default=None,
                   help="Run only this experiment (default: all 8).")
    p.add_argument("--model", default=None,
                   help="Run only this model (used by PBS array).")
    p.add_argument("--seed", type=int, default=None,
                   help="Run only this seed (used by PBS array).")
    p.add_argument("--test-chips", default=TEST_CHIPS)
    p.add_argument("--device", default=None,
                   help="cpu / cuda. Passed through to train.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--build-all-caches", action="store_true",
                   help="Build all 8 filtered caches (CPU one-off) and exit.")
    p.add_argument("--scope", default="all", choices=["final", "all"],
                   help="Cache scope (default 'all' to match exp6).")
    args = p.parse_args()

    if args.build_all_caches:
        build_all_caches(scope=args.scope)
        return

    experiments = ([EXP_BY_NAME[args.experiment]] if args.experiment
                   else EXPERIMENTS)
    models = [args.model] if args.model else MODELS
    seeds  = [args.seed] if args.seed is not None else SEEDS

    # Helpful sanity check: refuse to run if a needed cache is missing.
    missing = [e.cache_stem for e in experiments
               if not paths.cache_path(e.cache_stem).exists()]
    if missing:
        print("Missing caches (build them first via --build-all-caches):",
              file=sys.stderr)
        for s in missing:
            print(f"  {s}.npz", file=sys.stderr)
        sys.exit(2)

    run_sweep(experiments, models, seeds,
              epochs=args.epochs, device=args.device,
              test_chips=args.test_chips)


if __name__ == "__main__":
    main()
