"""RQ1 · deployed→+MAD→+spatA3 preprocessing ablation on CoV. [Cat A] Report §RQ1 Tab. 5.X.

exp8: MAD-based threshold rule across k values and layer subsets,
on the ntcRaw NTC policy.

Same chip-generalisation regime as exp6/exp7 (train on 5 dose-response
chips, test on 2 held-out SD chips).  Differs from exp7 in that
Layers A, B, D use ``median +- k * MAD`` thresholds instead of the
fixed-percentile cuts; Layer C is unchanged (no statistical
threshold).  Single ``k`` parameter shared across A/B/D within each
configuration.

The sweep has two halves:

1. **ABCD-only sweep across 6 k values** (the original exp8 run,
   already complete on HPC).  Answers: what is the best k when the
   full ABCD stack is active?

2. **Layer-progression sweep** ({A, AB, ABC, ABCD} x {1.5, 1.645, 2.0}).
   Answers: at each k, does adding C and D on top of A or AB still
   help, or does it start hurting like the percentile-rule profile
   for FCN/ResNet?  ABCD configs reuse the cells from sweep 1, so
   only 9 new cells (A/AB/ABC at the three k values) actually need
   to run.

Experiments (15 total = 6 ABCD + 9 layer-subset cells)
-----------------------------------------------------
    exp8_ABCD_ntcRaw_madk1.5    [DONE]
    exp8_ABCD_ntcRaw_madk1.645  [DONE]
    exp8_ABCD_ntcRaw_madk2.0    [DONE]
    exp8_ABCD_ntcRaw_madk2.5    [DONE]
    exp8_ABCD_ntcRaw_madk3.0    [DONE]
    exp8_ABCD_ntcRaw_madk3.5    [DONE]
    exp8_A_ntcRaw_madk1.5       [NEW]
    exp8_A_ntcRaw_madk1.645     [NEW]
    exp8_A_ntcRaw_madk2.0       [NEW]
    exp8_AB_ntcRaw_madk1.5      [NEW]
    exp8_AB_ntcRaw_madk1.645    [NEW]
    exp8_AB_ntcRaw_madk2.0      [NEW]
    exp8_ABC_ntcRaw_madk1.5     [NEW]
    exp8_ABC_ntcRaw_madk1.645   [NEW]
    exp8_ABC_ntcRaw_madk2.0     [NEW]

All configurations use the ntcRaw NTC policy (NTC well untouched
by Layer D pre-cleaning), since ntcRaw won the exp7 ablation.

Models
------
Same 5 well-behaved time-domain architectures as exp7:
``ann, cnn1d, fcn, resnet, inception``.

Usage::

    # Build all 15 caches (CPU one-off; skips ones already on disk):
    python -m lacewing.classification.experiments.run_exp8 --build-all-caches

    # Run a single (experiment, model, seed) cell (used by PBS):
    python -m lacewing.classification.experiments.run_exp8 \\
        --experiment exp8_AB_ntcRaw_madk1.645 --model ann --seed 0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass

from ..core import paths


# k values from the original ABCD-only sweep (all 6 already done on HPC).
K_VALUES = [1.5, 1.645, 2.0, 2.5, 3.0, 3.5]

# Layer subsets to sweep in the layer-progression extension.  ABCD is
# included so the existing ABCD runs feed the layer-progression plot
# without re-running anything.
LAYERS_PROGRESSION = ["A", "AB", "ABC", "ABCD"]

# k values to use for the layer-progression extension.  These are the
# three k values where the ABCD-only sweep showed the strongest
# pre-processing benefit -- the more permissive k (>= 2.5) hurt on
# ABCD, but we still want to check whether they help on the simpler
# layer subsets, hence k=2.0 is included as the "more permissive than
# percentile" tail of the sweep.
K_VALUES_PROGRESSION = [1.5, 1.645, 2.0]


@dataclass(frozen=True)
class ExpConfig:
    name:            str
    layers:          str
    clean_ntc_first: bool
    k:               float
    cache_stem:      str


def _cache_stem(layers: str, clean_ntc: bool, k: float,
                scope: str = "all") -> str:
    ntc = "ntcD" if clean_ntc else "ntcRaw"
    k_str = f"{k}".replace(".", "p")
    return f"dataset_{scope}_filt_{layers.lower()}_{ntc}_madk{k_str}"


def _exp_name(layers: str, clean_ntc: bool, k: float) -> str:
    ntc = "ntcD" if clean_ntc else "ntcRaw"
    return f"exp8_{layers}_{ntc}_madk{k}"


def _build_experiments() -> list[ExpConfig]:
    """Combine the original ABCD-only sweep with the new
    layer-progression sweep.  Dedupe so ABCD@{1.5, 1.645, 2.0} are
    listed exactly once.
    """
    seen: set[tuple[str, bool, float]] = set()
    configs: list[ExpConfig] = []

    def _add(layers: str, k: float) -> None:
        key = (layers, False, k)
        if key in seen:
            return
        seen.add(key)
        configs.append(ExpConfig(
            name=_exp_name(layers, False, k),
            layers=layers,
            clean_ntc_first=False,
            k=k,
            cache_stem=_cache_stem(layers, False, k),
        ))

    # 1. Original ABCD-only sweep (k in {1.5, 1.645, 2.0, 2.5, 3.0, 3.5}).
    for k in K_VALUES:
        _add("ABCD", k)
    # 2. Layer-progression extension at k in {1.5, 1.645, 2.0}.
    for k in K_VALUES_PROGRESSION:
        for layers in LAYERS_PROGRESSION:
            _add(layers, k)
    return configs


EXPERIMENTS: list[ExpConfig] = _build_experiments()
EXP_BY_NAME = {e.name: e for e in EXPERIMENTS}


MODELS = ["ann", "cnn1d", "fcn", "resnet", "inception"]
SEEDS  = list(range(10))   # 10 seeds per cell (matches exp7 best-3-of-10)

TEST_CHIPS = ",".join([
    "D20240719_E03_C44_F4500KHz_U_COV_SD",
    "D20240814_E00_C00_F4500KHz_U_PnG_Bead",
])


def build_all_caches(scope: str = "all") -> None:
    """Build every filtered cache the experiments consume.

    Idempotent: any cache .npz already on disk is skipped.  After the
    original ABCD-only sweep this should add 9 new caches
    (A/AB/ABC at k in {1.5, 1.645, 2.0}).
    """
    from lacewing.classification.data import build_filtered_dataset_mad as bfm
    for exp in EXPERIMENTS:
        stem = _cache_stem(exp.layers, exp.clean_ntc_first, exp.k, scope=scope)
        out  = paths.cache_path(stem)
        if out.exists():
            print(f"[skip] {stem}.npz already exists.")
            continue
        print(f"\n[build] {stem}  (layers={exp.layers}, "
              f"clean_ntc={exp.clean_ntc_first}, k={exp.k})")
        bfm.build_filtered_mad(
            scope=scope, layers=exp.layers,
            clean_ntc_first=exp.clean_ntc_first, k=exp.k, out_name=stem,
        )


def _run(cmd: list[str]) -> int:
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    return subprocess.call(cmd)


def run_one(exp: ExpConfig, model: str, seed: int, *,
            epochs: int, device: str | None,
            test_chips: str) -> int:
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
    print(f"exp8: {len(experiments)} experiments x {len(models)} models "
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
    print(f"\nexp8 sweep finished in {elapsed/60:.1f} min")
    if failures:
        print(f"Failures ({len(failures)}):")
        for f in failures:
            print(f"  exp={f[0]}  model={f[1]}  seed={f[2]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment",
                   choices=[e.name for e in EXPERIMENTS],
                   default=None,
                   help="Run only this experiment (default: all 6).")
    p.add_argument("--model", default=None,
                   help="Run only this model (used by PBS array).")
    p.add_argument("--seed", type=int, default=None,
                   help="Run only this seed (used by PBS array).")
    p.add_argument("--test-chips", default=TEST_CHIPS)
    p.add_argument("--device", default=None,
                   help="cpu / cuda. Passed through to train.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--build-all-caches", action="store_true",
                   help="Build all 6 filtered caches (CPU one-off) and exit.")
    p.add_argument("--scope", default="all", choices=["final", "all"])
    args = p.parse_args()

    if args.build_all_caches:
        build_all_caches(scope=args.scope)
        return

    experiments = ([EXP_BY_NAME[args.experiment]] if args.experiment
                   else EXPERIMENTS)
    models = [args.model] if args.model else MODELS
    seeds  = [args.seed] if args.seed is not None else SEEDS

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
