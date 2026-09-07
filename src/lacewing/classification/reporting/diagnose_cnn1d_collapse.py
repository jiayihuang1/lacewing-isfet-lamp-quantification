"""Plot cnn1d training curves on filtered caches to diagnose the
collapse pattern observed in some exp7 conditions.

For each cnn1d run in exp6 and the exp7 conditions where cnn1d had
issues, plot per-epoch:
  - train_loss (log scale)
  - train_acc + val_acc

If our hypothesis is right, the collapsed seeds will show:
  - loss frozen at ~0.28 (initialisation value) across all epochs
  - train_acc oscillating between ~0.20 and ~0.80 (corresponding to
    flipping between "predict all negative" and "predict all positive")
  - val_acc doing the same in anti-phase

while the working seeds show:
  - loss dropping from ~0.10 to <0.05 in the first 5 epochs
  - train_acc converging to ~0.96+ within 5 epochs

Run::

    python -m lacewing.classification.reporting.diagnose_cnn1d_collapse
"""
from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..core import paths


# Experiments to inspect.
EXPERIMENTS = [
    "exp6_final_to_sd",
    "exp7_A_ntcRaw",
    "exp7_A_ntcD",
    "exp7_AB_ntcRaw",
    "exp7_AB_ntcD",
    "exp7_ABC_ntcRaw",
    "exp7_ABC_ntcD",
    "exp7_ABCD_ntcRaw",
    "exp7_ABCD_ntcD",
]


def find_cnn1d_runs(exp_dir: Path) -> dict[int, Path]:
    """Return {seed: run_dir} for cnn1d runs in this experiment."""
    out = {}
    for run_dir in (exp_dir / "runs").glob("*"):
        if not run_dir.is_dir():
            continue
        m = re.match(
            r"^\d{8}-\d{6}_(?P<model>.+?)_(?:raw|spectrogram)_sd_test_fold.+_seed(?P<seed>\d+)$",
            run_dir.name,
        )
        if not m or m.group("model") != "cnn1d":
            continue
        out[int(m.group("seed"))] = run_dir
    return out


def read_metrics(run_dir: Path) -> dict:
    """Read metrics.csv → dict of {col: list[float]}."""
    with (run_dir / "metrics.csv").open() as fh:
        r = csv.DictReader(fh)
        rows = list(r)
    keys = rows[0].keys()
    return {k: [float(r[k]) if k != "epoch" else int(r[k]) for r in rows]
            for k in keys}


def main() -> None:
    fig, axes = plt.subplots(
        len(EXPERIMENTS), 2,
        figsize=(13, 2.2 * len(EXPERIMENTS)),
        sharex=True,
    )

    colours = {0: "#1f77b4", 1: "#e9a13b", 2: "#c0392b"}

    for row_i, exp_name in enumerate(EXPERIMENTS):
        exp_dir = paths.experiment_dir(exp_name)
        cnn1d_runs = find_cnn1d_runs(exp_dir)
        ax_loss = axes[row_i, 0]
        ax_acc  = axes[row_i, 1]
        ax_loss.set_title(f"{exp_name}  —  train_loss", fontsize=9.5,
                          loc="left")
        ax_acc.set_title(f"{exp_name}  —  train_acc / val_acc",
                         fontsize=9.5, loc="left")

        for seed in sorted(cnn1d_runs):
            d = read_metrics(cnn1d_runs[seed])
            colour = colours.get(seed, "#888")
            ax_loss.plot(d["epoch"], d["train_loss"], color=colour,
                         linewidth=1.0, label=f"seed {seed}")
            ax_acc.plot(d["epoch"], d["train_acc"], color=colour,
                        linewidth=1.0, linestyle="-",
                        label=f"seed {seed} train")
            ax_acc.plot(d["epoch"], d["val_acc"], color=colour,
                        linewidth=1.0, linestyle="--", alpha=0.6,
                        label=f"seed {seed} val")

        ax_loss.set_yscale("log")
        ax_loss.set_ylim(0.005, 0.35)
        ax_loss.grid(alpha=0.3)
        # Mark the suspect "stuck" loss level.
        ax_loss.axhline(0.282, color="#888", linestyle=":", linewidth=0.6,
                        alpha=0.6)

        ax_acc.set_ylim(0, 1.0)
        ax_acc.grid(alpha=0.3)
        ax_acc.axhline(0.5, color="#888", linestyle=":", linewidth=0.6,
                       alpha=0.6)

        if row_i == 0:
            ax_loss.legend(fontsize=7, frameon=False, ncol=3,
                           loc="upper right")
            ax_acc.legend(fontsize=6, frameon=False, ncol=2,
                          loc="lower right")

    for ax in axes[-1]:
        ax.set_xlabel("epoch")

    fig.suptitle("cnn1d training curves across all exp7 conditions "
                 "(line = seed, solid = train, dashed = val)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out = paths.CLASSIFICATION_RESULTS / "exp7_cnn1d_diagnosis.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    # Also print a one-line per-(exp, seed) verdict.
    print()
    print(f"{'experiment':<22} {'seed':>4}  {'final_loss':>10}  "
          f"{'final_train_acc':>16}  {'final_val_acc':>14}  verdict")
    print("-" * 90)
    for exp_name in EXPERIMENTS:
        runs = find_cnn1d_runs(paths.experiment_dir(exp_name))
        for seed in sorted(runs):
            d = read_metrics(runs[seed])
            final_loss = d["train_loss"][-1]
            final_tacc = d["train_acc"][-1]
            final_vacc = d["val_acc"][-1]
            # Heuristic: stuck if final_loss > 0.20 (random/init level)
            verdict = ("STUCK" if final_loss > 0.20
                       else "ok" if final_loss < 0.05
                       else "borderline")
            print(f"{exp_name:<22} {seed:>4}  {final_loss:>10.5f}  "
                  f"{final_tacc:>16.4f}  {final_vacc:>14.4f}  {verdict}")


if __name__ == "__main__":
    main()
