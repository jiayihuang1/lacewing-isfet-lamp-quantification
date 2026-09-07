"""Per-seed/per-fold scatter plots so error bars can be sanity-checked.

Addresses the Week-4 supervisor TODO:
    "Double check the error bars in classification result plots, why is
     there such a big error bar for some of them? Is it seeding issue?
     or is the results just inconsistent?"

Two outputs:

1) Per-experiment scatter
   results/<experiment>/per_run_scatter.png
   Every (model, split, fold, seed) run gets its own dot. Bar-chart-style
   mean+std overlay so the relationship between aggregate stats and
   individual runs is visible. Use this to spot:
     - one bad fold dragging the std up
     - per-seed inconsistency (within a (model, split, fold), do the
       3 seeds bunch up or scatter?)
     - bimodal failures (e.g. one fold collapses to 50%, rest are 85%)

2) Cross-experiment scatter
   results/comparison_across_experiments_per_run.png
   Same idea but every experiment gets a column per model, with all of
   its runs as dots — the companion to comparison_across_experiments.png
   that already exists.

Run:
    python -m lacewing.classification.plot_per_seed                       # default exp
    python -m lacewing.classification.plot_per_seed --experiment exp5_all_chipkfold
    python -m lacewing.classification.plot_per_seed --cross-experiment    # cross-exp plot
"""
from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core import paths
from .compare_experiments import EXPERIMENTS
from .plots import MODEL_LABEL, MODEL_ORDER
from .report import PAPER3_REFERENCE


def _per_run_dataframe(experiment: str) -> pd.DataFrame:
    csv = paths.metrics_csv(experiment)
    if not csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv)
    # Keep every row — these ARE the individual runs.
    return df


def _jitter(n: int, rng: np.random.Generator, width: float = 0.18) -> np.ndarray:
    return (rng.random(n) - 0.5) * 2.0 * width


def plot_per_experiment(experiment: str) -> None:
    """One scatter per model showing every run; mean+std overlay."""
    df = _per_run_dataframe(experiment)
    if df.empty:
        print(f"No metrics for {experiment}.")
        return

    rng = np.random.default_rng(0)
    splits_present = list(df["split"].unique())
    # Map split -> integer offset within a model's slot
    split_offset = {s: (i - (len(splits_present) - 1) / 2) * 0.30
                    for i, s in enumerate(splits_present)}
    split_color = {"random":   "#1F3A6B",
                   "chip":     "#E67E22",
                   "chipkfold": "#27AE60"}

    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    if not models_present:
        print(f"No known models in {experiment}.")
        return

    fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(models_present)), 6))
    x_base = np.arange(len(models_present))

    for split in splits_present:
        sub = df[df["split"] == split]
        for i, m in enumerate(models_present):
            ms = sub[sub["model"] == m]
            if ms.empty:
                continue
            xs = np.full(len(ms), x_base[i] + split_offset[split])
            xs = xs + _jitter(len(ms), rng, width=0.08)
            ax.scatter(xs, ms["pixel_acc"],
                       s=42, alpha=0.75,
                       color=split_color.get(split, "grey"),
                       edgecolor="white", linewidth=0.5,
                       label=split if i == 0 else None,
                       zorder=3)
            # mean+std overlay (black T-bar)
            mean = float(ms["pixel_acc"].mean())
            std = float(ms["pixel_acc"].std(ddof=1)) if len(ms) > 1 else 0.0
            ax.errorbar(x_base[i] + split_offset[split], mean,
                        yerr=std, fmt="_", color="black",
                        capsize=6, elinewidth=1.5, markersize=18,
                        zorder=4)

    # Paper 3 reference points
    for i, m in enumerate(models_present):
        ref = PAPER3_REFERENCE.get(m)
        if ref is not None:
            ax.scatter([x_base[i]], [ref], marker="*", s=140,
                       color="#C0392B", edgecolor="black", linewidth=0.6,
                       zorder=5,
                       label="Paper 3 reference" if i == 0 else None)

    ax.set_xticks(x_base)
    ax.set_xticklabels([MODEL_LABEL[m] for m in models_present], rotation=15)
    ax.set_ylabel("Pixel-level accuracy (one dot = one training run)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{experiment} — every individual run, mean±std as T-bar")
    ax.axhline(0.5, color="grey", lw=0.5, ls=":")
    ax.grid(axis="y", alpha=0.3)
    # Custom legend with split colours
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        seen = set()
        dedup = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        ax.legend([h for h, _ in dedup], [l for _, l in dedup],
                  loc="lower right", fontsize=9)
    plt.tight_layout()
    out = paths.experiment_dir(experiment) / "per_run_scatter.png"
    plt.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_cross_experiment_per_run() -> None:
    """Companion to comparison_across_experiments.png with every run shown."""
    rng = np.random.default_rng(0)
    aggs = []
    for name, label, color in EXPERIMENTS:
        df = _per_run_dataframe(name)
        if not df.empty:
            aggs.append((name, label, color, df))
    if not aggs:
        print("No experiment results found.")
        return

    models_present = []
    for _, _, _, df in aggs:
        for m in df["model"].unique():
            if m not in models_present and m in MODEL_ORDER:
                models_present.append(m)
    models_present.sort(key=lambda m: MODEL_ORDER.index(m))
    if not models_present:
        print("No known models across experiments.")
        return

    n_exps = len(aggs)
    slot_w = 0.85 / n_exps
    x_base = np.arange(len(models_present))

    fig, ax = plt.subplots(figsize=(max(13, 1.6 * len(models_present)), 6))
    for i, (name, label, color, df) in enumerate(aggs):
        offset = (i - n_exps / 2 + 0.5) * slot_w
        for j, m in enumerate(models_present):
            ms = df[df["model"] == m]
            if ms.empty:
                continue
            xs = np.full(len(ms), x_base[j] + offset)
            xs = xs + _jitter(len(ms), rng, width=slot_w * 0.25)
            ax.scatter(xs, ms["pixel_acc"], s=36, alpha=0.7,
                       color=color, edgecolor="white", linewidth=0.4,
                       label=label if j == 0 else None, zorder=3)
            mean = float(ms["pixel_acc"].mean())
            std = float(ms["pixel_acc"].std(ddof=1)) if len(ms) > 1 else 0.0
            ax.errorbar(x_base[j] + offset, mean,
                        yerr=std, fmt="_", color="black",
                        capsize=4, elinewidth=1.0, markersize=14,
                        zorder=4)

    # Paper 3 reference
    ref_offset = (n_exps / 2) * slot_w  # just to the right of the last exp
    for j, m in enumerate(models_present):
        ref = PAPER3_REFERENCE.get(m)
        if ref is not None:
            ax.scatter([x_base[j] + ref_offset], [ref], marker="*", s=120,
                       color="#C0392B", edgecolor="black", linewidth=0.6,
                       zorder=5,
                       label="Paper 3 ref" if j == 0 else None)

    ax.set_xticks(x_base)
    ax.set_xticklabels([MODEL_LABEL[m] for m in models_present], rotation=15)
    ax.set_ylabel("Pixel-level accuracy (one dot = one training run)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Paper 3 reproduction — every individual run across experiments")
    ax.axhline(0.5, color="grey", lw=0.5, ls=":")
    ax.grid(axis="y", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    dedup = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend([h for h, _ in dedup], [l for _, l in dedup],
              loc="lower right", fontsize=9)
    plt.tight_layout()
    out = paths.CLASSIFICATION_RESULTS / "comparison_across_experiments_per_run.png"
    plt.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default=paths.DEFAULT_EXPERIMENT)
    p.add_argument("--cross-experiment", action="store_true",
                   help="Skip the per-experiment plot and only do the "
                        "cross-experiment scatter.")
    p.add_argument("--all", action="store_true",
                   help="Generate per-experiment scatter for every "
                        "experiment listed in compare_experiments.EXPERIMENTS, "
                        "plus the cross-experiment plot.")
    args = p.parse_args()

    if args.all:
        for name, _label, _color in EXPERIMENTS:
            plot_per_experiment(name)
        plot_cross_experiment_per_run()
        return

    if not args.cross_experiment:
        plot_per_experiment(args.experiment)
    plot_cross_experiment_per_run()


if __name__ == "__main__":
    main()
