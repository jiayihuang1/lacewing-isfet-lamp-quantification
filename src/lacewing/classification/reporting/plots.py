"""Generate post-sweep comparison plots for the writeup / meeting slides.

Run after a sweep + report.py:
    python -m lacewing.classification.plots
    python -m lacewing.classification.plots --experiment exp2_final_chipfold

Outputs into results/<experiment>/:
    accuracy_vs_paper3.png   - per-model accuracy bar chart vs Paper 3 ref
    train_time_vs_acc.png    - train-time vs accuracy scatter
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core import paths
from .report import PAPER3_REFERENCE


# Plot order (worst-to-best on Paper 3) - keeps the bar chart readable.
MODEL_ORDER = ["autoencoder", "fcn", "ann", "cnn1d", "inception", "resnet",
               "cnn2d_spec"]
MODEL_LABEL = {
    "ann": "ANN",
    "cnn1d": "1D-DCNN",
    "fcn": "FCN",
    "resnet": "ResNet1D",
    "inception": "InceptionTime",
    "autoencoder": "Autoencoder",
    "cnn2d_spec": "2D-CNN spec.",
}


def _per_model(df: pd.DataFrame, split: str = "random") -> pd.DataFrame:
    """Mean ± std per model on the given split, ordered by MODEL_ORDER."""
    sub = df[df["split"] == split]
    if sub.empty:
        return sub.assign(acc_mean=np.nan, acc_std=np.nan, well_mean=np.nan,
                          n_params=np.nan, train_time=np.nan,
                          paper3=np.nan, delta=np.nan, order=np.nan)
    g = (sub.groupby("model")
         .agg(acc_mean=("pixel_acc", "mean"),
              acc_std=("pixel_acc", "std"),
              well_mean=("well_acc", "mean"),
              n_params=("n_parameters", "first"),
              train_time=("train_time_sec", "mean"))
         .reset_index())
    g["paper3"] = g["model"].map(PAPER3_REFERENCE)
    g["delta"] = g["acc_mean"] - g["paper3"]
    g["order"] = g["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return g.sort_values("order").reset_index(drop=True)


def plot_accuracy_vs_paper3(df: pd.DataFrame, out_path,
                            split: str = "random",
                            split_label: str = "random split") -> None:
    g = _per_model(df, split)
    if g.empty:
        print(f"  no '{split}' rows in this experiment, skipping {out_path.name}")
        return
    x = np.arange(len(g))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ours = ax.bar(x - width / 2, g["acc_mean"], width,
                  yerr=g["acc_std"], capsize=4,
                  color="#1F3A6B", label=f"This work ({split_label})")
    paper = ax.bar(x + width / 2, g["paper3"], width,
                   color="#C0392B", alpha=0.85,
                   label="Paper 3 (Tripathi 2023)")

    for bars, vals in [(ours, g["acc_mean"]), (paper, g["paper3"])]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in g["model"]], rotation=15)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Pixel-level accuracy")
    ax.set_title(f"Paper 3 reproduction — pixel accuracy by model "
                 f"({split_label})")
    ax.axhline(0.5, color="grey", lw=0.5, linestyle=":")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_acc_vs_train_time(df: pd.DataFrame, out_path) -> None:
    g = _per_model(df, "random")
    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = g["n_params"] / 500            # marker scaling
    sc = ax.scatter(g["train_time"], g["acc_mean"], s=sizes,
                    c=range(len(g)), cmap="viridis", edgecolor="k", lw=0.7)
    for _, row in g.iterrows():
        ax.annotate(MODEL_LABEL[row["model"]],
                    (row["train_time"], row["acc_mean"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Mean training time per run (sec, log scale)")
    ax.set_ylabel("Pixel-level accuracy")
    ax.set_title("Accuracy vs training time (marker size = parameter count)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", default=paths.DEFAULT_EXPERIMENT)
    args = p.parse_args()

    csv = paths.metrics_csv(args.experiment)
    if not csv.exists():
        print(f"No results yet at {csv}.")
        return
    df = pd.read_csv(csv)
    out = paths.experiment_dir(args.experiment)

    splits = sorted(df["split"].unique())
    for split in splits:
        label = "random pixel split" if split == "random" else "chip-fold CV"
        plot_accuracy_vs_paper3(df, out / f"accuracy_vs_paper3_{split}.png",
                                split=split, split_label=label)

    plot_acc_vs_train_time(df, out / "train_time_vs_acc.png")


if __name__ == "__main__":
    main()
