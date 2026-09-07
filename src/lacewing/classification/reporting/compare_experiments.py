"""Cross-experiment comparison plot: exp1 vs exp2 vs exp3 vs Paper 3.

Reads each experiment's per_method_metrics.csv, aggregates per model, and
produces a grouped bar chart. Output goes to results/comparison_across_experiments.png.

Run:
    python -m lacewing.classification.compare_experiments
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core import paths
from .plots import MODEL_LABEL, MODEL_ORDER
from .report import PAPER3_REFERENCE


EXPERIMENTS = [
    ("exp1_final_random",   "exp1 — 5 Final chips, random",                "#1F3A6B"),
    ("exp2_final_chipfold", "exp2 — 5 Final chips, leave-one-chip-out CV", "#E67E22"),
    ("exp3_all_random",     "exp3 — 30 chips, random",                     "#27AE60"),
    ("exp4_all_chipfold",   "exp4 — 30 chips, leave-one-chip-out CV",      "#8E44AD"),
    # exp5: 5-fold chip-level CV on the all-data cache (~6 chips per fold
    # held out). Same idea as exp4 but tests on a larger held-out group
    # so the across-fold variance reflects model generalisation rather
    # than which single chip happened to be unlucky. cnn2d_spec dropped.
    ("exp5_all_chipkfold",  "exp5 — 30 chips, 5-fold chip CV (no spec)",   "#16A085"),
    # exp6: train on 5 Final CoV chips → test on 2 held-out SD chips
    # (COV_SD + PnG_Bead). Protocol-shift test: different sample
    # preparation, no overlap with training set.
    ("exp6_final_to_sd",    "exp6 — 5 Finals → 2 SD chips (protocol shift)", "#D35400"),
]


def _agg(experiment: str) -> pd.DataFrame:
    csv = paths.metrics_csv(experiment)
    if not csv.exists():
        return pd.DataFrame()
    # exp6's run_id and fold both contain a literal comma (two test chips
    # joined by ","), which shifts pandas' default parse. Hand-stitch the
    # two stray fields back into their parent columns.
    text = csv.read_text().splitlines()
    header = text[0].split(",")
    n_expected = len(header)
    rows = []
    for line in text[1:]:
        parts = line.split(",")
        if len(parts) == n_expected:
            rows.append(parts)
        elif len(parts) == n_expected + 2:
            # run_id has 1 extra comma (cols 0-1); fold has 1 extra (cols 7-8)
            stitched = (
                [",".join(parts[0:2])]    # run_id
                + parts[2:7]              # experiment..split
                + [",".join(parts[7:9])]  # fold
                + parts[9:]
            )
            rows.append(stitched)
        else:
            continue  # skip malformed lines silently
    df = pd.DataFrame(rows, columns=header)
    df["pixel_acc"] = pd.to_numeric(df["pixel_acc"], errors="coerce")
    g = (df.groupby("model")
         .agg(acc=("pixel_acc", "mean"),
              std=("pixel_acc", "std"))
         .reset_index())
    g["order"] = g["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    g = g.dropna(subset=["order"])
    return g.sort_values("order").reset_index(drop=True)


def main() -> None:
    aggs = [(name, label, color, _agg(name)) for name, label, color in EXPERIMENTS]
    aggs = [(n, l, c, a) for n, l, c, a in aggs if not a.empty]
    if not aggs:
        print("No experiment results found.")
        return

    # Common x-axis: union of all model names, ordered
    models_present = []
    for _, _, _, a in aggs:
        for m in a["model"]:
            if m not in models_present:
                models_present.append(m)
    models_present.sort(key=lambda m: MODEL_ORDER.index(m))

    n_groups = len(aggs) + 1   # +1 for Paper 3 reference
    bar_w = 0.85 / n_groups
    x = np.arange(len(models_present))

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (_, label, color, agg) in enumerate(aggs):
        ys, errs = [], []
        for m in models_present:
            row = agg[agg["model"] == m]
            ys.append(float(row["acc"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row["std"].iloc[0]) if not row.empty else 0.0)
        offset = (i - n_groups / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, ys, bar_w, yerr=errs, capsize=3,
                      color=color, label=label)
        for bar, v in zip(bars, ys):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Paper 3 reference
    ref_ys = [PAPER3_REFERENCE.get(m, np.nan) for m in models_present]
    offset = (n_groups - 1 - n_groups / 2 + 0.5) * bar_w
    bars = ax.bar(x + offset, ref_ys, bar_w, color="#C0392B",
                  label="Paper 3 (Tripathi 2023)")
    for bar, v in zip(bars, ref_ys):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in models_present], rotation=15)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Pixel-level accuracy")
    ax.set_title("Paper 3 reproduction — pixel accuracy across experiments  "
                 "(start_type='temperature', no reference-pulse leakage)")
    ax.axhline(0.5, color="grey", lw=0.5, ls=":")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    out = paths.CLASSIFICATION_RESULTS / "comparison_across_experiments.png"
    plt.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
