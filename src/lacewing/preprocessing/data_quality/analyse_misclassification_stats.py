"""Compare per-pixel statistics of misclassified vs correctly-classified
pixels on the exp6 held-out set.

Motivation
----------
The exp6 test set has 21,058 pixels across the two held-out SD chips
(COV_SD + PnG_Bead).  Seven architectures x three seeds = 21 runs, each
producing a per-pixel prediction in ``predictions.npz``.  This script
asks: when we group pixels by *how often they were misclassified across
the 21 runs*, do their per-pixel statistics differ from pixels that
were always classified correctly?

If yes, the statistic on which they differ is a candidate signal for a
preprocessing filter that would catch the failures before training.
If no, the failures are not pixel-quality issues and preprocessing is
not the right lever.

Buckets
-------
For each (chip, well, pixel) triple in the test set, count
``n_wrong`` = number of the 21 runs that misclassified that pixel.

- ``always_correct``     n_wrong == 0
- ``sometimes_wrong``    1 <= n_wrong <  21
- ``always_wrong``       n_wrong == 21

We additionally split by ``y_true`` (positive vs negative) because the
two classes have very different signal shapes.

Statistics computed per pixel
-----------------------------
- dynamic_range      max(x) - min(x)
- baseline_std       std of the first ``--baseline`` samples
- absolute_drift     |x[-1] - x[0]|
- abs_max            max(|x|)
- mean_abs           mean(|x|)

CPU-only, no GPU.

Usage::

    python -m lacewing.preprocessing.data_quality.analyse_misclassification_stats
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lacewing.classification import paths as cls_paths


THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


STAT_LABELS = {
    "dynamic_range":  "Trace dynamic range (max - min)",
    "baseline_std":   "Baseline-segment std",
    "absolute_drift": "Absolute end-to-end drift |x[-1] - x[0]|",
    "abs_max":        "Max |x|",
    "mean_abs":       "Mean |x|",
}


BUCKETS = ["always_correct", "sometimes_wrong", "always_wrong"]
BUCKET_COLOURS = {
    "always_correct":  "#3b78bf",
    "sometimes_wrong": "#e9a13b",
    "always_wrong":    "#c0392b",
}

# Models whose exp6 failures are dominated by known architectural issues
# (cnn2d_spec seed-1 collapse; autoencoder polarity flip).  Including
# them in the run pool puts every pixel into the `sometimes_wrong`
# bucket and washes out the signal.  Excluded by default; pass
# --include-pathological to keep them.
DEFAULT_EXCLUDED_MODELS = ("cnn2d_spec", "cnn2d", "autoencoder")


# --------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------

def _model_name_from_run_dir(name: str) -> str:
    """Extract the model name from a run directory name.

    Run dirs look like ``20260518-165156_ann_raw_sd_test_fold...seed0``;
    the model name is the second token.
    """
    parts = name.split("_")
    if len(parts) < 2:
        return ""
    # Some models have multi-token names (cnn2d_spec).
    if len(parts) >= 3 and parts[1] == "cnn2d" and parts[2] == "spec":
        return "cnn2d_spec"
    return parts[1]


def collect_predictions(
    exp_dir: Path,
    excluded_models: tuple[str, ...] = DEFAULT_EXCLUDED_MODELS,
) -> list[dict]:
    """Read every predictions.npz under exp_dir/runs/* (filtered).

    Returns a list of dicts (one per run) with the run name and the
    arrays from the npz, excluding runs whose model name matches
    ``excluded_models``.
    """
    runs = sorted(exp_dir.glob("runs/*/predictions.npz"))
    if not runs:
        raise FileNotFoundError(f"no predictions.npz found under {exp_dir}/runs/")

    out = []
    skipped: list[str] = []
    for p in runs:
        model = _model_name_from_run_dir(p.parent.name)
        if model in excluded_models:
            skipped.append(f"{p.parent.name} (model={model})")
            continue
        z = np.load(p, allow_pickle=True)
        out.append({
            "run":      p.parent.name,
            "model":    model,
            "chip_id":  z["chip_id"],
            "well_id":  z["well_id"],
            "pixel_id": z["pixel_id"],
            "y_true":   z["y_true"],
            "y_pred":   z["y_pred"],
            "x_raw":    z["x_raw"],
        })
    if skipped:
        print(f"  excluded {len(skipped)} runs from pathological models "
              f"({', '.join(excluded_models)}):")
        for s in skipped:
            print(f"    - {s}")
    return out


def build_pixel_index(runs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the canonical (chip, well, pixel) index of the test set.

    Verifies every run has the same set of (chip, well, pixel) triples
    (so n_wrong can be computed by summing across runs at the same
    array positions).  Returns the canonical chip_id / well_id /
    pixel_id of the first run, and asserts the rest match.
    """
    chip0  = runs[0]["chip_id"]
    well0  = runs[0]["well_id"]
    pix0   = runs[0]["pixel_id"]

    for r in runs[1:]:
        if not (np.array_equal(r["chip_id"], chip0)
                and np.array_equal(r["well_id"], well0)
                and np.array_equal(r["pixel_id"], pix0)):
            raise ValueError(
                f"run {r['run']} has a different (chip, well, pixel) index "
                "than the first run; cannot align."
            )
    return chip0, well0, pix0


def per_pixel_stats(X: np.ndarray, baseline_samples: int) -> pd.DataFrame:
    baseline = X[:, :baseline_samples]
    return pd.DataFrame({
        "dynamic_range":  X.max(axis=1) - X.min(axis=1),
        "baseline_std":   baseline.std(axis=1),
        "absolute_drift": np.abs(X[:, -1] - X[:, 0]),
        "abs_max":        np.abs(X).max(axis=1),
        "mean_abs":       np.abs(X).mean(axis=1),
    })


# --------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------

def plot_bucket_histograms(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    """One 2x3 grid: a histogram per statistic, overlaying buckets."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    flat = axes.flat
    for ax, (stat, label) in zip(flat, STAT_LABELS.items()):
        # Use a shared bin set across buckets so the comparison is fair.
        all_vals = df[stat].values
        if len(all_vals) == 0:
            continue
        lo, hi = np.quantile(all_vals, [0.001, 0.999])
        bins = np.linspace(lo, hi, 60)
        for bucket in BUCKETS:
            sub = df.loc[df["bucket"] == bucket, stat].values
            if len(sub) == 0:
                continue
            ax.hist(sub, bins=bins, alpha=0.55, label=f"{bucket} (n={len(sub)})",
                    color=BUCKET_COLOURS[bucket], density=True)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("density")
        ax.legend(fontsize=8, frameon=False)

    # Leave the 6th cell blank or fill with the bucket-count summary
    ax = flat[len(STAT_LABELS)] if len(STAT_LABELS) < len(flat) else None
    if ax is not None:
        ax.axis("off")
        counts = df["bucket"].value_counts().reindex(BUCKETS).fillna(0).astype(int)
        text = "\n".join(f"{b:18s}  {int(counts[b]):6d}" for b in BUCKETS)
        ax.text(0.05, 0.5, "Pixel counts\n" + "-" * 28 + "\n" + text,
                family="monospace", fontsize=10, va="center")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_bucket_boxes(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    """One row of boxplots: one per statistic, x-axis is bucket."""
    fig, axes = plt.subplots(1, len(STAT_LABELS),
                             figsize=(3.2 * len(STAT_LABELS), 4.5))
    if len(STAT_LABELS) == 1:
        axes = [axes]
    for ax, (stat, label) in zip(axes, STAT_LABELS.items()):
        data = [df.loc[df["bucket"] == b, stat].values for b in BUCKETS]
        bp = ax.boxplot(data, showfliers=False, widths=0.6, patch_artist=True)
        for patch, b in zip(bp["boxes"], BUCKETS):
            patch.set_facecolor(BUCKET_COLOURS[b])
            patch.set_alpha(0.65)
        ax.set_xticks(range(1, len(BUCKETS) + 1))
        ax.set_xticklabels([b.replace("_", "\n") for b in BUCKETS], fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="exp6_final_to_sd",
                        help="experiment folder under classification/results/")
    parser.add_argument("--baseline", type=int, default=30,
                        help="leading samples treated as baseline (default 30)")
    parser.add_argument("--include-pathological", action="store_true",
                        help="include cnn2d_spec and autoencoder runs "
                             "(default excludes them; see DEFAULT_EXCLUDED_MODELS)")
    args = parser.parse_args()

    exp_dir = cls_paths.experiment_dir(args.experiment)
    print(f"Reading runs under {exp_dir}/runs/")
    excluded = () if args.include_pathological else DEFAULT_EXCLUDED_MODELS
    runs = collect_predictions(exp_dir, excluded_models=excluded)
    print(f"  found {len(runs)} runs")

    chip_id, well_id, pixel_id = build_pixel_index(runs)
    y_true = runs[0]["y_true"]
    x_raw  = runs[0]["x_raw"]
    n_pix  = len(chip_id)
    print(f"  test set: {n_pix} pixels across {len(set(chip_id))} chips")

    # Count misclassifications per pixel across all 21 runs.
    n_wrong = np.zeros(n_pix, dtype=np.int32)
    for r in runs:
        n_wrong += (r["y_pred"] != r["y_true"]).astype(np.int32)
    n_runs = len(runs)

    # Bucketise.
    bucket = np.full(n_pix, "sometimes_wrong", dtype=object)
    bucket[n_wrong == 0]      = "always_correct"
    bucket[n_wrong == n_runs] = "always_wrong"

    # Per-pixel statistics.
    stats = per_pixel_stats(x_raw, args.baseline)
    stats["chip"]       = chip_id
    stats["well"]       = well_id
    stats["pixel"]      = pixel_id
    stats["y_true"]     = y_true
    stats["n_wrong"]    = n_wrong
    stats["bucket"]     = bucket

    # ----------------------------------------------------------------
    # Summary CSV: pixel counts per (chip, label, bucket).
    # ----------------------------------------------------------------
    summary = (stats
        .groupby(["chip", "y_true", "bucket"])
        .size()
        .unstack("bucket", fill_value=0)
        .reindex(columns=BUCKETS, fill_value=0))
    summary["total"] = summary.sum(axis=1)
    for b in BUCKETS:
        summary[f"{b}_frac"] = summary[b] / summary["total"]
    csv_path = RESULTS_DIR / "bucket_counts.csv"
    summary.to_csv(csv_path)
    print(f"\nBucket counts:\n{summary[BUCKETS + ['total']]}\n")

    # ----------------------------------------------------------------
    # Plots: aggregate (all pixels), then per (chip, label).
    # ----------------------------------------------------------------
    print("Plotting overall + per-(chip, label) breakdowns...")
    plot_bucket_histograms(stats, "All test pixels (both chips, both labels)",
                           RESULTS_DIR / "hist_overall.png")
    plot_bucket_boxes(stats, "All test pixels",
                      RESULTS_DIR / "box_overall.png")

    for (chip, label), grp in stats.groupby(["chip", "y_true"]):
        tag = f"{chip}_label{label}"
        title = f"{chip}   y_true={label}   (n={len(grp)})"
        plot_bucket_histograms(grp, title, RESULTS_DIR / f"hist_{tag}.png")
        plot_bucket_boxes(grp, title, RESULTS_DIR / f"box_{tag}.png")

    # ----------------------------------------------------------------
    # Per-statistic separability report: median in each bucket,
    # and an effect-size (Cohen's d) between always_correct vs always_wrong.
    # ----------------------------------------------------------------
    rows = []
    for (chip, label), grp in stats.groupby(["chip", "y_true"]):
        for stat in STAT_LABELS:
            a = grp.loc[grp["bucket"] == "always_correct", stat].values
            w = grp.loc[grp["bucket"] == "always_wrong",   stat].values
            row = {
                "chip": chip, "y_true": label, "stat": stat,
                "median_always_correct":  float(np.median(a)) if len(a) else np.nan,
                "median_always_wrong":    float(np.median(w)) if len(w) else np.nan,
                "n_always_correct":       len(a),
                "n_always_wrong":         len(w),
            }
            if len(a) > 1 and len(w) > 1:
                pooled = np.sqrt((a.var(ddof=1) + w.var(ddof=1)) / 2.0)
                if pooled > 0:
                    row["cohens_d"] = (w.mean() - a.mean()) / pooled
                else:
                    row["cohens_d"] = 0.0
            else:
                row["cohens_d"] = np.nan
            rows.append(row)
    sep = pd.DataFrame(rows)
    sep_path = RESULTS_DIR / "separability.csv"
    sep.to_csv(sep_path, index=False)

    print(f"\nSeparability (Cohen's d between always_correct and always_wrong):")
    # show only |d| >= 0.2 (small effect or bigger), sorted by abs value
    interesting = sep.dropna(subset=["cohens_d"]).copy()
    interesting["abs_d"] = interesting["cohens_d"].abs()
    interesting = interesting.sort_values("abs_d", ascending=False)
    show = interesting[interesting["abs_d"] >= 0.2]
    if len(show):
        print(show[["chip", "y_true", "stat",
                    "median_always_correct", "median_always_wrong",
                    "cohens_d"]].to_string(index=False))
    else:
        print("  (none with |d| >= 0.2 - no per-pixel statistic separates "
              "always-correct from always-wrong)")

    print(f"\nOutputs in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
