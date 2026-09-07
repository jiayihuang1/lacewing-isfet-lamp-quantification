"""Compare exp7 preprocessing-filter experiments against the exp6 baseline.

For the five well-behaved time-domain models (ANN, 1D-DCNN, FCN,
ResNet1D, InceptionTime), aggregate pixel and well accuracy across
seeds for:
  - exp6_final_to_sd   (unfiltered baseline)
  - exp7_A_ntcRaw      (Layer A on positive wells, NTC untouched)
  - exp7_A_ntcD        (Layer A on positive wells, NTC D-cleaned)
  - exp7_AB_ntcRaw
  - exp7_AB_ntcD
  - exp7_ABC_ntcRaw
  - exp7_ABC_ntcD
  - exp7_ABCD_ntcRaw
  - exp7_ABCD_ntcD

Writes:
  - results/exp7_summary.csv   (one row per (experiment, model)
                                with mean+std over 3 seeds)
  - results/exp7_comparison.png  (two-panel figure: per-model grouped
                                  bars + per-condition averaged bars)

Note: per_method_metrics.csv is unsafe to read directly because the
``fold`` column contains a literal comma in the SD-test chip names,
which breaks any csv parser that doesn't quote-aware split.  We
parse the run directory name and read ``test_metrics.json``
directly per run.

Usage::

    python -m lacewing.classification.reporting.compare_exp7
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..core import paths


# --------------------------------------------------------------------
# Experiment configuration
# --------------------------------------------------------------------

EXPERIMENTS = [
    ("exp6_final_to_sd", "exp6 baseline\n(no filter)"),
    ("exp7_A_ntcRaw",    "A\nntcRaw"),
    ("exp7_A_ntcD",      "A\nntcD"),
    ("exp7_AB_ntcRaw",   "AB\nntcRaw"),
    ("exp7_AB_ntcD",     "AB\nntcD"),
    ("exp7_ABC_ntcRaw",  "ABC\nntcRaw"),
    ("exp7_ABC_ntcD",    "ABC\nntcD"),
    ("exp7_ABCD_ntcRaw", "ABCD\nntcRaw"),
    ("exp7_ABCD_ntcD",   "ABCD\nntcD"),
]

MODELS = ["ann", "cnn1d", "fcn", "resnet", "inception"]
SEEDS  = [0, 1, 2]

# Colour scale: baseline neutral grey, then progressive blue-orange for
# layer depth, with darker shade = ntcD variant.
EXP_COLOURS = {
    "exp6_final_to_sd": "#666666",
    "exp7_A_ntcRaw":    "#a6cee3",
    "exp7_A_ntcD":      "#1f78b4",
    "exp7_AB_ntcRaw":   "#b2df8a",
    "exp7_AB_ntcD":     "#33a02c",
    "exp7_ABC_ntcRaw":  "#fdbf6f",
    "exp7_ABC_ntcD":    "#ff7f00",
    "exp7_ABCD_ntcRaw": "#fb9a99",
    "exp7_ABCD_ntcD":   "#e31a1c",
}

# Output paths.
SUMMARY_CSV = paths.CLASSIFICATION_RESULTS / "exp7_summary.csv"
COMPARISON_PNG = paths.CLASSIFICATION_RESULTS / "exp7_comparison.png"


# --------------------------------------------------------------------
# Parse a run directory name into (model, seed)
# --------------------------------------------------------------------

# Match e.g.
#   20260523-025157_ann_raw_sd_test_foldD2024...,D2024..._seed0
_RUN_NAME_RE = re.compile(
    r"^\d{8}-\d{6}"               # timestamp
    r"_(?P<model>[a-z0-9_]+?)"    # model name (lazy)
    r"_(?P<features>raw|spectrogram)"  # features tag (helps anchor the model)
    r"_sd_test"
    r"_fold[^_]*"                 # chip list with literal commas, but '_' is the delimiter
    r"(?:,[^_]*)*"                # extra ',chip' components if any
    r"_seed(?P<seed>\d+)$"
)


def parse_run_name(name: str) -> tuple[str, int] | None:
    """Return (model, seed) parsed from a run directory name, or None."""
    # The 'fold' part has chip names that include underscores!  e.g.
    # "...foldD20240719_E03_C44_F4500KHz_U_COV_SD,D20240814_E00_C00_..._PnG_Bead_seed0"
    # so we can't use a simple underscore-splitter.  Anchor on
    # '_seed<digit>$' at the end and on '_<features>_sd_test_fold' in
    # the middle, then take everything between timestamp_ and _<features>
    # as the model.
    m = re.match(
        r"^\d{8}-\d{6}_(.+?)_(raw|spectrogram)_sd_test_fold.+_seed(\d+)$",
        name,
    )
    if not m:
        return None
    return m.group(1), int(m.group(3))


# --------------------------------------------------------------------
# Collect per-run metrics
# --------------------------------------------------------------------

def collect_runs(experiment_dir: Path) -> list[dict]:
    """Walk experiment_dir/runs/* and return one dict per run."""
    out = []
    for run_dir in sorted((experiment_dir / "runs").glob("*")):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_name(run_dir.name)
        if parsed is None:
            print(f"  [skip] unparsable: {run_dir.name}")
            continue
        model, seed = parsed
        json_path = run_dir / "test_metrics.json"
        if not json_path.exists():
            print(f"  [skip] no test_metrics.json: {run_dir.name}")
            continue
        with json_path.open() as fh:
            metrics = json.load(fh)
        out.append({
            "model":      model,
            "seed":       seed,
            "pixel_acc":  metrics["pixel"]["accuracy"],
            "pixel_f1":   metrics["pixel"]["f1"],
            "pixel_auroc": metrics["pixel"]["auroc"],
            "well_acc":   metrics["well"]["accuracy"],
            "run_dir":    run_dir.name,
        })
    return out


def aggregate(runs: list[dict], model: str) -> dict | None:
    """Mean + std over seeds for one (experiment, model) cell."""
    sub = [r for r in runs if r["model"] == model]
    if not sub:
        return None
    pix = [r["pixel_acc"] for r in sub]
    well = [r["well_acc"] for r in sub]
    return {
        "n_seeds":         len(sub),
        "pixel_acc_mean":  statistics.mean(pix),
        "pixel_acc_std":   statistics.stdev(pix) if len(pix) > 1 else 0.0,
        "well_acc_mean":   statistics.mean(well),
        "well_acc_std":    statistics.stdev(well) if len(well) > 1 else 0.0,
    }


# --------------------------------------------------------------------
# Build the summary table
# --------------------------------------------------------------------

def build_summary() -> list[dict]:
    """Return rows: one per (experiment, model)."""
    rows = []
    for exp_name, _ in EXPERIMENTS:
        runs = collect_runs(paths.experiment_dir(exp_name))
        if not runs:
            print(f"  [warn] no runs for {exp_name}")
            continue
        for model in MODELS:
            agg = aggregate(runs, model)
            if agg is None:
                continue
            rows.append({
                "experiment": exp_name,
                "model":      model,
                **agg,
            })
    return rows


def write_summary_csv(rows: list[dict]) -> None:
    import csv
    fields = ["experiment", "model", "n_seeds",
              "pixel_acc_mean", "pixel_acc_std",
              "well_acc_mean", "well_acc_std"]
    with SUMMARY_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {SUMMARY_CSV}")


# --------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------

def plot_comparison(rows: list[dict]) -> None:
    """Two-panel figure:
       (a) Per-model grouped bars, one bar per (model, experiment).
       (b) Per-experiment averaged across models.
    """
    # Re-index by (experiment, model) for quick lookup.
    idx: dict[tuple[str, str], dict] = {
        (r["experiment"], r["model"]): r for r in rows
    }

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 11),
        gridspec_kw={"height_ratios": [1.6, 1.0]},
    )

    # --- Panel A: grouped bars per model ---
    n_exps = len(EXPERIMENTS)
    bar_width = 0.85 / n_exps
    x = np.arange(len(MODELS))

    for i, (exp_name, exp_label) in enumerate(EXPERIMENTS):
        means = []
        stds = []
        for m in MODELS:
            agg = idx.get((exp_name, m))
            means.append(agg["pixel_acc_mean"] if agg else np.nan)
            stds.append(agg["pixel_acc_std"] if agg else 0.0)
        offset = (i - (n_exps - 1) / 2) * bar_width
        bars = ax1.bar(
            x + offset, means, bar_width,
            yerr=stds, capsize=2,
            color=EXP_COLOURS[exp_name],
            label=exp_label.replace("\n", " "),
            edgecolor="white", linewidth=0.4,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in MODELS])
    ax1.set_ylabel("Pixel accuracy on SD test (mean ± std over 3 seeds)")
    ax1.set_title("exp7 preprocessing-filter ablation vs exp6 baseline — per-model results")
    ax1.set_ylim(0.5, 1.0)
    ax1.axhline(0.95, color="grey", linestyle=":", linewidth=0.7, alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8, ncol=5, framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3)

    # --- Panel B: per-experiment averaged across models ---
    exp_means = []
    exp_stds  = []
    exp_labels = []
    for exp_name, exp_label in EXPERIMENTS:
        per_model_means = [idx[(exp_name, m)]["pixel_acc_mean"]
                           for m in MODELS
                           if (exp_name, m) in idx]
        if not per_model_means:
            continue
        exp_means.append(statistics.mean(per_model_means))
        exp_stds.append(
            statistics.stdev(per_model_means) if len(per_model_means) > 1 else 0.0
        )
        exp_labels.append(exp_label)

    xb = np.arange(len(exp_labels))
    colours = [EXP_COLOURS[exp_name] for exp_name, _ in EXPERIMENTS
               if any((exp_name, m) in idx for m in MODELS)]
    ax2.bar(xb, exp_means, yerr=exp_stds, capsize=4,
            color=colours, edgecolor="white")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(exp_labels, fontsize=9)
    ax2.set_ylabel("Pixel accuracy averaged over 5 models")
    ax2.set_title("Pipeline-level comparison (averaged across the 5 well-behaved models)")
    ax2.set_ylim(0.5, 1.0)
    baseline = exp_means[0]
    ax2.axhline(baseline, color="#666666", linestyle="--",
                linewidth=0.8, alpha=0.8,
                label=f"exp6 baseline = {baseline:.4f}")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # Annotate values on top of bars.
    for xi, (m, s) in enumerate(zip(exp_means, exp_stds)):
        ax2.text(xi, m + s + 0.005, f"{m:.4f}",
                 ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(COMPARISON_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {COMPARISON_PNG}")
    plt.close(fig)


# --------------------------------------------------------------------
# Headline printout
# --------------------------------------------------------------------

def print_headline(rows: list[dict]) -> None:
    """Pretty per-experiment summary."""
    idx: dict[tuple[str, str], dict] = {
        (r["experiment"], r["model"]): r for r in rows
    }
    print()
    print(f"{'experiment':<22} {'model':<12} {'pixel_acc':>14} {'well_acc':>14}")
    print("-" * 70)
    for exp_name, _ in EXPERIMENTS:
        print()
        per_model_pix = []
        for m in MODELS:
            agg = idx.get((exp_name, m))
            if agg is None:
                continue
            per_model_pix.append(agg["pixel_acc_mean"])
            print(f"{exp_name:<22} {m:<12} "
                  f"{agg['pixel_acc_mean']:>8.4f} ± {agg['pixel_acc_std']:.4f}  "
                  f"{agg['well_acc_mean']:>8.4f} ± {agg['well_acc_std']:.4f}")
        if per_model_pix:
            avg = statistics.mean(per_model_pix)
            print(f"{exp_name:<22} {'(avg)':<12} "
                  f"{avg:>8.4f}")


def main() -> None:
    rows = build_summary()
    if not rows:
        raise SystemExit("no results found")
    write_summary_csv(rows)
    print_headline(rows)
    plot_comparison(rows)


if __name__ == "__main__":
    main()
