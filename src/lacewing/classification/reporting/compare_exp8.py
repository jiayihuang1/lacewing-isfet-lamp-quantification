"""exp8 reporting: aggregate per-(experiment, model) pixel accuracies
across seeds, both all-10 and best-3-of-10 (selected by lowest final
train_loss), and plot accuracy vs k.

Run::

    python -m lacewing.classification.reporting.compare_exp8
"""
from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..core import paths


K_VALUES = [1.5, 1.645, 2.0, 2.5, 3.0, 3.5]

# Layer-progression sub-sweep (extends exp8 with the {A, AB, ABC}
# subsets at k in {1.5, 1.645, 2.0}).  ABCD configs for those k values
# are already in the ABCD-only sweep above.
LAYERS_PROGRESSION = ["A", "AB", "ABC", "ABCD"]
K_VALUES_PROGRESSION = [1.5, 1.645, 2.0]

EXPERIMENTS = (
    [(f"exp8_ABCD_ntcRaw_madk{k}", f"ABCD k={k}") for k in K_VALUES] +
    [(f"exp8_{L}_ntcRaw_madk{k}", f"{L} k={k}")
     for k in K_VALUES_PROGRESSION
     for L in LAYERS_PROGRESSION
     if L != "ABCD"]  # ABCD@k in {1.5, 1.645, 2.0} already listed above
)
# Optional baselines for comparison
BASELINES = [
    ("exp6_final_to_sd", "exp6 baseline\n(no filter)"),
    ("exp7_ABCD_ntcRaw", "exp7 ABCD ntcRaw\n(percentile, k=1.645 equiv)"),
]

MODELS = ["ann", "cnn1d", "fcn", "resnet", "inception"]

MODEL_COLOURS = {
    "ann":       "#1f78b4",
    "cnn1d":     "#33a02c",
    "fcn":       "#ff7f00",
    "resnet":    "#e31a1c",
    "inception": "#6a3d9a",
}


def parse_run_name(name: str) -> tuple[str, int] | None:
    m = re.match(
        r"^\d{8}-\d{6}_(.+?)_(?:raw|spectrogram)_sd_test_fold.+_seed(\d+)$",
        name,
    )
    if not m:
        return None
    return m.group(1), int(m.group(2))


def final_train_loss(run_dir: Path) -> float | None:
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        return None
    last = None
    with csv_path.open() as fh:
        r = csv.DictReader(fh)
        for row in r:
            last = row
    if last is None:
        return None
    try:
        return float(last["train_loss"])
    except (KeyError, ValueError):
        return None


def collect_runs(experiment_dir: Path) -> list[dict]:
    out = []
    runs_dir = experiment_dir / "runs"
    if not runs_dir.exists():
        return out
    for run_dir in sorted(runs_dir.glob("*")):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_name(run_dir.name)
        if parsed is None:
            continue
        model, seed = parsed
        json_path = run_dir / "test_metrics.json"
        if not json_path.exists():
            continue
        with json_path.open() as fh:
            metrics = json.load(fh)
        out.append({
            "model":      model,
            "seed":       seed,
            "pixel_acc":  metrics["pixel"]["accuracy"],
            "well_acc":   metrics["well"]["accuracy"],
            "final_loss": final_train_loss(run_dir),
            "run_dir":    run_dir.name,
        })
    return out


def aggregate_all(runs: list[dict], model: str) -> dict | None:
    sub = [r for r in runs if r["model"] == model]
    if not sub:
        return None
    pix = [r["pixel_acc"] for r in sub]
    return {
        "n_seeds":         len(sub),
        "pixel_acc_mean":  statistics.mean(pix),
        "pixel_acc_std":   statistics.stdev(pix) if len(pix) > 1 else 0.0,
    }


def aggregate_best3(runs: list[dict], model: str) -> dict | None:
    sub = [r for r in runs if r["model"] == model]
    if not sub:
        return None
    with_loss = [r for r in sub if r["final_loss"] is not None]
    if len(with_loss) >= 3:
        picked = sorted(with_loss,
                        key=lambda r: (r["final_loss"], r["seed"]))[:3]
        n_pick = 3
    else:
        picked = sub
        n_pick = len(sub)
    pix = [r["pixel_acc"] for r in picked]
    return {
        "n_seeds":         n_pick,
        "n_available":     len(sub),
        "pixel_acc_mean":  statistics.mean(pix),
        "pixel_acc_std":   statistics.stdev(pix) if len(pix) > 1 else 0.0,
        "picked_seeds":    sorted(r["seed"] for r in picked),
    }


def build_summary() -> list[dict]:
    rows = []
    for exp_name, _ in EXPERIMENTS + BASELINES:
        runs = collect_runs(paths.experiment_dir(exp_name))
        for model in MODELS:
            a_all   = aggregate_all(runs, model)
            a_best3 = aggregate_best3(runs, model)
            if a_all is None or a_best3 is None:
                continue
            rows.append({
                "experiment":           exp_name,
                "model":                model,
                "n_all":                a_all["n_seeds"],
                "all_pixel_acc_mean":   a_all["pixel_acc_mean"],
                "all_pixel_acc_std":    a_all["pixel_acc_std"],
                "n_best3":              a_best3["n_seeds"],
                "best3_pixel_acc_mean": a_best3["pixel_acc_mean"],
                "best3_pixel_acc_std":  a_best3["pixel_acc_std"],
                "picked_seeds":         a_best3["picked_seeds"],
            })
    return rows


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    fields = ["experiment", "model",
              "n_all",   "all_pixel_acc_mean",   "all_pixel_acc_std",
              "n_best3", "best3_pixel_acc_mean", "best3_pixel_acc_std",
              "picked_seeds"]
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["picked_seeds"] = ",".join(str(s) for s in r["picked_seeds"])
            w.writerow(r)
    print(f"wrote {out_path}")


EXP_COLOURS = {
    "exp6_final_to_sd":           "#666666",
    "exp8_ABCD_ntcRaw_madk1.5":   "#a6cee3",
    "exp8_ABCD_ntcRaw_madk1.645": "#1f78b4",
    "exp8_ABCD_ntcRaw_madk2.0":   "#b2df8a",
    "exp8_ABCD_ntcRaw_madk2.5":   "#33a02c",
    "exp8_ABCD_ntcRaw_madk3.0":   "#fdbf6f",
    "exp8_ABCD_ntcRaw_madk3.5":   "#ff7f00",
}


def plot_side_by_side(rows: list[dict], out_path: Path) -> None:
    """Grouped bar chart, exp7-style: x-axis = models, inner groups
    = unfiltered exp6 baseline + the 6 MAD-k configurations.

    The exp7 ABCD,ntcRaw (percentile) locked configuration is drawn
    as a horizontal reference line per panel so the MAD vs
    percentile comparison is visible at a glance.
    """
    idx = {(r["experiment"], r["model"]): r for r in rows}

    exp_list = [
        ("exp6_final_to_sd",           "exp6 baseline (no filter)"),
    ] + [
        (f"exp8_ABCD_ntcRaw_madk{k}",  f"MAD k={k}")
        for k in K_VALUES
    ]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(MODELS))
    n_exps = len(exp_list)
    bar_width = 0.85 / n_exps

    for i, (exp_name, exp_label) in enumerate(exp_list):
        means, stds = [], []
        for m in MODELS:
            r = idx.get((exp_name, m))
            means.append(100 * r["best3_pixel_acc_mean"] if r else np.nan)
            stds.append(100 * r["best3_pixel_acc_std"] if r else 0.0)
        offset = (i - (n_exps - 1) / 2) * bar_width
        ax.bar(
            x + offset, means, bar_width,
            yerr=stds, capsize=2,
            color=EXP_COLOURS.get(exp_name, "#888888"),
            label=exp_label,
            edgecolor="white", linewidth=0.4,
        )
        # Label each bar with its value, drawn INSIDE the bar
        # (vertical text, near the top) so we save vertical space
        # and the number stays attached to its bar.  Black bold text
        # stays legible against every bar colour in the palette
        # (the white labels we tried first vanished on pale bars).
        for xi, mean in zip(x + offset, means):
            if not np.isnan(mean):
                ax.text(xi, mean - 0.5, f"{mean:.1f}",
                        ha="center", va="top",
                        fontsize=13, rotation=90, color="#000000",
                        fontweight="bold")

    # exp7 ABCD,ntcRaw (percentile) reference horizontal line
    exp7_means = [
        idx[("exp7_ABCD_ntcRaw", m)]["best3_pixel_acc_mean"]
        for m in MODELS if ("exp7_ABCD_ntcRaw", m) in idx
    ]
    if exp7_means:
        exp7_meanmean = 100 * statistics.mean(exp7_means)
        ax.axhline(exp7_meanmean, color="#d62728", linestyle="--",
                   linewidth=1.2,
                   label=f"exp7 ABCD,ntcRaw (percentile): {exp7_meanmean:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in MODELS])
    ax.set_ylabel("Pixel accuracy on SD test (%)")
    ax.set_title("exp8 — MAD-based threshold sweep (ABCD, ntcRaw): "
                 "per-model bars across $k$, exp6 unfiltered baseline + "
                 "exp7 percentile baseline shown")
    ax.set_ylim(50, 100)
    ax.axhline(95, color="grey", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=7, ncol=4, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")
    plt.close(fig)


LAYER_COLOURS = {
    "exp6_baseline": "#666666",
    "A":             "#a6cee3",
    "AB":            "#b2df8a",
    "ABC":           "#fdbf6f",
    "ABCD":          "#fb9a99",
}

PANEL_LETTERS = ["a", "b", "c", "d", "e", "f"]


def plot_layer_progression(rows: list[dict], out_path: Path) -> None:
    """exp7-style grouped-bar plot, but with one panel per MAD k value.

    Inside each panel: x-axis = 5 models, inner groups = exp6 baseline
    + {A, AB, ABC, ABCD}.  Shows whether at each k the layer
    progression A -> AB -> ABC -> ABCD monotonically helps (the
    question the percentile-rule exp7 ablation raised for FCN/ResNet).
    """
    idx = {(r["experiment"], r["model"]): r for r in rows}

    n_panels = len(K_VALUES_PROGRESSION)
    # Match exp7's figsize: 13 × 5 per panel (exp7 uses 13×10 for 2 panels).
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 5.0 * n_panels))
    if n_panels == 1:
        axes = [axes]

    for panel_i, (ax, k) in enumerate(zip(axes, K_VALUES_PROGRESSION)):
        # exp6 baseline + 4 layer subsets at this k
        entries: list[tuple[str, str, str]] = [
            ("exp6_final_to_sd", "exp6_baseline", "exp6 baseline (no filter)"),
        ]
        for L in LAYERS_PROGRESSION:
            exp_name = f"exp8_{L}_ntcRaw_madk{k}"
            entries.append((exp_name, L, L))

        x = np.arange(len(MODELS))
        n_groups = len(entries)
        bar_width = 0.85 / n_groups

        for i, (exp_name, colour_key, label) in enumerate(entries):
            means, stds = [], []
            for m in MODELS:
                r = idx.get((exp_name, m))
                means.append(100 * r["best3_pixel_acc_mean"] if r else np.nan)
                stds.append(100 * r["best3_pixel_acc_std"] if r else 0.0)
            offset = (i - (n_groups - 1) / 2) * bar_width
            ax.bar(
                x + offset, means, bar_width,
                yerr=stds, capsize=2,
                color=LAYER_COLOURS.get(colour_key, "#888888"),
                label=label,
                edgecolor="white", linewidth=0.4,
            )
            # Value label INSIDE each bar (vertical, near the top) so
            # the per-bar number is visible without adding vertical
            # whitespace above the bar.  Black bold so it stays legible
            # against every bar colour in the palette.
            for xi, mean in zip(x + offset, means):
                if not np.isnan(mean):
                    ax.text(xi, mean - 0.5, f"{mean:.1f}",
                            ha="center", va="top",
                            fontsize=13, rotation=90, color="#000000",
                            fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in MODELS])
        ax.set_ylabel("Pixel accuracy on SD test (%)")
        ax.set_title(f"({PANEL_LETTERS[panel_i]}) MAD k={k}",
                     fontsize=11, loc="left")
        ax.set_ylim(50, 100)
        ax.axhline(95, color="grey", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, ncol=n_groups, framealpha=0.9)

    fig.suptitle("exp8 — best-3-seeds per cell (selected by lowest final train_loss)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")
    plt.close(fig)


def print_headline(rows: list[dict]) -> None:
    idx = {(r["experiment"], r["model"]): r for r in rows}
    print()
    print(f"{'experiment':<32} {'model':<10}  "
          f"{'all (mean±std, n)':<22}  "
          f"{'best3 (mean±std, n)':<22}  delta")
    print("-" * 110)
    for exp_name, _ in EXPERIMENTS + BASELINES:
        for model in MODELS:
            r = idx.get((exp_name, model))
            if r is None:
                continue
            a  = f"{r['all_pixel_acc_mean']:.4f} ± {r['all_pixel_acc_std']:.4f} ({r['n_all']:>2})"
            b  = f"{r['best3_pixel_acc_mean']:.4f} ± {r['best3_pixel_acc_std']:.4f} ({r['n_best3']})"
            d  = r['best3_pixel_acc_mean'] - r['all_pixel_acc_mean']
            print(f"{exp_name:<32} {model:<10}  {a:<22}  {b:<22}  {d:+.4f}")
        rows_e = [idx[(exp_name, m)] for m in MODELS if (exp_name, m) in idx]
        if rows_e:
            avg_all = statistics.mean(r["all_pixel_acc_mean"]   for r in rows_e)
            avg_b3  = statistics.mean(r["best3_pixel_acc_mean"] for r in rows_e)
            print(f"{exp_name:<32} {'(avg)':<10}  {avg_all:.4f}                  "
                  f"{avg_b3:.4f}                 {avg_b3-avg_all:+.4f}")
            print()


def main() -> None:
    rows = build_summary()
    if not rows:
        raise SystemExit("no results found")
    csv_path  = paths.CLASSIFICATION_RESULTS / "exp8_best3_summary.csv"
    png_path  = paths.CLASSIFICATION_RESULTS / "exp8_best3_comparison.png"
    layer_png = paths.CLASSIFICATION_RESULTS / "exp8_layer_progression.png"
    write_summary_csv(rows, csv_path)
    print_headline(rows)
    plot_side_by_side(rows, png_path)
    plot_layer_progression(rows, layer_png)


if __name__ == "__main__":
    main()
