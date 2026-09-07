"""Side-by-side comparison of exp6/exp7 aggregated by

  (a) all 10 seeds  (the headline number)
  (b) best 3 seeds (lowest final train_loss per cell)

Selecting best-3-of-10 introduces selection bias and should not be
the headline result.  This script exists to make the magnitude of
the cnn1d collapse explicit: comparing (a) vs (b) shows how much
the headline is dragged down by collapsed seeds.

For non-cnn1d models, (a) and (b) should be very similar because
all 10 seeds tend to converge.  For cnn1d, (b) drops the
non-trained seeds and shows the model's potential when it does
work.

Run::

    python -m lacewing.classification.reporting.compare_exp7_best3
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
    """Per-run dicts including pixel_acc + final_train_loss."""
    out = []
    for run_dir in sorted((experiment_dir / "runs").glob("*")):
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
    """Pick the 3 seeds with the lowest final train_loss.  Ties are
    broken by seed number.  If fewer than 3 seeds have a usable
    final_loss, fall back to all available seeds and flag it."""
    sub = [r for r in runs if r["model"] == model]
    if not sub:
        return None
    with_loss = [r for r in sub if r["final_loss"] is not None]
    if len(with_loss) >= 3:
        picked = sorted(with_loss, key=lambda r: (r["final_loss"], r["seed"]))[:3]
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
    for exp_name, _ in EXPERIMENTS:
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


def plot_side_by_side(rows: list[dict], out_path: Path) -> None:
    """Two-row best-3 bar chart: top = ntcRaw variants, bottom = ntcD variants.

    The exp6 baseline (no ntc filter) is shown on both panels for reference."""
    idx = {(r["experiment"], r["model"]): r for r in rows}

    ntc_raw_exps = [e for e in EXPERIMENTS
                    if e[0] == "exp6_final_to_sd" or e[0].endswith("_ntcRaw")]
    ntc_d_exps   = [e for e in EXPERIMENTS
                    if e[0] == "exp6_final_to_sd" or e[0].endswith("_ntcD")]

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    x = np.arange(len(MODELS))

    for ax, panel_exps, panel_label in [
        (axes[0], ntc_raw_exps, "(a) ntcRaw variants"),
        (axes[1], ntc_d_exps,   "(b) ntcD variants"),
    ]:
        n_exps = len(panel_exps)
        bar_width = 0.85 / n_exps
        for i, (exp_name, exp_label) in enumerate(panel_exps):
            means = []
            stds = []
            for m in MODELS:
                r = idx.get((exp_name, m))
                means.append(100 * r["best3_pixel_acc_mean"] if r else np.nan)
                stds.append(100 * r["best3_pixel_acc_std"] if r else 0.0)
            offset = (i - (n_exps - 1) / 2) * bar_width
            ax.bar(
                x + offset, means, bar_width,
                yerr=stds, capsize=2,
                color=EXP_COLOURS[exp_name],
                label=exp_label.replace("\n", " "),
                edgecolor="white", linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in MODELS])
        ax.set_ylabel("Pixel accuracy on SD test (%)")
        ax.set_title(panel_label, fontsize=11, loc="left")
        ax.set_ylim(50, 100)
        ax.axhline(95, color="grey", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, ncol=len(panel_exps), framealpha=0.9)

    fig.suptitle("exp7 — best-3-seeds per cell (selected by lowest final train_loss)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")
    plt.close(fig)


def print_headline(rows: list[dict]) -> None:
    idx = {(r["experiment"], r["model"]): r for r in rows}
    print()
    print(f"{'experiment':<22} {'model':<10}  "
          f"{'all (mean±std, n)':<22}  "
          f"{'best3 (mean±std, n)':<22}  delta")
    print("-" * 100)
    for exp_name, _ in EXPERIMENTS:
        for model in MODELS:
            r = idx.get((exp_name, model))
            if r is None:
                continue
            a  = f"{r['all_pixel_acc_mean']:.4f} ± {r['all_pixel_acc_std']:.4f} ({r['n_all']:>2})"
            b  = f"{r['best3_pixel_acc_mean']:.4f} ± {r['best3_pixel_acc_std']:.4f} ({r['n_best3']})"
            d  = r['best3_pixel_acc_mean'] - r['all_pixel_acc_mean']
            print(f"{exp_name:<22} {model:<10}  {a:<22}  {b:<22}  {d:+.4f}")
        # avg row across models, both columns
        rows_e = [idx[(exp_name, m)] for m in MODELS if (exp_name, m) in idx]
        if rows_e:
            avg_all = statistics.mean(r["all_pixel_acc_mean"]   for r in rows_e)
            avg_b3  = statistics.mean(r["best3_pixel_acc_mean"] for r in rows_e)
            print(f"{exp_name:<22} {'(avg)':<10}  {avg_all:.4f}                  {avg_b3:.4f}                 {avg_b3-avg_all:+.4f}")
            print()


def main() -> None:
    rows = build_summary()
    if not rows:
        raise SystemExit("no results found")
    csv_path = paths.CLASSIFICATION_RESULTS / "exp7_best3_summary.csv"
    png_path = paths.CLASSIFICATION_RESULTS / "exp7_best3_comparison.png"
    write_summary_csv(rows, csv_path)
    print_headline(rows)
    plot_side_by_side(rows, png_path)


if __name__ == "__main__":
    main()
