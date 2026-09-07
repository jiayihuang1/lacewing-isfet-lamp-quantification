"""Aggregate results/<experiment>/per_method_metrics.csv into a summary table.

Usage:
    python -m lacewing.classification.report                          # default exp
    python -m lacewing.classification.report --experiment exp2_final_chipfold
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from ..core import paths


# Paper 3 Table 2 reference numbers (pixel-level accuracy)
PAPER3_REFERENCE = dict(
    ann=0.5758,
    cnn1d=0.6033,
    fcn=0.5724,
    resnet=0.6967,
    inception=0.6905,
    autoencoder=0.5580,
    cnn2d_spec=0.8484,
)


def summarise(experiment: str = paths.DEFAULT_EXPERIMENT) -> pd.DataFrame:
    csv = paths.metrics_csv(experiment)
    if not csv.exists():
        raise FileNotFoundError(
            f"No results at {csv}. Run a sweep targeting --experiment "
            f"{experiment} first.")
    df = pd.read_csv(csv)

    summary = (df.groupby(["model", "split"])
               .agg(n_runs=("run_id", "count"),
                    pixel_acc_mean=("pixel_acc", "mean"),
                    pixel_acc_std=("pixel_acc", "std"),
                    pixel_f1_mean=("pixel_f1", "mean"),
                    pixel_auroc_mean=("pixel_auroc", "mean"),
                    well_acc_mean=("well_acc", "mean"),
                    n_params_mean=("n_parameters", "mean"),
                    train_time_mean=("train_time_sec", "mean"))
               .reset_index()
               .round(4))
    summary["paper3_acc"] = summary["model"].map(PAPER3_REFERENCE)
    summary["delta_vs_paper3"] = (summary["pixel_acc_mean"] -
                                  summary["paper3_acc"]).round(4)

    out = paths.experiment_dir(experiment) / "summary_table.csv"
    summary.to_csv(out, index=False)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", default=paths.DEFAULT_EXPERIMENT)
    args = p.parse_args()

    try:
        summary = summarise(args.experiment)
    except FileNotFoundError as e:
        print(e); sys.exit(1)
    print(f"\n=== {args.experiment} ===")
    print(summary.to_string(index=False))
    print(f"\nWritten: {paths.experiment_dir(args.experiment) / 'summary_table.csv'}")


if __name__ == "__main__":
    main()
