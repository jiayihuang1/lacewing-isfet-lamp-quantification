"""Eval · master scoreboard writer. [Cat A] Emits results/scoreboards/rq3_cov.csv.

Scoreboard CSV: one row per method, keyed by method_id.

Columns (Week 12 rewrite — 4 primaries + 1 filter + 1 sign-check):

  method_id
  config_json
  per_well_mae_min             — accuracy (primary)
  slope_proximity_min_per_decade — calibration (primary)
  slope_min_per_decade         — raw slope (sign-check; qLAMP ref ≈ -3.13)
  per_well_r2                  — correlation quality (primary)
  within_well_cov              — precision (primary)
  spearman_r_well_means        — sanity filter (pass at |ρ| ≥ 0.9)
  passes_spearman_filter       — boolean derived from the above
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lacewing.quantification.eval.resolution_metrics import (
    per_well_mae,
    slope_proximity,
    slope_min_per_decade,
    per_well_r2,
    within_well_cov,
    spearman_r_well_means,
    passes_spearman_filter,
)
from lacewing.quantification.eval.schema import PredictionArrays, LabelArrays


def compute_all_metrics(preds: PredictionArrays, labels: LabelArrays, split: str = "test") -> dict:
    """Compute the full metric portfolio for one (preds, labels) run."""
    rho = spearman_r_well_means(preds, labels, split=split)
    return {
        "per_well_mae_min": per_well_mae(preds, labels, split=split),
        "slope_proximity_min_per_decade": slope_proximity(preds, labels, split=split),
        "slope_min_per_decade": slope_min_per_decade(preds, labels, split=split),
        "per_well_r2": per_well_r2(preds, labels, split=split),
        "within_well_cov": within_well_cov(preds, labels, split=split),
        "spearman_r_well_means": rho,
        "passes_spearman_filter": passes_spearman_filter(rho),
    }


def append_scoreboard_row(
    csv_path: Path,
    method_id: str,
    config_json: dict,
    metrics: dict,
) -> None:
    row = {"method_id": method_id, "config_json": json.dumps(config_json, sort_keys=True)}
    row.update(metrics)
    df_new = pd.DataFrame([row])
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["method_id"] != method_id]
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = df_new
    df.to_csv(csv_path, index=False)


def read_scoreboard(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)
