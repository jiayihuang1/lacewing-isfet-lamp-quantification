"""Run-directory + structured-logging utilities for classification experiments.

A "run" is one (model, features, split, fold, seed) execution. Each run lives
in its own directory under results/runs/<run_id>/ containing:

  config.yaml          - exact training config (also logged by train.py)
  metrics.csv          - per-epoch train/val metrics
  test_metrics.json    - final test metrics
  predictions.npz      - test-set predictions + truth + pixel/well IDs
  checkpoints/best.pt  - lowest-val-loss checkpoint
  checkpoints/last.pt  - final-epoch checkpoint
  train.log            - full text log
  git_state.txt        - git SHA + modified files at run start
  env.txt              - pip freeze + python version

The aggregate index at results/per_method_metrics.csv has one row per
completed run for cross-run analysis.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from . import paths


def make_run_id(model: str, features: str, split: str, fold: str | int,
                seed: int, tag: str | None = None) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{ts}_{model}_{features}_{split}_fold{fold}_seed{seed}"
    return f"{base}_{tag}" if tag else base


def make_run_dir(run_id: str, experiment: str = paths.DEFAULT_EXPERIMENT
                 ) -> Path:
    run_dir = paths.runs_dir(experiment) / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logger(run_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """File + stream logger keyed to a run directory."""
    logger = logging.getLogger(f"run.{run_dir.name}")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_h = logging.FileHandler(run_dir / "train.log", mode="w", encoding="utf-8")
    file_h.setFormatter(fmt)
    logger.addHandler(file_h)

    stream_h = logging.StreamHandler(sys.stdout)
    stream_h.setFormatter(fmt)
    logger.addHandler(stream_h)

    logger.propagate = False
    return logger


def save_config(run_dir: Path, config: dict[str, Any]) -> None:
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def snapshot_git_state(run_dir: Path) -> None:
    """Write current git SHA + dirty-file list. Best-effort only."""
    out = run_dir / "git_state.txt"
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=paths.PROJECT_ROOT, text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=paths.PROJECT_ROOT, text=True,
        )
        out.write_text(f"sha: {sha}\n\nmodified:\n{status}", encoding="utf-8")
    except Exception as e:
        out.write_text(f"git snapshot failed: {e}\n", encoding="utf-8")


def snapshot_env(run_dir: Path) -> None:
    """Write python version + pip freeze. Best-effort only."""
    out = run_dir / "env.txt"
    lines = [f"python: {sys.version}", f"platform: {platform.platform()}", ""]
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True,
        )
        lines.append(freeze)
    except Exception as e:
        lines.append(f"pip freeze failed: {e}")
    out.write_text("\n".join(lines), encoding="utf-8")


def append_aggregate_row(row: dict[str, Any],
                         experiment: str = paths.DEFAULT_EXPERIMENT) -> None:
    """Append one completed-run row to this experiment's aggregate CSV."""
    csv_path = paths.metrics_csv(experiment)
    write_header = not csv_path.exists()
    keys = list(row.keys())
    with open(csv_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(keys) + "\n")
        f.write(",".join(str(row[k]) for k in keys) + "\n")


def save_test_metrics(run_dir: Path, metrics: dict[str, Any]) -> None:
    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
