"""Project paths for the regression strand.

Sibling of Analysis/classification/core/paths.py.  Reuses the
classification cache as the source for per-pixel signal arrays (the
filtered MAD k=1.5 ABCD ntcRaw cache is the locked preprocessing),
then attaches qLAMP-derived TTP labels per pixel.
"""
from __future__ import annotations

from pathlib import Path
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT

# src/lacewing/quantification/regression_shared/core/paths.py
# -> package src/lacewing/ is parents[3]
LACEWING_PKG_DIR = LACEWING_PKG_DIR  # was: parents[3]
# Reuse the classification cache dir directly; we read the existing
# locked-preprocessing cache from there.
CLASSIFICATION_CACHE_DIR = (
    LACEWING_PKG_DIR / "classification" / "data" / "cache"
)

# Regression-side cache (relabelled with per-pixel TTP).
REGRESSION_CACHE_DIR = (
    LACEWING_PKG_DIR / "quantification" / "regression_shared" / "data" / "cache"
)
REGRESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Locked preprocessing config (chosen 2026-06-22 with supervisor).
LOCKED_CACHE_STEM = "dataset_all_filt_abcd_ntcRaw_madk1p5"


def classification_cache_path(stem: str = LOCKED_CACHE_STEM) -> Path:
    return CLASSIFICATION_CACHE_DIR / f"{stem}.npz"


def regression_cache_path(stem: str) -> Path:
    return REGRESSION_CACHE_DIR / f"{stem}.npz"


RESULTS_DIR = (
    LACEWING_PKG_DIR / "quantification" / "methods" / "results"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def experiment_dir(experiment: str) -> Path:
    return RESULTS_DIR / experiment
