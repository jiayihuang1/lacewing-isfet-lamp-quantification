"""Project path resolution for the classification module.

Walks parent directories from this file to locate the project root, then
exposes canonical paths for data, titan, results, and cached datasets.

Importing this module also inserts the titan package onto sys.path so
`from titan...` works regardless of cwd.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
CORE_DIR = THIS_FILE.parent
CLASSIFICATION_DIR = CORE_DIR.parent
LACEWING_PKG_DIR = CLASSIFICATION_DIR.parent
# src/lacewing/classification/core/paths.py -> repo root is parents[4]
_REPO_ROOT_DEFAULT = THIS_FILE.parents[4]

_env_data = os.environ.get("LACEWING_DATA_ROOT")
PROJECT_ROOT = Path(_env_data) if _env_data else _REPO_ROOT_DEFAULT

_env_titan = os.environ.get("LACEWING_TITAN_PATH")
TITAN_PKG = Path(_env_titan) if _env_titan else PROJECT_ROOT / "Code" / "titan-signal-processing"
if TITAN_PKG.exists():
    if str(TITAN_PKG) not in sys.path:
        sys.path.insert(0, str(TITAN_PKG))
else:
    warnings.warn(
        f"titan-signal-processing not found at {TITAN_PKG}. Raw-chip processing "
        "and cache-building will fail; downstream analysis of pre-built caches "
        "will still work. Clone titan-signal-processing separately and set "
        "LACEWING_TITAN_PATH to its directory to enable full functionality.",
        stacklevel=2,
    )

DATA_ROOT = PROJECT_ROOT / "Data" / "24_CoV_Quantification"
OTHERS_DIR = DATA_ROOT / "Others"

CLASSIFICATION_RESULTS = CLASSIFICATION_DIR / "results"
DATASET_CACHE_DIR = CLASSIFICATION_DIR / "data" / "cache"
NTC_INVENTORY_CSV = CLASSIFICATION_DIR / "data" / "ntc_inventory.csv"

CLASSIFICATION_RESULTS.mkdir(exist_ok=True)
DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-experiment paths. Every sweep targets a named "experiment" so its
# runs, aggregate CSV, and plots stay in one folder, cleanly separated
# from other experiments.
# ---------------------------------------------------------------------------

DEFAULT_EXPERIMENT = "exp1_final_random"


def experiment_dir(experiment: str = DEFAULT_EXPERIMENT) -> Path:
    d = CLASSIFICATION_RESULTS / experiment
    (d / "runs").mkdir(parents=True, exist_ok=True)
    return d


def runs_dir(experiment: str = DEFAULT_EXPERIMENT) -> Path:
    return experiment_dir(experiment) / "runs"


def metrics_csv(experiment: str = DEFAULT_EXPERIMENT) -> Path:
    return experiment_dir(experiment) / "per_method_metrics.csv"


def cache_path(name: str = "dataset_per_pixel") -> Path:
    return DATASET_CACHE_DIR / f"{name}.npz"


# 5 Final-dataset chips (mirrors Analysis/quantification/quantification.py)
FINAL_CHIPS: dict[str, Path] = {
    "1e5": DATA_ROOT / "1e5" / "D20240821_E01_C07_F4500KHz_U_1e5",
    "1e6": DATA_ROOT / "1e6" / "D20240821_E01_C09_F4500KHz_U_1e6",
    "1e7": DATA_ROOT / "1e7" / "D20240820_E02_C03_F4500KHz_U_1e7",
    "1e8": DATA_ROOT / "1e8" / "D20240822_E02_C00_F4500KHz_U_1e8",
    "1e9": DATA_ROOT / "1e9" / "D20240822_E02_C00_F4500KHz_U_1e9",
}

# Standard 6-well layout (per supervisor's chip diagram, 2026-05-05):
#   wells 0-3: positive (E1-R1, E1-R2, E2-R1, E2-R2)
#   well 4:    PTC (positive template control - positive)
#   well 5:    NTC (no template control - negative)
POSITIVE_WELL_INDICES = (0, 1, 2, 3, 4)
NTC_WELL_INDEX = 5
N_WELLS = 6
