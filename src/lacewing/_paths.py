"""Central path resolution for the lacewing package.

All modules that need to locate the repo root, raw `Data/` folder,
titan clones, or any package-internal directory should import from
here rather than walking `Path(__file__).resolve().parents[N]` on
their own. Two reasons:

  1. `parents[N]` counts break every time the directory tree moves
     (as it did during the `Analysis/` → `src/lacewing/` restructure).
  2. Every path can be overridden with an environment variable so the
     package is portable across dev machines and HPC without editing
     source.

Environment-variable overrides (all optional):

  LACEWING_DATA_ROOT        Parent of the `Data/` folder. Default:
                            repo root (`<pkg-parent>/..`), so raw data
                            is expected at `<repo>/Data/...`.
  LACEWING_TITAN_PATH       Path to the CoV titan clone (main branch).
                            Default: `<data-root>/Code/titan-signal-processing`.
  LACEWING_TITAN_MULTI_PATH Path to the Matthew_Multi titan clone.
                            Default: `<data-root>/Code/titan-signal-processing-multi`.

Legacy layout support: because the release rename `Analysis/` →
`src/lacewing/` was purely a package-name change (subdir structure
preserved), the `LACEWING_PKG_DIR` constant is what old `PROJECT_ROOT
/ "Analysis"` used to point at. New code should use `LACEWING_PKG_DIR`
instead of `PROJECT_ROOT / "Analysis"`.
"""
from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Absolute anchors, computed once at import time.
# ---------------------------------------------------------------------------

# src/lacewing/_paths.py -> src/lacewing/ is parent, src/ is parents[1],
# repo root is parents[2].
LACEWING_PKG_DIR: Path = Path(__file__).resolve().parent
SRC_DIR: Path = LACEWING_PKG_DIR.parent
_REPO_ROOT_DEFAULT: Path = SRC_DIR.parent


def _env_path(var: str, default: Path) -> Path:
    val = os.environ.get(var)
    return Path(val) if val else default


# Data root: where the raw `Data/24_CoV_Quantification/` etc. subtrees live.
# Default: repo root. Override with LACEWING_DATA_ROOT.
DATA_ROOT: Path = _env_path("LACEWING_DATA_ROOT", _REPO_ROOT_DEFAULT)

# Titan clones. Default: under DATA_ROOT/Code/ (matches how the working tree
# has always been laid out). Override with the two LACEWING_TITAN_* vars.
TITAN_PATH: Path = _env_path("LACEWING_TITAN_PATH", DATA_ROOT / "Code" / "titan-signal-processing")
TITAN_MULTI_PATH: Path = _env_path("LACEWING_TITAN_MULTI_PATH", DATA_ROOT / "Code" / "titan-signal-processing-multi")


# ---------------------------------------------------------------------------
# Legacy convenience: everything that used to live under `Analysis/…` now
# lives under `LACEWING_PKG_DIR/…`. Old code paths like
#   PROJECT_ROOT / "Analysis" / "classification" / "results"
# should become
#   LACEWING_PKG_DIR / "classification" / "results"
# ---------------------------------------------------------------------------


def install_titan_on_path() -> bool:
    """Insert TITAN_PATH on sys.path if it exists. Returns True on success."""
    import sys

    if TITAN_PATH.exists():
        if str(TITAN_PATH) not in sys.path:
            sys.path.insert(0, str(TITAN_PATH))
        return True
    return False


def install_titan_multi_on_path() -> bool:
    """Insert TITAN_MULTI_PATH on sys.path if it exists. Returns True on success."""
    import sys

    if not (TITAN_MULTI_PATH / "titan").exists():
        return False
    inside = str(TITAN_MULTI_PATH / "titan")
    parent = str(TITAN_MULTI_PATH)
    if inside not in sys.path:
        sys.path.insert(0, inside)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return True
