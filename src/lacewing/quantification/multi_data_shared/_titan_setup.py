"""Pin this package to the Matthew_Multi titan clone.

Import this module before any ``import titan`` happens in this package.
Missing clone is a soft failure — module import succeeds but any function
that actually needs the multi-clone titan will fail on call.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = DATA_ROOT  # was: parents[2]
_env = os.environ.get("LACEWING_TITAN_MULTI_PATH")
TITAN_MULTI_ROOT = Path(_env) if _env else PROJECT_ROOT / "Code" / "titan-signal-processing-multi"


def _install_path() -> None:
    if not (TITAN_MULTI_ROOT / "titan").exists():
        warnings.warn(
            f"titan-signal-processing-multi clone not found at {TITAN_MULTI_ROOT}. "
            "Raw KP-chip processing will fail; analysis of pre-built caches "
            "still works. To enable: `git clone --branch Matthew_Multi "
            "https://github.com/gmat0110/titan-signal-processing.git "
            f"{TITAN_MULTI_ROOT}` and/or set LACEWING_TITAN_MULTI_PATH.",
            stacklevel=2,
        )
        return

    inside_titan = str(TITAN_MULTI_ROOT / "titan")
    parent_of_titan = str(TITAN_MULTI_ROOT)

    if inside_titan not in sys.path:
        sys.path.insert(0, inside_titan)
    if parent_of_titan not in sys.path:
        sys.path.insert(0, parent_of_titan)


_install_path()
