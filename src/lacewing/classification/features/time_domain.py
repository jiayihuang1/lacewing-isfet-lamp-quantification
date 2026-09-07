"""Time-domain features = the raw 450-sample trace, untouched."""
from __future__ import annotations

import numpy as np


def transform(X: np.ndarray) -> np.ndarray:
    """Identity. Models 1-6 take the raw trace.

    Returns X unchanged but contiguous + float32 for torch compatibility.
    """
    return np.ascontiguousarray(X, dtype=np.float32)
