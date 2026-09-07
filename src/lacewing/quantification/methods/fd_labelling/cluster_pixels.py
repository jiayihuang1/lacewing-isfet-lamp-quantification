"""Per-well k-means clustering of pixel traces (shape-based).

Used by the F-D manual-supervised segmentation labelling pipeline to pick
representative traces per well: cluster pixels into k groups, then label
one representative per group, then broadcast the label to all pixels in
that group.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


BASELINE_WINDOW_SAMPLES = 60  # first ~4 min at SAMPLES_PER_MIN=15


def _normalise_trace(traces: np.ndarray) -> np.ndarray:
    """Per-trace baseline-subtract + amplitude-scale.

    baseline = mean of first BASELINE_WINDOW_SAMPLES.
    Divide by max(|centered|) so shape (not level, not amplitude) drives
    clustering. Guard against constant traces (max ~= 0).
    """
    baseline = traces[:, :BASELINE_WINDOW_SAMPLES].mean(axis=1, keepdims=True)
    centered = traces - baseline
    max_abs = np.abs(centered).max(axis=1, keepdims=True)
    max_abs = np.where(max_abs < 1e-6, 1.0, max_abs)  # avoid division by zero
    return (centered / max_abs).astype(np.float32)


def cluster_well(traces: np.ndarray, k: int = 4, random_state: int = 0) -> np.ndarray:
    """Cluster pixel traces of one well into k shape-based groups.

    Parameters
    ----------
    traces : (n_pixels, T) float
        Raw ISFET traces for one well.
    k : int
        Number of clusters. Must be <= n_pixels.
    random_state : int
        For KMeans reproducibility.

    Returns
    -------
    (n_pixels,) int32
        Cluster ids in [0, k).

    Raises
    ------
    ValueError
        If n_pixels < k.
    """
    n_pixels = traces.shape[0]
    if n_pixels < k:
        raise ValueError(
            f"well has fewer pixels than clusters (n_pixels={n_pixels}, k={k})"
        )
    features = _normalise_trace(traces)
    # n_init="auto" is sklearn 1.4+ (present in your venv).
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    ids = km.fit_predict(features)
    return ids.astype(np.int32)
