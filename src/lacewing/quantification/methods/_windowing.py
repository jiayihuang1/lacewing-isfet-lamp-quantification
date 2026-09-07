"""Sliding-window utilities for CPD framings.

Adapts the Paper 18 (Li et al. 2024) sliding-window classifier framing
(§6, Algorithm 1) — slide a window across the trace, classify each window,
aggregate the per-window probabilities into a single change-location.
"""
from __future__ import annotations

import numpy as np


def slide_windows(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Slide a length-`window` window over the last axis of x with `stride`.

    Args:
        x: shape (B, T).
        window: samples per window.
        stride: samples between window starts.

    Returns:
        (B, N_windows, window) where N_windows = (T - window) // stride + 1.
    """
    if x.ndim != 2:
        raise ValueError(f"expected (B, T); got {x.shape}")
    B, T = x.shape
    n_windows = (T - window) // stride + 1
    out = np.empty((B, n_windows, window), dtype=x.dtype)
    for i in range(n_windows):
        s = i * stride
        out[:, i, :] = x[:, s : s + window]
    return out


def window_centres_min(n_windows: int, window: int, stride: int, samples_per_min: int = 15) -> np.ndarray:
    """Centre time of each window in minutes."""
    starts = np.arange(n_windows) * stride
    centres_samples = starts + (window - 1) / 2
    return (centres_samples / samples_per_min).astype(np.float32)


def aggregate_to_ttp_first_positive(
    window_probs: np.ndarray,
    threshold: float,
    k_consecutive: int,
    window: int,
    stride: int,
    samples_per_min: int = 15,
) -> np.ndarray:
    """TTP = first window index where P > threshold for K consecutive windows.

    Args:
        window_probs: shape (B, N_windows), post-amp probability per window.
        threshold: e.g. 0.5.
        k_consecutive: K in the "K consecutive over threshold" rule.

    Returns:
        (B,) TTP estimate in minutes; fallback = centre of last window if
        no K-run of positives exists.
    """
    B, N = window_probs.shape
    centres = window_centres_min(N, window, stride, samples_per_min)
    above = window_probs > threshold  # (B, N)
    ttp = np.full(B, centres[-1], dtype=np.float32)
    for b in range(B):
        run = 0
        for i in range(N):
            if above[b, i]:
                run += 1
                if run >= k_consecutive:
                    ttp[b] = centres[i - k_consecutive + 1]
                    break
            else:
                run = 0
    return ttp


def aggregate_to_ttp_closest_to_zero(
    minutes_until_ttp: np.ndarray,
    window: int,
    stride: int,
    samples_per_min: int = 15,
) -> np.ndarray:
    """TTP = centre of the window whose predicted minutes-until-TTP is closest to zero."""
    B, N = minutes_until_ttp.shape
    centres = window_centres_min(N, window, stride, samples_per_min)
    idx = np.argmin(np.abs(minutes_until_ttp), axis=1)  # (B,)
    return centres[idx]
