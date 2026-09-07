"""STFT spectrogram features per Paper 3 §IV.B.

Per-pixel pipeline:
    nperseg  = 22                  # window length
    noverlap = 11                  # 50% overlap
    window   = 'hann'              # Hanning weighting
    Output of scipy.signal.spectrogram on a 450-sample input is
       12 frequency bins x 39 time windows.
    Crop to the lowest 10 frequency bins (high-frequency = noise for
    DNA amplification) -> final feature shape (10, 39).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import spectrogram


NPERSEG = 22
NOVERLAP = 11
WINDOW = "hann"
N_FREQ_BINS_KEPT = 10   # crop highest 2 of 12 bins


def transform(X: np.ndarray) -> np.ndarray:
    """X: (N, 450) -> (N, 10, 39) float32.

    Each row is one pixel; STFT applied per row.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D (N, T), got {X.shape}")

    f, t, Sxx = spectrogram(
        X, fs=1.0, nperseg=NPERSEG, noverlap=NOVERLAP, window=WINDOW, axis=1,
    )
    # Sxx shape: (N, n_freq, n_time)
    Sxx = Sxx[:, :N_FREQ_BINS_KEPT, :]
    return np.ascontiguousarray(Sxx, dtype=np.float32)
