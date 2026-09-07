"""Windowing + aggregation utility tests."""
from __future__ import annotations

import numpy as np
import pytest

from lacewing.quantification.methods._windowing import (
    slide_windows,
    window_centres_min,
    aggregate_to_ttp_first_positive,
    aggregate_to_ttp_closest_to_zero,
)


def test_slide_windows_shape():
    x = np.zeros((4, 450), dtype=np.float32)
    w = slide_windows(x, window=60, stride=5)
    # (450 - 60) / 5 + 1 = 79 windows
    assert w.shape == (4, 79, 60)


def test_slide_windows_content():
    x = np.arange(20, dtype=np.float32)[None, :]  # (1, 20)
    w = slide_windows(x, window=5, stride=5)
    assert w.shape == (1, 4, 5)
    np.testing.assert_array_equal(w[0, 0], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(w[0, 1], [5, 6, 7, 8, 9])


def test_window_centres_min():
    c = window_centres_min(n_windows=79, window=60, stride=5, samples_per_min=15)
    assert c.shape == (79,)
    # First window centre: (60 - 1) / 2 = 29.5 samples = 29.5/15 min
    assert c[0] == pytest.approx((60 - 1) / 2 / 15, rel=1e-4)


def test_aggregate_to_ttp_first_positive():
    # 4 windows, threshold 0.5, k=2 consecutive → first pos-and-next-also-pos
    probs = np.array([[0.1, 0.4, 0.6, 0.7]])
    ttp = aggregate_to_ttp_first_positive(
        probs, threshold=0.5, k_consecutive=2,
        window=5, stride=5, samples_per_min=15,
    )
    # First run of >=2 consecutive above-threshold starts at window index 2.
    # Window 2 centre = (5-1)/2 + 2*5 = 12 samples → 12/15 min = 0.8 min
    assert ttp.shape == (1,)
    assert ttp[0] == pytest.approx(12 / 15, rel=1e-4)


def test_aggregate_to_ttp_first_positive_no_positives():
    probs = np.array([[0.1, 0.2, 0.3, 0.4]])
    ttp = aggregate_to_ttp_first_positive(
        probs, threshold=0.5, k_consecutive=2,
        window=5, stride=5, samples_per_min=15,
    )
    # Fallback: last window centre.
    assert np.isfinite(ttp[0])


def test_aggregate_to_ttp_closest_to_zero():
    # 4 windows with predicted "minutes until TTP" — window closest to 0 wins.
    mins = np.array([[5.0, 3.0, -0.5, -2.0]])
    ttp = aggregate_to_ttp_closest_to_zero(
        mins, window=5, stride=5, samples_per_min=15,
    )
    # argmin(|mins|) = 2. Window 2 centre = 12 samples → 12/15 min.
    assert ttp[0] == pytest.approx(12 / 15, rel=1e-4)
