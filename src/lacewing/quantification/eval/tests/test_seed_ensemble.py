"""Seed-ensemble smoke tests."""
import numpy as np
import pytest

from lacewing.quantification.eval.seed_ensemble import (
    average_arrays_across_seeds,
)


def test_average_two_arrays() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    out = average_arrays_across_seeds([a, b])
    expected = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)


def test_average_single_array_is_identity() -> None:
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    out = average_arrays_across_seeds([a])
    np.testing.assert_allclose(out, a)


def test_shape_mismatch_raises() -> None:
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        average_arrays_across_seeds([a, b])
