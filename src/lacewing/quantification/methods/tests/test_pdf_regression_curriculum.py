"""Curriculum-σ annealing: σ decreases from sigma_start to sigma_end."""
import numpy as np
import pytest

from lacewing.quantification.methods.pdf_regression import curriculum_sigma_at


def test_first_epoch_matches_sigma_start() -> None:
    sigma = curriculum_sigma_at(epoch=1, total_epochs=10, sigma_start=12.0, sigma_end=3.0)
    assert sigma == pytest.approx(12.0)


def test_last_epoch_matches_sigma_end() -> None:
    sigma = curriculum_sigma_at(epoch=10, total_epochs=10, sigma_start=12.0, sigma_end=3.0)
    assert sigma == pytest.approx(3.0)


def test_middle_epoch_is_linear_interpolation() -> None:
    # epoch 5 of 10 → linearly 4/9 of the way from 12 to 3 → 12 - 4/9 * 9 = 8.0
    sigma = curriculum_sigma_at(epoch=5, total_epochs=10, sigma_start=12.0, sigma_end=3.0)
    assert sigma == pytest.approx(8.0)


def test_single_epoch_returns_end() -> None:
    """Edge case: total_epochs=1 should return sigma_end."""
    sigma = curriculum_sigma_at(epoch=1, total_epochs=1, sigma_start=12.0, sigma_end=3.0)
    assert sigma == pytest.approx(3.0)


def test_monotonic_annealing() -> None:
    sigmas = [
        curriculum_sigma_at(e, total_epochs=10, sigma_start=12.0, sigma_end=3.0)
        for e in range(1, 11)
    ]
    for a, b in zip(sigmas, sigmas[1:]):
        assert a >= b, "sigma should be monotonically non-increasing"
