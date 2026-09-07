"""F-A σ-schedule tests."""
import math
import pytest

from lacewing.quantification.methods.pdf_regression import (
    curriculum_sigma_at,
    cosine_sigma_at,
    adaptive_sigma_step,
)


def test_cosine_sigma_endpoints() -> None:
    # At epoch 1, should be close to sigma_start; at epoch=E_total, ~ sigma_end
    s_start = cosine_sigma_at(1, 40, 12.0, 3.0)
    s_end = cosine_sigma_at(40, 40, 12.0, 3.0)
    assert abs(s_start - 12.0) < 1.0    # very close to start
    assert abs(s_end - 3.0) < 0.5       # very close to end


def test_cosine_sigma_smoother_than_linear_at_early_epochs() -> None:
    # At epoch 5 (of 40), cosine should still be near σ_start; linear will be ~10
    cos_5 = cosine_sigma_at(5, 40, 12.0, 3.0)
    lin_5 = curriculum_sigma_at(5, 40, 12.0, 3.0)
    assert cos_5 > lin_5


def test_adaptive_sharpens_on_plateau() -> None:
    # 3 epochs of near-flat val loss should sharpen sigma
    val_losses = [0.5, 0.499, 0.4995, 0.4990]
    new_sigma = adaptive_sigma_step(val_losses, current_sigma=10.0)
    assert new_sigma == pytest.approx(10.0 * 0.8, abs=1e-6)


def test_adaptive_broadens_on_spike() -> None:
    # Val loss spikes by > 20% from previous
    val_losses = [0.4, 0.5]  # +25% jump
    new_sigma = adaptive_sigma_step(val_losses, current_sigma=8.0)
    assert new_sigma == pytest.approx(8.0 * 1.2, abs=1e-6)


def test_adaptive_holds_on_moderate_improvement() -> None:
    # 3% improvement — not a plateau, not a spike
    val_losses = [0.5, 0.485]
    new_sigma = adaptive_sigma_step(val_losses, current_sigma=6.0)
    assert new_sigma == 6.0


def test_adaptive_clamps() -> None:
    assert adaptive_sigma_step([0.5, 0.499, 0.4995, 0.4990], current_sigma=3.5) == 3.0
    assert adaptive_sigma_step([0.4, 0.5], current_sigma=14.0) == 15.0
