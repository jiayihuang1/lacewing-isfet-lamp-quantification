"""TTA augmentation helper tests (no model loading)."""
import numpy as np

from lacewing.quantification.eval.tta_inference import (
    generate_augmentations,
    TIME_SHIFTS,
    AMPLITUDE_SCALES,
)


def test_num_augmentations_is_grid_product() -> None:
    X = np.zeros((2, 10), dtype=np.float32)
    augs = generate_augmentations(X)
    assert len(augs) == len(TIME_SHIFTS) * len(AMPLITUDE_SCALES)


def test_identity_augmentation_is_input() -> None:
    """The (0-shift, 1.0-scale) augmentation should be identical to input."""
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    augs = generate_augmentations(X)
    # Find the identity entry.
    identity = next(a for (shift, scale, a) in augs if shift == 0 and scale == 1.0)
    np.testing.assert_allclose(identity, X)


def test_amplitude_scale_multiplies() -> None:
    X = np.array([[1.0, 2.0]], dtype=np.float32)
    augs = generate_augmentations(X)
    scaled = next(a for (shift, scale, a) in augs if shift == 0 and scale == 1.05)
    np.testing.assert_allclose(scaled, X * 1.05)


def test_time_shift_rolls() -> None:
    X = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    augs = generate_augmentations(X)
    shifted = next(a for (shift, scale, a) in augs if shift == 1 and scale == 1.0)
    np.testing.assert_allclose(shifted, np.roll(X, 1, axis=-1))
