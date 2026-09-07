"""Smoke tests for --labels-mode manual in unet_segmentation.py."""
import numpy as np
import pytest
from pathlib import Path
import tempfile

from lacewing.quantification.methods.unet_segmentation import (
    _load_manual_labels_for_pixels,
)


def test_load_manual_labels_matches_by_pixel_id() -> None:
    """Given a set of cache pixel ids, return the matching subset + label matrix."""
    with tempfile.TemporaryDirectory() as td:
        npz_path = Path(td) / "manual.npz"
        # Say the cache has pixels 5, 10, 15 labelled.
        pixel_ids = np.array([5, 10, 15], dtype=np.int32)
        labels = np.array([
            [0]*100 + [1]*50 + [2]*100 + [3]*200,   # (450,)
            [0]*50  + [1]*50 + [2]*100 + [3]*250,
            [0]*0   + [1]*50 + [2]*100 + [3]*300,
        ], dtype=np.int8)
        np.savez_compressed(npz_path, pixel_ids=pixel_ids, labels=labels,
                            well_keys=np.array(["a", "b", "c"], dtype="U64"),
                            train_pixel_mask=np.array([True, True, False]),
                            holdout_pixel_mask=np.array([False, False, True]))

        # Cache has pixels 3, 5, 7, 10, 12 in the training split.
        cache_pixel_ids = np.array([3, 5, 7, 10, 12], dtype=np.int32)
        matched_mask, matched_labels = _load_manual_labels_for_pixels(cache_pixel_ids, npz_path)

        # Expect matched_mask == [False, True, False, True, False] (cache pixels 5 and 10 have labels)
        assert matched_mask.shape == (5,)
        np.testing.assert_array_equal(matched_mask, [False, True, False, True, False])
        assert matched_labels.shape == (2, 450)
        # Row 0 of matched_labels = labels for cache pixel 5 (= labels[0] from manual)
        np.testing.assert_array_equal(matched_labels[0], labels[0])
        # Row 1 = labels for cache pixel 10 (= labels[1])
        np.testing.assert_array_equal(matched_labels[1], labels[1])


def test_load_manual_labels_no_overlap_returns_empty() -> None:
    with tempfile.TemporaryDirectory() as td:
        npz_path = Path(td) / "manual.npz"
        pixel_ids = np.array([5, 10, 15], dtype=np.int32)
        labels = np.zeros((3, 450), dtype=np.int8)
        np.savez_compressed(npz_path, pixel_ids=pixel_ids, labels=labels,
                            well_keys=np.array(["a", "b", "c"], dtype="U64"),
                            train_pixel_mask=np.array([True, True, True]),
                            holdout_pixel_mask=np.array([False, False, False]))

        cache_pixel_ids = np.array([1, 2, 3], dtype=np.int32)
        matched_mask, matched_labels = _load_manual_labels_for_pixels(cache_pixel_ids, npz_path)
        assert matched_mask.shape == (3,)
        assert not matched_mask.any()
        assert matched_labels.shape == (0, 450)


def test_monotonicity_loss_zero_on_monotonic_sequence():
    from lacewing.quantification.methods.unet_segmentation import _monotonicity_loss
    import torch
    # Perfect monotonic: prob(class c) = 1 at t = c*10..(c+1)*10
    B, C, T = 1, 4, 40
    logits = torch.full((B, C, T), -10.0)
    for c in range(C):
        logits[0, c, c*10:(c+1)*10] = 10.0
    loss = _monotonicity_loss(logits)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_monotonicity_loss_positive_on_reversed_sequence():
    from lacewing.quantification.methods.unet_segmentation import _monotonicity_loss
    import torch
    # Reversed: prob(class 3) at start, prob(class 0) at end
    B, C, T = 1, 4, 40
    logits = torch.full((B, C, T), -10.0)
    for c in range(C):
        logits[0, C-1-c, c*10:(c+1)*10] = 10.0
    loss = _monotonicity_loss(logits)
    assert loss.item() > 0.5
