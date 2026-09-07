"""Test that pdf_regression selects best.pt by val TTP MAE, not val_loss."""
from __future__ import annotations

import numpy as np

from lacewing.quantification.methods.pdf_regression import select_best_epoch


def test_select_by_val_ttp_mae_not_val_loss():
    """val_loss dips at epoch 1 but val TTP MAE bottoms at epoch 3 — pick 3."""
    val_loss = np.array([0.020, 0.015, 0.016, 0.017, 0.018])
    val_ttp_mae = np.array([17.0, 9.2, 7.6, 7.0, 7.3])
    best_epoch = select_best_epoch(val_loss=val_loss, val_ttp_mae=val_ttp_mae)
    assert best_epoch == 3, f"expected epoch 3 (min val TTP MAE=7.0); got {best_epoch}"


def test_select_first_epoch_when_flat():
    val_loss = np.array([0.02, 0.02, 0.02])
    val_ttp_mae = np.array([5.0, 5.0, 5.0])
    best = select_best_epoch(val_loss=val_loss, val_ttp_mae=val_ttp_mae)
    assert best == 0
