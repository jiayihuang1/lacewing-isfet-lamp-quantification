import numpy as np
import pytest
from lacewing.quantification.eval.fd_boundary_metrics import (
    compute_boundary_errors, boundary_weighted_accuracy,
)


def test_boundary_errors_exact_match():
    T = 20
    pred = np.array([0]*5 + [1]*5 + [2]*5 + [3]*5, dtype=np.int8)
    true = pred.copy()
    errs = compute_boundary_errors(pred, true)
    assert errs == {"bd_error": 0, "dr_error": 0, "rp_error": 0}


def test_boundary_errors_offset_by_2():
    pred = np.array([0]*7 + [1]*3 + [2]*7 + [3]*3, dtype=np.int8)  # boundaries at 7, 10, 17
    true = np.array([0]*5 + [1]*5 + [2]*5 + [3]*5, dtype=np.int8)  # boundaries at 5, 10, 15
    errs = compute_boundary_errors(pred, true)
    assert errs == {"bd_error": 2, "dr_error": 0, "rp_error": 2}


def test_boundary_errors_class_missing_returns_sentinel():
    pred = np.array([0]*10 + [3]*10, dtype=np.int8)  # no drift, no rising
    true = np.array([0]*5 + [1]*5 + [2]*5 + [3]*5, dtype=np.int8)
    errs = compute_boundary_errors(pred, true)
    assert errs["bd_error"] == -1  # class 1 (drift) missing from pred
    assert errs["dr_error"] == -1  # class 2 (rising) missing from pred


def test_boundary_weighted_acc_penalises_boundary_shifts():
    T = 100
    true = np.concatenate([np.full(50, 1), np.full(50, 2)]).astype(np.int8)  # boundary at 50
    pred_off = np.concatenate([np.full(45, 1), np.full(55, 2)]).astype(np.int8)  # boundary at 45 → 5 samples off
    acc_off = boundary_weighted_accuracy(pred_off, true, K=15, weight=5.0)
    pred_flat_wrong = np.concatenate([np.full(50, 1), np.full(45, 2), np.full(5, 3)]).astype(np.int8)  # boundary OK but tail wrong
    acc_flat = boundary_weighted_accuracy(pred_flat_wrong, true, K=15, weight=5.0)
    # Both wrong on 5 samples, but boundary-off is inside boundary window (higher weight),
    # so boundary-off should score LOWER
    assert acc_off < acc_flat


def test_boundary_weighted_acc_perfect_is_1():
    true = np.array([0]*20 + [1]*20 + [2]*20 + [3]*20, dtype=np.int8)
    pred = true.copy()
    acc = boundary_weighted_accuracy(pred, true, K=15, weight=5.0)
    assert acc == pytest.approx(1.0, abs=1e-6)
