import numpy as np
import pytest
import torch
from lacewing.quantification.methods.monotonic_crf import MonotonicCRF


def test_transitions_matrix_shape_and_values():
    crf = MonotonicCRF(n_classes=4)
    t = crf.transitions
    assert t.shape == (4, 4)
    # Allowed = 0
    for c in range(4):
        assert t[c, c].item() == 0.0
        if c + 1 < 4:
            assert t[c, c+1].item() == 0.0
    # Forbidden = very negative
    assert t[3, 0].item() < -1e10
    assert t[2, 0].item() < -1e10
    assert t[0, 2].item() < -1e10  # skip 0→2 also forbidden


def test_decode_forces_monotonic_path():
    crf = MonotonicCRF(n_classes=4)
    # Emissions that would favour a non-monotonic path if not constrained:
    # push class 3 hard at t=0, class 0 hard at t=1 (backward)
    B, T, C = 1, 4, 4
    emissions = torch.zeros(B, T, C)
    emissions[0, 0, 3] = 10.0
    emissions[0, 1, 0] = 10.0
    emissions[0, 2, 2] = 5.0
    emissions[0, 3, 3] = 5.0
    path = crf.decode(emissions)  # (B, T)
    assert path.shape == (B, T)
    # Must be non-decreasing
    diffs = path[0, 1:] - path[0, :-1]
    assert (diffs >= 0).all(), f"Non-monotonic path: {path[0].tolist()}"


def test_nll_finite_on_valid_label_path():
    crf = MonotonicCRF(n_classes=4)
    B, T, C = 2, 40, 4
    emissions = torch.randn(B, T, C)
    # Build a valid monotonic label path
    labels = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        labels[b, 10:] = 1
        labels[b, 20:] = 2
        labels[b, 30:] = 3
    loss = crf(emissions, labels)
    assert torch.isfinite(loss)
    assert loss.item() > 0  # NLL is positive


def test_nll_inf_on_invalid_backward_label_path():
    crf = MonotonicCRF(n_classes=4)
    B, T, C = 1, 20, 4
    emissions = torch.randn(B, T, C)
    labels = torch.zeros(B, T, dtype=torch.long)
    labels[0, 0] = 3
    labels[0, 1] = 0  # backward transition — score should be -inf → NLL inf
    loss = crf(emissions, labels)
    assert torch.isinf(loss)


def test_forward_algorithm_matches_brute_force_small_case():
    crf = MonotonicCRF(n_classes=3)
    B, T, C = 1, 3, 3
    emissions = torch.tensor([[[0.5, 0.3, 0.2],
                               [0.1, 0.6, 0.3],
                               [0.2, 0.2, 0.6]]])
    log_Z = crf._forward_algorithm(emissions)
    # Brute-force: enumerate all valid monotonic paths of length 3, compute path scores, log-sum-exp
    from itertools import product
    valid_paths = []
    for p in product(range(C), repeat=T):
        # Check monotonic (non-decreasing)
        if all(p[i+1] >= p[i] for i in range(T-1)):
            valid_paths.append(p)
    scores = []
    for p in valid_paths:
        s = sum(emissions[0, t, p[t]].item() for t in range(T))
        scores.append(s)
    expected_log_Z = torch.logsumexp(torch.tensor(scores), dim=0)
    assert log_Z[0].item() == pytest.approx(expected_log_Z.item(), abs=1e-4)
