"""Extraction-rule tests for F-D 4-class segmentation."""
import numpy as np
import pytest

from lacewing.quantification.methods.unet_segmentation import (
    _predict_ttp_from_argmax,
)
from lacewing.quantification.eval.fd_extraction_sweep import (
    _extract_smoothed_prob,
    _extract_largest_block,
    _extract_viterbi_monotonic,
)


def test_kcons_1_matches_original_first_index() -> None:
    # 3 pixels, T=10 timesteps. Class 2 = "rising".
    argmax = np.array([
        [0, 0, 0, 2, 0, 2, 2, 2, 2, 2],  # first rising at t=3 (spurious)
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],  # first rising at t=4
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # no rising at all
    ], dtype=np.int8)
    ttp = _predict_ttp_from_argmax(argmax, target_cls=frozenset({2}), k_consecutive=1)
    # SAMPLES_PER_MIN = 15
    np.testing.assert_allclose(ttp, [3/15, 4/15, 9/15], atol=1e-6)


def test_kcons_3_ignores_single_spurious_rising() -> None:
    argmax = np.array([
        [0, 0, 0, 2, 0, 2, 2, 2, 2, 2],  # kcons=3 → skip t=3 (isolated), pick t=5
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],  # kcons=3 → pick t=4 (3 consecutive)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # no rising at all → fallback
    ], dtype=np.int8)
    ttp = _predict_ttp_from_argmax(argmax, target_cls=frozenset({2}), k_consecutive=3)
    np.testing.assert_allclose(ttp, [5/15, 4/15, 9/15], atol=1e-6)


def test_kcons_5_stricter() -> None:
    argmax = np.array([
        [0, 0, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2],  # kcons=5 → skip t=3-5 (only 3 consec), pick t=7
    ], dtype=np.int8)
    ttp = _predict_ttp_from_argmax(argmax, target_cls=frozenset({2}), k_consecutive=5)
    np.testing.assert_allclose(ttp, [7/15], atol=1e-6)


def test_fallback_when_never_reaches_kcons() -> None:
    argmax = np.array([
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # never K consecutive class-2
    ], dtype=np.int8)
    ttp = _predict_ttp_from_argmax(argmax, target_cls=frozenset({2}), k_consecutive=2)
    # Fallback to last index / SAMPLES_PER_MIN
    np.testing.assert_allclose(ttp, [9/15], atol=1e-6)


def test_multi_class_target_supported() -> None:
    argmax = np.array([
        [0, 0, 1, 1, 2, 2, 3, 3, 3, 3],  # target = {1, 2}: first at t=2
    ], dtype=np.int8)
    ttp = _predict_ttp_from_argmax(argmax, target_cls=frozenset({1, 2}), k_consecutive=1)
    np.testing.assert_allclose(ttp, [2/15], atol=1e-6)


def test_smoothed_prob_ignores_single_spike() -> None:
    # 1 pixel, T=20, class 2 has a single-spike at t=3 (softmax=0.9),
    # sustained rise from t=10 (softmax ramps 0.3→0.9).
    T = 20
    sm = np.zeros((1, 4, T), dtype=np.float16)
    sm[0, 0, :] = 0.9   # baseline dominant early
    sm[0, 2, 3] = 0.9   # single spike (spurious)
    sm[0, 0, 3] = 0.05
    for t in range(10, T):
        sm[0, 2, t] = 0.3 + 0.06 * (t - 10)  # ramps up
        sm[0, 0, t] = 1.0 - sm[0, 2, t]
    ttp = _extract_smoothed_prob(sm, target_class=2, window=5, threshold=0.5)
    # Smoothing kills the single spike; first index where smoothed > 0.5 is around t=13
    assert 12 / 15 <= ttp[0] <= 15 / 15


def test_largest_block_picks_longest_run() -> None:
    argmax = np.array([
        [0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 0, 2, 2, 0, 0],
        # runs of class 2: [t=2 len 1], [t=5-9 len 5], [t=11-12 len 2]
        # → largest starts at t=5
    ], dtype=np.int8)
    ttp = _extract_largest_block(argmax, target_class=2)
    np.testing.assert_allclose(ttp, [5/15], atol=1e-6)


def test_viterbi_enforces_monotonicity() -> None:
    # Softmax has a strong "backward" spike (rising → drift → rising) that
    # per-timestep argmax would pick up, but Viterbi with monotonicity should
    # not decode back-transitions.
    T = 10
    sm = np.zeros((1, 4, T), dtype=np.float16)
    # Set class 1 (drift) high for t=0-4, class 2 (rising) high for t=5-9,
    # but with a spurious class-1 spike at t=7.
    for t in range(0, 5):
        sm[0, 1, t] = 0.9; sm[0, 0, t] = 0.05; sm[0, 2, t] = 0.03; sm[0, 3, t] = 0.02
    for t in range(5, T):
        sm[0, 2, t] = 0.85; sm[0, 3, t] = 0.05; sm[0, 1, t] = 0.05; sm[0, 0, t] = 0.05
    sm[0, 1, 7] = 0.85; sm[0, 2, 7] = 0.05
    ttp = _extract_viterbi_monotonic(sm, rising_class=2)
    # Viterbi should NOT go back to class 1 at t=7; TTP stays at t=5.
    np.testing.assert_allclose(ttp, [5/15], atol=1e-6)
