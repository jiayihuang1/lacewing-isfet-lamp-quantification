"""F-A extraction-rule sweep: 4 rules on saved density outputs."""
import numpy as np
import pytest

from lacewing.quantification.eval.fa_extraction_sweep import (
    extract_ttp_hard_argmax,
    extract_ttp_expected_value,
    extract_ttp_first_moment_threshold,
    extract_ttp_soft_argmax_top5,
    SAMPLES_PER_MIN,
)


def _one_hot_at(idx: int, T: int = 450) -> np.ndarray:
    d = np.zeros((1, T), dtype=np.float32)
    d[0, idx] = 1.0
    return d


def test_hard_argmax_returns_peak_time() -> None:
    density = _one_hot_at(150)  # 150 / 15 = 10 min
    ttp = extract_ttp_hard_argmax(density)
    assert ttp.shape == (1,)
    assert ttp[0] == pytest.approx(10.0)


def test_expected_value_matches_argmax_for_delta() -> None:
    density = _one_hot_at(150)
    ttp = extract_ttp_expected_value(density)
    assert ttp[0] == pytest.approx(10.0)


def test_expected_value_is_between_two_peaks() -> None:
    density = np.zeros((1, 450), dtype=np.float32)
    density[0, 90] = 1.0    # 6 min
    density[0, 150] = 1.0   # 10 min
    ttp = extract_ttp_expected_value(density)
    # Expected value = (6 + 10) / 2 = 8 min.
    assert ttp[0] == pytest.approx(8.0, abs=0.1)


def test_first_moment_threshold_returns_early_time() -> None:
    # Peaked density → first-moment > 0.5 happens near the peak.
    density = _one_hot_at(150)
    ttp = extract_ttp_first_moment_threshold(density)
    assert ttp[0] == pytest.approx(10.0, abs=1e-3)


def test_soft_argmax_top5_matches_argmax_for_delta() -> None:
    density = _one_hot_at(150)
    ttp = extract_ttp_soft_argmax_top5(density)
    assert ttp[0] == pytest.approx(10.0)
