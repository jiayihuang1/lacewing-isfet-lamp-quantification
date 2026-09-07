"""Per-well k-means clustering tests."""
import numpy as np
import pytest

from lacewing.quantification.methods.fd_labelling.cluster_pixels import (
    cluster_well,
    BASELINE_WINDOW_SAMPLES,
)


def _synthetic_family(n_pixels: int, shape_fn, T: int = 450, seed: int = 0) -> np.ndarray:
    """Build n_pixels traces of length T, each a small noisy variation of shape_fn(t)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    base = shape_fn(t)
    return base[None, :] + rng.normal(0, 0.05, size=(n_pixels, T)).astype(np.float32)


def test_kmeans_returns_k_labels() -> None:
    T = 450
    traces = _synthetic_family(20, lambda t: np.zeros(T, dtype=np.float32))
    out = cluster_well(traces, k=4, random_state=0)
    assert out.shape == (20,)
    assert out.dtype == np.int32
    # k-means may not use all k centroids if the data doesn't support them; we
    # just verify all returned ids are within [0, k).
    assert (out >= 0).all() and (out < 4).all()


def test_kmeans_deterministic_with_same_random_state() -> None:
    T = 450
    traces = _synthetic_family(30, lambda t: np.sin(t / 20), seed=42)
    a = cluster_well(traces, k=4, random_state=7)
    b = cluster_well(traces, k=4, random_state=7)
    np.testing.assert_array_equal(a, b)


def test_kmeans_separates_two_shape_families() -> None:
    T = 450
    # Family A: rise around t=100.
    fam_a = _synthetic_family(15, lambda t: 1.0 / (1.0 + np.exp(-(t - 100) / 20)), seed=1)
    # Family B: rise around t=300.
    fam_b = _synthetic_family(15, lambda t: 1.0 / (1.0 + np.exp(-(t - 300) / 20)), seed=2)
    traces = np.vstack([fam_a, fam_b])  # (30, 450)
    labels = cluster_well(traces, k=2, random_state=0)
    # Whatever cluster ids are assigned, all of fam_a's pixels should share one id
    # and all of fam_b's pixels should share the other.
    unique_a = set(labels[:15].tolist())
    unique_b = set(labels[15:].tolist())
    assert len(unique_a) == 1
    assert len(unique_b) == 1
    assert unique_a != unique_b


def test_under_populated_well_raises() -> None:
    T = 450
    traces = _synthetic_family(3, lambda t: np.zeros(T, dtype=np.float32))
    with pytest.raises(ValueError, match="fewer pixels than clusters"):
        cluster_well(traces, k=4, random_state=0)


def test_baseline_window_constant_matches_spec() -> None:
    assert BASELINE_WINDOW_SAMPLES == 60
