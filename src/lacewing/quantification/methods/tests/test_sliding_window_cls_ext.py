"""F-B extension tests: soft labels + backbone dispatch."""
import numpy as np
import pytest

from lacewing.quantification.methods.sliding_window_cls import _WindowDataset


def test_hard_labels_when_soft_sigma_zero() -> None:
    """soft_label_sigma_samples=0 -> binary labels identical to legacy behaviour."""
    X = np.zeros((2, 100), dtype=np.float32)
    ttp_min = np.array([2.0, 4.0], dtype=np.float32)  # ttp_idx = 30, 60 samples
    ds = _WindowDataset(X, ttp_min, window=20, stride=5, soft_label_sigma_samples=0.0)
    # Verify labels are strictly 0.0 or 1.0.
    unique = np.unique(ds.labels)
    assert set(unique.tolist()) <= {0.0, 1.0}


def test_soft_labels_have_intermediate_values() -> None:
    """soft_label_sigma_samples>0 -> labels smoothly transition."""
    X = np.zeros((1, 100), dtype=np.float32)
    ttp_min = np.array([2.0], dtype=np.float32)  # ttp_idx = 30
    ds = _WindowDataset(X, ttp_min, window=20, stride=5, soft_label_sigma_samples=5.0)
    # Sigmoid around ttp_idx should have windows with labels in (0.05, 0.95).
    intermediate = ((ds.labels > 0.05) & (ds.labels < 0.95)).any()
    assert intermediate, f"expected intermediate labels; got unique={np.unique(ds.labels)}"


def test_soft_label_is_monotonic_in_time() -> None:
    """Windows further from TTP in one direction should have monotonic labels."""
    X = np.zeros((1, 200), dtype=np.float32)
    ttp_min = np.array([5.0], dtype=np.float32)  # ttp_idx = 75
    ds = _WindowDataset(X, ttp_min, window=20, stride=5, soft_label_sigma_samples=5.0)
    row = ds.labels[0]
    # Labels should be non-decreasing across windows (time increases).
    diffs = np.diff(row)
    assert (diffs >= -1e-6).all(), f"labels not monotonic; row={row}"


def test_backbone_dispatch_default_is_cnn_gru_par() -> None:
    """Sanity: build_model(window) default returns something that outputs (B,)."""
    from lacewing.quantification.methods.sliding_window_cls import build_model
    import torch
    m = build_model(window=60)
    x = torch.zeros(4, 1, 60)
    out = m(x)
    assert out.shape == (4,)


def test_backbone_dispatch_bigru() -> None:
    from lacewing.quantification.methods.sliding_window_cls import build_model
    import torch
    m = build_model(window=60, backbone="bigru")
    x = torch.zeros(4, 1, 60)
    out = m(x)
    assert out.shape == (4,)
