"""Schema round-trip + validation tests."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pytest

from lacewing.quantification.eval.schema import (
    PredictionArrays,
    LabelArrays,
    write_predictions,
    write_labels,
    read_predictions,
    read_labels,
    validate_alignment,
)


def test_predictions_roundtrip(tmp_path):
    ttp = np.array([8.5, 10.6, 15.0, 17.0, 21.0], dtype=np.float32)
    path = tmp_path / "predictions.npz"
    write_predictions(path, ttp_pred_min=ttp)
    result = read_predictions(path)
    np.testing.assert_array_equal(result.ttp_pred_min, ttp)
    assert result.ttp_pred_min.dtype == np.float32


def test_predictions_extras_roundtrip(tmp_path):
    ttp = np.array([8.5, 10.6], dtype=np.float32)
    density = np.random.default_rng(0).random((2, 450)).astype(np.float32)
    path = tmp_path / "predictions.npz"
    write_predictions(path, ttp_pred_min=ttp, ttp_pred_extras={"density": density})
    result = read_predictions(path)
    np.testing.assert_array_equal(result.extras["density"], density)


def test_labels_roundtrip(tmp_path):
    n = 5
    path = tmp_path / "labels.npz"
    write_labels(
        path,
        ttp_true_min=np.array([8.5, 10.6, 15.0, 17.0, 21.0], dtype=np.float32),
        chip_id=np.array(["c0", "c0", "c1", "c1", "c2"]),
        well_id=np.array([0, 1, 0, 1, 0], dtype=np.int32),
        log10_concentration=np.array([9.0, 8.0, 7.0, 6.0, 5.0], dtype=np.float32),
        split=np.array(["train", "train", "val", "val", "test"]),
    )
    result = read_labels(path)
    assert len(result.ttp_true_min) == n
    assert result.chip_id[0] == "c0"


def test_validate_alignment_ok():
    preds = PredictionArrays(ttp_pred_min=np.zeros(3, dtype=np.float32), extras={})
    labels = LabelArrays(
        ttp_true_min=np.zeros(3, dtype=np.float32),
        chip_id=np.array(["c0", "c0", "c0"]),
        well_id=np.zeros(3, dtype=np.int32),
        log10_concentration=np.zeros(3, dtype=np.float32),
        split=np.array(["train", "train", "train"]),
    )
    validate_alignment(preds, labels)  # should not raise


def test_validate_alignment_mismatch():
    preds = PredictionArrays(ttp_pred_min=np.zeros(3, dtype=np.float32), extras={})
    labels = LabelArrays(
        ttp_true_min=np.zeros(2, dtype=np.float32),
        chip_id=np.array(["c0", "c0"]),
        well_id=np.zeros(2, dtype=np.int32),
        log10_concentration=np.zeros(2, dtype=np.float32),
        split=np.array(["train", "train"]),
    )
    with pytest.raises(ValueError, match="pixel count"):
        validate_alignment(preds, labels)
