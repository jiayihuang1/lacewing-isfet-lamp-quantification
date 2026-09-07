"""Scoreboard append + idempotency + read tests."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from lacewing.quantification.eval.schema import PredictionArrays, LabelArrays
from lacewing.quantification.eval.scoreboard import (
    compute_all_metrics,
    append_scoreboard_row,
    read_scoreboard,
)


def _mkdata():
    ttp = np.array([16.0, 15.0, 14.0, 13.0, 12.0], dtype=np.float32)
    preds = PredictionArrays(ttp_pred_min=ttp, extras={})
    labels = LabelArrays(
        ttp_true_min=ttp.copy(),
        chip_id=np.array(["c0"] * 5),
        well_id=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        log10_concentration=np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32),
        split=np.array(["test"] * 5),
    )
    return preds, labels


def test_compute_all_metrics_returns_flat_dict():
    preds, labels = _mkdata()
    m = compute_all_metrics(preds, labels)
    # Must contain the 4 primaries + slope sign-check + Spearman filter columns.
    for key in ("per_well_mae_min", "slope_proximity_min_per_decade",
                "slope_min_per_decade", "per_well_r2",
                "within_well_cov", "spearman_r_well_means",
                "passes_spearman_filter"):
        assert key in m, f"missing metric: {key}"


def test_append_scoreboard_creates_file(tmp_path):
    preds, labels = _mkdata()
    m = compute_all_metrics(preds, labels)
    p = tmp_path / "scoreboard.csv"
    append_scoreboard_row(p, method_id="rule_ttp", config_json={"kind": "rule"}, metrics=m)
    df = read_scoreboard(p)
    assert len(df) == 1
    assert df.iloc[0]["method_id"] == "rule_ttp"


def test_append_scoreboard_overwrites_by_method_id(tmp_path):
    preds, labels = _mkdata()
    m = compute_all_metrics(preds, labels)
    p = tmp_path / "scoreboard.csv"
    append_scoreboard_row(p, method_id="rule_ttp", config_json={"v": 1}, metrics=m)
    m2 = dict(m)
    m2["per_well_mae_min"] = -99.0  # sentinel
    append_scoreboard_row(p, method_id="rule_ttp", config_json={"v": 2}, metrics=m2)
    df = read_scoreboard(p)
    assert len(df) == 1
    assert df.iloc[0]["per_well_mae_min"] == -99.0
    assert json.loads(df.iloc[0]["config_json"])["v"] == 2


def test_append_multiple_methods(tmp_path):
    preds, labels = _mkdata()
    m = compute_all_metrics(preds, labels)
    p = tmp_path / "scoreboard.csv"
    for method in ("rule_ttp", "rule_sdm", "rule_cy0"):
        append_scoreboard_row(p, method_id=method, config_json={}, metrics=m)
    df = read_scoreboard(p)
    assert set(df["method_id"]) == {"rule_ttp", "rule_sdm", "rule_cy0"}
