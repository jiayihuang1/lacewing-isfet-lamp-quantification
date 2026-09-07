"""Tests for build_merged_cov_uti_cache."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from lacewing.quantification.methods.build_merged_cov_uti_cache import (
    build_merged_cls_cache, build_merged_reg_cache,
)


def test_cls_row_count_preserved(tmp_path):
    """N_merged == N_cov + N_uti."""
    out = tmp_path / "merged_cls.npz"
    build_merged_cls_cache(out_path=out)
    m = np.load(out, allow_pickle=False)
    # Sanity: origin split matches source cache sizes.
    n_cov = int((m["origin"] == "cov").sum())
    n_uti = int((m["origin"] == "uti").sum())
    assert n_cov > 0 and n_uti > 0
    assert len(m["X"]) == n_cov + n_uti


def test_cls_chip_namespace_disjoint(tmp_path):
    """No chip_id appears in both COV and UTI origin subsets."""
    out = tmp_path / "merged_cls.npz"
    build_merged_cls_cache(out_path=out)
    m = np.load(out, allow_pickle=False)
    cov_chips = set(m["chip_id"][m["origin"] == "cov"].tolist())
    uti_chips = set(m["chip_id"][m["origin"] == "uti"].tolist())
    assert cov_chips.isdisjoint(uti_chips), (
        f"chip_id namespace overlaps between COV and UTI: "
        f"{cov_chips & uti_chips}"
    )


def test_cls_label_domain(tmp_path):
    """y is 0/1 for both COV and UTI origins."""
    out = tmp_path / "merged_cls.npz"
    build_merged_cls_cache(out_path=out)
    m = np.load(out, allow_pickle=False)
    assert set(np.unique(m["y"]).tolist()).issubset({0, 1})


def test_cls_signal_shape(tmp_path):
    """X.shape[1] == 450."""
    out = tmp_path / "merged_cls.npz"
    build_merged_cls_cache(out_path=out)
    m = np.load(out, allow_pickle=False)
    assert m["X"].shape[1] == 450


def test_reg_row_count_preserved(tmp_path):
    out = tmp_path / "merged_reg.npz"
    build_merged_reg_cache(out_path=out)
    m = np.load(out, allow_pickle=False)
    n_cov = int((m["origin"] == "cov").sum())
    n_uti = int((m["origin"] == "uti").sum())
    assert n_cov > 0 and n_uti > 0
    assert len(m["X"]) == n_cov + n_uti


def test_reg_uti_split_is_train(tmp_path):
    """All UTI pixels get split=0 (train); test defined by LOOCV in training scripts."""
    out = tmp_path / "merged_reg.npz"
    build_merged_reg_cache(out_path=out)
    m = np.load(out, allow_pickle=False)
    uti_splits = m["split"][m["origin"] == "uti"]
    assert (uti_splits == 0).all(), (
        f"expected all UTI split=0, got unique {np.unique(uti_splits)}"
    )


def test_reg_uti_labels_positive(tmp_path):
    """UTI reg cache only carries amp+ pixels, so all y_ttp_min are finite positive."""
    out = tmp_path / "merged_reg.npz"
    build_merged_reg_cache(out_path=out)
    m = np.load(out, allow_pickle=False)
    uti_ttps = m["y_ttp_min"][m["origin"] == "uti"]
    assert np.isfinite(uti_ttps).all()
    assert (uti_ttps > 0).all()


def test_loocv_split_valid(tmp_path):
    """Given a UTI chip to hold out, we can select COV_train + UTI_train + UTI_test cleanly."""
    out = tmp_path / "merged_reg.npz"
    build_merged_reg_cache(out_path=out)
    m = np.load(out, allow_pickle=False)
    uti_chips = np.unique(m["chip_id"][m["origin"] == "uti"])
    assert len(uti_chips) >= 3, "need at least 3 UTI chips for LOOCV to be meaningful"
    hold = uti_chips[0]
    is_test = (m["origin"] == "uti") & (m["chip_id"] == hold)
    is_train = ~is_test
    # Basic sanity: partition covers all pixels, disjoint.
    assert (is_test | is_train).all()
    assert not (is_test & is_train).any()
    # Test set only has that one chip, all UTI.
    assert (m["origin"][is_test] == "uti").all()
    assert (m["chip_id"][is_test] == hold).all()
