"""Broadcast tests: boundaries → per-timestep labels + holdout selection."""
import numpy as np
import pytest

from lacewing.quantification.methods.fd_labelling.broadcast_labels import (
    boundaries_to_labels,
    pick_holdout_wells,
)


def test_full_4_classes() -> None:
    lbl = boundaries_to_labels(50, 150, 300, n_samples=450)
    assert lbl.shape == (450,)
    assert lbl.dtype == np.int8
    assert (lbl[:50] == 0).all()      # baseline
    assert (lbl[50:150] == 1).all()   # drift
    assert (lbl[150:300] == 2).all()  # rising
    assert (lbl[300:450] == 3).all()  # post-amp


def test_no_baseline() -> None:
    lbl = boundaries_to_labels(0, 100, 300, n_samples=450)
    # No pixels labelled baseline.
    assert (lbl == 0).sum() == 0
    assert (lbl[:100] == 1).all()
    assert (lbl[100:300] == 2).all()
    assert (lbl[300:] == 3).all()


def test_no_drift() -> None:
    lbl = boundaries_to_labels(50, 50, 300, n_samples=450)
    # No pixels labelled drift.
    assert (lbl == 1).sum() == 0
    assert (lbl[:50] == 0).all()
    assert (lbl[50:300] == 2).all()
    assert (lbl[300:] == 3).all()


def test_no_post_amp() -> None:
    lbl = boundaries_to_labels(50, 150, 450, n_samples=450)
    assert (lbl == 3).sum() == 0
    assert (lbl[:50] == 0).all()
    assert (lbl[50:150] == 1).all()
    assert (lbl[150:450] == 2).all()


def test_only_rising() -> None:
    lbl = boundaries_to_labels(0, 0, 450, n_samples=450)
    assert (lbl == 2).all()


def test_pick_holdout_deterministic() -> None:
    wells = [f"chipA::{i}" for i in range(30)]
    a = pick_holdout_wells(wells, seed=42, n_holdout=6)
    b = pick_holdout_wells(wells, seed=42, n_holdout=6)
    assert a == b
    assert len(a) == 6
    # Different seed → different selection (probably).
    c = pick_holdout_wells(wells, seed=99, n_holdout=6)
    assert a != c


def test_pick_holdout_returns_subset() -> None:
    wells = [f"chipA::{i}" for i in range(30)]
    hold = pick_holdout_wells(wells, seed=0, n_holdout=6)
    assert set(hold).issubset(set(wells))
