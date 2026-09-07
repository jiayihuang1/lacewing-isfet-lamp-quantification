"""Labelling UI logic tests (excludes actual matplotlib interaction)."""
import pytest

from lacewing.quantification.methods.fd_labelling.labelling_ui import (
    CLS_BASELINE, CLS_DRIFT, CLS_RISING, CLS_POST_AMP,
    validate_boundaries,
    quit_requested,
)


def test_class_constants() -> None:
    assert CLS_BASELINE == 0
    assert CLS_DRIFT == 1
    assert CLS_RISING == 2
    assert CLS_POST_AMP == 3


def test_validate_boundaries_valid() -> None:
    validate_boundaries(0, 100, 300)         # canonical: all 4 classes present
    validate_boundaries(0, 0, 300)           # no drift
    validate_boundaries(0, 100, 450)         # no post-amp
    validate_boundaries(0, 0, 450)           # only rising
    validate_boundaries(30, 30, 450)         # no drift, has baseline
    validate_boundaries(100, 200, 200)       # rising is empty? No — b2==b3 means no rising.
    # Actually the last one is DEGENERATE — rising is empty (b2==b3 means rising is zero-width).
    # We keep it allowed here; enforcement of "rising must be non-empty" is a
    # LABELLER concern, not a boundary-shape concern. But the test above shows
    # what shape is accepted. Correct: an "invalid amp" label bypasses this via
    # status="invalid" instead of stacking boundaries.


def test_validate_boundaries_rejects_non_monotonic() -> None:
    with pytest.raises(ValueError, match="monotonic"):
        validate_boundaries(100, 50, 300)


def test_validate_boundaries_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="range"):
        validate_boundaries(-1, 100, 300)
    with pytest.raises(ValueError, match="range"):
        validate_boundaries(0, 100, 500)  # >450
    with pytest.raises(ValueError, match="range"):
        validate_boundaries(451, 452, 453)


def test_quit_flag_default_false() -> None:
    # State survives across imports; ensure it starts False in a fresh session.
    assert quit_requested() is False
