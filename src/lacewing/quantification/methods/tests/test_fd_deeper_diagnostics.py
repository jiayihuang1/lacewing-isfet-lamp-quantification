import numpy as np
import pandas as pd
import pytest
from lacewing.quantification.eval.fd_deeper_diagnostics import (
    compute_test_boundary_errors,
    compute_per_well_errors,
    time_flip_test,
)


def test_boundary_errors_df_columns():
    """Verify the returned DataFrame has all expected columns."""
    df = compute_test_boundary_errors(seed=0, ckpt="last.pt")
    assert set(df.columns) >= {"pixel_idx", "well_id", "first_rising_pred", "first_rising_true", "dr_error_samples", "dr_error_signed_samples"}
    assert len(df) == 5014, f"Expected 5014 test pixels, got {len(df)}"


def test_per_well_errors_5_wells():
    df = compute_per_well_errors(seed=0, ckpt="last.pt")
    assert len(df) == 5, f"Expected 5 SD wells, got {len(df)}"
    assert set(df.columns) >= {"seed", "well", "true_ttp_min", "median_pred_ttp_min", "signed_error_min", "mae_min"}


def test_time_flip_returns_expected_keys():
    result = time_flip_test(seed=0, n_samples=5)
    assert "correlation_normal_vs_flipped_complement" in result
    assert "median_abs_diff" in result
    assert isinstance(result["correlation_normal_vs_flipped_complement"], (float, np.floating))
