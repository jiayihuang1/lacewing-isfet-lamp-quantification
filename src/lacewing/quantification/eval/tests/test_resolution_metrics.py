"""Tests for the RQ2 metric portfolio (Week 12 rewrite).

Portfolio: per_well_mae, slope_proximity, per_well_r2, within_well_cov, and
spearman_r_well_means as a sanity filter. All metrics operate on per-well
aggregates, not per-pixel predictions.
"""
from __future__ import annotations

import numpy as np
import pytest

from lacewing.quantification.eval.schema import PredictionArrays, LabelArrays
from lacewing.quantification.eval.resolution_metrics import (
    per_well_mae,
    slope_proximity,
    slope_min_per_decade,
    per_well_r2,
    within_well_cov,
    spearman_r_well_means,
    passes_spearman_filter,
    QLAMP_SLOPE_SD,
)


def _make_multi_well(
    concentrations,
    well_means_pred,
    well_stds_pred,
    well_means_true=None,
    n_wells_per_conc=1,
    n_pixels_per_well=200,
    seed=0,
):
    """Build synthetic predictions with a controllable per-well structure.

    Each concentration gets n_wells_per_conc wells, each with n_pixels_per_well
    pixels. Predictions within a well are drawn from N(well_mean_pred, well_std_pred).
    True TTPs default to the well-mean prediction so per_well_mae is zero at the
    limit.
    """
    if well_means_true is None:
        well_means_true = well_means_pred
    rng = np.random.default_rng(seed)
    preds = []
    trues = []
    chip_ids = []
    well_ids = []
    log_cs = []
    for i, (c, m, s, mt) in enumerate(
        zip(concentrations, well_means_pred, well_stds_pred, well_means_true)
    ):
        for w in range(n_wells_per_conc):
            preds.append(rng.normal(m, s, n_pixels_per_well).astype(np.float32))
            trues.append(np.full(n_pixels_per_well, mt, dtype=np.float32))
            chip_ids.extend(["c0"] * n_pixels_per_well)
            well_ids.extend([w + i * n_wells_per_conc] * n_pixels_per_well)
            log_cs.extend([c] * n_pixels_per_well)
    ttp_pred = np.concatenate(preds)
    ttp_true = np.concatenate(trues)
    n = len(ttp_pred)
    return (
        PredictionArrays(ttp_pred_min=ttp_pred, extras={}),
        LabelArrays(
            ttp_true_min=ttp_true,
            chip_id=np.array(chip_ids),
            well_id=np.array(well_ids, dtype=np.int32),
            log10_concentration=np.array(log_cs, dtype=np.float32),
            split=np.array(["test"] * n),
        ),
    )


# ---------------------------------------------------------------------------
# per_well_mae
# ---------------------------------------------------------------------------

def test_per_well_mae_zero_when_perfect():
    # Well means match true TTPs exactly.
    preds, labels = _make_multi_well(
        concentrations=[5.0, 6.0, 7.0, 8.0, 9.0],
        well_means_pred=[21.0, 17.0, 15.0, 11.0, 9.0],
        well_stds_pred=[0.1, 0.1, 0.1, 0.1, 0.1],
        well_means_true=[21.0, 17.0, 15.0, 11.0, 9.0],
    )
    assert per_well_mae(preds, labels) < 0.05  # noise floor from N(0, 0.1)


def test_per_well_mae_matches_absolute_offset():
    # Predictions systematically 2 min above true.
    preds, labels = _make_multi_well(
        concentrations=[5.0, 6.0, 7.0],
        well_means_pred=[13.0, 12.0, 11.0],
        well_stds_pred=[0.01, 0.01, 0.01],
        well_means_true=[11.0, 10.0, 9.0],
    )
    assert per_well_mae(preds, labels) == pytest.approx(2.0, abs=0.05)


# ---------------------------------------------------------------------------
# slope_proximity / slope_min_per_decade
# ---------------------------------------------------------------------------

def test_slope_matches_qlamp_when_target_hit():
    # Design predictions with well-means that follow qLAMP slope exactly.
    concs = [5.0, 6.0, 7.0, 8.0, 9.0]
    means = [21.0 + QLAMP_SLOPE_SD * (c - 5.0) for c in concs]
    preds, labels = _make_multi_well(
        concentrations=concs,
        well_means_pred=means,
        well_stds_pred=[0.01] * 5,
    )
    raw_slope = slope_min_per_decade(preds, labels)
    assert raw_slope == pytest.approx(QLAMP_SLOPE_SD, abs=0.01)
    assert slope_proximity(preds, labels) < 0.02


def test_slope_proximity_penalises_compression():
    # Well-means only span 1 min over 4 decades (slope ~ -0.25 vs qLAMP -3.13).
    preds, labels = _make_multi_well(
        concentrations=[5.0, 6.0, 7.0, 8.0, 9.0],
        well_means_pred=[16.0, 15.75, 15.5, 15.25, 15.0],
        well_stds_pred=[0.01] * 5,
    )
    prox = slope_proximity(preds, labels)
    # Predicted slope ~ -0.25; qLAMP -3.13 → proximity ~ 2.88.
    assert prox > 2.5


# ---------------------------------------------------------------------------
# per_well_r2
# ---------------------------------------------------------------------------

def test_r2_high_when_linear():
    concs = [5.0, 6.0, 7.0, 8.0, 9.0]
    means = [21.0 + QLAMP_SLOPE_SD * (c - 5.0) for c in concs]
    preds, labels = _make_multi_well(
        concentrations=concs,
        well_means_pred=means,
        well_stds_pred=[0.01] * 5,
    )
    assert per_well_r2(preds, labels) > 0.99


def test_r2_low_when_scrambled():
    # Well means don't track concentration linearly at all.
    preds, labels = _make_multi_well(
        concentrations=[5.0, 6.0, 7.0, 8.0, 9.0],
        well_means_pred=[15.0, 12.0, 16.0, 11.0, 17.0],  # zig-zag
        well_stds_pred=[0.01] * 5,
    )
    assert per_well_r2(preds, labels) < 0.3


# ---------------------------------------------------------------------------
# within_well_cov
# ---------------------------------------------------------------------------

def test_within_well_cov_zero_when_broadcast():
    # Zero within-well spread (like rule-based).
    preds, labels = _make_multi_well(
        concentrations=[5.0, 6.0, 7.0],
        well_means_pred=[15.0, 12.0, 10.0],
        well_stds_pred=[0.0, 0.0, 0.0],
    )
    assert within_well_cov(preds, labels) < 1e-6


def test_within_well_cov_scales_with_noise():
    tight = _make_multi_well(
        concentrations=[5.0, 6.0, 7.0],
        well_means_pred=[15.0, 12.0, 10.0],
        well_stds_pred=[0.1, 0.1, 0.1],
    )
    loose = _make_multi_well(
        concentrations=[5.0, 6.0, 7.0],
        well_means_pred=[15.0, 12.0, 10.0],
        well_stds_pred=[2.0, 2.0, 2.0],
    )
    assert within_well_cov(*loose) > within_well_cov(*tight)


# ---------------------------------------------------------------------------
# spearman + filter
# ---------------------------------------------------------------------------

def test_spearman_negative_for_monotonic_decrease_with_conc():
    # True qLAMP behaviour: TTP DECREASES as log-conc increases → ρ ≈ -1.
    preds, labels = _make_multi_well(
        concentrations=[5.0, 6.0, 7.0, 8.0, 9.0],
        well_means_pred=[21.0, 17.0, 15.0, 11.0, 9.0],
        well_stds_pred=[0.01] * 5,
    )
    rho = spearman_r_well_means(preds, labels)
    assert rho == pytest.approx(-1.0, abs=0.01)
    assert passes_spearman_filter(rho)  # ρ = -1.0 ≤ -0.9


def test_spearman_filter_rejects_anti_correlated_below_threshold():
    # ρ = -0.3 (F-C anti-correlation regression pathology) should FAIL the filter.
    # Build a set where well_mean_pred is a weakly negatively correlated fn of log-c.
    preds, labels = _make_multi_well(
        concentrations=[5.0, 6.0, 7.0, 8.0, 9.0],
        well_means_pred=[12.0, 14.0, 13.0, 15.0, 12.5],  # noisy, low |ρ|
        well_stds_pred=[0.01] * 5,
    )
    rho = spearman_r_well_means(preds, labels)
    assert abs(rho) < 0.9
    assert not passes_spearman_filter(rho)


def test_spearman_filter_rejects_sign_flip():
    # A method that predicts TTP INCREASING with concentration (wrong sign)
    # must FAIL the filter — TTP going up with concentration is biologically
    # impossible, so ρ ≈ +1 gets rejected under the signed threshold.
    preds, labels = _make_multi_well(
        concentrations=[5.0, 6.0, 7.0, 8.0, 9.0],
        well_means_pred=[9.0, 11.0, 15.0, 17.0, 21.0],  # WRONG sign
        well_stds_pred=[0.01] * 5,
    )
    rho = spearman_r_well_means(preds, labels)
    assert rho == pytest.approx(1.0, abs=0.01)
    assert not passes_spearman_filter(rho)
    # The raw slope is positive, which independently confirms the pathology.
    assert slope_min_per_decade(preds, labels) > 0


# ---------------------------------------------------------------------------
# Multi-well aggregation
# ---------------------------------------------------------------------------

def test_metrics_scale_correctly_with_multiple_wells_per_conc():
    # 3 wells per concentration, all identical well-means → metrics stable.
    preds, labels = _make_multi_well(
        concentrations=[5.0, 6.0, 7.0],
        well_means_pred=[15.0, 12.0, 10.0],
        well_stds_pred=[0.1, 0.1, 0.1],
        n_wells_per_conc=3,
    )
    # Slope from well-means: (10 - 15) / (7 - 5) = -2.5
    assert slope_min_per_decade(preds, labels) == pytest.approx(-2.5, abs=0.05)
