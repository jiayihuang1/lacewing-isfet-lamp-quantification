"""Eval · per-well MAE, slope-proximity, per-well R², within-well CoV, Spearman-ρ filter. [Cat A] Report §RQ2/RQ3 evaluation.

RQ2 quantification metric portfolio (Week 12 rewrite).

All metrics are computed on **per-well aggregates**, not per-pixel predictions.
This matches how the Lacewing ISFET-LAMP device is deployed (predictions are
averaged across ~1,350 pixels per well before a call is made).

The portfolio is intentionally minimal — four primaries + one filter:

  1. per_well_mae_min           — accuracy       (min)
  2. slope_proximity_min_per_decade — calibration (min/decade)
  3. per_well_r2                — correlation    (dimensionless)
  4. within_well_cov            — precision      (dimensionless)
  5. spearman_r_well_means      — sanity filter  (pass if ≥ 0.9)

Distinguishability (σ-separation / interval-overlap / pairwise AUC on well-means)
is intentionally NOT scored today — only 1 well per concentration is available on
the SD test chip, which makes any between-well variance estimate meaningless.
Deferred until multi-well test data lands (mid-Aug supervisor dataset).

Reference constants derived from qLAMP well-mean TTPs on the SD test chip
(see docs/superpowers/plans/2026-07-06-decision-log.md).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from lacewing.quantification.eval.schema import PredictionArrays, LabelArrays

# qLAMP well-mean TTP slope on the SD test chip, min/decade.
# Computed once from the SD labels (5 wells at log10-conc 5..9, well-mean qLAMP TTPs).
# See labels.npz on any P2 run — the fit is very close to but not perfectly linear.
QLAMP_SLOPE_SD = -3.1336  # min per log10-decade
QLAMP_INTERCEPT_SD = 36.3774  # min at log10-conc = 0 (reference only)

# Spearman ρ threshold for the sanity filter. Any method whose well-mean
# predictions don't monotonically track concentration below this threshold
# is flagged as broken and excluded from the winner comparison.
SPEARMAN_FILTER_THRESHOLD = 0.9


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def _well_group_key(chip_id: np.ndarray, well_id: np.ndarray) -> np.ndarray:
    """Return an integer well index that uniquely identifies (chip, well) pairs.

    We can't hash (chip_id, well_id) pairs into a single ndarray directly, so
    we assemble a string key and then unique-ify.
    """
    keys = np.array([f"{c}|{w}" for c, w in zip(chip_id, well_id.astype(str))])
    _, inv = np.unique(keys, return_inverse=True)
    return inv


def _per_well_aggregate(
    preds: PredictionArrays,
    labels: LabelArrays,
    split: str = "test",
) -> dict:
    """Aggregate per-pixel predictions to per-well means and record per-well
    within-well std for the CoV metric.

    Returns a dict with:
      log_conc         — (n_wells,) log10 concentration of each well
      well_mean_pred   — (n_wells,) mean predicted TTP within each well
      well_std_pred    — (n_wells,) std of predicted TTP within each well (ddof=1)
      well_mean_true   — (n_wells,) mean qLAMP TTP within each well
      n_pixels_per_well — (n_wells,)
    """
    mask = labels.split == split
    ttp_pred = preds.ttp_pred_min[mask]
    ttp_true = labels.ttp_true_min[mask]
    log_c = labels.log10_concentration[mask]
    chip_id = labels.chip_id[mask]
    well_id = labels.well_id[mask]

    well_idx = _well_group_key(chip_id, well_id)
    unique_wells = np.unique(well_idx)

    well_mean_pred = np.zeros(len(unique_wells), dtype=np.float64)
    well_std_pred = np.zeros(len(unique_wells), dtype=np.float64)
    well_mean_true = np.zeros(len(unique_wells), dtype=np.float64)
    well_log_c = np.zeros(len(unique_wells), dtype=np.float64)
    well_n_pixels = np.zeros(len(unique_wells), dtype=np.int64)

    for i, w in enumerate(unique_wells):
        sel = well_idx == w
        well_mean_pred[i] = float(ttp_pred[sel].mean())
        well_std_pred[i] = float(ttp_pred[sel].std(ddof=1)) if sel.sum() > 1 else 0.0
        well_mean_true[i] = float(ttp_true[sel].mean())
        well_log_c[i] = float(log_c[sel][0])
        well_n_pixels[i] = int(sel.sum())

    return {
        "log_conc": well_log_c,
        "well_mean_pred": well_mean_pred,
        "well_std_pred": well_std_pred,
        "well_mean_true": well_mean_true,
        "n_pixels_per_well": well_n_pixels,
    }


# ----------------------------------------------------------------------------
# Primary metrics
# ----------------------------------------------------------------------------

def per_well_mae(preds: PredictionArrays, labels: LabelArrays, split: str = "test") -> float:
    """Median absolute error between well-mean predicted TTP and well-mean qLAMP TTP.

    This is the primary accuracy metric. Reports the typical error a deployment
    call would make on a single well: "our method is off by X minutes."

    Beat rule-based when per_well_mae < rule_TTP's ~2.7 min.
    """
    agg = _per_well_aggregate(preds, labels, split=split)
    errs = np.abs(agg["well_mean_pred"] - agg["well_mean_true"])
    return float(np.mean(errs))


def slope_proximity(
    preds: PredictionArrays,
    labels: LabelArrays,
    split: str = "test",
    qlamp_slope: float = QLAMP_SLOPE_SD,
) -> float:
    """Absolute difference between the linear slope of predicted well-mean TTP vs
    log10-concentration, and the qLAMP reference slope (~ -3.13 min/decade on SD).

    Zero = perfect match to qLAMP dynamic range. Larger = compressed (or reversed)
    dynamic range. Reported in min/decade so it's directly interpretable.
    """
    agg = _per_well_aggregate(preds, labels, split=split)
    if len(agg["log_conc"]) < 2:
        return float("nan")
    # Use unique-concentration well-means. If multiple wells per concentration
    # exist, average them first (so the slope estimator isn't biased by well
    # count imbalance).
    concs, inv = np.unique(agg["log_conc"], return_inverse=True)
    conc_means = np.array([agg["well_mean_pred"][inv == i].mean() for i in range(len(concs))])
    if len(concs) < 2:
        return float("nan")
    slope, _ = np.polyfit(concs, conc_means, 1)
    return float(abs(slope - qlamp_slope))


def slope_min_per_decade(
    preds: PredictionArrays,
    labels: LabelArrays,
    split: str = "test",
) -> float:
    """Raw slope of well-mean predicted TTP vs log10-concentration, min/decade.

    Reported alongside slope_proximity so the sign and magnitude are visible.
    A slope near qLAMP's -3.13 is good; a slope near 0 is compressed; a positive
    slope is a sign-flipped pathology.
    """
    agg = _per_well_aggregate(preds, labels, split=split)
    if len(agg["log_conc"]) < 2:
        return float("nan")
    concs, inv = np.unique(agg["log_conc"], return_inverse=True)
    conc_means = np.array([agg["well_mean_pred"][inv == i].mean() for i in range(len(concs))])
    slope, _ = np.polyfit(concs, conc_means, 1)
    return float(slope)


def per_well_r2(
    preds: PredictionArrays,
    labels: LabelArrays,
    split: str = "test",
) -> float:
    """Coefficient of determination of the linear fit of well-mean predicted TTP
    to log10-concentration.

    r² = 1 - SS_res / SS_tot. Bounded (-inf, 1]. Near 1 = well-means lie on a
    straight line with the fitted slope. Low values mean the calibration curve
    is noisy or non-linear.

    Sign of the fit is captured by Spearman ρ, not r².
    """
    agg = _per_well_aggregate(preds, labels, split=split)
    if len(agg["log_conc"]) < 2:
        return float("nan")
    concs, inv = np.unique(agg["log_conc"], return_inverse=True)
    conc_means = np.array([agg["well_mean_pred"][inv == i].mean() for i in range(len(concs))])
    slope, intercept = np.polyfit(concs, conc_means, 1)
    y_pred = slope * concs + intercept
    ss_res = float(np.sum((conc_means - y_pred) ** 2))
    ss_tot = float(np.sum((conc_means - conc_means.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def within_well_cov(
    preds: PredictionArrays,
    labels: LabelArrays,
    split: str = "test",
) -> float:
    """Mean coefficient of variation of per-pixel predictions within a well,
    averaged across wells.

    CoV_well = std(pixel_preds within well) / |mean(pixel_preds within well)|.
    Reported dimensionless. Rule-based emits one landmark per well broadcast to
    all pixels, so CoV_rule = 0 by construction. ML methods have real CoVs
    reflecting per-pixel prediction noise.

    Lower is better — a lower CoV means predictions are consistent across
    pixels within the same well.
    """
    agg = _per_well_aggregate(preds, labels, split=split)
    means = agg["well_mean_pred"]
    stds = agg["well_std_pred"]
    # Guard against zero means (would give inf); only include wells with
    # nonzero mean prediction.
    safe = np.abs(means) > 1e-6
    if not safe.any():
        return float("nan")
    covs = stds[safe] / np.abs(means[safe])
    return float(np.mean(covs))


# ----------------------------------------------------------------------------
# Sanity filter
# ----------------------------------------------------------------------------

def spearman_r_well_means(
    preds: PredictionArrays,
    labels: LabelArrays,
    split: str = "test",
) -> float:
    """Spearman rank correlation between well-mean predicted TTP and log10-concentration.

    Used as a pass/fail sanity filter. Biology requires ρ ≤ -0.9 (TTP must
    decrease monotonically with concentration — higher template → faster amp →
    shorter TTP). Methods with ρ > -0.9 are flagged as broken (F-C anti-
    correlation, F-B collapse, TCN/GRU-uni sign-flip, transformer_patch failure)
    and excluded from the winner comparison — their scoreboard rows are kept
    for auditability.
    """
    agg = _per_well_aggregate(preds, labels, split=split)
    if len(agg["log_conc"]) < 2:
        return float("nan")
    rho, _ = spearmanr(agg["well_mean_pred"], agg["log_conc"])
    return float(rho)


def passes_spearman_filter(rho: float, threshold: float = SPEARMAN_FILTER_THRESHOLD) -> bool:
    """Return True if a method's well-mean Spearman ρ is strong AND correctly signed.

    Biology-consistent calibration requires ρ ≤ -threshold (default -0.9):
      * strongly negative → TTP decreases monotonically with concentration ✓
      * near zero → framing produces no monotonic signal ✗
      * strongly positive → sign-flip: TTP INCREASES with concentration
        (biologically impossible for ISFET-LAMP on qLAMP-TTP) ✗

    Both no-signal and sign-flip pathologies fail this single check — no
    separate slope-sign diagnosis needed.
    """
    if not np.isfinite(rho):
        return False
    return rho <= -threshold
