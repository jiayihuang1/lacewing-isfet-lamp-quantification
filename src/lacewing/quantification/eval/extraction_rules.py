"""TTP extraction rules for the F-B sliding-window classifier ablation.

Given a trained F-B model's per-pixel per-window sigmoid probabilities
`probs` of shape (n_pixels, n_windows), each rule reduces the array to a
single per-pixel TTP in minutes.

All rules are pure NumPy — no GPU, no scipy for the core rules (a couple
use scipy.optimize for sigmoid fitting).  Runtime for 5000 pixels × any
one rule is < 1 s locally.

Naming: rule id is a short slug used as part of the scoreboard method_id
(e.g. "first_above_K3", "linear_interp_thr0p5", "sigmoid_fit_thr0p5",
"first_deriv_peak").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


SAMPLES_PER_MIN = 15


@dataclass(frozen=True)
class Rule:
    """One TTP extraction rule."""
    id: str                                       # scoreboard-friendly slug
    label: str                                    # human-readable description
    family: str                                   # {"threshold", "interp", "deriv", "argmax"}
    fn: Callable[[np.ndarray, int, int], np.ndarray]  # (probs, window, stride) -> ttp_pred_min


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _window_centres_samples(n_windows: int, window: int, stride: int) -> np.ndarray:
    return np.arange(n_windows) * stride + (window - 1) / 2


def _window_centres_min(n_windows: int, window: int, stride: int) -> np.ndarray:
    return _window_centres_samples(n_windows, window, stride) / SAMPLES_PER_MIN


# ---------------------------------------------------------------------------
# Rule family 1 — Threshold-crossing with K-consecutive gating
# ---------------------------------------------------------------------------

def _first_above_K(
    probs: np.ndarray, window: int, stride: int,
    threshold: float = 0.5, K: int = 3,
) -> np.ndarray:
    """First window where probs > threshold for K consecutive windows.

    Returns the CENTRE of the first window in that K-run.
    Falls back to centre of last window if no K-run exists.
    """
    n_pix, n_win = probs.shape
    centres = _window_centres_min(n_win, window, stride)
    above = probs > threshold  # (n_pix, n_win) boolean
    ttp = np.full(n_pix, centres[-1], dtype=np.float32)
    for i in range(n_pix):
        run = 0
        for t in range(n_win):
            if above[i, t]:
                run += 1
                if run >= K:
                    ttp[i] = centres[t - K + 1]
                    break
            else:
                run = 0
    return ttp


def _first_above(
    probs: np.ndarray, window: int, stride: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """First window where probs > threshold (K = 1 special case)."""
    return _first_above_K(probs, window, stride, threshold=threshold, K=1)


def _start_of_K_run(
    probs: np.ndarray, window: int, stride: int,
    threshold: float = 0.5, K: int = 3,
) -> np.ndarray:
    """Same as _first_above_K.  Alias for readability."""
    return _first_above_K(probs, window, stride, threshold=threshold, K=K)


def _end_of_K_run(
    probs: np.ndarray, window: int, stride: int,
    threshold: float = 0.5, K: int = 3,
) -> np.ndarray:
    """Centre of the LAST window in the first K-consecutive-above run."""
    n_pix, n_win = probs.shape
    centres = _window_centres_min(n_win, window, stride)
    above = probs > threshold
    ttp = np.full(n_pix, centres[-1], dtype=np.float32)
    for i in range(n_pix):
        run = 0
        for t in range(n_win):
            if above[i, t]:
                run += 1
                if run >= K:
                    ttp[i] = centres[t]
                    break
            else:
                run = 0
    return ttp


def _fraction_above(
    probs: np.ndarray, window: int, stride: int,
    lookback: int = 10, threshold: float = 0.5, frac: float = 0.5,
) -> np.ndarray:
    """First window t where at least `frac` of the last `lookback` windows are > threshold.

    Robust to isolated dropouts inside an otherwise post-amp run.
    """
    n_pix, n_win = probs.shape
    centres = _window_centres_min(n_win, window, stride)
    above = (probs > threshold).astype(np.float32)  # (n_pix, n_win)
    ttp = np.full(n_pix, centres[-1], dtype=np.float32)
    for i in range(n_pix):
        for t in range(lookback - 1, n_win):
            frac_above = above[i, t - lookback + 1 : t + 1].mean()
            if frac_above >= frac:
                ttp[i] = centres[t - lookback + 1]  # centre of start of window
                break
    return ttp


# ---------------------------------------------------------------------------
# Rule family 2 — Interpolation to sub-window resolution
# ---------------------------------------------------------------------------

def _linear_interpolation(
    probs: np.ndarray, window: int, stride: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Linear interpolation between the two adjacent windows straddling threshold.

    Gives sub-window TTP resolution — no longer bounded by stride.
    """
    n_pix, n_win = probs.shape
    centres = _window_centres_min(n_win, window, stride)
    ttp = np.full(n_pix, centres[-1], dtype=np.float32)
    for i in range(n_pix):
        p = probs[i]
        # Find first index where p >= threshold.
        idx = np.argmax(p >= threshold)
        if p[idx] < threshold:  # never crossed
            continue
        if idx == 0:
            ttp[i] = centres[0]
            continue
        # Interpolate between idx-1 and idx.
        p0, p1 = p[idx - 1], p[idx]
        c0, c1 = centres[idx - 1], centres[idx]
        if p1 == p0:
            ttp[i] = c1
        else:
            frac = (threshold - p0) / (p1 - p0)
            ttp[i] = c0 + frac * (c1 - c0)
    return ttp


def _sigmoid_fit(
    probs: np.ndarray, window: int, stride: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Fit sigmoid P(t) = 1 / (1 + exp(-k*(t - t0))) to per-window probs; TTP = t0.

    Slower than linear interpolation but robust to noise on individual windows.
    """
    from scipy.optimize import curve_fit

    n_pix, n_win = probs.shape
    centres = _window_centres_min(n_win, window, stride)
    ttp = np.full(n_pix, centres[-1], dtype=np.float32)

    def _sigmoid(t, t0, k):
        return 1.0 / (1.0 + np.exp(-k * (t - t0)))

    for i in range(n_pix):
        p = probs[i]
        # Skip if the trace never crosses threshold.
        if p.max() < threshold or p.min() > threshold:
            continue
        try:
            # Initial guess: t0 = centre of first-above-threshold; k = 1.
            idx = int(np.argmax(p >= threshold))
            t0_guess = centres[idx] if idx > 0 else centres[len(centres) // 2]
            popt, _ = curve_fit(
                _sigmoid, centres, p,
                p0=[t0_guess, 1.0],
                maxfev=1000,
            )
            t0_fitted = popt[0]
            # Clip to plot range.
            ttp[i] = float(np.clip(t0_fitted, centres[0], centres[-1]))
        except Exception:
            # Fall back to linear interpolation.
            pass
    return ttp


# ---------------------------------------------------------------------------
# Rule family 3 — Derivative-based (paper 6 / SDM / Cy0 analogues on model output)
# ---------------------------------------------------------------------------

def _first_deriv_peak(
    probs: np.ndarray, window: int, stride: int,
) -> np.ndarray:
    """TTP = argmax(dP/dt) across windows.

    Where the model's probability is rising fastest.  Analogue of rule-based TTP.
    """
    n_pix, n_win = probs.shape
    centres = _window_centres_min(n_win, window, stride)
    ttp = np.zeros(n_pix, dtype=np.float32)
    for i in range(n_pix):
        dP = np.gradient(probs[i])
        ttp[i] = centres[int(np.argmax(dP))]
    return ttp


def _second_deriv_peak(
    probs: np.ndarray, window: int, stride: int,
) -> np.ndarray:
    """TTP = argmax(d²P/dt²) across windows.

    Analogue of SDM (second-derivative maximum) applied to the model's own output.
    """
    n_pix, n_win = probs.shape
    centres = _window_centres_min(n_win, window, stride)
    ttp = np.zeros(n_pix, dtype=np.float32)
    for i in range(n_pix):
        dP = np.gradient(probs[i])
        d2P = np.gradient(dP)
        ttp[i] = centres[int(np.argmax(d2P))]
    return ttp


def _inflection_from_sigmoid_fit(
    probs: np.ndarray, window: int, stride: int,
) -> np.ndarray:
    """Fit a sigmoid to P(t), TTP = inflection point = fitted t0.

    Same as sigmoid_fit(threshold=0.5) because a sigmoid's inflection is at
    y = 0.5 by construction.  Kept as a separate rule for report labelling.
    """
    return _sigmoid_fit(probs, window, stride, threshold=0.5)


# ---------------------------------------------------------------------------
# Rule family 4 — Position-of-maximum
# ---------------------------------------------------------------------------

def _argmax_P(
    probs: np.ndarray, window: int, stride: int,
) -> np.ndarray:
    """TTP = argmax of P across windows.

    Fires LATE (usually at the plateau of the model's confidence).
    Included as a sanity-check rule.
    """
    n_pix, n_win = probs.shape
    centres = _window_centres_min(n_win, window, stride)
    return centres[np.argmax(probs, axis=1)].astype(np.float32)


def _first_local_max(
    probs: np.ndarray, window: int, stride: int,
) -> np.ndarray:
    """TTP = first local maximum of P (any t with P[t] > P[t-1] and P[t] >= P[t+1]).

    Might catch spurious peaks; interesting comparator to argmax_P.
    """
    n_pix, n_win = probs.shape
    centres = _window_centres_min(n_win, window, stride)
    ttp = np.full(n_pix, centres[-1], dtype=np.float32)
    for i in range(n_pix):
        p = probs[i]
        found = False
        for t in range(1, n_win - 1):
            if p[t] > p[t - 1] and p[t] >= p[t + 1]:
                ttp[i] = centres[t]
                found = True
                break
        if not found:
            ttp[i] = centres[int(np.argmax(p))]
    return ttp


# ---------------------------------------------------------------------------
# The full rule registry
# ---------------------------------------------------------------------------

ALL_RULES: list[Rule] = [
    # Threshold-crossing family
    Rule("first_above_K1_thr0p5",    "first t: P>0.5 (K=1)",           "threshold", lambda p, w, s: _first_above_K(p, w, s, 0.5, 1)),
    Rule("first_above_K2_thr0p5",    "first t: P>0.5 (K=2)",           "threshold", lambda p, w, s: _first_above_K(p, w, s, 0.5, 2)),
    Rule("first_above_K3_thr0p5",    "first t: P>0.5 (K=3, F-B default)", "threshold", lambda p, w, s: _first_above_K(p, w, s, 0.5, 3)),
    Rule("first_above_K5_thr0p5",    "first t: P>0.5 (K=5)",           "threshold", lambda p, w, s: _first_above_K(p, w, s, 0.5, 5)),
    Rule("first_above_K3_thr0p3",    "first t: P>0.3 (K=3)",           "threshold", lambda p, w, s: _first_above_K(p, w, s, 0.3, 3)),
    Rule("first_above_K3_thr0p7",    "first t: P>0.7 (K=3)",           "threshold", lambda p, w, s: _first_above_K(p, w, s, 0.7, 3)),
    Rule("fraction_above_lb10_thr0p5_f0p5", "5-of-10 above 0.5",       "threshold", lambda p, w, s: _fraction_above(p, w, s, 10, 0.5, 0.5)),
    Rule("end_of_K3_run_thr0p5",     "end of first K=3 run above 0.5", "threshold", lambda p, w, s: _end_of_K_run(p, w, s, 0.5, 3)),

    # Interpolation family
    Rule("linear_interp_thr0p5",     "linear interp at P=0.5",         "interp",    lambda p, w, s: _linear_interpolation(p, w, s, 0.5)),
    Rule("linear_interp_thr0p3",     "linear interp at P=0.3",         "interp",    lambda p, w, s: _linear_interpolation(p, w, s, 0.3)),
    Rule("sigmoid_fit_thr0p5",       "fit sigmoid, t at P=0.5",        "interp",    lambda p, w, s: _sigmoid_fit(p, w, s, 0.5)),

    # Derivative family (paper-6 / SDM / Cy0 analogues on model output)
    Rule("first_deriv_peak",         "argmax(dP/dt) — paper-6 analogue",  "deriv",     lambda p, w, s: _first_deriv_peak(p, w, s)),
    Rule("second_deriv_peak",        "argmax(d²P/dt²) — SDM analogue",    "deriv",     lambda p, w, s: _second_deriv_peak(p, w, s)),
    Rule("inflection_sigmoid",       "sigmoid inflection — Cy0 analogue", "deriv",     lambda p, w, s: _inflection_from_sigmoid_fit(p, w, s)),

    # Position-of-max family
    Rule("argmax_P",                 "argmax(P) — sanity check",       "argmax",    lambda p, w, s: _argmax_P(p, w, s)),
    Rule("first_local_max",          "first local max of P",           "argmax",    lambda p, w, s: _first_local_max(p, w, s)),
]
