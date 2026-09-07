"""Extraction-rule sweep for F-D 4-class segmentation.

Reads existing per-timestep predictions cache (argmax + softmax), applies
4 different rules, appends one scoreboard row per (rule, K, seed).

Usage:
    python -m lacewing.quantification.eval.fd_extraction_sweep
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lacewing.quantification.eval.schema import PredictionArrays, read_labels, validate_alignment
from lacewing.quantification.eval.scoreboard import (
    compute_all_metrics,
    append_scoreboard_row,
)
from lacewing.quantification.methods.unet_segmentation import (
    _predict_ttp_from_argmax,
    SAMPLES_PER_MIN,
    N_SAMPLES,
)
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT

# Expose k-consecutive extraction under the fd_extraction_sweep namespace for API consistency.
_extract_kconsecutive = _predict_ttp_from_argmax

SCOREBOARD = LACEWING_PKG_DIR / "quantification" / "eval" / "scoreboard.csv"
RESULTS_ROOT = LACEWING_PKG_DIR / "quantification" / "methods" / "results"

# (dir_name, ttp_classes)
TARGETS = [
    ("p1_fd_manual4_spatA3",       frozenset({2})),         # manual: rising = 2
    ("p1_fd_manual4_spatA3_bb_gru", frozenset({2})),        # F-D backbone swap: GRU
    ("p1_fd_manual4_spatA3_bb_cnn_gru_par", frozenset({2})),  # F-D backbone swap: CNN-GRU-par
    ("p1_fd_unet_seg_ow30_spatA3", frozenset({1, 2})),      # rule-based: onset-window + amp
]


def _extract_smoothed_prob(
    softmax: np.ndarray,
    target_class: int = 2,
    window: int = 5,
    threshold: float = 0.5,
) -> np.ndarray:
    """Smooth per-timestep P(target_class) via moving average, take first
    index where smoothed prob > threshold.

    Args:
        softmax: (N, 4, T) softmax posteriors
        target_class: which class is "rising"
        window: moving-average window in samples
        threshold: probability cutoff

    Returns:
        (N,) TTP in minutes
    """
    N, _, T = softmax.shape
    probs = softmax[:, target_class, :].astype(np.float32)  # (N, T)
    # Moving average with reflection padding to preserve edges.
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.empty_like(probs)
    for i in range(N):
        smoothed[i] = np.convolve(probs[i], kernel, mode="same")
    ttp_pred = np.full(N, N_SAMPLES - 1, dtype=np.float32)
    above = smoothed > threshold
    for i in range(N):
        idxs = np.where(above[i])[0]
        if len(idxs):
            ttp_pred[i] = idxs[0]
    return (ttp_pred / SAMPLES_PER_MIN).astype(np.float32)


def _extract_largest_block(
    argmax: np.ndarray,
    target_class: int = 2,
) -> np.ndarray:
    """Find the largest contiguous run of target_class in each row; return
    its start index converted to minutes.  If no target_class present,
    fallback to N_SAMPLES - 1 / SAMPLES_PER_MIN.
    """
    N, T = argmax.shape
    ttp_pred = np.full(N, N_SAMPLES - 1, dtype=np.float32)
    for i in range(N):
        row = argmax[i] == target_class
        if not row.any():
            continue
        # Find run boundaries via diff.
        diffs = np.diff(row.astype(np.int8))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        if row[0]:
            starts = np.insert(starts, 0, 0)
        if row[-1]:
            ends = np.append(ends, T)
        lengths = ends - starts
        biggest = int(np.argmax(lengths))
        ttp_pred[i] = starts[biggest]
    return (ttp_pred / SAMPLES_PER_MIN).astype(np.float32)


def _extract_viterbi_monotonic(
    softmax: np.ndarray,
    rising_class: int = 2,
    n_classes: int = 4,
    eps: float = 1e-8,
) -> np.ndarray:
    """Constrained-Viterbi decode: allow only non-decreasing class transitions.

    Emission log-probs = log(softmax + eps).  Transition log-probs =
    0 for c_{t+1} >= c_t (allowed), -inf for c_{t+1} < c_t (forbidden).

    Returns (N,) TTP in minutes = first t where decoded path reaches
    rising_class.  Fallback: last-index if never reaches rising_class.
    """
    N, C, T = softmax.shape
    logp = np.log(softmax.astype(np.float32) + eps)   # (N, C, T)
    ttp_pred = np.full(N, N_SAMPLES - 1, dtype=np.float32)
    NEG_INF = -1e18

    for i in range(N):
        # dp[c, t] = best log-prob ending in class c at t
        dp = np.full((C, T), NEG_INF, dtype=np.float32)
        bt = np.zeros((C, T), dtype=np.int8)
        dp[:, 0] = logp[i, :, 0]  # start anywhere at t=0
        for t in range(1, T):
            for c in range(C):
                # allowed predecessors: c' <= c
                best_prev = NEG_INF
                best_c = 0
                for cp in range(c + 1):
                    if dp[cp, t - 1] > best_prev:
                        best_prev = dp[cp, t - 1]
                        best_c = cp
                dp[c, t] = best_prev + logp[i, c, t]
                bt[c, t] = best_c
        # Backtrack from best terminal class
        path = np.zeros(T, dtype=np.int8)
        path[T - 1] = int(np.argmax(dp[:, T - 1]))
        for t in range(T - 1, 0, -1):
            path[t - 1] = bt[path[t], t]
        # First index where path == rising_class
        idxs = np.where(path == rising_class)[0]
        if len(idxs):
            ttp_pred[i] = idxs[0]
    return (ttp_pred / SAMPLES_PER_MIN).astype(np.float32)


def _load_cache(seed_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    p = seed_dir / "test_time_predictions.npz"
    if not p.exists():
        return None
    with np.load(p) as npz:
        return npz["argmax"], npz["softmax"]


def _score(method_id: str, ttp_pred: np.ndarray, labels, config: dict) -> None:
    preds = PredictionArrays(ttp_pred_min=ttp_pred)
    validate_alignment(preds, labels)
    metrics = compute_all_metrics(preds, labels, split="test")
    append_scoreboard_row(SCOREBOARD, method_id=method_id, config_json=config, metrics=metrics)
    flag = "✓" if metrics["passes_spearman_filter"] else "✗"
    print(f"  [ok] {method_id}: MAE={metrics['per_well_mae_min']:.2f} ρ={metrics['spearman_r_well_means']:+.2f} {flag}")


def main() -> None:
    for dir_name, target_cls in TARGETS:
        root = RESULTS_ROOT / dir_name
        if not root.exists():
            continue
        print(f"=== {dir_name} (target_cls={sorted(target_cls)}) ===")
        for seed_dir in sorted(root.iterdir()):
            if not seed_dir.is_dir():
                continue
            cached = _load_cache(seed_dir)
            if cached is None:
                print(f"  [skip] {seed_dir.name}: no test_time_predictions.npz")
                continue
            argmax, softmax = cached
            labels = read_labels(seed_dir / "labels.npz")
            cfg = json.loads((seed_dir / "config.json").read_text())
            seed_num = int(seed_dir.name.replace("seed", ""))

            # Rule 1a-c: K-consecutive with K ∈ {2, 3, 5}
            for K in (2, 3, 5):
                ttp = _predict_ttp_from_argmax(argmax, target_cls=target_cls, k_consecutive=K)
                _score(f"{dir_name}_kcons{K}_seed{seed_num}", ttp, labels,
                       {**cfg, "extraction_rule": f"kcons{K}"})

            # Rule 2: smoothed probability (rising class only for both label modes)
            rising_cls = 2 if 2 in target_cls else max(target_cls)
            ttp = _extract_smoothed_prob(softmax, target_class=rising_cls, window=5, threshold=0.5)
            _score(f"{dir_name}_smoothedP_seed{seed_num}", ttp, labels,
                   {**cfg, "extraction_rule": "smoothed_prob_w5_thr0.5"})

            # Rule 3: largest contiguous block
            ttp = _extract_largest_block(argmax, target_class=rising_cls)
            _score(f"{dir_name}_largestBlock_seed{seed_num}", ttp, labels,
                   {**cfg, "extraction_rule": "largest_block"})

            # Rule 4: Viterbi monotonic
            ttp = _extract_viterbi_monotonic(softmax, rising_class=rising_cls)
            _score(f"{dir_name}_viterbiMono_seed{seed_num}", ttp, labels,
                   {**cfg, "extraction_rule": "viterbi_monotonic"})


if __name__ == "__main__":
    main()
