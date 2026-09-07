"""RQ1 · classifier evaluation on held-out chips. [Cat A] Report §RQ1.

Pixel-level + well-level metrics + confusion matrices.

Used both inside the training loop (for per-epoch val metrics) and
standalone for re-evaluating saved predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


@dataclass
class PixelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    n_pos: int
    n_neg: int


def pixel_metrics(y_true: np.ndarray, y_score: np.ndarray,
                  threshold: float = 0.5) -> PixelMetrics:
    y_pred = (y_score >= threshold).astype(int)
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = float("nan")
    return PixelMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auroc=auroc,
        n_pos=int((y_true == 1).sum()),
        n_neg=int((y_true == 0).sum()),
    )


def well_majority_vote(y_true: np.ndarray, y_score: np.ndarray,
                       chip_id: np.ndarray, well_id: np.ndarray,
                       threshold: float = 0.5
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-pixel votes into per-well predictions.

    A well is positive iff > 50% of its pixels vote positive (Paper 3 §V).

    Returns (well_keys, well_truth, well_pred) - aligned arrays.
    """
    keys = np.array([f"{c}::{w}" for c, w in zip(chip_id, well_id)])
    unique = np.unique(keys)
    truths, preds = [], []
    for k in unique:
        mask = (keys == k)
        truths.append(int(y_true[mask][0]))     # all pixels in a well share label
        pos_frac = float((y_score[mask] >= threshold).mean())
        preds.append(int(pos_frac > 0.5))
    return unique, np.array(truths), np.array(preds)


def well_metrics(y_true: np.ndarray, y_score: np.ndarray,
                 chip_id: np.ndarray, well_id: np.ndarray,
                 threshold: float = 0.5) -> dict:
    keys, truth, pred = well_majority_vote(
        y_true, y_score, chip_id, well_id, threshold)
    return dict(
        n_wells=int(len(keys)),
        accuracy=float(accuracy_score(truth, pred)) if len(keys) else float("nan"),
        precision=float(precision_score(truth, pred, zero_division=0)),
        recall=float(recall_score(truth, pred, zero_division=0)),
        f1=float(f1_score(truth, pred, zero_division=0)),
    )


def save_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray,
                            well_truth: np.ndarray, well_pred: np.ndarray,
                            out_dir: Path) -> None:
    """Two PNGs: pixel-level CM and well-level CM."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    out_dir.mkdir(parents=True, exist_ok=True)

    cm_pix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm_pix, display_labels=["NTC", "Pos"]).plot(ax=ax)
    ax.set_title("Pixel-level confusion matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "cm_pixel.png", dpi=120)
    plt.close(fig)

    if len(well_truth) > 0:
        cm_well = confusion_matrix(well_truth, well_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(cm_well, display_labels=["NTC", "Pos"]).plot(ax=ax)
        ax.set_title("Well-level confusion matrix (majority vote)")
        fig.tight_layout()
        fig.savefig(out_dir / "cm_well.png", dpi=120)
        plt.close(fig)
