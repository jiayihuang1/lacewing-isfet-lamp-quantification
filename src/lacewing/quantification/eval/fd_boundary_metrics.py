"""Boundary-timing metrics for F-D 4-class segmentation failure analysis.

The F-D U-Net collapses on 66.5% of test pixels (backward class
transitions) and picks up spurious early "rising" blips as TTP. Standard
per-timestep accuracy doesn't distinguish "off by 2 samples at a critical
boundary" from "off by 2 samples in the middle of a long flat segment" —
both cost one wrong timestep, but only the former matters for TTP
extraction. This module adds:

  1. compute_boundary_errors — per-pixel |pred_boundary - true_boundary|
     in samples, for each of the 3 class transitions (baseline->drift,
     drift->rising, rising->post-amp). Sentinel -1 when the class is
     absent from either the prediction or the ground truth.

  2. boundary_weighted_accuracy — per-pixel accuracy that up-weights
     timesteps near a true class boundary (+/- K samples), since errors
     there are the ones that actually move TTP.

  3. main() — CLI that walks the F-D seed dirs (base U-Net + backbone
     variants), recomputes these metrics on the internal validation split
     used during training (see unet_segmentation.main's
     `--internal-val-frac` logic), and writes an aggregate scoreboard CSV.

Usage
-----
    python -m lacewing.quantification.eval.fd_boundary_metrics
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from lacewing.quantification.methods.unet_segmentation import (
    CLS_ONSET_WINDOW as _CLS_DRIFT,  # class 1 in manual schema = "drift"
    CLS_AMP as _CLS_RISING,          # class 2 in manual schema = "rising"
    CLS_PLATEAU as _CLS_POST_AMP,    # class 3 in manual schema = "post-amp"
    _load_manual_labels_for_pixels,
    build_model,
)
from lacewing.quantification.methods.unet_segmentation_backbones import build_fd_backbone
from lacewing.quantification.regression_shared.data import dataset as reg_ds
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT

RESULTS_ROOT = LACEWING_PKG_DIR / "quantification" / "methods" / "results"
OUT_CSV = LACEWING_PKG_DIR / "quantification" / "eval" / "fd_boundary_scoreboard.csv"
MANUAL_LABELS_PATH = LACEWING_PKG_DIR / "quantification" / "methods" / "fd_labelling" / "manual_labels.npz"
CACHE_STEM = "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3"

# Boundary classes: (name, class_a -> class_b) — a boundary is the first
# timestep where the sequence reaches class_b.
_BOUNDARIES = [
    ("bd_error", 1),  # baseline(0) -> drift(1)
    ("dr_error", 2),  # drift(1) -> rising(2)
    ("rp_error", 3),  # rising(2) -> post-amp(3)
]

# Method dirs to sweep in main().
SEED_DIRS = [
    "p1_fd_manual4_spatA3",
    "p1_fd_manual4_spatA3_bb_gru",
    "p1_fd_manual4_spatA3_bb_cnn_gru_par",
]


def _first_index(arr: np.ndarray, cls: int) -> int | None:
    idxs = np.where(arr == cls)[0]
    if len(idxs) == 0:
        return None
    return int(idxs[0])


def compute_boundary_errors(pred_argmax: np.ndarray, true_label: np.ndarray) -> dict[str, int]:
    """Per-pixel boundary-timing errors for the 3 class transitions.

    Parameters
    ----------
    pred_argmax, true_label : (T,) int arrays for a single pixel.

    Returns
    -------
    dict with keys bd_error, dr_error, rp_error (sample-count int).
    Value is -1 (sentinel) if the target class is absent from either
    array — that pixel/boundary should be filtered out of aggregation.
    """
    out: dict[str, int] = {}
    for key, cls in _BOUNDARIES:
        p_idx = _first_index(pred_argmax, cls)
        t_idx = _first_index(true_label, cls)
        if p_idx is None or t_idx is None:
            out[key] = -1
        else:
            out[key] = abs(p_idx - t_idx)
    return out


def boundary_weighted_accuracy(
    pred_argmax: np.ndarray,
    true_label: np.ndarray,
    K: int = 15,
    weight: float = 5.0,
) -> float:
    """Per-pixel accuracy with up-weighted boundary-adjacent timesteps.

    Timesteps within K samples of any true class-transition boundary get
    weight `weight`; all others get weight 1.0.

    Parameters
    ----------
    pred_argmax, true_label : (T,) int arrays for a single pixel.
    K : window half-width in samples.
    weight : weight applied to boundary-adjacent timesteps.

    Returns
    -------
    Scalar float in [0, 1].
    """
    T = len(true_label)
    t = np.arange(T)

    # True boundary indices = first index where true_label reaches each
    # class in {1, 2, 3} (i.e. the 3 transitions), when present.
    boundary_idxs = []
    for cls in (1, 2, 3):
        idx = _first_index(true_label, cls)
        if idx is not None:
            boundary_idxs.append(idx)

    weights = np.ones(T, dtype=np.float64)
    if boundary_idxs:
        near_boundary = np.zeros(T, dtype=bool)
        for b in boundary_idxs:
            near_boundary |= np.abs(t - b) <= K
        weights[near_boundary] = weight

    correct = (pred_argmax == true_label).astype(np.float64)
    return float((correct * weights).sum() / weights.sum())


# ---------------------------------------------------------------------------
# CLI: recompute on the internal-val split for each seed dir / checkpoint.
# ---------------------------------------------------------------------------

def _internal_val_split(seed: int, internal_val_frac: float = 0.1):
    """Reproduce the internal train/val split used by unet_segmentation.main
    for --labels-mode manual, returning (X_va, y_va_labels)."""
    arr = reg_ds.make_split(CACHE_STEM, seed=seed, val_n_chips=1, normalise_y=False)
    matched_mask, matched_labels = _load_manual_labels_for_pixels(arr.pixel_tr, MANUAL_LABELS_PATH)
    n_matched = int(matched_mask.sum())
    X_labelled = arr.X_tr[matched_mask]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_matched)
    n_val_internal = max(1, int(n_matched * internal_val_frac))
    val_idx = np.sort(perm[:n_val_internal])

    X_va = X_labelled[val_idx]
    y_va_labels = matched_labels[val_idx]
    return X_va, y_va_labels


def _predict_argmax_for_pixels(model, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    n_pixels, T = X.shape
    out = np.empty((n_pixels, T), dtype=np.int8)
    with torch.no_grad():
        for i in range(n_pixels):
            x = torch.from_numpy(X[i].astype(np.float32))[None, None, :].to(device)
            logits = model(x)
            out[i] = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int8)
    return out


def _backbone_for_dir(dir_name: str) -> str:
    if dir_name.endswith("_bb_gru"):
        return "gru"
    if dir_name.endswith("_bb_cnn_gru_par"):
        return "cnn_gru_par"
    return "unet"


def _aggregate_row(method_id: str, seed: int, ckpt: str, n_val_pixels: int,
                    bd_errs: np.ndarray, dr_errs: np.ndarray, rp_errs: np.ndarray,
                    weighted_accs: np.ndarray) -> dict:
    def _stats(errs: np.ndarray, prefix: str) -> dict:
        valid = errs[errs >= 0]
        if len(valid) == 0:
            return {f"median_{prefix}_error": None, f"p90_{prefix}_error": None}
        return {
            f"median_{prefix}_error": float(np.median(valid)),
            f"p90_{prefix}_error": float(np.percentile(valid, 90)),
        }

    row = {
        "method_id": method_id,
        "seed": seed,
        "ckpt": ckpt,
        "n_val_pixels": n_val_pixels,
    }
    row.update(_stats(bd_errs, "bd"))
    dr_stats = _stats(dr_errs, "dr")
    row.update(dr_stats)
    dr_valid = dr_errs[dr_errs >= 0]
    row["dr_within_5samp_frac"] = (
        float((dr_valid <= 5).mean()) if len(dr_valid) else None
    )
    row.update(_stats(rp_errs, "rp"))
    row["boundary_weighted_acc"] = float(np.mean(weighted_accs)) if len(weighted_accs) else None
    return row


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows: list[dict] = []

    for dir_name in SEED_DIRS:
        root = RESULTS_ROOT / dir_name
        if not root.exists():
            print(f"[skip] {root} does not exist")
            continue
        backbone = _backbone_for_dir(dir_name)
        for seed_dir in sorted(root.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                continue
            seed = int(seed_dir.name.replace("seed", ""))
            ckpt_dir = seed_dir / "checkpoints"
            if not ckpt_dir.exists():
                print(f"  [skip] {seed_dir}: no checkpoints/")
                continue

            X_va, y_va = _internal_val_split(seed)
            n_val_pixels = len(X_va)

            for ckpt_name in ("best.pt", "last.pt"):
                ckpt_path = ckpt_dir / ckpt_name
                if not ckpt_path.exists():
                    print(f"  [skip] {ckpt_path}: missing")
                    continue
                model = build_fd_backbone(backbone, n_classes=4).to(device)
                state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state["model_state"])

                pred_argmax = _predict_argmax_for_pixels(model, X_va, device)

                bd_errs = np.empty(n_val_pixels, dtype=np.int64)
                dr_errs = np.empty(n_val_pixels, dtype=np.int64)
                rp_errs = np.empty(n_val_pixels, dtype=np.int64)
                weighted_accs = np.empty(n_val_pixels, dtype=np.float64)
                for i in range(n_val_pixels):
                    errs = compute_boundary_errors(pred_argmax[i], y_va[i])
                    bd_errs[i] = errs["bd_error"]
                    dr_errs[i] = errs["dr_error"]
                    rp_errs[i] = errs["rp_error"]
                    weighted_accs[i] = boundary_weighted_accuracy(pred_argmax[i], y_va[i])

                method_id = f"{dir_name}_seed{seed}_{ckpt_name.replace('.pt', '')}"
                row = _aggregate_row(
                    method_id, seed, ckpt_name, n_val_pixels,
                    bd_errs, dr_errs, rp_errs, weighted_accs,
                )
                rows.append(row)
                print(
                    f"  [ok] {method_id}: n={n_val_pixels} "
                    f"median_dr_error={row['median_dr_error']} "
                    f"boundary_weighted_acc={row['boundary_weighted_acc']:.4f}"
                )

    if not rows:
        print("[warn] no rows computed — nothing written")
        return

    fieldnames = [
        "method_id", "seed", "ckpt", "n_val_pixels",
        "median_bd_error", "p90_bd_error",
        "median_dr_error", "p90_dr_error", "dr_within_5samp_frac",
        "median_rp_error", "p90_rp_error",
        "boundary_weighted_acc",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[ok] wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
