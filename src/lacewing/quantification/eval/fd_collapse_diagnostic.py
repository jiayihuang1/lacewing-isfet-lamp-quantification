"""F-D collapse diagnostic: why do seeds 1 and 2 predict ~21 min for every SD well?

Analyses:
  1. Class-2 (rising) first-index distribution in each seed's training labels.
  2. Per-well pixel counts and class balance across training subsets.
  3. Model behaviour on TRAINING pixels at inference time.
  4. Overlap of training-pixel sets across the 3 seeds.

Writes a markdown report to docs/superpowers/plans/2026-07-22-fd-collapse-diagnostic.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lacewing.quantification.methods import unet_segmentation as u
from lacewing.quantification.methods.unet_segmentation import _load_manual_labels_for_pixels
from lacewing.quantification.regression_shared.data import dataset as reg_ds
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


RESULTS = LACEWING_PKG_DIR / "quantification" / "methods" / "results" / "p1_fd_manual4_spatA3"
OUT = Path("docs/superpowers/plans/2026-07-22-fd-collapse-diagnostic.md")


def _first_rising_index(labels_2d: np.ndarray, rising_class: int = 2) -> np.ndarray:
    """(N, T) label matrix → (N,) first index where label==rising_class; -1 if absent."""
    mask = labels_2d == rising_class
    first = np.where(mask.any(axis=1),
                     mask.argmax(axis=1),
                     np.full(labels_2d.shape[0], -1))
    return first


def _analyse_seed(seed: int) -> dict:
    arr = reg_ds.make_split("regress_all_filt_abcd_ntcRaw_madk1p5_spatA3", seed=seed)
    mask, labels_mat = _load_manual_labels_for_pixels(
        arr.pixel_tr, LACEWING_PKG_DIR / "quantification" / "methods" / "fd_labelling" / "manual_labels.npz"
    )
    matched_pixel_ids = arr.pixel_tr[mask]

    # Class-2 first index distribution
    first_rising = _first_rising_index(labels_mat, rising_class=2)
    first_rising = first_rising[first_rising >= 0]  # drop pixels with no rising

    # Model behaviour on training pixels
    ck_path = RESULTS / f"seed{seed}" / "checkpoints" / "last.pt"
    model = u.build_model()
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    X_tr_matched = arr.X_tr[mask]

    train_pred_first = []
    with torch.no_grad():
        for i in range(min(len(X_tr_matched), 500)):  # subsample for speed
            x = torch.from_numpy(X_tr_matched[i].astype(np.float32))[None, None, :]
            pred = model(x).argmax(dim=1).squeeze(0).numpy()
            idxs = np.where(pred == 2)[0]
            train_pred_first.append(int(idxs[0]) if len(idxs) else -1)
    train_pred_first = np.array(train_pred_first)
    train_pred_valid = train_pred_first[train_pred_first >= 0]

    return {
        "seed": seed,
        "n_matched_pixels": int(mask.sum()),
        "matched_pixel_ids": matched_pixel_ids,
        "train_label_first_rising_mean": float(first_rising.mean()) if len(first_rising) else float("nan"),
        "train_label_first_rising_std": float(first_rising.std()) if len(first_rising) else float("nan"),
        "train_label_first_rising_min": int(first_rising.min()) if len(first_rising) else -1,
        "train_label_first_rising_max": int(first_rising.max()) if len(first_rising) else -1,
        "train_pred_first_rising_mean": float(train_pred_valid.mean()) if len(train_pred_valid) else float("nan"),
        "train_pred_first_rising_std": float(train_pred_valid.std()) if len(train_pred_valid) else float("nan"),
        "train_pred_no_rising_count": int((train_pred_first < 0).sum()),
    }


def main() -> None:
    stats = [_analyse_seed(s) for s in (0, 1, 2)]

    # Cross-seed pixel-ID overlap
    ids = [set(s["matched_pixel_ids"].tolist()) for s in stats]
    overlap_01 = len(ids[0] & ids[1])
    overlap_02 = len(ids[0] & ids[2])
    overlap_12 = len(ids[1] & ids[2])
    overlap_all = len(ids[0] & ids[1] & ids[2])

    lines = [
        "# F-D collapse diagnostic",
        "",
        "**Date:** 2026-07-22",
        "**Question:** Why do seeds 1 and 2 predict ~21 min TTP for every SD well, while seed 0 produces a non-collapsed but ordering-broken result?",
        "",
        "## Per-seed training subset statistics",
        "",
        "| Seed | Matched pixels | Label class-2 first-index (mean / std) | Model pred class-2 first-index on train (mean / std) | No-rising pred count |",
        "|---|---:|---|---|---:|",
    ]
    for s in stats:
        lines.append(
            f"| {s['seed']} | {s['n_matched_pixels']} "
            f"| {s['train_label_first_rising_mean']:.1f} / {s['train_label_first_rising_std']:.1f} "
            f"(range {s['train_label_first_rising_min']}-{s['train_label_first_rising_max']}) "
            f"| {s['train_pred_first_rising_mean']:.1f} / {s['train_pred_first_rising_std']:.1f} "
            f"| {s['train_pred_no_rising_count']} |"
        )
    lines += [
        "",
        "## Training-pixel overlap between seeds",
        "",
        f"- seed 0 ∩ seed 1: {overlap_01} pixels",
        f"- seed 0 ∩ seed 2: {overlap_02} pixels",
        f"- seed 1 ∩ seed 2: {overlap_12} pixels",
        f"- all three: {overlap_all} pixels",
        "",
        "## Interpretation",
        "",
        "(Fill in after running: look for narrow label-first-rising distributions vs wide model-pred distributions, cross-seed disjoint pixel subsets, etc.)",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
