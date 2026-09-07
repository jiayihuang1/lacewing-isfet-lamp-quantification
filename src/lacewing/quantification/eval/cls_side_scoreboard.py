"""Aggregate `cls_side_eval.npz` outputs across the two 5-seed contenders
(A1 alpha=0.5 SSL 50-ep and A3 alpha=0.5 SSL 100-ep) into a summary table.

For each seed and each contender, reports:
  - per-pixel accuracy, precision (amp+), recall (amp+), F1 (amp+)
  - per-well accuracy (majority-vote of per-pixel prob >= 0.5)

For each contender, reports the mean and std across 5 seeds.

Usage
-----
    python -m lacewing.quantification.eval.cls_side_scoreboard

Writes to `Analysis/quantification/eval/cls_side_scoreboard.csv`.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT

CONTENDERS = [
    ("A1_alpha0.5_SSL_50ep", "p3_a1_alpha0.5_unet_w120_stride10_ptcontrastive"),
    ("A3_alpha0.5_SSL_100ep", "p3_a3_alpha0.5_unet_w120_stride10_ptcontrastive_epochs100"),
]
RESULTS_ROOT = LACEWING_PKG_DIR / "quantification" / "methods" / "results"


def per_seed_metrics(prob: np.ndarray, label: np.ndarray, well: np.ndarray,
                     thr: float = 0.5) -> dict:
    pred = (prob >= thr).astype(np.uint8)
    pos = label == 1
    neg = label == 0
    tp = int((pred[pos] == 1).sum())
    fn = int((pred[pos] == 0).sum())
    tn = int((pred[neg] == 0).sum())
    fp = int((pred[neg] == 1).sum())
    acc_px = (pred == label).mean()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # Per-well accuracy: majority-vote per-pixel decisions per well; well
    # label = 1 if it has any amp+ pixels, else 0
    per_well_correct = 0
    per_well_n = 0
    for w in np.unique(well):
        wmask = well == w
        well_pred = 1 if pred[wmask].mean() >= 0.5 else 0
        well_true = int(label[wmask].max())
        per_well_correct += int(well_pred == well_true)
        per_well_n += 1
    acc_well = per_well_correct / per_well_n

    return {
        "acc_pixel": float(acc_px),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc_well": acc_well,
        "n_wells": per_well_n,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main() -> None:
    rows = []
    for label, cfg in CONTENDERS:
        for seed in range(5):
            eval_path = RESULTS_ROOT / cfg / f"seed{seed}" / "cls_side_eval.npz"
            if not eval_path.exists():
                print(f"[skip] {label} seed{seed}: {eval_path} missing")
                continue
            d = np.load(eval_path, allow_pickle=True)
            m = per_seed_metrics(d["cls_prob"], d["cls_label"], d["well_id"])
            m["contender"] = label
            m["seed"] = seed
            rows.append(m)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No cls_side_eval.npz files found. Run cls_side_eval.py first.")
        return

    # Order columns
    cols = ["contender", "seed", "acc_pixel", "precision", "recall", "f1",
            "acc_well", "n_wells", "tp", "fp", "tn", "fn"]
    df = df[cols]

    out_csv = LACEWING_PKG_DIR / "quantification" / "eval" / "cls_side_scoreboard.csv"
    df.to_csv(out_csv, index=False)
    print(f"[write] {out_csv}")

    # Per-contender summary
    print("\n=== per-contender means over 5 seeds (passing seeds only) ===")
    print(f"{'contender':<24s} {'acc_px':>8s} {'precision':>10s} {'recall':>8s} {'f1':>8s} {'acc_well':>10s}")
    for label, _ in CONTENDERS:
        sub = df[df["contender"] == label]
        if len(sub) == 0:
            continue
        print(f"{label:<24s} "
              f"{sub['acc_pixel'].mean():>8.4f} "
              f"{sub['precision'].mean():>10.4f} "
              f"{sub['recall'].mean():>8.4f} "
              f"{sub['f1'].mean():>8.4f} "
              f"{sub['acc_well'].mean():>10.4f}")

    print("\n=== per-seed table ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
