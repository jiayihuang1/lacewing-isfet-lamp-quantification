"""Eval · joint-framework classification-head evaluation. [Cat A] Report §RQ3 classification-side.

Score the classification-head output of the joint framework on the
SD test chip, using a mixed test set of amp+ (wells 0-4) and NTC
(well 5) pixels drawn from the classification cache with matching
`_madk1p5_spatA3` preprocessing.

The joint framework was trained on `regress_all_filt_abcd_ntcRaw_madk1p5_spatA3`
(regression cache, amp+ only). The classification cache
`dataset_all_filt_abcd_ntcRaw_madk1p5_spatA3` uses identical
preprocessing but keeps NTC pixels, so we can score the cls-head on
a proper mixed test set that includes negatives.

Per-pixel cls_prob is the mean over the pixel's sliding windows (same
convention the joint framework uses internally). Per-well cls decision
is majority-vote of per-pixel cls_prob >= 0.5.

Usage
-----
    python -m lacewing.quantification.eval.cls_side_eval \\
        --run-dir Analysis/quantification/methods/results/p3_a1_alpha0.5_unet_w120_stride10_ptcontrastive/seed0 \\
        --window 120 --stride 10

Outputs to `<run-dir>/cls_side_eval.npz`:
    cls_prob         (N_pixels,)  float32   per-pixel cls probability
    cls_label        (N_pixels,)  uint8     1 = amp+, 0 = NTC
    chip_id          (N_pixels,)  <U64
    well_id          (N_pixels,)  uint8
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lacewing.quantification.methods.joint_cls_quant import (
    JointModel, build_joint_model, SAMPLES_PER_MIN,
)
from lacewing.quantification.methods._windowing import slide_windows

CLS_CACHE = Path(
    "Analysis/classification/data/cache/"
    "dataset_all_filt_abcd_ntcRaw_madk1p5_spatA3.npz"
)
SD_CHIP = "D20240719_E03_C44_F4500KHz_U_COV_SD"


def load_sd_test_set() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, chip, well) for the SD test chip: 5017 amp+ + 1311 NTC."""
    d = np.load(CLS_CACHE, allow_pickle=True)
    chip = d["chip_id"]
    mask = chip == SD_CHIP
    if not mask.any():
        raise RuntimeError(f"SD chip {SD_CHIP} not found in {CLS_CACHE}")
    X = d["X"][mask].astype(np.float32)
    y = d["y"][mask].astype(np.uint8)
    well = d["well_id"][mask].astype(np.uint8)
    return X, y, chip[mask], well


def run_cls_inference(
    ckpt_path: Path,
    X: np.ndarray,
    window: int,
    stride: int,
    device: str = "cpu",
    batch_size: int = 256,
    arch: str = "a1",
    alpha: float = 0.5,
) -> np.ndarray:
    """Run cls-head inference on X (N, 450), return per-pixel cls_prob (N,)."""
    model = build_joint_model(arch=arch, window=window, alpha=alpha)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model = model.to(device).eval()

    test_windows = slide_windows(X, window, stride)  # (N, W_count, W)
    n, n_wins, _ = test_windows.shape
    flat = test_windows.reshape(-1, window)

    cls_flat = np.zeros(len(flat), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            end = min(start + batch_size, len(flat))
            xb = torch.from_numpy(flat[start:end]).unsqueeze(1).to(device)
            _, c_logit = model(xb)
            cls_flat[start:end] = torch.sigmoid(c_logit).cpu().numpy()

    # Per-pixel = mean over its windows (matches training-time convention)
    cls_probs_per_win = cls_flat.reshape(n, n_wins)
    return cls_probs_per_win.mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to a joint-framework seed directory (contains checkpoints/best.pt and config.json)")
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ckpt", default="best.pt", choices=["best.pt", "last.pt"])
    args = parser.parse_args()

    # Load config to know arch + alpha
    cfg = json.loads((args.run_dir / "config.json").read_text())
    arch = cfg["arch"]
    alpha = float(cfg["alpha"])
    ckpt_path = args.run_dir / "checkpoints" / args.ckpt

    print(f"[load] {args.run_dir}  (arch={arch}, alpha={alpha}, ckpt={args.ckpt})")
    X, y, chip, well = load_sd_test_set()
    print(f"[data] SD test chip: {len(X)} pixels ({(y==1).sum()} amp+, {(y==0).sum()} NTC)")

    cls_prob = run_cls_inference(
        ckpt_path, X, args.window, args.stride,
        device=args.device, batch_size=args.batch_size,
        arch=arch, alpha=alpha,
    )

    out_path = args.run_dir / "cls_side_eval.npz"
    np.savez_compressed(
        out_path,
        cls_prob=cls_prob.astype(np.float32),
        cls_label=y,
        chip_id=chip,
        well_id=well,
    )
    print(f"[save] {out_path}")

    # Print a quick summary
    thr = 0.5
    pred = (cls_prob >= thr).astype(np.uint8)
    acc = (pred == y).mean()
    pos_mask = y == 1
    neg_mask = y == 0
    tp = int((pred[pos_mask] == 1).sum())
    fn = int((pred[pos_mask] == 0).sum())
    tn = int((pred[neg_mask] == 0).sum())
    fp = int((pred[neg_mask] == 1).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"[summary @ thr={thr}]")
    print(f"  per-pixel accuracy:  {acc:.4f}  ({int((pred==y).sum())}/{len(y)})")
    print(f"  precision (amp+):    {precision:.4f}")
    print(f"  recall (amp+):       {recall:.4f}")
    print(f"  F1 (amp+):           {f1:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")


if __name__ == "__main__":
    main()
