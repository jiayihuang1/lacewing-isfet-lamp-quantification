"""F-D retrain on UTI cache with plate-anchored manual-TTP labels (A5).

Loads Analysis/quantification/methods/results/uti_seg_manual_labels_v1/cache.npz,
splits by CHIP (hold one UTI chip out as test), and trains the 1D U-Net
segmentation model.

The label schema matches the manual F-D convention:
    0 = baseline
    1 = drift
    2 = rising
    3 = plateau
TTP extraction at test time = first timestep where argmax ∈ {rising}.

Output dir: Analysis/quantification/methods/results/fd_uti_plate_labels/
    seed{S}_holdout_{chip_tag}/
        config.json
        history.json
        predictions.npz     — per-pixel predicted TTP + argmax array
        labels.npz          — per-pixel true TTP (from b_drift_end / SAMPLES_PER_MIN)
        checkpoints/best.pt, last.pt

Usage
-----
    python -m lacewing.quantification.methods.fd_train_uti \\
        --seed 0 --holdout-chip uti_260728_EC_SD --epochs 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lacewing.quantification.methods.unet_segmentation import (
    _TraceDatasetFromLabels, _train, _predict_ttp, build_model,
    SAMPLES_PER_MIN,
)
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


CACHE_PATH = LACEWING_PKG_DIR / "quantification" / "methods" / "results" / "uti_seg_manual_labels_v1" / "cache.npz"
RESULTS_ROOT = LACEWING_PKG_DIR / "quantification" / "methods" / "results" / "fd_uti_plate_labels"


# Class indices in the UTI cache:
#   0 = baseline, 1 = drift, 2 = rising, 3 = plateau.
# TTP = start of "rising" (drift→rising boundary = user's TTP).
UTI_TTP_CLASSES = frozenset({2, 3})  # rising OR plateau — first hit = TTP
# Rationale: if the model predicts drift then rising, first "rising" sample is TTP.
# Including plateau catches the case where the model skips rising and jumps to plateau
# for very high-conc / fast wells.


def load_uti_cache() -> dict:
    if not CACHE_PATH.exists():
        raise SystemExit(
            f"Missing {CACHE_PATH}. "
            "Run build_plate_segmentation_cache.py first."
        )
    npz = np.load(CACHE_PATH, allow_pickle=False)
    return {
        "X": npz["X"],
        "labels": npz["labels"],
        "chip_tag": npz["chip_tag"],
        "well_id": npz["well_id"],
        "log10_conc": npz["log10_conc"],
        "b_drift_end": npz["b_drift_end"],
        "is_reliable": npz["is_reliable"],
        "time_min": npz["time_min"],
        "pixel_shift": npz["pixel_shift"],
    }


def split_by_chip(cache: dict, holdout_chip: str, val_chip: str | None = None,
                  seed: int = 0) -> dict:
    """Split cache by chip.

    Test  = holdout_chip
    Val   = val_chip (if given) OR random 10% of train pixels
    Train = the rest
    """
    chip = cache["chip_tag"]

    te_mask = (chip == holdout_chip)
    if val_chip:
        va_mask = (chip == val_chip)
        tr_mask = ~(te_mask | va_mask)
    else:
        # No dedicated val chip — random split within train.
        rng = np.random.default_rng(seed)
        tr_pool = np.where(~te_mask)[0]
        rng.shuffle(tr_pool)
        n_val = max(1, int(len(tr_pool) * 0.1))
        va_idx = np.sort(tr_pool[:n_val])
        tr_idx = np.sort(tr_pool[n_val:])
        tr_mask = np.zeros_like(te_mask); tr_mask[tr_idx] = True
        va_mask = np.zeros_like(te_mask); va_mask[va_idx] = True

    def _select(mask):
        return {
            "X":            cache["X"][mask],
            "labels":       cache["labels"][mask],
            "chip_tag":     cache["chip_tag"][mask],
            "well_id":      cache["well_id"][mask],
            "log10_conc":   cache["log10_conc"][mask],
            "b_drift_end":  cache["b_drift_end"][mask],
            "is_reliable":  cache["is_reliable"][mask],
            "pixel_shift":  cache["pixel_shift"][mask],
        }

    return {"train": _select(tr_mask), "val": _select(va_mask), "test": _select(te_mask)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-channels", type=int, default=8)
    p.add_argument("--holdout-chip", type=str, default="uti_260728_EC_SD",
                   help="Which UTI chip to hold out as test set.")
    p.add_argument("--val-chip", type=str, default=None,
                   help="Optional: hold out a second chip as val. If None, use random 10%% of train.")
    p.add_argument("--exclude-unreliable", action="store_true",
                   help="Drop supervisor-flagged unreliable pixels from all splits.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading UTI cache from {CACHE_PATH}")
    cache = load_uti_cache()
    print(f"  {cache['X'].shape[0]} pixels, {cache['X'].shape[1]} samples")
    print(f"  chips present: {sorted(set(cache['chip_tag']))}")

    if args.exclude_unreliable:
        mask = cache["is_reliable"]
        n_before = len(cache["X"])
        for k, v in cache.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n_before:
                cache[k] = v[mask]
        print(f"  excluded {int((~mask).sum())} unreliable pixels; {len(cache['X'])} remain")

    if args.holdout_chip not in set(cache["chip_tag"]):
        raise SystemExit(f"holdout chip '{args.holdout_chip}' not in cache. "
                         f"Available: {sorted(set(cache['chip_tag']))}")

    split = split_by_chip(cache, args.holdout_chip, args.val_chip, args.seed)
    print(f"  split: train={len(split['train']['X'])}  "
          f"val={len(split['val']['X'])}  test={len(split['test']['X'])}")

    # Datasets
    train_ds = _TraceDatasetFromLabels(split["train"]["X"], split["train"]["labels"])
    val_ds   = _TraceDatasetFromLabels(split["val"]["X"],   split["val"]["labels"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # Model
    model = build_model(base_channels=args.base_channels).to(args.device)
    n_params = sum(pi.numel() for pi in model.parameters())
    print(f"  model: 1D U-Net (base_channels={args.base_channels}, "
          f"n_params={n_params:,})")

    # Output dir
    out_dir = RESULTS_ROOT / f"seed{args.seed}_holdout_{args.holdout_chip}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Reset the _train persistent best-tracker between runs
    if hasattr(_train, "_best_seen"):
        delattr(_train, "_best_seen")

    print(f"\nTraining for {args.epochs} epochs on {args.device}...")
    history = _train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        ckpt_dir=ckpt_dir,
    )

    # ----- test-time inference on the held-out chip -----
    print(f"\nRunning test inference on {args.holdout_chip}...")
    # Load best checkpoint
    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    print(f"  loaded best.pt (epoch {best_ckpt['epoch']}, val_loss={best_ckpt['val_loss']:.4f})")

    X_te = split["test"]["X"]
    y_true_ttp_min = split["test"]["b_drift_end"].astype(np.float32) / SAMPLES_PER_MIN
    chip_te = split["test"]["chip_tag"]
    well_te = split["test"]["well_id"]
    log10_te = split["test"]["log10_conc"]

    ttp_pred_min = _predict_ttp(model, X_te, device=args.device,
                                ttp_classes=UTI_TTP_CLASSES)
    # Also save the per-timestep argmax for downstream diagnostics
    argmax_pred = np.empty((len(X_te), X_te.shape[1]), dtype=np.int8)
    model.eval()
    with torch.no_grad():
        for i in range(len(X_te)):
            x = torch.from_numpy(X_te[i].astype(np.float32))[None, None, :].to(args.device)
            preds = model(x).argmax(dim=1).squeeze(0).cpu().numpy()
            argmax_pred[i] = preds

    # ----- save results -----
    np.savez_compressed(
        out_dir / "predictions.npz",
        ttp_pred_min=ttp_pred_min.astype(np.float32),
        argmax_per_timestep=argmax_pred,
    )
    np.savez_compressed(
        out_dir / "labels.npz",
        ttp_true_min=y_true_ttp_min.astype(np.float32),
        chip_id=chip_te,
        well_id=well_te.astype(np.int32),
        log10_concentration=log10_te.astype(np.float32),
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "cache": str(CACHE_PATH),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "base_channels": args.base_channels,
        "holdout_chip": args.holdout_chip,
        "val_chip": args.val_chip,
        "exclude_unreliable": args.exclude_unreliable,
        "n_train": len(split["train"]["X"]),
        "n_val": len(split["val"]["X"]),
        "n_test": len(split["test"]["X"]),
        "ttp_classes_for_extraction": sorted(UTI_TTP_CLASSES),
    }, indent=2))

    # Print per-well test summary
    print(f"\n=== Test set results (holdout = {args.holdout_chip}) ===")
    print(f"{'well':>4}  {'log10':>5}  {'n_pix':>5}  {'true TTP':>9}  {'pred TTP':>9}  {'|err|':>6}")
    print("-" * 55)
    for wid in sorted(set(well_te.tolist())):
        m = well_te == wid
        n = int(m.sum())
        true_med = float(np.median(y_true_ttp_min[m]))
        pred_med = float(np.median(ttp_pred_min[m]))
        err = abs(pred_med - true_med)
        log10 = float(log10_te[m][0]) if n > 0 and np.isfinite(log10_te[m][0]) else float("nan")
        print(f"  {wid:>2}  {log10:>5.2f}  {n:>5}  {true_med:>9.2f}  {pred_med:>9.2f}  {err:>6.2f}")

    print(f"\n[ok] wrote {out_dir}")


if __name__ == "__main__":
    main()
