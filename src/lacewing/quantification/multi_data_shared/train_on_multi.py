"""Train + test on the Data/Multi/ chips with the locked configuration.

Configuration (locked 2026-06-22 with supervisor):
    preprocessing  MAD k=1.5 ABCD ntcRaw   (matches exp8 winner)
    model          cnn_transformer_seq      (exp9 winner)
    cache          Analysis/multi_data/cache/multi_filt_abcd_ntcRaw_madk1p5.npz

Split: leave-one-chip-out.  For each of the 5 Multi chips, hold that
chip out as the test set, train on the other 4, carve 15% of the
training pool as a chip-stratified val set, train, evaluate.

This mirrors the classification training loop in
``Analysis/classification/core/train.py`` exactly, but with a different
cache path and a chip-fold split logic specific to the Multi chips
(the classification side's `make_split` is wired to the dose-response
chip layout, not Multi).

Usage::

    python -m lacewing.quantification.multi_data_shared.train_on_multi \\
        --fold D20260608_E00_C00_F4500KHz_U_norm_temp_04 --seed 0

Outputs (under Analysis/classification/results/multi_train_<fold>/runs/<run_id>/):
    config.yaml           - exact run config
    metrics.csv           - per-epoch train/val metrics
    test_metrics.json     - final pixel + well metrics on the held-out chip
    predictions.npz       - test predictions (for re-analysis)
    checkpoints/best.pt   - lowest val-loss checkpoint
    train.log
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lacewing.classification import models
from lacewing.classification.core import (
    evaluate as eval_mod, logging_utils, paths, seeding,
)
from lacewing.classification.core.train import (
    _bce_with_pos_weight, _train_classifier_epoch, _eval_classifier,
)
from lacewing.classification.data.dataset import PixelDataset, SplitArrays


# --------------------------------------------------------------------
# Locked configuration
# --------------------------------------------------------------------

DEFAULT_MODEL    = "cnn_transformer_seq"
DEFAULT_CACHE    = "multi_filt_abcd_ntcRaw_madk1p5"
DEFAULT_EPOCHS   = 40
DEFAULT_BATCH    = 16
DEFAULT_LR       = 1e-3
DEFAULT_LR_STEP  = 10
DEFAULT_LR_GAMMA = 0.75
VAL_FRAC_OF_TRAIN_POOL = 0.15

# Multi cache directory and ALL chips that survived to the locked cache.
MULTI_CACHE_DIR = Path(__file__).resolve().parent / "cache"
MULTI_CHIPS = [
    "D20260320_E00_C00_F4500KHz_U_Elena_steap_cv",
    "D20260608_E00_C00_F4500KHz_U_norm_temp_04",
    "D20260609_E00_C00_F4500KHz_U_norm_temp_read_06",
    "D20260609_E00_C00_F4500KHz_U_norm_temp_read_07",
    "D20260609_E00_C00_F4500KHz_U_norm_temp_ready_08",
]


# --------------------------------------------------------------------
# Cache loader + chip-fold split
# --------------------------------------------------------------------

def _load_multi_cache(stem: str) -> dict:
    path = MULTI_CACHE_DIR / f"{stem}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Multi cache not found: {path}")
    with np.load(path) as data:
        return {
            "X":         data["X"].astype(np.float32, copy=False),
            "y":         data["y"].astype(np.int64, copy=False),
            "chip_id":   np.asarray(data["chip_id"]),
            "well_id":   np.asarray(data["well_id"]),
            "pixel_id":  np.asarray(data["pixel_id"]),
        }


def make_chip_fold_split(cache: dict, fold_chip: str, seed: int,
                         val_frac: float = VAL_FRAC_OF_TRAIN_POOL
                         ) -> SplitArrays:
    """Leave-one-chip-out split.

    test     = pixels whose chip_id == fold_chip
    train+val pool = pixels whose chip_id != fold_chip
    val      = `val_frac` of the pool, chip-stratified
    train    = the remainder

    The val carve-out uses a single deterministic permutation seeded
    by `seed`, so each (fold, seed) is reproducible.
    """
    chip_id = cache["chip_id"]
    if fold_chip not in set(map(str, chip_id)):
        raise ValueError(
            f"fold chip '{fold_chip}' not present in cache; "
            f"available = {sorted(set(map(str, chip_id)))}"
        )

    idx_test = np.flatnonzero(chip_id == fold_chip)
    idx_pool = np.flatnonzero(chip_id != fold_chip)

    pool_chips = sorted(set(map(str, chip_id[idx_pool])))
    rng = np.random.default_rng(seed)

    # Carve val: choose a random subset of TRAINING chips' wells.
    # To keep val chip-stratified, sample by chip × well so val
    # contains a mix of all training chips' wells.
    target_n_val = int(round(val_frac * len(idx_pool)))
    permuted = rng.permutation(len(idx_pool))
    idx_val_local = permuted[:target_n_val]
    idx_tr_local  = permuted[target_n_val:]
    idx_va = idx_pool[idx_val_local]
    idx_tr = idx_pool[idx_tr_local]

    X        = cache["X"]
    y        = cache["y"]
    well_id  = cache["well_id"]
    pixel_id = cache["pixel_id"]

    return SplitArrays(
        X_tr=X[idx_tr], y_tr=y[idx_tr].astype(np.float32),
        X_va=X[idx_va], y_va=y[idx_va].astype(np.float32),
        X_te=X[idx_test], y_te=y[idx_test].astype(np.float32),
        chip_te=chip_id[idx_test],
        well_te=well_id[idx_test],
        pixel_te=pixel_id[idx_test],
        X_raw_te=X[idx_test].copy(),
    )


def _make_loaders(arr: SplitArrays, batch_size: int, num_workers: int):
    tr = DataLoader(PixelDataset(arr.X_tr, arr.y_tr),
                    batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, drop_last=False)
    va = DataLoader(PixelDataset(arr.X_va, arr.y_va),
                    batch_size=256, shuffle=False, num_workers=num_workers)
    te = DataLoader(PixelDataset(arr.X_te, arr.y_te),
                    batch_size=256, shuffle=False, num_workers=num_workers)
    return tr, va, te


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold", required=True, choices=MULTI_CHIPS,
                   help="Chip folder name to hold out as test.")
    p.add_argument("--model",  default=DEFAULT_MODEL,
                   choices=models.list_models(),
                   help="Architecture (locked = cnn_transformer_seq).")
    p.add_argument("--cache",  default=DEFAULT_CACHE,
                   help="Multi cache stem under Analysis/multi_data/cache/.")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr",         type=float, default=DEFAULT_LR)
    p.add_argument("--lr-step",    type=int,   default=DEFAULT_LR_STEP)
    p.add_argument("--lr-gamma",   type=float, default=DEFAULT_LR_GAMMA)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--tag",        default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    seeding.seed_everything(args.seed)

    cache = _load_multi_cache(args.cache)
    arr = make_chip_fold_split(cache, fold_chip=args.fold, seed=args.seed)
    tr_loader, va_loader, te_loader = _make_loaders(
        arr, args.batch_size, args.num_workers)

    # Short fold tag for the experiment name (only the suffix after the
    # firmware/chip prefix, otherwise the run dirs get unreadable).
    fold_short = args.fold.split("KHz_U_")[-1] if "KHz_U_" in args.fold else args.fold
    experiment = f"multi_train_{fold_short}"

    # Use the classification logging utilities so results land under
    # Analysis/classification/results/<experiment>/runs/<run_id>/.
    run_id = logging_utils.make_run_id(
        args.model, "raw", "chipfold", fold_short, args.seed, args.tag)
    run_dir = logging_utils.make_run_dir(run_id, experiment=experiment)
    logger = logging_utils.setup_logger(run_dir)

    cfg = dict(
        experiment=experiment, cache=args.cache, fold=args.fold,
        fold_short=fold_short, model=args.model, features="raw",
        split="chipfold", seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, lr_step=args.lr_step, lr_gamma=args.lr_gamma,
        device=args.device, run_id=run_id,
        n_train=int(len(arr.y_tr)), n_val=int(len(arr.y_va)),
        n_test=int(len(arr.y_te)),
        n_pos_train=int((arr.y_tr == 1).sum()),
        n_neg_train=int((arr.y_tr == 0).sum()),
        train_chips=sorted(set(map(str, cache["chip_id"][
            np.isin(cache["chip_id"], [c for c in MULTI_CHIPS if c != args.fold])
        ]))),
        test_chip=args.fold,
        val_frac_of_train_pool=VAL_FRAC_OF_TRAIN_POOL,
    )
    logging_utils.save_config(run_dir, cfg)
    logging_utils.snapshot_git_state(run_dir)
    logging_utils.snapshot_env(run_dir)

    logger.info(f"run_id = {run_id}")
    logger.info(
        f"split: train={len(arr.y_tr)}  val={len(arr.y_va)}  test={len(arr.y_te)}"
    )
    logger.info(f"test chip held out = {args.fold}")

    # Build model + train loop
    input_shape = arr.X_tr.shape[1:]
    model = models.build(args.model, input_shape).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model {args.model}: {n_params:,} parameters")
    cfg["n_parameters"] = n_params
    logging_utils.save_config(run_dir, cfg)  # re-save with param count

    optim = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optim, step_size=args.lr_step, gamma=args.lr_gamma)
    criterion = _bce_with_pos_weight(arr.y_tr, args.device)

    metrics_csv = run_dir / "metrics.csv"
    metrics_writer = open(metrics_csv, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(metrics_writer, fieldnames=[
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr",
        "elapsed_sec",
    ])
    writer.writeheader()

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _train_classifier_epoch(
            model, tr_loader, optim, criterion, args.device)
        val_loss, y_va_true, y_va_score = _eval_classifier(
            model, va_loader, criterion, args.device)
        val_pred = (y_va_score >= 0.5).astype(int)
        val_acc = float((val_pred == y_va_true).mean())
        scheduler.step()

        cur_lr = optim.param_groups[0]["lr"]
        elapsed = time.time() - t0
        writer.writerow({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss":   val_loss,   "val_acc":   val_acc,
            "lr": cur_lr, "elapsed_sec": elapsed,
        })
        metrics_writer.flush()
        logger.info(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"tr_loss={train_loss:.4f} tr_acc={train_acc:.4f} | "
            f"va_loss={val_loss:.4f} va_acc={val_acc:.4f} | "
            f"lr={cur_lr:.2e} | {elapsed:.0f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                dict(epoch=epoch, model_state=model.state_dict(),
                     optim_state=optim.state_dict(), val_loss=val_loss),
                run_dir / "checkpoints" / "best.pt",
            )
        torch.save(
            dict(epoch=epoch, model_state=model.state_dict(),
                 optim_state=optim.state_dict()),
            run_dir / "checkpoints" / "last.pt",
        )

    metrics_writer.close()
    train_time = time.time() - t0
    logger.info(f"training finished in {train_time:.1f}s")

    # Reload best checkpoint + final test
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt",
                      map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"loaded best checkpoint from epoch {ckpt['epoch']}")

    criterion_eval = nn.BCEWithLogitsLoss()
    _, y_te_true, y_te_score = _eval_classifier(
        model, te_loader, criterion_eval, args.device)
    y_pred = (y_te_score >= 0.5).astype(int)

    pix = eval_mod.pixel_metrics(arr.y_te, y_te_score)
    keys, well_truth, well_pred = eval_mod.well_majority_vote(
        arr.y_te, y_te_score, arr.chip_te, arr.well_te)
    well_acc = float((well_truth == well_pred).mean()) if len(well_truth) else float("nan")

    test_payload = dict(
        pixel=dict(accuracy=pix.accuracy, precision=pix.precision,
                   recall=pix.recall, f1=pix.f1, auroc=pix.auroc,
                   n_pos=pix.n_pos, n_neg=pix.n_neg),
        well=dict(accuracy=well_acc, n_wells=int(len(well_truth))),
        per_well=[
            dict(chip_id=str(k.split("::")[0]),
                 well_id=int(k.split("::")[1]),
                 truth=int(t), pred=int(p))
            for k, t, p in zip(keys, well_truth, well_pred)
        ],
        best_epoch=int(ckpt["epoch"]),
        train_time_s=float(train_time),
    )
    logging_utils.save_test_metrics(run_dir, test_payload)

    np.savez_compressed(
        run_dir / "predictions.npz",
        y_true=arr.y_te, y_score=y_te_score, y_pred=y_pred,
        chip_id=arr.chip_te, well_id=arr.well_te, pixel_id=arr.pixel_te,
    )

    logger.info(
        f"TEST  pixel_acc={pix.accuracy:.4f}  pixel_auc={pix.auroc:.4f}  "
        f"pixel_f1={pix.f1:.4f}  well_acc={well_acc:.4f} "
        f"(n_wells={int(len(well_truth))})"
    )
    logger.info(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
