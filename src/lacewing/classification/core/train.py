"""RQ1 · classifier training loop. [Cat A] Report §RQ1.

Train one Paper 3 classifier on the per-pixel dataset.

Usage examples
--------------
    # Single run, default seed/split
    python -m lacewing.classification.train --model cnn2d_spec

    # Specific config
    python -m lacewing.classification.train \
        --model cnn1d --features raw --split random --seed 0

    # Chip-fold cross-validation: hold out the 1e7 chip
    python -m lacewing.classification.train \
        --model cnn2d_spec --split chip --fold 1e7

    # Loop the runner over all 7 models for a sweep:
    for m in ann cnn1d fcn resnet inception autoencoder cnn2d_spec; do
        python -m lacewing.classification.train --model $m
    done

Outputs (under results/runs/<run_id>/):
    config.yaml           - exact run config
    metrics.csv           - per-epoch train/val metrics
    test_metrics.json     - final pixel + well metrics on the test split
    predictions.npz       - test predictions (for later re-analysis)
    checkpoints/best.pt   - lowest val-loss checkpoint
    checkpoints/last.pt   - final-epoch checkpoint
    cm_pixel.png          - pixel confusion matrix
    cm_well.png           - well confusion matrix
    train_curves.png      - loss + acc curves
    train.log             - full text log
    git_state.txt, env.txt
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from . import logging_utils, paths, seeding
from . import evaluate as eval_mod
from .. import models
from ..data import dataset


# Paper 3 §IV.D
DEFAULTS = dict(
    epochs=40,
    batch_size=16,
    lr=1e-3,
    lr_step=10,
    lr_gamma=0.75,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=models.list_models())
    p.add_argument("--features", default=None,
                   help="Override the model's default feature kind. "
                        "Most models pick raw or spectrogram automatically.")
    p.add_argument("--split", default="random",
                   choices=["random", "chip", "chipkfold", "sd_test"])
    p.add_argument("--fold", default="0",
                   help="For --split chip, the held-out chip key "
                        "(1e5/1e6/1e7/1e8/1e9). For --split chipkfold, "
                        "the fold index 0..K-1 (K=5 by default). "
                        "For --split sd_test, a comma-separated list of "
                        "test-chip folder names (default: the two "
                        "supervisor-selected SD chips). "
                        "Ignored for random.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--lr-step", type=int, default=DEFAULTS["lr_step"])
    p.add_argument("--lr-gamma", type=float, default=DEFAULTS["lr_gamma"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--tag", default=None,
                   help="Optional human-readable suffix for the run dir.")
    p.add_argument("--experiment", default=paths.DEFAULT_EXPERIMENT,
                   help="Experiment subfolder under results/ for this run.")
    p.add_argument("--cache", default="dataset_per_pixel",
                   help="Dataset cache name (under data/cache/<name>.npz).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-epoch training/eval helpers
# ---------------------------------------------------------------------------

def _bce_with_pos_weight(y_tr: np.ndarray, device: str) -> nn.Module:
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return nn.BCEWithLogitsLoss()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32,
                              device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def _train_classifier_epoch(model, loader, optim, criterion, device):
    model.train()
    losses, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        losses += loss.item() * y.size(0)
        pred = (torch.sigmoid(logits) >= 0.5).float()
        correct += int((pred == y).sum())
        total += y.size(0)
    return losses / total, correct / total


@torch.no_grad()
def _eval_classifier(model, loader, criterion, device):
    model.eval()
    losses, ys, scores = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses += loss.item() * y.size(0)
        ys.append(y.cpu().numpy())
        scores.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.concatenate(ys)
    y_score = np.concatenate(scores)
    return losses / len(y_true), y_true, y_score


# Autoencoder gets its own loop because it trains on positives only and
# classifies via reconstruction MSE.

def _train_ae_epoch(model, loader, optim, device):
    model.train()
    total = 0
    sse = 0.0
    for x, _ in loader:
        x = x.to(device)
        optim.zero_grad()
        recon = model(x)
        loss = ((recon - x) ** 2).mean()
        loss.backward()
        optim.step()
        sse += loss.item() * x.size(0)
        total += x.size(0)
    return sse / total


@torch.no_grad()
def _ae_recon_errors(model, loader, device):
    model.eval()
    ys, errs = [], []
    for x, y in loader:
        x = x.to(device)
        recon = model(x)
        e = ((recon - x) ** 2).mean(dim=(1, 2)).cpu().numpy()
        errs.append(e); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(errs)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_curves(metrics_csv: Path, out_path: Path) -> None:
    rows = list(csv.DictReader(open(metrics_csv, encoding="utf-8")))
    if not rows:
        return
    epochs = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [float(r["train_loss"]) for r in rows], label="train")
    axes[0].plot(epochs, [float(r["val_loss"]) for r in rows], label="val")
    axes[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [float(r["train_acc"]) for r in rows], label="train")
    axes[1].plot(epochs, [float(r["val_acc"]) for r in rows], label="val")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    seeding.seed_everything(args.seed)

    features = args.features or models.features_for(args.model)

    # Load data
    cache_path = paths.cache_path(args.cache)
    arr = dataset.make_split(features=features, split=args.split,
                             fold=args.fold, seed=args.seed,
                             cache_path=cache_path)
    tr_loader, va_loader, te_loader = dataset.make_loaders(
        arr, batch_size=args.batch_size, num_workers=args.num_workers)

    # Run dir + logger
    run_id = logging_utils.make_run_id(
        args.model, features, args.split, args.fold, args.seed, args.tag)
    run_dir = logging_utils.make_run_dir(run_id, experiment=args.experiment)
    logger = logging_utils.setup_logger(run_dir)

    # Persist config + environment snapshots
    cfg = dict(
        experiment=args.experiment, cache=args.cache,
        model=args.model, features=features, split=args.split,
        fold=args.fold, seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, lr_step=args.lr_step, lr_gamma=args.lr_gamma,
        device=args.device, run_id=run_id,
        n_train=int(len(arr.y_tr)), n_val=int(len(arr.y_va)),
        n_test=int(len(arr.y_te)),
        n_pos_train=int((arr.y_tr == 1).sum()),
        n_neg_train=int((arr.y_tr == 0).sum()),
    )
    logging_utils.save_config(run_dir, cfg)
    logging_utils.snapshot_git_state(run_dir)
    logging_utils.snapshot_env(run_dir)

    logger.info(f"run_id = {run_id}")
    logger.info(f"config = {cfg}")

    # Build model
    input_shape = arr.X_tr.shape[1:]
    model = models.build(args.model, input_shape).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model {args.model}: {n_params:,} parameters")
    cfg["n_parameters"] = n_params

    optim = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optim, step_size=args.lr_step, gamma=args.lr_gamma)

    is_autoencoder = (args.model == "autoencoder")
    if not is_autoencoder:
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
        if is_autoencoder:
            # AE trains on positive samples only
            pos_idx = np.where(arr.y_tr == 1)[0]
            if len(pos_idx) == 0:
                logger.error("Autoencoder needs positives in train set.")
                break
            X_pos = arr.X_tr[pos_idx]; y_pos = arr.y_tr[pos_idx]
            ae_arr = dataset.SplitArrays(
                X_tr=X_pos, y_tr=y_pos,
                X_va=arr.X_va, y_va=arr.y_va,
                X_te=arr.X_te, y_te=arr.y_te,
                chip_te=arr.chip_te, well_te=arr.well_te, pixel_te=arr.pixel_te)
            tr_loader_pos, _, _ = dataset.make_loaders(
                ae_arr, batch_size=args.batch_size,
                num_workers=args.num_workers)
            train_loss = _train_ae_epoch(model, tr_loader_pos, optim,
                                         args.device)
            train_acc = float("nan")

            # Val: reconstruction error -> classify with current threshold
            # = mean of train MSE (refined post-training)
            _, val_errs = _ae_recon_errors(model, va_loader, args.device)
            val_loss = float(val_errs.mean())
            # Pseudo-accuracy from current threshold
            thr = float(np.percentile(val_errs[arr.y_va == 1], 95)) \
                if (arr.y_va == 1).any() else float(val_errs.mean())
            pred = (val_errs < thr).astype(int)
            val_acc = float((pred == arr.y_va).mean())
        else:
            train_loss, train_acc = _train_classifier_epoch(
                model, tr_loader, optim, criterion, args.device)
            val_loss, vy, vs = _eval_classifier(
                model, va_loader, criterion, args.device)
            val_acc = float(((vs >= 0.5).astype(int) == vy).mean())

        scheduler.step()
        elapsed = time.time() - t0
        cur_lr = optim.param_groups[0]["lr"]

        writer.writerow(dict(
            epoch=epoch, train_loss=f"{train_loss:.6f}",
            train_acc=f"{train_acc:.4f}",
            val_loss=f"{val_loss:.6f}", val_acc=f"{val_acc:.4f}",
            lr=f"{cur_lr:.6f}", elapsed_sec=f"{elapsed:.1f}",
        ))
        metrics_writer.flush()
        logger.info(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={cur_lr:.5f} | {elapsed:.1f}s")

        # Checkpointing
        if val_loss < best_val:
            best_val = val_loss
            torch.save(dict(epoch=epoch, model_state=model.state_dict(),
                            optim_state=optim.state_dict(),
                            val_loss=val_loss),
                       run_dir / "checkpoints" / "best.pt")
        torch.save(dict(epoch=epoch, model_state=model.state_dict(),
                        optim_state=optim.state_dict(),
                        val_loss=val_loss),
                   run_dir / "checkpoints" / "last.pt")

    metrics_writer.close()
    train_time = time.time() - t0
    logger.info(f"training finished in {train_time:.1f}s")

    # Reload best checkpoint
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt",
                      map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"loaded best checkpoint from epoch {ckpt['epoch']}")

    # Final test evaluation. The autoencoder path differs because it
    # classifies via reconstruction MSE rather than a logit; we convert
    # both paths to a (y_score in [0,1], threshold) pair so downstream
    # metric code is identical.
    if is_autoencoder:
        # Threshold for "is positive" = the 95th-percentile recon error
        # of POSITIVE validation samples (Paper 3 §IV.C: tail of in-class
        # error distribution).
        _, val_errs = _ae_recon_errors(model, va_loader, args.device)
        pos_errs = val_errs[arr.y_va == 1]
        err_threshold = (float(np.percentile(pos_errs, 95)) if len(pos_errs)
                         else float(val_errs.mean()))
        _, test_errs = _ae_recon_errors(model, te_loader, args.device)
        # Map "low error -> positive" into a 0..1 score where higher = more
        # positive: y_score = sigmoid((err_threshold - err) / scale).
        scale = max(test_errs.std(), 1e-6)
        y_score = 1.0 / (1.0 + np.exp(-(err_threshold - test_errs) / scale))
        decision_threshold = 0.5
    else:
        criterion_eval = nn.BCEWithLogitsLoss()
        _, _, y_score = _eval_classifier(model, te_loader,
                                         criterion_eval, args.device)
        decision_threshold = 0.5

    y_pred = (y_score >= decision_threshold).astype(int)
    pix = eval_mod.pixel_metrics(arr.y_te, y_score,
                                 threshold=decision_threshold)
    well = eval_mod.well_metrics(arr.y_te, y_score, arr.chip_te,
                                 arr.well_te,
                                 threshold=decision_threshold)

    test_metrics = dict(
        pixel=asdict(pix),
        well=well,
        train_time_sec=train_time,
        n_parameters=n_params,
        decision_threshold=decision_threshold,
    )
    logging_utils.save_test_metrics(run_dir, test_metrics)
    logger.info(f"TEST pixel: {asdict(pix)}")
    logger.info(f"TEST well : {well}")

    # Save predictions. For sd_test we also persist the raw 450-sample
    # input traces so the failure-analysis viewer can render them without
    # re-reading the cache; for other splits we keep the file lean by
    # writing raw traces only if available.
    pred_payload = dict(
        y_true=arr.y_te.astype(np.uint8),
        y_score=y_score.astype(np.float32),
        y_pred=y_pred.astype(np.uint8),
        chip_id=arr.chip_te, well_id=arr.well_te, pixel_id=arr.pixel_te,
    )
    if arr.X_raw_te is not None:
        pred_payload["x_raw"] = arr.X_raw_te.astype(np.float32)
    np.savez_compressed(run_dir / "predictions.npz", **pred_payload)

    # Failure dump: misclassified pixels only (FP + FN), with raw trace,
    # score, and pixel metadata. Used by the failure_viewer.html.
    if arr.X_raw_te is not None:
        wrong_mask = (y_pred != arr.y_te.astype(np.uint8))
        if wrong_mask.any():
            np.savez_compressed(
                run_dir / "failure_dump.npz",
                y_true=arr.y_te[wrong_mask].astype(np.uint8),
                y_score=y_score[wrong_mask].astype(np.float32),
                y_pred=y_pred[wrong_mask].astype(np.uint8),
                chip_id=arr.chip_te[wrong_mask],
                well_id=arr.well_te[wrong_mask],
                pixel_id=arr.pixel_te[wrong_mask],
                x_raw=arr.X_raw_te[wrong_mask].astype(np.float32),
            )
            logger.info(f"failure_dump: {int(wrong_mask.sum())} "
                        f"misclassified pixels saved.")
        else:
            logger.info("failure_dump: no misclassified pixels — skipped.")

    # Confusion matrices + curves
    _, well_truth, well_pred = eval_mod.well_majority_vote(
        arr.y_te, y_score, arr.chip_te, arr.well_te,
        threshold=decision_threshold)
    eval_mod.save_confusion_matrices(arr.y_te, y_pred, well_truth, well_pred,
                                     run_dir)
    _plot_curves(metrics_csv, run_dir / "train_curves.png")

    # Append to aggregate index
    logging_utils.append_aggregate_row(dict(
        run_id=run_id, experiment=args.experiment, cache=args.cache,
        model=args.model, features=features,
        split=args.split, fold=args.fold, seed=args.seed,
        epochs=args.epochs, n_parameters=n_params,
        train_time_sec=round(train_time, 1),
        pixel_acc=round(pix.accuracy, 4),
        pixel_precision=round(pix.precision, 4),
        pixel_recall=round(pix.recall, 4),
        pixel_f1=round(pix.f1, 4),
        pixel_auroc=round(pix.auroc, 4),
        well_acc=round(well["accuracy"], 4),
        well_f1=round(well["f1"], 4),
        n_test_pixels=int(len(arr.y_te)),
        n_wells_tested=well["n_wells"],
    ), experiment=args.experiment)
    logger.info(f"DONE. results in {run_dir}")


if __name__ == "__main__":
    main()
