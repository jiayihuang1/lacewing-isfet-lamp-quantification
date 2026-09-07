"""Train one model on the per-pixel TTP regression cache.

Mirrors lacewing.classification.core.train but with a regression head
and MSE/Huber loss.  Same models (gru, transformer, ann, cnn1d, ...)
work out of the box because each model factory returns a head that
outputs one scalar — we just interpret that scalar as a normalised
TTP prediction instead of a logit.

Usage examples
--------------
    python -m lacewing.quantification.regression_shared.core.train \\
        --model cnn_transformer_seq --seed 0 \\
        --cache regress_all_filt_abcd_ntcRaw_madk1p5

    # Huber instead of MSE:
    python -m lacewing.quantification.regression_shared.core.train \\
        --model cnn_transformer_seq --seed 0 --loss huber \\
        --cache regress_all_filt_abcd_ntcRaw_madk1p5

Outputs (under results/<experiment>/runs/<run_id>/):
    config.yaml       - exact run config
    metrics.csv       - per-epoch train/val MAE+MSE
    test_metrics.json - final per-pixel + per-well test metrics (minutes)
    predictions.npz   - test predictions (de-normalised)
    checkpoints/best.pt - best val-MAE checkpoint
    checkpoints/last.pt
    train_curves.png
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from lacewing.classification import models
from lacewing.quantification.regression_shared.core import paths
from lacewing.quantification.regression_shared.data import dataset as ds


DEFAULTS = {
    "epochs":     40,
    "batch_size": 16,
    "lr":         1e-3,
    "lr_step":    10,
    "lr_gamma":   0.75,
    "loss":       "huber",          # 'huber' or 'mse'
    "huber_delta": 1.0,             # delta in *normalised* y units
    "experiment": "regress_default",
    "cache":      "regress_all_filt_abcd_ntcRaw_madk1p5",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   choices=[m for m in models.list_models()
                            if m not in {"autoencoder", "cnn2d_spec"}],
                   help="Architecture from the classification model "
                        "registry; we reuse the same backbones with a "
                        "scalar regression interpretation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--lr-step", type=int, default=DEFAULTS["lr_step"])
    p.add_argument("--lr-gamma", type=float, default=DEFAULTS["lr_gamma"])
    p.add_argument("--loss", choices=["mse", "huber"],
                   default=DEFAULTS["loss"])
    p.add_argument("--huber-delta", type=float,
                   default=DEFAULTS["huber_delta"],
                   help="Delta for HuberLoss, in NORMALISED y units "
                        "(default 1.0 = roughly 1 std of train TTPs).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--experiment", default=DEFAULTS["experiment"])
    p.add_argument("--cache", default=DEFAULTS["cache"])
    p.add_argument("--val-n-chips", type=int, default=1,
                   help="Number of dose-response chips held out as val.")
    p.add_argument("--no-normalise-y", action="store_true",
                   help="Train on raw-minute targets (default normalises).")
    p.add_argument("--tag", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, optim, criterion, device):
    model.train()
    sse_norm, sae_norm, total = 0.0, 0.0, 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        optim.zero_grad()
        pred = model(x).squeeze(-1)
        loss = criterion(pred, y)
        loss.backward()
        optim.step()
        with torch.no_grad():
            sse_norm += float(((pred - y) ** 2).sum())
            sae_norm += float((pred - y).abs().sum())
        total += y.size(0)
    return sse_norm / total, sae_norm / total


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    """Returns (loss_mean, sse_norm/n, sae_norm/n, y_true, y_pred)."""
    model.eval()
    loss_sum, sse_norm, sae_norm, total = 0.0, 0.0, 0.0, 0
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        pred = model(x).squeeze(-1)
        loss = criterion(pred, y)
        loss_sum += float(loss) * y.size(0)
        sse_norm += float(((pred - y) ** 2).sum())
        sae_norm += float((pred - y).abs().sum())
        total += y.size(0)
        ys.append(y.cpu().numpy())
        preds.append(pred.cpu().numpy())
    return (loss_sum / total,
            sse_norm / total, sae_norm / total,
            np.concatenate(ys), np.concatenate(preds))


# ---------------------------------------------------------------------------
# Metrics + reporting (minutes)
# ---------------------------------------------------------------------------

def _denormalise(y_norm: np.ndarray, mean: float, std: float) -> np.ndarray:
    return y_norm * std + mean


def _per_well_predictions(pred_min: np.ndarray, chip: np.ndarray,
                          well: np.ndarray) -> dict:
    """Aggregate per-pixel predictions to per-well means."""
    keys = np.array([f"{c}::{w}" for c, w in zip(chip, well)])
    out: dict[str, list[float]] = {}
    for k, p in zip(keys, pred_min):
        out.setdefault(k, []).append(float(p))
    return {k: float(np.mean(v)) for k, v in out.items()}


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, R^2 (both raw and 'against mean baseline')."""
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # MAE of the constant-mean predictor on the same y_true:
    mae_baseline_constant_mean = float(
        np.mean(np.abs(y_true - y_true.mean()))
    )
    return {
        "mae_min":  mae,
        "rmse_min": rmse,
        "r2":       r2,
        "mae_baseline_constant_mean": mae_baseline_constant_mean,
        "n":        int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_curves(metrics_csv: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    epochs, tr_mae, va_mae = [], [], []
    with metrics_csv.open() as fh:
        for row in csv.DictReader(fh):
            epochs.append(int(row["epoch"]))
            tr_mae.append(float(row["tr_mae_norm"]))
            va_mae.append(float(row["va_mae_norm"]))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs, tr_mae, label="train MAE (norm)", color="#1f78b4")
    ax.plot(epochs, va_mae, label="val MAE (norm)",   color="#e31a1c")
    ax.set_xlabel("epoch"); ax.set_ylabel("MAE in normalised y units")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    ax.set_title("regression train curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ----- seeding -----
    from lacewing.classification.core.seeding import seed_everything
    seed_everything(args.seed)

    # ----- load + split data -----
    print(f"Loading cache: {args.cache}")
    arr = ds.make_split(args.cache, seed=args.seed,
                         val_n_chips=args.val_n_chips,
                         normalise_y=not args.no_normalise_y)
    print(f"  n_tr={len(arr.X_tr)}  n_va={len(arr.X_va)}  n_te={len(arr.X_te)}")
    print(f"  y_mean (train, min) = {arr.y_mean:.3f}  "
          f"y_std (train, min) = {arr.y_std:.3f}")
    print(f"  val chips:  {sorted(set(arr.chip_va.tolist()))}")
    print(f"  test chips: {sorted(set(arr.chip_te.tolist()))}")

    tr_loader, va_loader, te_loader = ds.make_loaders(
        arr, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # ----- model + loss + optim -----
    input_shape = (1, arr.X_tr.shape[1])
    model = models.build(args.model, input_shape).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built {args.model}  ({n_params:,} params)")

    if args.loss == "huber":
        criterion = nn.HuberLoss(delta=args.huber_delta, reduction="mean")
    else:
        criterion = nn.MSELoss(reduction="mean")

    optim = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optim, step_size=args.lr_step, gamma=args.lr_gamma)

    # ----- run dir -----
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{ts}_{args.model}_{args.loss}_seed{args.seed}"
    if args.tag:
        run_id += f"_{args.tag}"
    run_dir = paths.experiment_dir(args.experiment) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    cfg = {
        "model": args.model,
        "loss": args.loss,
        "huber_delta": args.huber_delta,
        "cache": args.cache,
        "experiment": args.experiment,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_step": args.lr_step,
        "lr_gamma": args.lr_gamma,
        "val_n_chips": args.val_n_chips,
        "normalise_y": not args.no_normalise_y,
        "y_mean_train": arr.y_mean,
        "y_std_train":  arr.y_std,
        "n_train": len(arr.X_tr),
        "n_val":   len(arr.X_va),
        "n_test":  len(arr.X_te),
        "val_chips_held_out": sorted(set(arr.chip_va.tolist())),
        "test_chips":         sorted(set(arr.chip_te.tolist())),
        "run_id": run_id,
        "n_parameters": n_params,
        "device": args.device,
    }
    (run_dir / "config.yaml").write_text(
        "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()),
        encoding="utf-8",
    )

    # ----- training loop -----
    metrics_csv = run_dir / "metrics.csv"
    fields = ["epoch", "lr",
              "tr_mse_norm", "tr_mae_norm",
              "va_loss",     "va_mse_norm", "va_mae_norm", "va_mae_min"]
    with metrics_csv.open("w", newline="") as fh:
        csv.DictWriter(fh, fields).writeheader()

    best_val_mae = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_mse_n, tr_mae_n = _train_epoch(
            model, tr_loader, optim, criterion, args.device)
        va_loss, va_mse_n, va_mae_n, y_va_true, y_va_pred = _eval_epoch(
            model, va_loader, criterion, args.device)
        scheduler.step()

        # De-normalise val MAE to minutes for an interpretable headline.
        if cfg["normalise_y"]:
            va_mae_min = va_mae_n * arr.y_std
        else:
            va_mae_min = va_mae_n

        cur_lr = optim.param_groups[0]["lr"]
        with metrics_csv.open("a", newline="") as fh:
            csv.DictWriter(fh, fields).writerow({
                "epoch": epoch, "lr": cur_lr,
                "tr_mse_norm": tr_mse_n, "tr_mae_norm": tr_mae_n,
                "va_loss": va_loss,
                "va_mse_norm": va_mse_n, "va_mae_norm": va_mae_n,
                "va_mae_min": va_mae_min,
            })
        print(f"epoch {epoch:02d}/{args.epochs}  "
              f"tr_mae_n={tr_mae_n:.4f}  va_mae_n={va_mae_n:.4f}  "
              f"va_mae_min={va_mae_min:.3f}  lr={cur_lr:.2e}")

        if va_mae_n < best_val_mae:
            best_val_mae = va_mae_n
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optim_state": optim.state_dict(),
                        "val_mae_norm": va_mae_n},
                       run_dir / "checkpoints" / "best.pt")
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optim_state": optim.state_dict()},
                   run_dir / "checkpoints" / "last.pt")

    train_time_s = time.time() - t0

    # ----- reload best, run test -----
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt",
                      map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nReloaded best checkpoint from epoch {ckpt['epoch']}.")

    # Test: predictions in NORMALISED units, then de-normalise to minutes.
    model.eval()
    preds_norm = []
    with torch.no_grad():
        for x, _ in te_loader:
            x = x.to(args.device)
            preds_norm.append(model(x).squeeze(-1).cpu().numpy())
    preds_norm = np.concatenate(preds_norm)
    if cfg["normalise_y"]:
        preds_min = _denormalise(preds_norm, arr.y_mean, arr.y_std)
    else:
        preds_min = preds_norm
    y_te_min = arr.y_te                         # already in minutes

    # Per-pixel
    pixel_metrics = _regression_metrics(y_te_min, preds_min)
    # Per-well (mean of per-pixel preds)
    well_pred_min = _per_well_predictions(preds_min, arr.chip_te, arr.well_te)
    # Per-well ground truth
    well_true = {}
    keys = [f"{c}::{w}" for c, w in zip(arr.chip_te, arr.well_te)]
    for k, y in zip(keys, y_te_min):
        well_true[k] = float(y)
    well_keys = sorted(set(keys))
    yw_true = np.array([well_true[k]      for k in well_keys])
    yw_pred = np.array([well_pred_min[k]  for k in well_keys])
    well_metrics = _regression_metrics(yw_true, yw_pred)

    test_payload = {
        "pixel": pixel_metrics,
        "well":  well_metrics,
        "per_well": [
            {"chip_id": k.split("::")[0],
             "well_id": int(k.split("::")[1]),
             "y_true_min": float(well_true[k]),
             "y_pred_min": float(well_pred_min[k])}
            for k in well_keys
        ],
        "best_epoch":    int(ckpt["epoch"]),
        "val_mae_norm_best": float(ckpt.get("val_mae_norm", best_val_mae)),
        "train_time_s":  float(train_time_s),
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(
        test_payload, indent=2), encoding="utf-8")

    # Predictions cache (de-normalised, in minutes)
    np.savez_compressed(
        run_dir / "predictions.npz",
        y_true_min=y_te_min, y_pred_min=preds_min,
        chip_id=arr.chip_te, well_id=arr.well_te, pixel_id=arr.pixel_te,
    )

    _plot_curves(metrics_csv, run_dir / "train_curves.png")

    print(f"\nDone.  Test pixel MAE = {pixel_metrics['mae_min']:.3f} min  "
          f"(baseline constant-mean = "
          f"{pixel_metrics['mae_baseline_constant_mean']:.3f} min)")
    print(f"      Test well  MAE = {well_metrics['mae_min']:.3f} min  "
          f"(n_wells = {well_metrics['n']})")
    print(f"      R^2 (pixel) = {pixel_metrics['r2']:.3f}    "
          f"R^2 (well) = {well_metrics['r2']:.3f}")
    print(f"      training time {train_time_s/60:.1f} min")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
