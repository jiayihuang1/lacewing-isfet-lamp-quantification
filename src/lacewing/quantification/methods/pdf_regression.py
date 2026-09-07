"""RQ2 · F-A probability-density regression framing. [Cat A] Report §RQ2 Tab. 6.2.

F6 — PDF regression for TTP extraction (Dempster 2024 framing).

Reframes per-timestep TTP detection from hard binary classification (F1.x)
to regression of a SOFT Gaussian density centred on TTP_qLAMP.  Same
backbone, only the head and the loss change.

Motivation
----------
F1.1 (per-timestep XGBoost) failed with negative R^2.  Boundary-diagnostic
analysis showed Cohen's d ~ 0.2 at +/- 1 min from TTP (label-precision
floor) but d > 6 at +/- 10 min (signal IS distinguishable at coarse
resolution).  The hard step label is mathematically sharp but the ISFET
signal is a gradual sigmoid; BCE on a hard step throws away the
proximity-to-boundary information that should be the training signal.

PDF regression replaces the hard step with a smooth Gaussian target.  MSE
loss preserves proximity information ('predict 0.55, target 0.59' is a
small loss; 'predict 0.55, target 1.0' is a large loss).

Pipeline
--------
1. Load spatA3 regression cache (X, y_ttp_min, chip_id, well_id, split).
2. Split via lacewing.quantification.regression_shared.data.dataset.make_split (1 chip held as val).
3. Build per-(pixel, t) Gaussian target:
       target[t] = exp(- (t - ttp_idx)^2 / (2 * sigma^2))
   where ttp_idx = TTP_min * SAMPLES_PER_MIN, sigma = SIGMA_SAMPLES
   (default 5 samples ~ 20 s; matches the methodology slide).
   Peak height = 1.0 by construction; NOT L1-normalised.
4. Train cnn1d-per-step backbone (option A in the plan): the existing
   cnn1d's encoder, with MaxPool removed and a per-timestep sigmoid head.
   Output shape: (B, T).
5. Loss: MSE between predicted density and Gaussian target, per-timestep.
6. TTP_pred per test pixel = argmax(predicted_density) / SAMPLES_PER_MIN.
7. Per-pixel / per-well MAE + R^2 + per-well dose-response payload, in the
   same schema as the regression baselines.

Outputs
-------
Analysis/quantification/methods/results/<experiment>/seed<N>/:
    config.json
    test_metrics.json
    predictions.npz   (per-pixel TTP_pred and the raw 450-dim density)
    train_curves.png
    metrics.csv
    checkpoints/best.pt, checkpoints/last.pt

Usage
-----
    python -m lacewing.quantification.methods.pdf_regression \\
        --seed 0

    # Tune sigma:
    python -m lacewing.quantification.methods.pdf_regression \\
        --seed 0 --sigma 10
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

from lacewing.quantification.regression_shared.data import dataset as reg_ds
from lacewing.quantification.eval.schema import write_predictions, write_labels


# Constants (match F1.1 conventions).
SAMPLES_PER_MIN = 15
N_SAMPLES = 450

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Checkpoint selection (Week 11 debrief priority-1 fix)
# ---------------------------------------------------------------------------

def select_best_epoch(val_loss: np.ndarray, val_ttp_mae: np.ndarray) -> int:
    """Return the epoch index (0-indexed) that minimises val TTP MAE.

    Week 11 debrief priority-1 fix. val_loss and val TTP MAE decouple on
    sparse Gaussian targets — the epoch that minimises val_loss is often
    a mode-collapsed flat prediction, while a later epoch with higher
    val_loss actually produces per-pixel peaks.
    """
    if len(val_ttp_mae) == 0:
        raise ValueError("val_ttp_mae is empty")
    return int(np.argmin(val_ttp_mae))


# ---------------------------------------------------------------------------
# Concentration label helpers
# ---------------------------------------------------------------------------

# Mapping from chip_id suffix to log10(copies/uL).
_CHIP_SUFFIX_TO_LOG10: dict[str, float] = {
    "1e5": 5.0,
    "1e6": 6.0,
    "1e7": 7.0,
    "1e8": 8.0,
    "1e9": 9.0,
}

# SD multiplex chip: well index -> log10(copies/uL).
# Wells: 0=E(1e9), 1=D(1e8), 2=C(1e7), 3=B(1e6), 4=A(1e5), 5=NTC
_SD_WELL_TO_LOG10: dict[int, float] = {0: 9.0, 1: 8.0, 2: 7.0, 3: 6.0, 4: 5.0}
_SD_CHIP_FOLDER = "D20240719_E03_C44_F4500KHz_U_COV_SD"


def _log10_concentration_array(chip_ids: np.ndarray,
                                well_ids: np.ndarray) -> np.ndarray:
    """Derive log10(copies/uL) per pixel from chip_id and well_id arrays.

    Dose-response chips: concentration encoded in chip_id suffix (e.g. _1e7).
    SD chip: concentration encoded in well index (0=E=1e9 ... 4=A=1e5).
    Unknown pixels: NaN.
    """
    chip_ids = np.asarray(chip_ids)
    well_ids = np.asarray(well_ids).astype(np.int32)
    out = np.full(len(chip_ids), np.nan, dtype=np.float32)
    for i, (cid, wid) in enumerate(zip(chip_ids, well_ids)):
        cid_str = str(cid)
        if cid_str == _SD_CHIP_FOLDER:
            out[i] = _SD_WELL_TO_LOG10.get(int(wid), np.nan)
        else:
            # Dose-response chip: last token is the concentration key.
            suffix = cid_str.rsplit("_", 1)[-1]
            out[i] = _CHIP_SUFFIX_TO_LOG10.get(suffix, np.nan)
    return out


# ---------------------------------------------------------------------------
# Soft Gaussian target construction
# ---------------------------------------------------------------------------

def build_gaussian_targets(y_ttp_min: np.ndarray, sigma_samples: float,
                           T: int = N_SAMPLES) -> np.ndarray:
    """One Gaussian-density target per pixel.

    target[i, t] = exp(- (t - ttp_idx_i)^2 / (2 * sigma^2))   in [0, 1]

    Peak amplitude is 1.0 at t = ttp_idx_i (the qLAMP-derived sample
    index).  Not normalised to integrate to 1 -- the MODEL has a sigmoid
    output that bounds it to [0, 1], so we want the target on the same
    scale.

    Inputs
        y_ttp_min     (N,) float32   qLAMP TTPs in minutes
        sigma_samples float          width parameter (in samples)
        T             int            trace length

    Returns
        targets       (N, T) float32
    """
    ttp_idx = (y_ttp_min.astype(np.float32) * float(SAMPLES_PER_MIN))   # (N,)
    t_grid = np.arange(T, dtype=np.float32)[None, :]                    # (1, T)
    sq = (t_grid - ttp_idx[:, None]) ** 2                               # (N, T)
    return np.exp(-sq / (2.0 * float(sigma_samples) ** 2)).astype(np.float32)


def curriculum_sigma_at(epoch: int, total_epochs: int,
                        sigma_start: float, sigma_end: float) -> float:
    """Linear anneal from sigma_start at epoch=1 to sigma_end at epoch=total_epochs.

    total_epochs=1 → return sigma_end (degenerate case).
    """
    if total_epochs <= 1:
        return float(sigma_end)
    frac = (epoch - 1) / (total_epochs - 1)     # 0.0 at epoch=1, 1.0 at final
    return float(sigma_start) + frac * (float(sigma_end) - float(sigma_start))


def cosine_sigma_at(epoch: int, total_epochs: int,
                    sigma_start: float, sigma_end: float) -> float:
    """Cosine-anneal σ from sigma_start (epoch=1) to sigma_end (epoch=total_epochs).

    Smoother tail than linear-anneal — stays near σ_start longer, tightens
    faster at the end.
    """
    if total_epochs <= 1:
        return float(sigma_end)
    frac = (epoch - 1) / (total_epochs - 1)
    return float(sigma_end) + 0.5 * (float(sigma_start) - float(sigma_end)) * (1 + math.cos(math.pi * frac))


def adaptive_sigma_step(val_losses: list[float], current_sigma: float,
                        plateau_window: int = 3, plateau_tol: float = 0.01,
                        spike_tol: float = 0.20, sharpen_by: float = 0.8,
                        broaden_by: float = 1.2,
                        clamp_min: float = 3.0, clamp_max: float = 15.0) -> float:
    """Loss-adaptive σ step.

    Called AFTER each epoch. Returns σ to use for the NEXT epoch.

    Rules:
      - Spike (latest val loss > previous * (1 + spike_tol)): broaden σ by
        `broaden_by`.
      - Else, plateau (last `plateau_window + 1` epochs all within `plateau_tol`
        relative of each other): sharpen σ by `sharpen_by`.
      - Otherwise: hold σ.
      - Result clamped to [clamp_min, clamp_max].
    """
    if len(val_losses) < 2:
        return float(current_sigma)
    # Spike check: latest > previous * (1 + spike_tol)
    if val_losses[-1] > val_losses[-2] * (1.0 + spike_tol):
        new_sigma = current_sigma * broaden_by
    elif len(val_losses) >= plateau_window + 1:
        # Plateau check: last (plateau_window + 1) all within plateau_tol
        recent = val_losses[-(plateau_window + 1):]
        rel_range = (max(recent) - min(recent)) / max(abs(min(recent)), 1e-8)
        if rel_range < plateau_tol:
            new_sigma = current_sigma * sharpen_by
        else:
            new_sigma = current_sigma
    else:
        new_sigma = current_sigma
    return float(max(clamp_min, min(clamp_max, new_sigma)))


# ---------------------------------------------------------------------------
# Backbone: cnn1d, adapted for per-timestep output
# ---------------------------------------------------------------------------

class CNN1DPerStep(nn.Module):
    """cnn1d with MaxPool removed and a per-timestep sigmoid head.

    Receptive field matches the original cnn1d's first two conv layers
    (kernel 3 + kernel 3 = 5 timesteps) with same-padding so T is
    preserved end-to-end.  Output shape: (batch, T).

    Parameter count is comparable to the original cnn1d (~70-80k).
    """

    def __init__(self, t: int = N_SAMPLES):
        super().__init__()
        self.t = t
        # Encoder: 1D conv stack, same-padding, no downsampling.
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        # Per-timestep head: small 1D conv refinement + projection to 1 channel.
        self.head = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T)  ->  density: (B, T) in [0, 1]"""
        z = self.encoder(x)                # (B, 32, T)
        out = self.head(z)                 # (B, 1, T)
        return torch.sigmoid(out).squeeze(1)


class P2BackboneDensity(nn.Module):
    """Wrap a P2 per-timestep backbone as an F-A density head.

    P2 backbones output (B, T) LOGITS.  F-A needs (B, T) DENSITY in [0, 1].
    Wrapping = sigmoid on the logits.
    """

    def __init__(self, backbone_id: str):
        super().__init__()
        from lacewing.quantification.methods.p2_backbones import build_p2_backbone
        self.backbone_id = backbone_id
        self.inner = build_p2_backbone(backbone_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.inner(x)             # (B, T)
        return torch.sigmoid(logits)


P2_BACKBONE_IDS = ["bigru", "gru", "ann", "cnn_gru_par", "transformer_patch", "unet", "tcn"]


def build_fa_backbone(backbone_id: str, t: int = N_SAMPLES) -> nn.Module:
    """Build F-A backbone: default (cnn1d) or one of the 7 P2 backbones."""
    if backbone_id == "cnn1d":
        return CNN1DPerStep(t=t)
    if backbone_id in P2_BACKBONE_IDS:
        return P2BackboneDensity(backbone_id)
    raise ValueError(
        f"Unknown F-A backbone {backbone_id!r}. Valid: "
        f"{['cnn1d'] + P2_BACKBONE_IDS}"
    )


# ---------------------------------------------------------------------------
# Dataset: (X, gaussian_target) pairs, with split-time TTP for eval
# ---------------------------------------------------------------------------

class PdfDataset(Dataset):
    def __init__(self, X: np.ndarray, target: np.ndarray):
        self.X = X.astype(np.float32, copy=False)
        self.target = target.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx][None, :])         # (1, T)
        y = torch.from_numpy(self.target[idx])             # (T,)
        return x, y


class WeightedMSE(nn.Module):
    """Per-timestep MSE weighted by the target value.

    Plain MSE on a sparse Gaussian target (mostly zeros with a small
    peak) collapses to the trivial constant solution: the model learns
    to output a low flat value everywhere because that minimises loss
    over the ~445 baseline timesteps at the cost of the ~5 peak ones.

    We re-weight the per-timestep squared error by (target + base) so
    the peak timesteps carry more gradient than the baseline ones.
    With base=0.1 and a target peaking at 1.0, peak timesteps get ~11x
    the weight of baseline ones.

    weight[t] = target[t] + base   (positive everywhere, larger near peak)
    loss      = mean( weight[t] * (pred[t] - target[t])^2 )
    """

    def __init__(self, base: float = 0.1):
        super().__init__()
        self.base = float(base)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = target + self.base
        return (weight * (pred - target) ** 2).mean()


class FocalMSE(nn.Module):
    """Per-timestep MSE weighted by |pred - target|^gamma.

    Focal-loss-style weighting: hard-to-predict timesteps (large residual)
    contribute quadratically more gradient than easy ones.  Complements
    WeightedMSE by upweighting on RESIDUAL rather than TARGET density.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = pred - target
        focal_w = err.abs().pow(self.gamma).detach()   # stop-grad on the weight
        return (focal_w * err.pow(2)).mean()


class KLDivergenceLoss(nn.Module):
    """KL divergence between predicted-density and Gaussian target as distributions.

    Softmax-normalises both `pred` and `target` along the last dim (T), then
    computes KL(target || pred).  Treats the PDF-regression output as a
    probability over time.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(pred, dim=-1) + self.eps
        t = torch.softmax(target, dim=-1) + self.eps
        return (t * (t.log() - p.log())).sum(dim=-1).mean()


def make_loaders(arr, sigma_samples: float, batch_size: int):
    """Wrap the SplitArrays output with Gaussian-target datasets."""
    targets_tr = build_gaussian_targets(arr.y_tr, sigma_samples)
    targets_va = build_gaussian_targets(arr.y_va, sigma_samples)
    # Test targets aren't used for loss; we evaluate against y_te directly.
    tr = DataLoader(PdfDataset(arr.X_tr, targets_tr),
                    batch_size=batch_size, shuffle=True)
    va = DataLoader(PdfDataset(arr.X_va, targets_va),
                    batch_size=256, shuffle=False)
    return tr, va


# ---------------------------------------------------------------------------
# Training + per-epoch eval
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, optim, criterion, device):
    model.train()
    total_loss, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        pred = model(x)                  # (B, T)
        loss = criterion(pred, y)
        loss.backward()
        optim.step()
        total_loss += float(loss) * y.size(0)
        total += y.size(0)
    return total_loss / total


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device, ttp_true_min: np.ndarray):
    """Returns (mean_loss, mae_min_argmax) for sanity-checking val curve."""
    model.eval()
    total_loss, total = 0.0, 0
    argmax_idx_all = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += float(loss) * y.size(0)
        total += y.size(0)
        argmax_idx_all.append(pred.argmax(dim=1).cpu().numpy())
    argmax_idx = np.concatenate(argmax_idx_all)
    ttp_pred_min = argmax_idx.astype(np.float32) / float(SAMPLES_PER_MIN)
    mae_min = float(np.mean(np.abs(ttp_pred_min - ttp_true_min)))
    return total_loss / total, mae_min


# ---------------------------------------------------------------------------
# Metrics + reporting (matches the regression baselines' schema)
# ---------------------------------------------------------------------------

def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    n_total = len(y_true)
    n_used = int(mask.sum())
    y_t = y_true[mask]; y_p = y_pred[mask]
    err = y_p - y_t
    mae = float(np.mean(np.abs(err))) if n_used else float("nan")
    rmse = float(np.sqrt(np.mean(err ** 2))) if n_used else float("nan")
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_t - y_t.mean()) ** 2)) if n_used else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae_baseline = (float(np.mean(np.abs(y_t - y_t.mean())))
                    if n_used else float("nan"))
    return {
        "mae_min":  mae,
        "rmse_min": rmse,
        "r2":       r2,
        "mae_baseline_constant_mean": mae_baseline,
        "n":        n_used,
        "n_total":  n_total,
        "n_nan_pred": n_total - n_used,
    }


def _per_well_predictions(pred_min: np.ndarray, chip: np.ndarray,
                          well: np.ndarray) -> dict[str, float]:
    keys = np.array([f"{c}::{w}" for c, w in zip(chip, well)])
    out: dict[str, list[float]] = {}
    for k, p in zip(keys, pred_min):
        if not np.isnan(p):
            out.setdefault(k, []).append(float(p))
    return {k: float(np.mean(v)) if v else float("nan")
            for k, v in out.items()}


def _plot_curves(metrics_csv: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    epochs, tr_loss, va_loss, va_mae = [], [], [], []
    with metrics_csv.open() as fh:
        for row in csv.DictReader(fh):
            epochs.append(int(row["epoch"]))
            tr_loss.append(float(row["tr_loss"]))
            va_loss.append(float(row["va_loss"]))
            va_mae.append(float(row["va_mae_min"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(epochs, tr_loss, label="train MSE", color="#1f78b4")
    axes[0].plot(epochs, va_loss, label="val MSE",   color="#e31a1c")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("density MSE")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=9)
    axes[0].set_title("PDF regression train curves")
    axes[1].plot(epochs, va_mae, color="#2ca02c")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val TTP MAE (min, argmax)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Val TTP MAE via argmax")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    import matplotlib.pyplot as _; _.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULTS = {
    "cache":         "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3",
    "experiment":    "f6_pdf_cnn1d_spatA3_wmse",
    "val_n_chips":   1,
    "epochs":        40,
    "batch_size":    32,
    "lr":            1e-3,
    "lr_step":       10,
    "lr_gamma":      0.75,
    "sigma":         5.0,        # samples (~20 s)
    "loss":          "wmse",     # "wmse" = weighted MSE, "mse" = plain MSE
    "wmse_base":     0.1,        # baseline weight added to target before MSE
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default=DEFAULTS["cache"])
    p.add_argument("--experiment", default=DEFAULTS["experiment"])
    p.add_argument("--val-n-chips", type=int, default=DEFAULTS["val_n_chips"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--lr-step", type=int, default=DEFAULTS["lr_step"])
    p.add_argument("--lr-gamma", type=float, default=DEFAULTS["lr_gamma"])
    p.add_argument("--sigma", type=float, default=DEFAULTS["sigma"],
                   help="Gaussian width in samples (15 samples = 1 min).")
    p.add_argument("--loss", choices=["mse", "wmse", "focal_mse", "kl"], default=DEFAULTS["loss"],
                   help="mse = plain per-timestep MSE; "
                        "wmse = MSE weighted by (target + wmse_base); "
                        "focal_mse = MSE weighted by |pred-target|^focal_gamma; "
                        "kl = KL divergence between softmax(pred) and softmax(target).")
    p.add_argument("--wmse-base", type=float, default=DEFAULTS["wmse_base"],
                   help="Baseline weight added to target in WeightedMSE. "
                        "Higher = more uniform weighting (->MSE); lower = "
                        "more peak-focused.")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal-MSE exponent gamma. Used only with --loss focal_mse.")
    p.add_argument("--curriculum-sigma", action="store_true",
                   help="Anneal Gaussian target σ from --sigma-start to --sigma-end over training.")
    p.add_argument("--sigma-start", type=float, default=12.0,
                   help="Starting σ for --curriculum-sigma (samples).")
    p.add_argument("--sigma-end", type=float, default=3.0,
                   help="Ending σ for --curriculum-sigma (samples).")
    p.add_argument("--sigma-schedule", default="linear",
                   choices=["linear", "adaptive", "cosine"],
                   help="Curriculum σ schedule (only used with --curriculum-sigma). "
                        "'linear' = curriculum_sigma_at (default, backward compat); "
                        "'cosine' = cosine_sigma_at; "
                        "'adaptive' = loss-adaptive step between epochs.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--tag", default=None)
    p.add_argument("--backbone", default="cnn1d",
                   choices=["cnn1d"] + P2_BACKBONE_IDS,
                   help="F-A backbone. Default 'cnn1d' (legacy, backward-compat). "
                        "The 7 P2 backbones are also available: ann, bigru, gru, "
                        "cnn_gru_par, transformer_patch, unet, tcn.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    from lacewing.classification.core.seeding import seed_everything
    seed_everything(args.seed)

    # ----- load + split -----
    print(f"Loading cache: {args.cache}")
    arr = reg_ds.make_split(args.cache, seed=args.seed,
                             val_n_chips=args.val_n_chips,
                             normalise_y=False)
    print(f"  n_tr_pixels={len(arr.X_tr)}  "
          f"n_va_pixels={len(arr.X_va)}  n_te_pixels={len(arr.X_te)}")
    print(f"  val chips:  {sorted(set(arr.chip_va.tolist()))}")
    print(f"  test chips: {sorted(set(arr.chip_te.tolist()))}")
    print(f"  train TTPs (unique): "
          f"{sorted(np.unique(arr.y_tr).round(3).tolist())}")

    # ----- build loaders -----
    tr_loader, va_loader = make_loaders(arr, args.sigma, args.batch_size)
    print(f"  Gaussian target sigma = {args.sigma} samples "
          f"(~{args.sigma * 4:.0f} s)")

    # ----- model + loss + optim -----
    model = build_fa_backbone(args.backbone, t=N_SAMPLES).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built F-A backbone={args.backbone}  ({n_params:,} params)")
    if args.loss == "wmse":
        criterion = WeightedMSE(base=args.wmse_base)
        print(f"  loss = WeightedMSE(base={args.wmse_base})")
    elif args.loss == "focal_mse":
        criterion = FocalMSE(gamma=args.focal_gamma)
        print(f"  loss = FocalMSE(gamma={args.focal_gamma})")
    elif args.loss == "kl":
        criterion = KLDivergenceLoss()
        print(f"  loss = KLDivergenceLoss")
    else:
        criterion = nn.MSELoss(reduction="mean")
        print(f"  loss = MSE")
    optim = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optim, step_size=args.lr_step, gamma=args.lr_gamma)

    # ----- run dir -----
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    loss_label = args.loss
    if args.loss == "wmse":
        loss_label = f"wmse{args.wmse_base:g}"
    elif args.loss == "focal_mse":
        loss_label = f"focal_mse_g{args.focal_gamma:g}"
    sched_tag = "" if args.sigma_schedule == "linear" else f"_sched{args.sigma_schedule}"
    if args.curriculum_sigma:
        run_dir = (RESULTS_ROOT / args.experiment /
                   f"seed{args.seed}_curriculum_s{args.sigma_start:g}_to_{args.sigma_end:g}_{loss_label}{sched_tag}{tag}")
    else:
        run_dir = (RESULTS_ROOT / args.experiment /
                   f"seed{args.seed}_sigma{args.sigma:g}_{loss_label}{tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    cfg = {
        "experiment":   args.experiment,
        "method":       f"F6 PDF regression ({args.backbone} backbone, soft Gaussian target)",
        "backbone":     args.backbone,
        "cache":        args.cache,
        "seed":         args.seed,
        "val_n_chips":  args.val_n_chips,
        "sigma_samples": args.sigma,
        "sigma_seconds": args.sigma * (60.0 / SAMPLES_PER_MIN),
        "loss": args.loss,
        "wmse_base": args.wmse_base if args.loss == "wmse" else None,
        "focal_gamma": args.focal_gamma if args.loss == "focal_mse" else None,
        "curriculum_sigma": bool(args.curriculum_sigma),
        "sigma_start": args.sigma_start if args.curriculum_sigma else None,
        "sigma_end":   args.sigma_end if args.curriculum_sigma else None,
        "sigma_schedule": args.sigma_schedule if args.curriculum_sigma else None,
        "samples_per_min": SAMPLES_PER_MIN,
        "n_samples":    N_SAMPLES,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "lr_step":      args.lr_step,
        "lr_gamma":     args.lr_gamma,
        "n_parameters": n_params,
        "device":       args.device,
        "n_train":      len(arr.X_tr),
        "n_val":        len(arr.X_va),
        "n_test":       len(arr.X_te),
        "val_chips_held_out": sorted(set(arr.chip_va.tolist())),
        "test_chips":         sorted(set(arr.chip_te.tolist())),
        "started_at":   datetime.now().isoformat(),
    }
    (run_dir / "config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8")

    # ----- training loop -----
    metrics_csv = run_dir / "metrics.csv"
    fields = ["epoch", "lr", "tr_loss", "va_loss", "va_mae_min"]
    with metrics_csv.open("w", newline="") as fh:
        csv.DictWriter(fh, fields).writeheader()

    # Week 11 debrief priority-1 fix: track val TTP MAE history to select
    # best checkpoint by val TTP MAE (argmax-derived), not val_loss.
    va_loss_history: list[float] = []
    va_ttp_mae_history: list[float] = []
    adaptive_sigma_history: list[float] = []  # val losses seen so far, for --sigma-schedule adaptive
    current_adaptive_sigma = args.sigma_start
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # Curriculum σ: rebuild loaders with the epoch's σ if enabled.
        if args.curriculum_sigma:
            if args.sigma_schedule == "cosine":
                sigma_this_epoch = cosine_sigma_at(
                    epoch=epoch, total_epochs=args.epochs,
                    sigma_start=args.sigma_start, sigma_end=args.sigma_end,
                )
            elif args.sigma_schedule == "adaptive":
                # current_adaptive_sigma is updated AFTER the epoch's val loss
                # is known (see below); this epoch uses whatever was decided
                # at the end of the previous epoch.
                sigma_this_epoch = current_adaptive_sigma
            else:  # "linear" (default, backward compat)
                sigma_this_epoch = curriculum_sigma_at(
                    epoch=epoch, total_epochs=args.epochs,
                    sigma_start=args.sigma_start, sigma_end=args.sigma_end,
                )
            tr_loader, va_loader = make_loaders(arr, sigma_this_epoch, args.batch_size)
            if epoch == 1 or epoch == args.epochs:
                print(f"  [curriculum:{args.sigma_schedule}] epoch {epoch}: σ = {sigma_this_epoch:.2f}")

        tr_loss = _train_epoch(model, tr_loader, optim, criterion, args.device)
        va_loss, va_mae_min = _eval_epoch(
            model, va_loader, criterion, args.device, arr.y_va)
        scheduler.step()

        va_loss_history.append(va_loss)
        va_ttp_mae_history.append(va_mae_min)

        if args.curriculum_sigma and args.sigma_schedule == "adaptive":
            adaptive_sigma_history.append(va_loss)
            current_adaptive_sigma = adaptive_sigma_step(
                adaptive_sigma_history, current_adaptive_sigma)

        cur_lr = optim.param_groups[0]["lr"]
        with metrics_csv.open("a", newline="") as fh:
            csv.DictWriter(fh, fields).writerow({
                "epoch":      epoch,
                "lr":         cur_lr,
                "tr_loss":    tr_loss,
                "va_loss":    va_loss,
                "va_mae_min": va_mae_min,
            })
        print(f"epoch {epoch:02d}/{args.epochs}  "
              f"tr_loss={tr_loss:.5f}  va_loss={va_loss:.5f}  "
              f"va_mae_min={va_mae_min:.3f}  lr={cur_lr:.2e}")

        # Save best.pt when this epoch minimises val TTP MAE (Week 11 fix).
        best_epoch_idx = select_best_epoch(
            val_loss=np.array(va_loss_history),
            val_ttp_mae=np.array(va_ttp_mae_history),
        )
        if best_epoch_idx == epoch - 1:  # epoch is 1-indexed; history is 0-indexed
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": va_loss, "val_ttp_mae": va_mae_min},
                       run_dir / "checkpoints" / "best.pt")
        # Always save last.
        torch.save({"epoch": epoch, "model_state": model.state_dict()},
                   run_dir / "checkpoints" / "last.pt")

    train_time_s = time.time() - t0

    # ----- reload best, run test -----
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt",
                      map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nReloaded best checkpoint from epoch {ckpt['epoch']}.")

    model.eval()
    # Test batch loader (no targets needed; predictions only).
    te_loader = DataLoader(
        PdfDataset(arr.X_te, np.zeros_like(arr.X_te)),
        batch_size=256, shuffle=False,
    )
    densities = []
    with torch.no_grad():
        for x, _ in te_loader:
            x = x.to(args.device)
            densities.append(model(x).cpu().numpy())
    densities = np.concatenate(densities, axis=0)   # (N_test, T)
    argmax_idx = densities.argmax(axis=1)
    ttp_pred_min = argmax_idx.astype(np.float32) / float(SAMPLES_PER_MIN)
    n_at_left_edge = int((argmax_idx == 0).sum())
    n_at_right_edge = int((argmax_idx == N_SAMPLES - 1).sum())
    print(f"  test pixels: {len(arr.X_te)}; "
          f"argmax at t=0: {n_at_left_edge}; "
          f"argmax at t={N_SAMPLES - 1}: {n_at_right_edge}")

    # ----- metrics -----
    pixel_metrics = _regression_metrics(arr.y_te, ttp_pred_min)
    well_pred_min = _per_well_predictions(ttp_pred_min, arr.chip_te, arr.well_te)
    keys = [f"{c}::{w}" for c, w in zip(arr.chip_te, arr.well_te)]
    well_true: dict[str, float] = {}
    for k, yv in zip(keys, arr.y_te):
        well_true[k] = float(yv)
    well_keys = sorted(set(keys))
    yw_true = np.array([well_true[k]
                        for k in well_keys
                        if not np.isnan(well_pred_min.get(k, np.nan))])
    yw_pred = np.array([well_pred_min[k]
                        for k in well_keys
                        if not np.isnan(well_pred_min.get(k, np.nan))])
    well_metrics = _regression_metrics(yw_true, yw_pred)

    # ----- write outputs -----
    test_payload = {
        "pixel": pixel_metrics,
        "well":  well_metrics,
        "per_well": [
            {"chip_id": k.split("::")[0],
             "well_id": int(k.split("::")[1]),
             "y_true_min": float(well_true[k]),
             "y_pred_min": float(well_pred_min.get(k, float("nan")))}
            for k in well_keys
        ],
        "best_epoch":     int(ckpt["epoch"]),
        "best_va_loss":   float(ckpt["val_loss"]),
        "best_va_ttp_mae": float(ckpt.get("val_ttp_mae", float("nan"))),
        "train_time_s": float(train_time_s),
        "sigma_samples": args.sigma,
        "n_test_argmax_at_left_edge":  n_at_left_edge,
        "n_test_argmax_at_right_edge": n_at_right_edge,
    }
    (run_dir / "test_metrics.json").write_text(
        json.dumps(test_payload, indent=2), encoding="utf-8")

    # Canonical P7 schema — predictions.npz replaces the legacy format.
    write_predictions(
        run_dir / "predictions.npz",
        ttp_pred_min=ttp_pred_min.astype(np.float32),
        ttp_pred_extras={"density": densities.astype(np.float32)},
    )

    # Derive log10(copies/uL) per pixel from chip_id + well_id.
    log10_conc = _log10_concentration_array(arr.chip_te, arr.well_te)

    write_labels(
        run_dir / "labels.npz",
        ttp_true_min=arr.y_te.astype(np.float32),
        chip_id=arr.chip_te,
        well_id=arr.well_te.astype(np.int32),
        log10_concentration=log10_conc,
        split=np.full(len(ttp_pred_min), "test", dtype="U16"),
    )

    _plot_curves(metrics_csv, run_dir / "train_curves.png")

    print(f"\nWrote {run_dir}/")
    print(f"  Test pixel MAE = {pixel_metrics['mae_min']:.3f} min  "
          f"(baseline constant-mean = "
          f"{pixel_metrics['mae_baseline_constant_mean']:.3f} min)")
    print(f"  Test well  MAE = {well_metrics['mae_min']:.3f} min  "
          f"(n_wells = {well_metrics['n']})")
    print(f"  R^2 (pixel) = {pixel_metrics['r2']:.3f}    "
          f"R^2 (well) = {well_metrics['r2']:.3f}")
    print(f"  training time = {train_time_s/60:.1f} min")


if __name__ == "__main__":
    main()
