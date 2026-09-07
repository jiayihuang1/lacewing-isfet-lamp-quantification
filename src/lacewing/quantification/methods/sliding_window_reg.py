"""RQ2 · F-C sliding-window regression framing. [Cat B] Report §RQ2 (documented dead branch: Spearman ρ sign-flip).

F-C — Sliding-window direct regression per-window.

Own design: soft target vs F-B's hard label. Each window's head outputs
`minutes_until_TTP` (a signed scalar); TTP = window whose predicted
distance-to-TTP is closest to zero.

Provenance
----------
Own design; natural interpolation between F-B (hard label) and F-A (soft PDF
target). No published paper prescribes this exact framing on LAMP.

Usage
-----
    python -m lacewing.quantification.methods.sliding_window_reg \\
        --seed 0 --window 60 --stride 5 --epochs 30

Output dir (canonical schema):
    Analysis/quantification/methods/results/
        p1_fc_slide_reg_w{window}_stride{stride}_spatA3/seed{seed}/
            config.json
            history.json
            predictions.npz
            labels.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from lacewing.classification.models.cnn_gru_parallel import CNNGRUParallel
from lacewing.quantification.eval.schema import write_predictions, write_labels
from lacewing.quantification.methods._windowing import (
    slide_windows,
    aggregate_to_ttp_closest_to_zero,
)
from lacewing.quantification.regression_shared.data import dataset as reg_ds
from lacewing.quantification.methods.pdf_regression import _log10_concentration_array


SAMPLES_PER_MIN = 15
N_SAMPLES = 450
CACHE_STEM = "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3"

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-window dataset
# ---------------------------------------------------------------------------

class _WindowRegressionDataset(Dataset):
    """Yields (1, W) windows with per-window signed distance-to-TTP labels.

    Label per window = TTP_min - window_centre_min (positive = TTP in future,
    negative = TTP already passed).
    """

    def __init__(self, X: np.ndarray, ttp_min: np.ndarray, window: int, stride: int):
        self.windows = slide_windows(X, window, stride)  # (n_pixels, N_windows, W)
        n_windows = self.windows.shape[1]
        centres_samples = np.arange(n_windows) * stride + (window - 1) / 2
        centres_min = centres_samples / SAMPLES_PER_MIN  # (N_windows,)
        # Label per window: signed distance TTP - window_centre (positive = TTP in future).
        self.labels = (ttp_min[:, None] - centres_min[None, :]).astype(np.float32)
        self.n_pixels = self.windows.shape[0]
        self.n_windows = n_windows

    def __len__(self) -> int:
        return self.n_pixels * self.n_windows

    def __getitem__(self, idx: int):
        p = idx // self.n_windows
        w = idx % self.n_windows
        x = torch.from_numpy(self.windows[p, w])[None, :]  # (1, W)
        y = torch.tensor(self.labels[p, w], dtype=torch.float32)
        return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(window: int) -> nn.Module:
    """CNN-GRU-par backbone for per-window regression.

    CNNGRUParallel forward returns (B,) scalars (head Linear(d_fusion, 1)
    then squeeze(-1)), so HuberLoss applies directly.
    """
    return CNNGRUParallel(input_len=window)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
) -> list[dict]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=1.0)
    history = []
    for e in range(epochs):
        # Train step
        model.train()
        total_loss = 0.0
        total_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)            # (B,) — already squeezed by CNNGRUParallel
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * yb.size(0)
            total_samples += yb.size(0)
        avg_train_loss = total_loss / max(total_samples, 1)

        # Validation step (MAE in minutes)
        model.eval()
        total_mae = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)        # (B,)
                total_mae += torch.abs(pred - yb).sum().item()
                n += yb.numel()
        val_mae = total_mae / max(n, 1)

        record = {
            "epoch": e,
            "train_loss": avg_train_loss,
            "val_mae_min": val_mae,
        }
        history.append(record)
        print(
            f"epoch {e + 1:02d}/{epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_mae_min={val_mae:.4f}"
        )
    return history


# ---------------------------------------------------------------------------
# Test-time prediction
# ---------------------------------------------------------------------------

def _predict_windows(
    model: nn.Module,
    X: np.ndarray,
    window: int,
    stride: int,
    device: str,
) -> np.ndarray:
    """Return (n_pixels, N_windows) predicted minutes_until_TTP scalars."""
    model.eval()
    ws = slide_windows(X, window, stride)   # (n_pixels, N_windows, W)
    n_p, n_w, _ = ws.shape
    out = np.empty((n_p, n_w), dtype=np.float32)
    with torch.no_grad():
        for i in range(n_p):
            # (N_windows, 1, W) — batch over windows for one pixel at a time.
            xb = torch.from_numpy(ws[i]).unsqueeze(1).to(device)  # (N_windows, 1, W)
            out[i] = model(xb).cpu().numpy()                       # (N_windows,)
    return out


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--max_train_pixels", type=int, default=None,
        help="Subsample training set to at most N pixels (for smoke-testing on CPU).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----- load + split -----
    print(f"Loading cache: {CACHE_STEM}")
    arr = reg_ds.make_split(
        CACHE_STEM,
        seed=args.seed,
        val_n_chips=1,
        normalise_y=False,
    )
    print(
        f"  n_tr={len(arr.X_tr)}  n_va={len(arr.X_va)}  n_te={len(arr.X_te)}"
    )

    X_tr, y_tr = arr.X_tr, arr.y_tr
    X_va, y_va = arr.X_va, arr.y_va

    # Optional subsampling for fast smoke-tests.
    if args.max_train_pixels is not None and len(X_tr) > args.max_train_pixels:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X_tr), size=args.max_train_pixels, replace=False)
        idx.sort()
        X_tr = X_tr[idx]
        y_tr = y_tr[idx]
        print(f"  [debug] subsampled train to {len(X_tr)} pixels")

    train_ds = _WindowRegressionDataset(X_tr, y_tr, args.window, args.stride)
    val_ds = _WindowRegressionDataset(X_va, y_va, args.window, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(
        f"  windows per pixel = {train_ds.n_windows}  "
        f"total train windows = {len(train_ds)}  "
        f"total val windows = {len(val_ds)}"
    )

    # ----- build model -----
    model = build_model(window=args.window).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built CNNGRUParallel(input_len={args.window})  ({n_params:,} params)")

    # ----- train -----
    history = _train(model, train_loader, val_loader, args.device, args.epochs, args.lr)

    # ----- test-time eval -----
    print("Running test-time window prediction...")
    preds_win = _predict_windows(model, arr.X_te, args.window, args.stride, args.device)
    print(f"  window preds shape: {preds_win.shape}")

    ttp_pred = aggregate_to_ttp_closest_to_zero(
        preds_win,
        window=args.window,
        stride=args.stride,
        samples_per_min=SAMPLES_PER_MIN,
    )
    print(f"  ttp_pred_min range: [{ttp_pred.min():.2f}, {ttp_pred.max():.2f}]")

    # Sanity clamp — spec range [0, 30] min for the 30-min ISFET window.
    assert np.all(np.isfinite(ttp_pred)), "non-finite TTP predictions"
    assert ttp_pred.min() >= 0 and ttp_pred.max() <= 30

    # ----- write outputs -----
    out_dir = (
        RESULTS_ROOT
        / f"p1_fc_slide_reg_w{args.window}_stride{args.stride}_spatA3"
        / f"seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "method": "P1.3 F-C sliding-window regression (own design)",
        "seed": args.seed,
        "window": args.window,
        "stride": args.stride,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_train_pixels": args.max_train_pixels,
        "device": args.device,
        "n_train_pixels": len(X_tr),
        "n_val_pixels": len(X_va),
        "n_test_pixels": len(arr.X_te),
        "n_params": n_params,
        "samples_per_min": SAMPLES_PER_MIN,
        "cache_stem": CACHE_STEM,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Canonical schema.
    write_predictions(
        out_dir / "predictions.npz",
        ttp_pred_min=ttp_pred.astype(np.float32),
    )

    log10_conc = _log10_concentration_array(arr.chip_te, arr.well_te)

    write_labels(
        out_dir / "labels.npz",
        ttp_true_min=arr.y_te.astype(np.float32),
        chip_id=arr.chip_te,
        well_id=arr.well_te.astype(np.int32),
        log10_concentration=log10_conc,
        split=np.full(len(arr.X_te), "test", dtype="U16"),
    )

    print(f"[ok] wrote {out_dir}/")
    print(f"  predictions.npz  labels.npz  config.json  history.json")


if __name__ == "__main__":
    main()
