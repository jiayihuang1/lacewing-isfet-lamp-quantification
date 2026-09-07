"""F-B confound-check: F-B framing on a chosen (P2-winning) backbone.

The P1 F-B result used CNN-GRU-par backbone.  The P1 F-E result used BiGRU.
Different backbones + different framings = confounded.  This runner keeps the
F-B framing (sliding-window classifier + first_above_K3 as the DEFAULT
extraction rule; extraction rule can be swept post-hoc via the F-B extraction
sweep) but swaps the backbone in.  Compare against F-E-on-same-backbone (from
Analysis/quantification/methods/p2_sweep.py) to isolate the framing effect.

Backbones (must produce a scalar logit per length-W window):
  * ann          — small MLP on flattened window (~9k params for W=60)
  * gru          — unidirectional GRU with mean-pool head
  * bigru        — bidirectional GRU with mean-pool head
  * cnn_gru_par  — CNN-GRU-par (the P1 F-B backbone; kept for reproducibility)
  * transformer_patch — patched transformer with mean-pool head
  * unet         — 1D U-Net encoder features + global-pool head
  * tcn          — TCN with global-pool head

Every backbone here consumes a windowed input `(B, 1, W)` and outputs a
single scalar per window: `(B,)` post-amp logit.

Output dir (canonical schema, with window_probs saved as extras):
    Analysis/quantification/methods/results/
        fb_confound_{backbone}_w{W}_stride{S}_spatA3/seed{seed}/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from lacewing.quantification.eval.schema import write_predictions, write_labels
from lacewing.quantification.methods._windowing import (
    aggregate_to_ttp_first_positive,
    slide_windows,
)
from lacewing.quantification.methods.pdf_regression import _log10_concentration_array
from lacewing.quantification.regression_shared.data import dataset as reg_ds


SAMPLES_PER_MIN = 15
N_SAMPLES = 450
CACHE_STEM = "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3"

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results"


# ---------------------------------------------------------------------------
# Per-pixel windowed dataset (identical to F-B's)
# ---------------------------------------------------------------------------

class _WindowedDataset(Dataset):
    """Yields (1, W) window slices with binary post-amp labels.

    Each pixel becomes `n_windows` samples in the dataset — one per window.
    Label = 1 if window centre-timestep >= qLAMP TTP index, else 0.
    """
    def __init__(self, X: np.ndarray, ttp_min: np.ndarray, window: int, stride: int):
        # X: (n_pixels, T); ttp_min: (n_pixels,) in minutes
        self.window = window
        self.stride = stride
        self.windows = slide_windows(X, window, stride)  # (n_pixels, n_windows, W)
        n_pixels, n_windows, _ = self.windows.shape

        # Window centre timestep index.
        window_centre_idx = np.arange(n_windows) * stride + (window - 1) / 2.0
        ttp_idx = (ttp_min * SAMPLES_PER_MIN).astype(np.float32)
        # (n_pixels, n_windows) binary labels
        self.labels = (window_centre_idx[None, :] >= ttp_idx[:, None]).astype(np.float32)

        # Flatten pixels x windows.
        self.flat_x = self.windows.reshape(n_pixels * n_windows, 1, window)
        self.flat_y = self.labels.reshape(n_pixels * n_windows)
        self.n = self.flat_x.shape[0]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.flat_x[idx]).float()  # (1, W)
        y = torch.tensor(self.flat_y[idx], dtype=torch.float32)
        return x, y


# ---------------------------------------------------------------------------
# Backbone factory — each backbone returns (B,) logits from (B, 1, W)
# ---------------------------------------------------------------------------

def _build_window_backbone(backbone_id: str, W: int) -> nn.Module:
    """Return a model with forward(x: (B, 1, W)) -> (B,) logit."""
    if backbone_id == "ann":
        from lacewing.classification.models.ann import ANN
        return ANN(input_len=W)
    if backbone_id == "gru":
        from lacewing.classification.models.gru import GRUClassifier
        # existing GRUClassifier is BIDIRECTIONAL by default; force uni.
        return GRUClassifier(input_len=W, bidirectional=False)
    if backbone_id == "bigru":
        from lacewing.classification.models.gru import GRUClassifier
        return GRUClassifier(input_len=W, bidirectional=True)
    if backbone_id == "cnn_gru_par":
        from lacewing.classification.models.cnn_gru_parallel import CNNGRUParallel
        return CNNGRUParallel(input_len=W)
    if backbone_id == "transformer_patch":
        # transformer_patch requires input_len divisible by patch_size (default 15).
        # For arbitrary W, we pick a patch_size that divides W.
        from lacewing.classification.models.transformer_patch import PatchTransformer
        patch_size = W
        for candidate in [15, 10, 6, 5, 3, 2, 1]:
            if W % candidate == 0:
                patch_size = candidate
                break
        return PatchTransformer(input_len=W, patch_size=patch_size)
    if backbone_id == "unet":
        # U-Net expects at least 16-samples input.  For W=60/90/120 fine.  For
        # W=30 we'd need base_channels smaller; but P1 winner determination
        # only requires one W.  Wrap U-Net with a global mean-pool to get (B,)
        # from (B, n_classes, T).
        from lacewing.classification.models.unet_1d import UNet1D
        return _UNetForWindow(UNet1D(in_channels=1, n_classes=1, base_channels=8))
    if backbone_id == "tcn":
        from lacewing.classification.models.tcn import TCN
        return _SqueezeLast(TCN(input_length=W, n_classes=1))
    raise ValueError(
        f"Unknown backbone {backbone_id!r}. "
        f"Valid: ann, gru, bigru, cnn_gru_par, transformer_patch, unet, tcn"
    )


class _UNetForWindow(nn.Module):
    """Wrap a 1D U-Net whose forward gives (B, 1, T) into (B,) via global mean-pool."""
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.unet(x)  # (B, 1, W)
        return h.mean(dim=(1, 2))  # (B,)


class _SqueezeLast(nn.Module):
    """Wrap a module whose forward gives (B, 1) into (B,)."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.m(x)  # (B, 1) or (B, n)
        return h.squeeze(-1)


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
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    for e in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)  # (B,)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * yb.size(0)
            total_samples += yb.size(0)
        avg_train_loss = total_loss / max(total_samples, 1)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_acc = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)  # (B,)
                val_loss_acc += loss_fn(logits, yb).item() * yb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == yb).sum().item()
                val_total += yb.numel()
        avg_val_loss = val_loss_acc / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        record = {
            "epoch": e,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
        }
        history.append(record)
        print(
            f"epoch {e + 1:02d}/{epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"val_acc={val_acc:.4f}"
        )
    return history


# ---------------------------------------------------------------------------
# Test-time prediction — return per-pixel per-window sigmoid probs
# ---------------------------------------------------------------------------

def _predict_windows(
    model: nn.Module,
    X: np.ndarray,
    window: int,
    stride: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Return (n_pixels, n_windows) sigmoid probabilities."""
    model.eval()
    ws = slide_windows(X, window, stride)  # (n_pixels, n_windows, W)
    n_pixels, n_windows, _ = ws.shape
    probs = np.empty((n_pixels, n_windows), dtype=np.float32)
    ws_flat = ws.reshape(n_pixels * n_windows, 1, window).astype(np.float32)
    with torch.no_grad():
        for start in range(0, ws_flat.shape[0], batch_size):
            end = min(start + batch_size, ws_flat.shape[0])
            xb = torch.from_numpy(ws_flat[start:end]).to(device)
            logits = model(xb)  # (B,)
            probs_flat = torch.sigmoid(logits).cpu().numpy()
            probs.flat[start:end] = probs_flat
    return probs


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--backbone", type=str, required=True,
        choices=["ann", "gru", "bigru", "cnn_gru_par", "transformer_patch", "unet", "tcn"],
        help="P2-winning backbone (or any of the P2 slate) to swap into F-B's framing.",
    )
    parser.add_argument("--window", type=int, default=60,
                        help="Window length in samples.  Default 60 = F-B W=60 argmax_P best.")
    parser.add_argument("--stride", type=int, default=5,
                        help="Sliding stride in samples.  Default 5.")
    parser.add_argument("--k_consecutive", type=int, default=3,
                        help="TTP extraction: K consecutive above-threshold rule.  Default 3 (F-B original).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_train_pixels", type=int, default=None,
                        help="Subsample training pixels for smoke test.")
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

    if args.max_train_pixels is not None and len(X_tr) > args.max_train_pixels:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X_tr), size=args.max_train_pixels, replace=False)
        idx.sort()
        X_tr = X_tr[idx]
        y_tr = y_tr[idx]
        print(f"  [debug] subsampled train to {len(X_tr)} pixels")

    train_ds = _WindowedDataset(X_tr, y_tr, args.window, args.stride)
    val_ds = _WindowedDataset(X_va, y_va, args.window, args.stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(
        f"  total train (pixel, window) rows = {len(train_ds)}  "
        f"total val rows = {len(val_ds)}"
    )

    # ----- build model -----
    model = _build_window_backbone(args.backbone, args.window).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built confound-check backbone {args.backbone!r} for W={args.window}  ({n_params:,} params)")

    # ----- train -----
    history = _train(model, train_loader, val_loader, args.device, args.epochs, args.lr)

    # ----- test-time eval -----
    print("Running test-time windowed prediction...")
    probs = _predict_windows(model, arr.X_te, args.window, args.stride, args.device, args.batch_size)
    print(f"  probs shape: {probs.shape}")

    # First-K-consecutive-positives TTP extraction (F-B default rule).
    ttp_pred = aggregate_to_ttp_first_positive(
        probs, threshold=0.5, k_consecutive=args.k_consecutive,
        window=args.window, stride=args.stride, samples_per_min=SAMPLES_PER_MIN,
    )
    print(f"  ttp_pred_min range: [{ttp_pred.min():.2f}, {ttp_pred.max():.2f}]")

    # ----- write outputs -----
    out_dir = (
        RESULTS_ROOT
        / f"fb_confound_{args.backbone}_w{args.window}_stride{args.stride}_spatA3"
        / f"seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "method": f"F-B confound-check — backbone={args.backbone}, W={args.window}, stride={args.stride}",
        "framing": "F-B (sliding-window classifier, per-window BCE, first-K-consecutive TTP)",
        "seed": args.seed,
        "backbone": args.backbone,
        "window": args.window,
        "stride": args.stride,
        "k_consecutive": args.k_consecutive,
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
        "n_samples": N_SAMPLES,
        "cache_stem": CACHE_STEM,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    write_predictions(
        out_dir / "predictions.npz",
        ttp_pred_min=ttp_pred.astype(np.float32),
        ttp_pred_extras={
            "window_probs": probs.astype(np.float32),
            "window": np.array([args.window], dtype=np.int32),
            "stride": np.array([args.stride], dtype=np.int32),
        },
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
