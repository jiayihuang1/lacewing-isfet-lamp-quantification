"""P2 backbone sweep on F-E's framing.

Fix F-E's framing (whole-trace input, per-timestep binary label, K=5
consecutive-positive TTP extraction) and dispatch on --backbone to swap
in one of 7 architectures.  Disambiguates "did F-E win because of the
framing or because of the BiGRU backbone?"

Backbones (see Analysis/quantification/methods/p2_backbones.py):
  * bigru              — the P1.5 F-E reference (100k params)
  * gru                — unidirectional, half of BiGRU (37k)
  * ann                — diagnostic no-context MLP per timestep (177)
  * cnn_gru_par        — CNN branch broadcast + BiGRU per-timestep (350k)
  * transformer_patch  — PatchTST with per-patch → per-timestep (101k)
  * unet               — 1D U-Net with n_classes=1 (492k)
  * tcn                — TCNPerTimestep with n_classes=1 (55k)

Usage
-----
    python -m lacewing.quantification.methods.p2_sweep \\
        --seed 0 --backbone gru --epochs 30

Output dir (canonical schema):
    Analysis/quantification/methods/results/
        p2_{backbone}_full_spatA3/seed{seed}/
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

from lacewing.quantification.eval.schema import write_predictions, write_labels
from lacewing.quantification.regression_shared.data import dataset as reg_ds
from lacewing.quantification.methods.pdf_regression import _log10_concentration_array
from lacewing.quantification.methods.p2_backbones import BACKBONES, build_p2_backbone


SAMPLES_PER_MIN = 15
N_SAMPLES = 450
CACHE_STEM = "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3"

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-pixel full-trace dataset (identical to F-E's)
# ---------------------------------------------------------------------------

class _FullTraceDataset(Dataset):
    """Yields (1, T) full traces with (T,) per-timestep binary labels.

    Label = 1 if timestep t >= ttp_idx (post-amp), else 0 (pre-amp).
    """

    def __init__(self, X: np.ndarray, ttp_min: np.ndarray):
        self.X = X.astype(np.float32)
        ttp_idx = (ttp_min * SAMPLES_PER_MIN).astype(np.float32)
        t_range = np.arange(N_SAMPLES, dtype=np.float32)
        self.labels = (t_range[None, :] >= ttp_idx[:, None]).astype(np.float32)
        self.n_pixels = X.shape[0]

    def __len__(self) -> int:
        return self.n_pixels

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])[None, :]  # (1, T)
        y = torch.from_numpy(self.labels[idx])       # (T,)
        return x, y


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
            logits = model(xb)  # (B, T)
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
                logits = model(xb)  # (B, T)
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
# Test-time prediction + TTP extraction (identical to F-E's)
# ---------------------------------------------------------------------------

def _predict_full_traces(
    model: nn.Module,
    X: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Return (n_pixels, T) sigmoid probabilities."""
    model.eval()
    n_pixels = X.shape[0]
    probs = np.empty((n_pixels, N_SAMPLES), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_pixels, batch_size):
            end = min(start + batch_size, n_pixels)
            xb = torch.from_numpy(X[start:end].astype(np.float32)).unsqueeze(1).to(device)
            logits = model(xb)
            probs[start:end] = torch.sigmoid(logits).cpu().numpy()
    return probs


def _extract_ttp(probs: np.ndarray, k_consecutive: int) -> np.ndarray:
    """First t where probs > 0.5 for k_consecutive consecutive timesteps."""
    n_pixels = probs.shape[0]
    above = probs > 0.5
    ttp_pred = np.full(n_pixels, (N_SAMPLES - 1) / SAMPLES_PER_MIN, dtype=np.float32)
    for i in range(n_pixels):
        run = 0
        for t in range(N_SAMPLES):
            if above[i, t]:
                run += 1
                if run >= k_consecutive:
                    ttp_pred[i] = (t - k_consecutive + 1) / SAMPLES_PER_MIN
                    break
            else:
                run = 0
    return ttp_pred


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
        choices=list(BACKBONES.keys()),
        help="Which backbone to sweep in.",
    )
    parser.add_argument("--k_consecutive", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--max_train_pixels", type=int, default=None,
        help="Subsample training set to at most N pixels (smoke-test only).",
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

    if args.max_train_pixels is not None and len(X_tr) > args.max_train_pixels:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X_tr), size=args.max_train_pixels, replace=False)
        idx.sort()
        X_tr = X_tr[idx]
        y_tr = y_tr[idx]
        print(f"  [debug] subsampled train to {len(X_tr)} pixels")

    train_ds = _FullTraceDataset(X_tr, y_tr)
    val_ds = _FullTraceDataset(X_va, y_va)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(
        f"  total train pixels = {len(train_ds)}  "
        f"total val pixels = {len(val_ds)}"
    )

    # ----- build model -----
    model = build_p2_backbone(args.backbone).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built P2 backbone {args.backbone!r}  ({n_params:,} params)")

    # ----- train -----
    history = _train(model, train_loader, val_loader, args.device, args.epochs, args.lr)

    # ----- test-time eval -----
    print("Running test-time full-trace prediction...")
    probs = _predict_full_traces(model, arr.X_te, args.device, args.batch_size)
    print(f"  probs shape: {probs.shape}")

    ttp_pred = _extract_ttp(probs, k_consecutive=args.k_consecutive)
    print(f"  ttp_pred_min range: [{ttp_pred.min():.2f}, {ttp_pred.max():.2f}]")

    # ----- write outputs -----
    out_dir = (
        RESULTS_ROOT
        / f"p2_{args.backbone}_full_spatA3"
        / f"seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "method": f"P2 backbone sweep on F-E framing — backbone={args.backbone}",
        "framing": "F-E (whole-trace, per-timestep BCE, K-consecutive TTP)",
        "seed": args.seed,
        "backbone": args.backbone,
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
        # Also save the raw per-timestep probs so we can experiment with
        # different K / thresholds later without retraining.
        ttp_pred_extras={
            "timestep_probs": probs.astype(np.float32),
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
