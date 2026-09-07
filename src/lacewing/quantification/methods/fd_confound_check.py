"""F-D confound-check: F-D framing on a chosen (P2-winning) backbone.

F-D framing: whole 450-sample trace → per-timestep 4-class softmax with
classes {pre-amp, onset-window, amp, plateau}.  TTP = first timestep where
argmax ∈ {onset-window, amp}.

P1 F-D used 1D U-Net specifically because that's what's natural for
segmentation.  To disambiguate 'F-D lost because U-Net wasn't the right
backbone' vs 'F-D lost because segmentation is the wrong framing', we
adapt the P2-winning backbone (or any of the P2 slate) to output
(B, 4, T) instead of (B, T) or (B, n_classes).

Backbone adapters:
  * unet         — native, just use n_classes=4
  * tcn          — TCNPerTimestep(n_classes=4)
  * bigru        — BiGRU whose per-timestep head projects to 4 classes
  * gru          — same as bigru but bidirectional=False
  * cnn_gru_par  — F-E's per-timestep CNN-GRU-par (from p2_backbones.py) with
                   4-class final linear
  * transformer_patch — F-E's per-timestep transformer (from p2_backbones.py)
                   with 4-class final linear
  * ann          — F-E's per-timestep ANN (each timestep MLP maps 1 -> 4)

Output dir (canonical schema):
    Analysis/quantification/methods/results/
        fd_confound_{backbone}_spatA3/seed{seed}/
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
from lacewing.quantification.methods.pdf_regression import _log10_concentration_array
from lacewing.quantification.regression_shared.data import dataset as reg_ds


SAMPLES_PER_MIN = 15
N_SAMPLES = 450
N_CLASSES = 4  # pre-amp / onset-window / amp / plateau
CLASS_PRE_AMP = 0
CLASS_ONSET = 1
CLASS_AMP = 2
CLASS_PLATEAU = 3
CACHE_STEM = "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3"

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results"


# ---------------------------------------------------------------------------
# Per-pixel full-trace dataset with per-timestep 4-class labels
# ---------------------------------------------------------------------------

def _build_per_timestep_labels(ttp_min: np.ndarray, onset_window_samples: int) -> np.ndarray:
    """Return (n_pixels, T) integer class labels in {0..3}.

    Per-pixel per-timestep from qLAMP TTP:
      * pre-amp: t < ttp_idx - onset_window_samples
      * onset-window: ttp_idx - onset_window_samples <= t < ttp_idx
      * amp: ttp_idx <= t < ttp_idx + 90 (6 min post-TTP)
      * plateau: t >= ttp_idx + 90
    """
    ttp_idx = (ttp_min * SAMPLES_PER_MIN).astype(np.float32)  # (n_pixels,)
    t_range = np.arange(N_SAMPLES, dtype=np.float32)          # (T,)
    labels = np.full((len(ttp_min), N_SAMPLES), CLASS_PRE_AMP, dtype=np.int64)

    for i, ti in enumerate(ttp_idx):
        onset_start = ti - onset_window_samples
        amp_end = ti + 90
        labels[i] = np.where(t_range < onset_start, CLASS_PRE_AMP, labels[i])
        labels[i] = np.where(
            (t_range >= onset_start) & (t_range < ti),
            CLASS_ONSET, labels[i],
        )
        labels[i] = np.where(
            (t_range >= ti) & (t_range < amp_end),
            CLASS_AMP, labels[i],
        )
        labels[i] = np.where(t_range >= amp_end, CLASS_PLATEAU, labels[i])
    return labels


class _FullTraceSegDataset(Dataset):
    """Yields (1, T) full traces with (T,) integer class labels."""
    def __init__(self, X: np.ndarray, ttp_min: np.ndarray, onset_window_samples: int):
        self.X = X.astype(np.float32)
        self.labels = _build_per_timestep_labels(ttp_min, onset_window_samples)
        self.n_pixels = X.shape[0]

    def __len__(self) -> int:
        return self.n_pixels

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])[None, :]        # (1, T)
        y = torch.from_numpy(self.labels[idx]).long()      # (T,)
        return x, y


# ---------------------------------------------------------------------------
# Backbone factory — each backbone returns (B, N_CLASSES, T) logits from (B, 1, T)
# ---------------------------------------------------------------------------

def _build_seg_backbone(backbone_id: str) -> nn.Module:
    """Return a model with forward(x: (B, 1, T)) -> (B, N_CLASSES, T) logits."""
    if backbone_id == "unet":
        from lacewing.classification.models.unet_1d import UNet1D
        return UNet1D(in_channels=1, n_classes=N_CLASSES, base_channels=8)
    if backbone_id == "tcn":
        from lacewing.classification.models.tcn import TCNPerTimestep
        return TCNPerTimestep(input_size=1, n_classes=N_CLASSES)
    if backbone_id == "bigru":
        return _RNNSeg(bidirectional=True)
    if backbone_id == "gru":
        return _RNNSeg(bidirectional=False)
    if backbone_id == "cnn_gru_par":
        return _CNNGRUParSeg()
    if backbone_id == "transformer_patch":
        return _TransformerPatchSeg()
    if backbone_id == "ann":
        return _ANNPerTimestepSeg()
    raise ValueError(
        f"Unknown backbone {backbone_id!r}. "
        f"Valid: ann, gru, bigru, cnn_gru_par, transformer_patch, unet, tcn"
    )


class _RNNSeg(nn.Module):
    """GRU or BiGRU with per-timestep 4-class head."""
    def __init__(self, hidden: int = 64, layers: int = 2, dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=1, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x.transpose(1, 2))  # (B, T, out_dim)
        logits = self.head(h)                # (B, T, N_CLASSES)
        return logits.transpose(1, 2)        # (B, N_CLASSES, T)


class _CNNGRUParSeg(nn.Module):
    """CNN broadcast + BiGRU per-timestep with 4-class head."""
    def __init__(self, input_len: int = N_SAMPLES, d_branch: int = 64,
                 hidden: int = 64, layers: int = 2, dropout: float = 0.1, d_fusion: int = 64):
        super().__init__()
        from lacewing.classification.models import cnn1d
        self.cnn_features = cnn1d.CNN1D(input_len).features
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            n_flat = self.cnn_features(dummy).shape[1]
        self.cnn_proj = nn.Linear(n_flat, d_branch)

        self.gru = nn.GRU(
            input_size=1, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.gru_proj = nn.Linear(2 * hidden, d_branch)

        self.head = nn.Sequential(
            nn.Linear(2 * d_branch, d_fusion), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fusion, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        cnn_feat = self.cnn_proj(self.cnn_features(x))       # (B, d_branch)
        cnn_bcast = cnn_feat.unsqueeze(1).expand(b, t, cnn_feat.size(-1))

        h, _ = self.gru(x.transpose(1, 2))                    # (B, T, 2*hidden)
        gru_feat = self.gru_proj(h)                           # (B, T, d_branch)

        fused = torch.cat([cnn_bcast, gru_feat], dim=-1)      # (B, T, 2*d_branch)
        logits = self.head(fused)                              # (B, T, N_CLASSES)
        return logits.transpose(1, 2)                          # (B, N_CLASSES, T)


class _TransformerPatchSeg(nn.Module):
    """PatchTST-style attention → per-timestep 4-class."""
    def __init__(self, input_len: int = N_SAMPLES, patch_size: int = 15,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 3,
                 dim_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        if input_len % patch_size != 0:
            raise ValueError(f"input_len={input_len} not divisible by patch_size={patch_size}")
        self.patch_size = patch_size
        n_patches = input_len // patch_size

        from lacewing.classification.models._positional import SinusoidalPositionalEncoding
        self.embed = nn.Linear(patch_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=n_patches)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        h = x.view(b, c, t // self.patch_size, self.patch_size).squeeze(1)
        h = self.embed(h)
        h = self.pos(h)
        h = self.encoder(h)                                  # (B, n_patches, d_model)
        h = h.repeat_interleave(self.patch_size, dim=1)      # (B, T, d_model)
        logits = self.head(h)                                # (B, T, N_CLASSES)
        return logits.transpose(1, 2)                        # (B, N_CLASSES, T)


class _ANNPerTimestepSeg(nn.Module):
    """Tiny MLP applied to each timestep independently → 4-class."""
    def __init__(self, hidden_dims: tuple[int, ...] = (16, 8)):
        super().__init__()
        dims = (1,) + tuple(hidden_dims)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], N_CLASSES))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        assert c == 1, x.shape
        x_flat = x.transpose(1, 2).reshape(b * t, 1)
        y_flat = self.mlp(x_flat)             # (B*T, N_CLASSES)
        return y_flat.view(b, t, N_CLASSES).transpose(1, 2)  # (B, N_CLASSES, T)


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
    loss_fn = nn.CrossEntropyLoss()  # per-timestep multi-class
    history = []
    for e in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)  # (B, N_CLASSES, T)
            loss = loss_fn(logits, yb)  # CE over class dim
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
                logits = model(xb)  # (B, N_CLASSES, T)
                val_loss_acc += loss_fn(logits, yb).item() * yb.size(0)
                preds = logits.argmax(dim=1)  # (B, T)
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
# Test-time prediction + TTP extraction (F-D rule)
# ---------------------------------------------------------------------------

def _predict_full_traces(
    model: nn.Module,
    X: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Return (n_pixels, T) integer class predictions."""
    model.eval()
    n_pixels = X.shape[0]
    preds = np.empty((n_pixels, N_SAMPLES), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, n_pixels, batch_size):
            end = min(start + batch_size, n_pixels)
            xb = torch.from_numpy(X[start:end].astype(np.float32)).unsqueeze(1).to(device)
            logits = model(xb)  # (B, N_CLASSES, T)
            preds[start:end] = logits.argmax(dim=1).cpu().numpy()
    return preds


def _extract_ttp(class_preds: np.ndarray) -> np.ndarray:
    """First t where class in {onset-window, amp}.  Fallback: t = N_SAMPLES - 1."""
    n_pixels = class_preds.shape[0]
    ttp_pred = np.full(n_pixels, (N_SAMPLES - 1) / SAMPLES_PER_MIN, dtype=np.float32)
    for i in range(n_pixels):
        row = class_preds[i]
        onset_or_amp = np.where(np.isin(row, [CLASS_ONSET, CLASS_AMP]))[0]
        if len(onset_or_amp) > 0:
            ttp_pred[i] = onset_or_amp[0] / SAMPLES_PER_MIN
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
        choices=["ann", "gru", "bigru", "cnn_gru_par", "transformer_patch", "unet", "tcn"],
    )
    parser.add_argument("--onset_window_samples", type=int, default=30,
                        help="Onset-window class width in samples (2 min = 30 samples).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_train_pixels", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----- load + split -----
    print(f"Loading cache: {CACHE_STEM}")
    arr = reg_ds.make_split(CACHE_STEM, seed=args.seed, val_n_chips=1, normalise_y=False)
    print(f"  n_tr={len(arr.X_tr)}  n_va={len(arr.X_va)}  n_te={len(arr.X_te)}")

    X_tr, y_tr = arr.X_tr, arr.y_tr
    X_va, y_va = arr.X_va, arr.y_va

    if args.max_train_pixels is not None and len(X_tr) > args.max_train_pixels:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X_tr), size=args.max_train_pixels, replace=False)
        idx.sort()
        X_tr = X_tr[idx]
        y_tr = y_tr[idx]
        print(f"  [debug] subsampled train to {len(X_tr)} pixels")

    train_ds = _FullTraceSegDataset(X_tr, y_tr, args.onset_window_samples)
    val_ds = _FullTraceSegDataset(X_va, y_va, args.onset_window_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"  total train pixels = {len(train_ds)}  total val pixels = {len(val_ds)}")

    # ----- build model -----
    model = _build_seg_backbone(args.backbone).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built F-D confound-check backbone {args.backbone!r}  ({n_params:,} params)")

    # ----- train -----
    history = _train(model, train_loader, val_loader, args.device, args.epochs, args.lr)

    # ----- test-time eval -----
    print("Running test-time full-trace prediction...")
    class_preds = _predict_full_traces(model, arr.X_te, args.device, args.batch_size)
    print(f"  class_preds shape: {class_preds.shape}")

    ttp_pred = _extract_ttp(class_preds)
    print(f"  ttp_pred_min range: [{ttp_pred.min():.2f}, {ttp_pred.max():.2f}]")

    # ----- write outputs -----
    out_dir = (
        RESULTS_ROOT
        / f"fd_confound_{args.backbone}_spatA3"
        / f"seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "method": f"F-D confound-check — backbone={args.backbone}",
        "framing": "F-D (whole-trace 4-class segmentation, per-timestep CE, first onset-or-amp TTP)",
        "seed": args.seed,
        "backbone": args.backbone,
        "onset_window_samples": args.onset_window_samples,
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
            "timestep_class_preds": class_preds.astype(np.int8),
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
