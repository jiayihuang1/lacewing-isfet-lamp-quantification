"""RQ2 · F-B sliding-window classifier framing (winner). [Cat A] Report §RQ2 Tab. 6.2, carried into RQ3.

F-B — Sliding-window classifier per-window (Paper 18 adaptation).

Adaptation of Li et al. 2024 §6 / Algorithm 1: slide a length-W window
across each pixel's trace, classify each window as pre-amp / post-amp,
aggregate to TTP via the first-K-consecutive-positive rule.

Backbone: CNN-GRU-par (parallel arrangement — best-hybrid pre-screen winner).
Framing knobs: W ∈ {30, 60, 90} samples, stride = W/12, K = 3.

Provenance
----------
Adaptation of Paper 18 (Li et al. 2024) §6 / Algorithm 1; adapted to LAMP
amplification onset detection with CNN-GRU-par backbone.

Usage
-----
    python -m lacewing.quantification.methods.sliding_window_cls \\
        --seed 0 --window 60 --stride 5 --epochs 30

Output dir (canonical schema):
    Analysis/quantification/methods/results/
        p1_fb_slide_cls_w{window}_stride{stride}_spatA3/seed{seed}/
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
    aggregate_to_ttp_first_positive,
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

class _WindowDataset(Dataset):
    """Yields (1, W) windows with per-window labels (hard or soft).

    Hard label (default, soft_label_sigma_samples=0):
        Label = 1 if the window centre sample index >= ttp_idx, else 0.

    Soft label (soft_label_sigma_samples > 0):
        Label = sigmoid((centre_idx - ttp_idx) / soft_label_sigma_samples).
        Smooth transition around TTP, hyperparameter controls transition width.
    """

    def __init__(self, X: np.ndarray, ttp_min: np.ndarray, window: int, stride: int,
                 soft_label_sigma_samples: float = 0.0):
        # X: (n_pixels, T);  ttp_min: (n_pixels,) in minutes
        self.windows = slide_windows(X, window, stride)  # (n_pixels, N_windows, W)
        ttp_idx = ttp_min * SAMPLES_PER_MIN              # convert to sample index
        n_windows = self.windows.shape[1]
        # Centre sample of each window (real-valued for broadcasting).
        centres = np.arange(n_windows) * stride + (window - 1) / 2.0  # (N_windows,)
        # (n_pixels, N_windows) float32 labels.
        if soft_label_sigma_samples > 0:
            dist = centres[None, :] - ttp_idx[:, None]   # (n_pixels, N_windows)
            self.labels = (1.0 / (1.0 + np.exp(-dist / soft_label_sigma_samples))).astype(np.float32)
        else:
            self.labels = (centres[None, :] >= ttp_idx[:, None]).astype(np.float32)
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

def build_model(window: int, backbone: str = "cnn_gru_par", dropout: float = 0.1) -> nn.Module:
    """Build the F-B per-window classifier.

    Dispatches to the shared F-B backbone factory. Backbones expose a
    ``(B, 1, window) -> (B,)`` scalar-logit signature.

    Note: dropout only takes effect for CNN-GRU-par (the current F-B
    default). For other backbones dropout is ignored; a warning is
    printed at construction time.
    """
    if backbone == "cnn_gru_par":
        # Wire dropout directly (HP tuning axis for B4).
        return CNNGRUParallel(input_len=window, dropout=dropout)
    from lacewing.quantification.methods._fb_backbones import build_fb_backbone
    if dropout != 0.1:
        print(f"  [warn] --dropout={dropout} ignored for backbone={backbone}")
    return build_fb_backbone(backbone, input_len=window)


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
    weight_decay: float = 0.0,
    ckpt_dir: Path | None = None,
) -> list[dict]:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    for e in range(epochs):
        # Train step
        model.train()
        total_loss = 0.0
        total_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)          # (B,)  — already squeezed by CNNGRUParallel
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * yb.size(0)
            total_samples += yb.size(0)
        avg_train_loss = total_loss / max(total_samples, 1)

        # Validation step
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_acc = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)      # (B,)
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
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": e + 1, "model_state": model.state_dict(),
                        "val_loss": avg_val_loss, "val_acc": val_acc},
                       ckpt_dir / "last.pt")
            if not hasattr(_train, "_best_seen") or avg_val_loss < _train._best_seen:
                _train._best_seen = avg_val_loss
                torch.save({"epoch": e + 1, "model_state": model.state_dict(),
                            "val_loss": avg_val_loss, "val_acc": val_acc},
                           ckpt_dir / "best.pt")
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
    """Return (n_pixels, N_windows) sigmoid probabilities."""
    model.eval()
    ws = slide_windows(X, window, stride)   # (n_pixels, N_windows, W)
    n_p, n_w, _ = ws.shape
    probs = np.empty((n_p, n_w), dtype=np.float32)
    with torch.no_grad():
        for i in range(n_p):
            # (N_windows, 1, W) — batch over windows for one pixel at a time.
            xb = torch.from_numpy(ws[i]).unsqueeze(1).to(device)  # (N_windows, 1, W)
            logits = model(xb)                                      # (N_windows,)
            probs[i] = torch.sigmoid(logits).cpu().numpy()
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
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--k_consecutive", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--max_train_pixels", type=int, default=None,
        help="Subsample training set to at most N pixels (for smoke-testing on CPU).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone", type=str, default="cnn_gru_par",
                        choices=["ann", "bigru", "cnn_gru_par", "gru", "tcn", "transformer_patch", "unet"],
                        help="F-B backbone. Default cnn_gru_par matches the current baseline.")
    parser.add_argument("--soft-label-sigma", type=float, default=0.0,
                        help="Soft-label sigma in samples. 0 = hard binary labels (default). "
                             ">0 = sigmoid smoothing around TTP.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="CNN-GRU-par dropout probability. Ignored for other backbones.")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="AdamW-style L2 weight decay.")
    parser.add_argument("--k_thr", type=float, default=0.5,
                        help="Threshold for first_above_K extraction rule. Default 0.5 = F-B legacy.")
    parser.add_argument("--cache-stem", type=str, default=CACHE_STEM,
                        help="Regression cache stem (default = SARS-CoV-2). "
                             "For new-data LOCO runs pass e.g. regress_conc_loco1.")
    parser.add_argument("--pretrained-encoder", type=str, default=None,
                        help="Optional path to an SSL-pretrained U-Net encoder "
                             "state_dict (e.g. p4_ssl_contrastive_unet_conc/"
                             "pretrained.pt). Only used when --backbone=unet.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----- load + split -----
    cache_stem = args.cache_stem
    print(f"Loading cache: {cache_stem}")
    arr = reg_ds.make_split(
        cache_stem,
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

    train_ds = _WindowDataset(X_tr, y_tr, args.window, args.stride,
                              soft_label_sigma_samples=args.soft_label_sigma)
    val_ds = _WindowDataset(X_va, y_va, args.window, args.stride,
                            soft_label_sigma_samples=args.soft_label_sigma)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(
        f"  windows per pixel = {train_ds.n_windows}  "
        f"total train windows = {len(train_ds)}  "
        f"total val windows = {len(val_ds)}"
    )

    # ----- build model -----
    model = build_model(window=args.window, backbone=args.backbone, dropout=args.dropout).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built backbone={args.backbone}(input_len={args.window})  ({n_params:,} params)")

    # ----- optionally load SSL-pretrained encoder weights -----
    pretrained_tag = ""
    if args.pretrained_encoder is not None:
        if args.backbone != "unet":
            raise SystemExit(f"--pretrained-encoder only supported for --backbone unet; "
                             f"got {args.backbone!r}")
        ckpt_path = Path(args.pretrained_encoder)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No pretrained encoder at {ckpt_path}")
        ssl_state = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        # SSL encoder is SSLEncoderUNet wrapping a UNet1D at self.unet. F-B's
        # _UNetWindowClassifier also has model.unet (same UNet1D shape). Strip
        # the "unet." prefix so keys align.
        stripped = {}
        for k, v in ssl_state.items():
            if k.startswith("unet."):
                stripped[k[len("unet."):]] = v
        # Drop any shape-mismatched keys (e.g. SSL n_classes=8 vs F-B n_classes=1)
        target_shapes = {k: v.shape for k, v in model.unet.state_dict().items()}
        for k in list(stripped.keys()):
            if k in target_shapes and stripped[k].shape != target_shapes[k]:
                del stripped[k]
        missing, unexpected = model.unet.load_state_dict(stripped, strict=False)
        print(f"  loaded pretrained encoder from {ckpt_path.name}: "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        # Tag out_dir with SSL identity: e.g. p4_ssl_contrastive_unet_conc -> _ptcontrastive_conc
        import re as _re
        enc_dir = ckpt_path.parent.name
        m = _re.match(r"^p4_ssl_(?P<obj>[a-z]+)_(?P<bb>[a-z_]+?)(?:_(?P<suffix>.+))?$",
                       enc_dir)
        if m and m.group("suffix"):
            pretrained_tag = f"_pt{m.group('obj')}_{m.group('suffix')}"
        elif m:
            pretrained_tag = f"_pt{m.group('obj')}"
        else:
            pretrained_tag = f"_pt{enc_dir.replace('p4_ssl_', '')}"

    # ----- construct out_dir BEFORE training so checkpoints can be saved -----
    # Build a compact tag reflecting the axes swept in W13.
    extra_tag = ""
    if args.backbone != "cnn_gru_par":
        extra_tag += f"_bb_{args.backbone}"
    if args.soft_label_sigma > 0:
        extra_tag += f"_softsig{args.soft_label_sigma:g}"
    if args.dropout != 0.1 or args.weight_decay != 0.0 or args.lr != 1e-3:
        extra_tag += f"_lr{args.lr:g}_do{args.dropout:g}_wd{args.weight_decay:g}"
    if args.k_thr != 0.5:
        extra_tag += f"_kthr{args.k_thr:g}"

    # Suffix the output dir with the cache stem when it's not the default
    # SARS-CoV-2 cache, so new-data LOCO runs don't collide with shipped runs.
    cache_suffix = ""
    if cache_stem != CACHE_STEM:
        # e.g. "regress_conc_loco1" -> "_conc_loco1"
        cache_suffix = "_" + cache_stem.replace("regress_", "")
    out_dir = (
        RESULTS_ROOT
        / f"p1_fb_slide_cls_w{args.window}_stride{args.stride}_spatA3{extra_tag}{pretrained_tag}{cache_suffix}"
        / f"seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- train -----
    # Reset _best_seen state to avoid cross-run contamination.
    if hasattr(_train, "_best_seen"):
        delattr(_train, "_best_seen")
    history = _train(model, train_loader, val_loader, args.device, args.epochs, args.lr,
                     weight_decay=args.weight_decay,
                     ckpt_dir=out_dir / "checkpoints")

    # ----- test-time eval -----
    print("Running test-time window prediction...")
    probs = _predict_windows(model, arr.X_te, args.window, args.stride, args.device)
    print(f"  window probs shape: {probs.shape}")

    ttp_pred = aggregate_to_ttp_first_positive(
        probs,
        threshold=args.k_thr,
        k_consecutive=args.k_consecutive,
        window=args.window,
        stride=args.stride,
        samples_per_min=SAMPLES_PER_MIN,
    )
    print(f"  ttp_pred_min range: [{ttp_pred.min():.2f}, {ttp_pred.max():.2f}]")

    # ----- write outputs -----
    cfg = {
        "method": "P1.2 F-B sliding-window classifier (Paper 18 / Li et al. 2024 adaptation)",
        "seed": args.seed,
        "window": args.window,
        "stride": args.stride,
        "k_consecutive": args.k_consecutive,
        "k_thr": args.k_thr,
        "backbone": args.backbone,
        "soft_label_sigma": args.soft_label_sigma,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
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
        "cache_stem": cache_stem,
        "pretrained_encoder": str(args.pretrained_encoder) if args.pretrained_encoder else None,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Canonical schema.  Save per-window probabilities in `extras` so the
    # F-B extraction-rule ablation can apply many post-processing rules to
    # the same trained model without re-training.
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
