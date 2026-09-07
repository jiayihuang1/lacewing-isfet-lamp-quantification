"""RQ3 · SSL finetuning stage. [Cat A] Report §RQ3.

P4 SSL fine-tuning: {masked, contrastive} × {unet, gru, cnn_gru_par}
                × {frozen, fullft} × {fa, fb}.

Loads a pretrained encoder from p4_ssl_{obj}_{bb}/pretrained.pt, attaches
the F-A or F-B downstream head, trains 1 seed per invocation (Task 11's
PBS array sweeps 3 seeds x 2 obj x 3 backbones x 2 protocols x 2 downstream),
and writes per-pixel test predictions in the shared eval schema.

Output dir: p4_{obj}_{bb}_{protocol}_{downstream}/seed{N}/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lacewing.quantification.methods.ssl_pretrain import build_ssl_encoder
from lacewing.quantification.methods._windowing import (
    slide_windows,
    aggregate_to_ttp_first_positive,
)
from lacewing.quantification.methods.pdf_regression import (
    build_gaussian_targets,
    curriculum_sigma_at,
    _log10_concentration_array,
)
from lacewing.quantification.regression_shared.data import dataset as reg_ds


SAMPLES_PER_MIN = 15
CACHE_STEM = "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3"


class FADensityHead(nn.Module):
    """Per-timestep sigmoid density head for F-A fine-tune."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T) -> (B, 1, T) -> sigmoid -> (B, T)
        return torch.sigmoid(self.proj(x).squeeze(1))


class FBWindowHead(nn.Module):
    """Per-window scalar logit head for F-B fine-tune (input from encoder on window)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T_window) -> (B, D) -> (B,)
        pooled = self.pool(x).squeeze(-1)
        return self.proj(pooled).squeeze(-1)


class FineTuneModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)


def build_finetune_model(
    objective: str,
    backbone: str,
    downstream: str,
    freeze: bool,
    pretrained_ckpt: Path | None = None,
) -> FineTuneModel:
    """Build encoder + head; optionally load pretrained weights + freeze.

    `objective` is not used to select the encoder architecture (backbone
    determines that) — it is retained so callers can name/route checkpoints
    per-objective (see `main()`), and so this function's signature matches
    the CLI's four fine-tuning axes.
    """
    encoder = build_ssl_encoder(backbone)
    if pretrained_ckpt is not None:
        state = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
        encoder.load_state_dict(state, strict=False)

    if freeze:
        for p in encoder.parameters():
            p.requires_grad = False

    # Introspect encoder output channels by running a dummy forward pass at
    # the downstream's native trace length (F-A: full 450-sample trace;
    # F-B: 120-sample window). The encoder architectures (1D-CNN / GRU) are
    # length-agnostic so this works whether or not pretrained_ckpt was
    # trained at a different length.
    encoder.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 450 if downstream == "fa" else 120)
        enc_out = encoder(dummy)
        in_ch = enc_out.size(1)

    if downstream == "fa":
        head: nn.Module = FADensityHead(in_ch)
    elif downstream == "fb":
        head = FBWindowHead(in_ch)
    else:
        raise ValueError(f"unknown downstream: {downstream!r}")

    return FineTuneModel(encoder, head)


def _make_optimizer(model: FineTuneModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    """frozen: only head params get gradients (encoder params require_grad=False
    already, so we hand the optimiser only the head). fullft: discriminative
    LR groups, encoder at 1/10 the head's learning rate."""
    if args.protocol == "fullft":
        return torch.optim.Adam([
            {"params": model.encoder.parameters(), "lr": args.lr / 10},
            {"params": model.head.parameters(), "lr": args.lr},
        ])
    return torch.optim.Adam(model.head.parameters(), lr=args.lr)


def _train_fa(model: FineTuneModel, arr, args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader, TensorDataset

    X_tr = torch.from_numpy(arr.X_tr).unsqueeze(1).float()
    y_tr = arr.y_tr

    tr_loader = DataLoader(
        TensorDataset(torch.arange(len(X_tr)), X_tr),
        batch_size=args.batch_size, shuffle=True,
    )

    opt = _make_optimizer(model, args)

    for epoch in range(1, args.epochs + 1):
        sigma = curriculum_sigma_at(epoch, args.epochs, sigma_start=12.0, sigma_end=3.0)
        model.train()
        tr_loss = 0.0
        n = 0
        for idx, x in tr_loader:
            x = x.to(args.device)
            targets_np = build_gaussian_targets(y_tr[idx.numpy()], sigma_samples=sigma, T=x.size(-1))
            targets = torch.from_numpy(targets_np).to(args.device)
            pred = model(x)  # (B, T)
            weights = targets + 0.1
            loss = (weights * (pred - targets) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += float(loss) * x.size(0)
            n += x.size(0)
        print(f"epoch {epoch:02d}/{args.epochs}  fa_tr_loss={tr_loss / n:.4f} sigma={sigma:.2f}")


def _train_fb(model: FineTuneModel, arr, args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader, TensorDataset

    windows_tr = slide_windows(arr.X_tr, args.window, args.stride)  # (N, W_count, W_samples)
    ttp_idx_tr = arr.y_tr * SAMPLES_PER_MIN
    n_wins = windows_tr.shape[1]
    centres = np.arange(n_wins) * args.stride + (args.window - 1) / 2.0
    labels_tr = (centres[None, :] >= ttp_idx_tr[:, None]).astype(np.float32)

    x_flat = torch.from_numpy(windows_tr.reshape(-1, args.window)).unsqueeze(1).float()
    y_flat = torch.from_numpy(labels_tr.reshape(-1)).float()

    tr_loader = DataLoader(TensorDataset(x_flat, y_flat), batch_size=args.batch_size, shuffle=True)

    opt = _make_optimizer(model, args)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        n = 0
        for x, y in tr_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            logit = model(x)
            loss = bce(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += float(loss) * x.size(0)
            n += x.size(0)
        print(f"epoch {epoch:02d}/{args.epochs}  fb_tr_loss={tr_loss / n:.4f}")


def _extract_ttp_fa(model: FineTuneModel, X_te: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    model.eval()
    N = len(X_te)
    ttps = np.zeros(N, dtype=np.float32)
    with torch.no_grad():
        for i in range(N):
            x = torch.from_numpy(X_te[i]).unsqueeze(0).unsqueeze(0).float().to(args.device)
            density = model(x).squeeze(0).cpu().numpy()
            ttps[i] = float(density.argmax()) / SAMPLES_PER_MIN
    return ttps


def _extract_ttp_fb(model: FineTuneModel, X_te: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    windows_te = slide_windows(X_te, args.window, args.stride)  # (N, W_count, W_samples)
    N, W_count, _ = windows_te.shape
    probs = np.zeros((N, W_count), dtype=np.float32)
    with torch.no_grad():
        for i in range(N):
            x = torch.from_numpy(windows_te[i]).unsqueeze(1).float().to(args.device)
            logit = model(x)
            probs[i] = torch.sigmoid(logit).cpu().numpy()
    return aggregate_to_ttp_first_positive(
        probs, threshold=args.k_thr, k_consecutive=args.k_consecutive,
        window=args.window, stride=args.stride, samples_per_min=SAMPLES_PER_MIN,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--objective", choices=["masked", "contrastive"], required=True)
    p.add_argument("--backbone", choices=["unet", "gru", "cnn_gru_par"], required=True)
    p.add_argument("--downstream", choices=["fa", "fb"], required=True)
    p.add_argument("--protocol", choices=["frozen", "fullft"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window", type=int, default=120)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--k_consecutive", type=int, default=3)
    p.add_argument("--k_thr", type=float, default=0.7)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--pretrained-dir", type=str, default=None,
        help="Override the auto-derived encoder-dir path. Point at a "
             "p4_ssl_{obj}_{bb}[_epochs{N}][_tau{v}] directory to fine-tune "
             "off a non-default pretrain (e.g. W16 τ sweep, longer epochs).",
    )
    args = p.parse_args()

    from lacewing.classification.core.seeding import seed_everything
    seed_everything(args.seed)

    arr = reg_ds.make_split(CACHE_STEM, seed=args.seed, normalise_y=False)

    if args.pretrained_dir is not None:
        pretrained_dir = Path(args.pretrained_dir)
    else:
        pretrained_dir = Path(
            f"Analysis/quantification/methods/results/p4_ssl_{args.objective}_{args.backbone}"
        )
    pretrained_ckpt = pretrained_dir / "pretrained.pt"
    if not pretrained_ckpt.exists():
        raise FileNotFoundError(f"No pretrained encoder at {pretrained_ckpt}")

    # Derive an output-dir suffix from the pretrained_dir's name so W16 sweeps
    # don't collide with the legacy p4_{obj}_{bb}_{proto}_{ds} naming used by
    # default (epochs=50, tau=0.1) runs. The suffix is anything appended to
    # the base "p4_ssl_{obj}_{bb}" — e.g. "_epochs150", "_tau0.3",
    # "_epochs150_tau0.3".
    base_prefix = f"p4_ssl_{args.objective}_{args.backbone}"
    if pretrained_dir.name.startswith(base_prefix):
        out_suffix = pretrained_dir.name[len(base_prefix):]
    else:
        out_suffix = ""

    freeze = args.protocol == "frozen"
    model = build_finetune_model(
        args.objective, args.backbone, args.downstream, freeze, pretrained_ckpt
    ).to(args.device)

    if args.downstream == "fa":
        _train_fa(model, arr, args)
        ttp_pred = _extract_ttp_fa(model, arr.X_te, args)
    else:
        _train_fb(model, arr, args)
        ttp_pred = _extract_ttp_fb(model, arr.X_te, args)

    out_dir = Path(
        f"Analysis/quantification/methods/results/"
        f"p4_{args.objective}_{args.backbone}_{args.protocol}_{args.downstream}{out_suffix}/seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_dir / "predictions.npz", ttp_pred_min=ttp_pred.astype(np.float32))
    from lacewing.quantification.eval.schema import write_labels
    log10_conc = _log10_concentration_array(arr.chip_te, arr.well_te)
    write_labels(
        out_dir / "labels.npz",
        ttp_true_min=arr.y_te.astype(np.float32),
        chip_id=arr.chip_te, well_id=arr.well_te,
        log10_concentration=log10_conc,
        split=np.array(["test"] * len(arr.y_te), dtype="U16"),
    )
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(f"[ok] wrote {out_dir}/")


if __name__ == "__main__":
    main()
