"""RQ3 · SSL pretraining stage (contrastive NT-Xent + masked reconstruction). [Cat A] Report §RQ3.

P4 SSL pretraining: masked-reconstruction + contrastive.

Backbones: unet, gru, cnn_gru_par (structurally similar to, but self-contained
from, the existing P2/F-D backbone registries — see
lacewing.quantification.methods.unet_segmentation_backbones).
Objectives:
  - masked-recon: 15% chunk masks of 30 samples, MSE on masked positions.
  - contrastive (TS2Vec-style): augmented pairs, NT-Xent loss.

Emits pretrained encoder checkpoint to
    Analysis/quantification/methods/results/p4_ssl_{obj}_{backbone}/pretrained.pt

Unlabelled corpus: loaded from the frozen preprocessing cache
    Analysis/regression/data/cache/regress_all_filt_abcd_ntcRaw_madk1p5_spatA3.npz
via lacewing.quantification.regression_shared.data.dataset.make_split, concatenating X_tr/X_va/X_te.
This is the full ~28k-pixel corpus regardless of split/label — appropriate
for corpus-level SSL pretraining (see _load_unlabelled_corpus docstring for
why this differs from the brief's original `load_cache` suggestion).

With `--corpus cov_uti` (W18), the UTI classifier cache
(Analysis/quantification/chip_pipeline/output/uti_classifier_cache.npz, all
pixels, ~39k) is concatenated after the COV pixels, growing the corpus to
COV+UTI+KP_03 (~67k pixels; COV reg cache 27.9k + UTI cls cache 39.3k). Default `--corpus cov` is unchanged/legacy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lacewing.quantification.methods.unet_segmentation import UNet1D
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT

UTI_CLS_CACHE = LACEWING_PKG_DIR / "quantification" / "chip_pipeline" / "output" / "uti_classifier_cache.npz"


class SSLEncoderUNet(nn.Module):
    """U-Net encoder that returns per-timestep features (B, D, T)."""
    def __init__(self, base_channels: int = 8):
        super().__init__()
        # Reuse existing UNet1D but set n_classes=base_channels so the final
        # 1x1 conv emits per-timestep features instead of class logits.
        self.unet = UNet1D(in_channels=1, n_classes=base_channels, base_channels=base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)  # (B, base_channels, T)


class SSLEncoderGRU(nn.Module):
    """Unidirectional GRU encoder → per-timestep hidden state (B, hidden, T)."""
    def __init__(self, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(1, hidden, num_layers=layers, batch_first=True,
                          bidirectional=False, dropout=dropout if layers > 1 else 0)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x.transpose(1, 2))    # (B, T, hidden)
        return h.transpose(1, 2)               # (B, hidden, T)


class SSLEncoderCNNGRUPar(nn.Module):
    """Parallel CNN + GRU encoder → concatenated per-timestep features.

    (B, 1, T) -> (B, cnn_channels[-1] + gru_hidden, T)
    """
    def __init__(self, cnn_channels: tuple[int, ...] = (16, 32, 64),
                 gru_hidden: int = 64, gru_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = 1
        for ch in cnn_channels:
            layers += [nn.Conv1d(prev, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU()]
            prev = ch
        self.cnn = nn.Sequential(*layers)
        self.gru = nn.GRU(1, gru_hidden, num_layers=gru_layers, batch_first=True,
                          bidirectional=False, dropout=dropout if gru_layers > 1 else 0)
        self.out_channels = cnn_channels[-1] + gru_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T = x.shape
        cnn_feat = self.cnn(x)                         # (B, cnn_ch, T) — per-timestep
        gru_out, _ = self.gru(x.transpose(1, 2))       # (B, T, gru_hidden)
        gru_feat = gru_out.transpose(1, 2)             # (B, gru_hidden, T)
        return torch.cat([cnn_feat, gru_feat], dim=1)  # (B, cnn_ch + gru_hidden, T)


def build_ssl_encoder(backbone_id: str) -> nn.Module:
    """Dispatcher: backbone_id in {'unet', 'gru', 'cnn_gru_par'} -> encoder module."""
    if backbone_id == "unet":
        return SSLEncoderUNet(base_channels=8)
    if backbone_id == "gru":
        return SSLEncoderGRU()
    if backbone_id == "cnn_gru_par":
        return SSLEncoderCNNGRUPar()
    raise ValueError(f"unknown backbone: {backbone_id!r}")


class MaskedReconstructionModel(nn.Module):
    """Encoder + small upsampling decoder -> reconstruct masked positions."""
    def __init__(self, encoder: nn.Module, decoder_channels: tuple[int, ...] = (16, 8)):
        super().__init__()
        self.encoder = encoder
        # Infer encoder output channels via a dummy forward.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 450)
            enc_out = encoder(dummy)
            in_ch = enc_out.size(1)
        layers = []
        prev = in_ch
        for ch in decoder_channels:
            layers += [nn.Conv1d(prev, ch, 3, padding=1), nn.ReLU()]
            prev = ch
        layers.append(nn.Conv1d(prev, 1, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)   # (B, 1, T)


class ContrastiveModel(nn.Module):
    """Encoder -> global-pool -> projection MLP -> L2-normalised embedding."""
    def __init__(self, encoder: nn.Module, projection_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 450)
            enc_out = encoder(dummy)
            in_ch = enc_out.size(1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Sequential(
            nn.Linear(in_ch, in_ch), nn.ReLU(),
            nn.Linear(in_ch, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z_pooled = self.pool(z).squeeze(-1)  # (B, in_ch)
        z_proj = self.project(z_pooled)      # (B, projection_dim)
        return F.normalize(z_proj, dim=1)


def mask_chunks(x: torch.Tensor, mask_ratio: float = 0.15, chunk_len: int = 30
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Random chunk-masking: replace mask_ratio of chunks with local-std noise.

    Contiguous chunks of `chunk_len` samples are masked (rather than
    single-timestep masks) so the model must learn amplification shape
    instead of trivially interpolating a single missing point.

    Masked positions are filled with Gaussian noise scaled to the local
    (window-padded) std of the trace, NOT zeros. Zero-filling is a trivial
    shortcut for near-baseline ISFET traces (baseline clusters near 0), which
    let the model achieve ~0 MSE by learning pred=0 everywhere instead of
    actually reconstructing amplification shape (observed on cx3: tr_loss=
    va_loss=0.0000 from epoch 1). Noise-filling removes that degenerate
    solution while still being fully corruptible/informative-void.

    Returns (masked_x, mask) where mask has shape (B, 1, T) and is 1 where masked.
    """
    B, C, T = x.shape
    n_chunks = T // chunk_len
    n_masked_chunks = max(1, int(round(n_chunks * mask_ratio)))
    mask = torch.zeros(B, 1, T, device=x.device)
    masked_x = x.clone()
    for b in range(B):
        chunk_ids = torch.randperm(n_chunks)[:n_masked_chunks]
        for cid in chunk_ids:
            start = int(cid) * chunk_len
            end = start + chunk_len
            mask[b, 0, start:end] = 1.0
            local_std = x[b, :, max(0, start - 30):min(T, end + 30)].std() + 1e-6
            noise = torch.randn(end - start, device=x.device) * local_std
            masked_x[b, :, start:end] = noise
    return masked_x, mask


def augment_pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two independently-augmented views of the same batch, for contrastive SSL."""
    return _augment_one(x), _augment_one(x)


def _augment_one(x: torch.Tensor) -> torch.Tensor:
    """Apply time-shift + additive drift + Gaussian noise + amplitude scale.

    Per the brief: (i) time-shift up to ±20 samples, (ii) additive low-freq
    sinusoidal drift (freq 0.005-0.05 Hz), (iii) additive Gaussian noise
    (sigma = 0.02 of signal std), (iv) amplitude scale in [0.85, 1.15].
    (Random end-crop (v) is omitted here in favour of the roll-based
    time-shift, which already perturbs alignment; all four applied
    augmentations are randomised per-sample so two calls give two views.)
    """
    B, C, T = x.shape
    out = x.clone()
    # 1. Time-shift up to +/-20 samples (roll)
    shifts = torch.randint(-20, 21, (B,), device=x.device)
    for b in range(B):
        out[b] = torch.roll(out[b], int(shifts[b].item()), dims=-1)
    # 2. Additive low-freq drift
    t = torch.arange(T, dtype=torch.float32, device=x.device) / T
    for b in range(B):
        freq = 0.005 + 0.045 * torch.rand(1, device=x.device).item()
        phase = 2 * np.pi * torch.rand(1, device=x.device).item()
        amp = 0.05 * out[b].std()
        drift = amp * torch.sin(2 * np.pi * freq * T * t + phase)
        out[b, 0] += drift
    # 3. Additive Gaussian noise
    out = out + 0.02 * out.std() * torch.randn_like(out)
    # 4. Amplitude scaling
    scale = 0.85 + 0.3 * torch.rand(B, 1, 1, device=x.device)
    out = out * scale
    return out


def nt_xent_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.1
                 ) -> torch.Tensor:
    """NT-Xent (InfoNCE) contrastive loss, SimCLR-style.

    z_a, z_b: (B, D) L2-normalised.
    Positive pair: (z_a[i], z_b[i]); negatives: all other pairs in the batch.
    """
    B = z_a.size(0)
    z = torch.cat([z_a, z_b], dim=0)   # (2B, D)
    sim = torch.mm(z, z.t()) / temperature   # (2B, 2B)
    # Mask self-similarities
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))
    # Positive index: i pairs with (i + B) mod 2B
    pos_idx = torch.arange(2 * B, device=z.device)
    pos_idx = (pos_idx + B) % (2 * B)
    return F.cross_entropy(sim, pos_idx)


def output_dir_name(objective: str, backbone: str, epochs: int, tau: float,
                     corpus: str = "cov") -> str:
    """Deterministic result-dir naming that preserves legacy paths for defaults.

    Defaults (epochs=50, tau=0.1, corpus=cov) → `p4_ssl_{obj}_{backbone}`
    (legacy). Otherwise append `_epochs{N}` and/or `_tau{value}` so W16
    sweeps don't collide with the existing 50-epoch encoders, and are
    discoverable by the scoreboard glob.

    `corpus="cov_uti"` (W18) uses a distinct suffix format --
    `_cov_uti_ep{N}` (epochs != 50) or bare `_cov_uti` (epochs == 50) --
    matching the path already hardcoded by Task 6's downstream consumers
    (w18_rq2_track_c_cov_uti.py / w18_rq2_track_c_uti_only.py and their
    PBS scripts expect exactly
    `p4_ssl_contrastive_unet_cov_uti_ep100`, NOT `..._epochs100_cov_uti`).
    The default `corpus="cov"` is untouched by this branch, so existing
    callers/tests are unaffected.
    """
    base = f"p4_ssl_{objective}_{backbone}"
    if corpus == "cov_uti":
        suffix = "_cov_uti"
        if epochs != 50:
            suffix += f"_ep{epochs}"
        if abs(tau - 0.1) > 1e-9:
            suffix += f"_tau{tau:g}"
        return base + suffix
    if corpus == "conc":
        # New-data KP concentration corpus (all 5 chips' pixels).
        suffix = "_conc"
        if epochs != 50:
            suffix += f"_ep{epochs}"
        if abs(tau - 0.1) > 1e-9:
            suffix += f"_tau{tau:g}"
        return base + suffix
    suffix = ""
    if epochs != 50:
        suffix += f"_epochs{epochs}"
    if abs(tau - 0.1) > 1e-9:
        # Match the tau tag used by the fine-tune stage: keep bare float.
        suffix += f"_tau{tau:g}"
    return base + suffix


def _load_unlabelled_corpus(corpus: str = "cov") -> np.ndarray:
    """Return (N, 450) unlabelled ISFET traces, N >= 25000.

    The brief's suggested `lacewing.classification.data.dataset.load_cache`
    API does not exist (that module only exposes a private `_load_cache`
    tied to the `dataset_per_pixel` classification cache). Per the brief's
    documented fallback, we instead use the frozen preprocessing cache
    (`regress_all_filt_abcd_ntcRaw_madk1p5_spatA3`, required by this
    project's global constraints) via
    `lacewing.quantification.regression_shared.data.dataset.make_split`, concatenating
    X_tr/X_va/X_te to recover every pixel in the cache regardless of its
    split or (TTP) label -- appropriate for corpus-level SSL, where labels
    are irrelevant. This cache alone has ~27.9k traces, above the 25k
    threshold.

    When `corpus == "cov_uti"` (W18), the UTI classifier cache (all pixels,
    labelled and unlabelled status alike -- every pixel is training material
    for unlabelled SSL) is concatenated after the COV pixels, growing the
    corpus to COV+UTI+KP_03 (~67k pixels; COV reg cache 27.9k + UTI cls cache 39.3k). The UTI cache is NOT filtered by
    amp+/amp- status: SSL pretraining has no notion of labels.
    """
    # Use _load_cache (raw) instead of make_split, so SSL sees every pixel
    # in the cache including NTC/PTC — labels are irrelevant for SSL, and
    # make_split now drops NaN-labelled pixels which would shrink the
    # unlabelled corpus without cause.
    from lacewing.quantification.regression_shared.data.dataset import _load_cache
    if corpus == "conc":
        # Pool ALL pixels from all 5 KP concentration-data LOCO caches.
        # Since every LOCO cache contains all 5 chips' pixels (just with
        # different split assignments), we load ONE cache and use every
        # pixel exactly once.
        data = _load_cache("regress_conc_loco1")
        return data["X"].astype(np.float32)
    data = _load_cache("regress_all_filt_abcd_ntcRaw_madk1p5_spatA3")
    X = data["X"].astype(np.float32)
    if corpus == "cov_uti":
        uti = np.load(UTI_CLS_CACHE, allow_pickle=False)
        uti_X = uti["X"].astype(np.float32)
        X = np.concatenate([X, uti_X], axis=0)
    return X


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--objective", choices=["masked", "contrastive"], required=True)
    p.add_argument("--backbone", choices=["unet", "gru", "cnn_gru_par"], required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--tau", type=float, default=0.1,
        help="NT-Xent temperature for contrastive objective (ignored for masked).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--corpus", choices=["cov", "cov_uti", "conc"], default="cov",
        help="Unlabelled corpus to pretrain on: 'cov' (legacy, default), "
             "'cov_uti' (COV+UTI+KP_03, W18), or 'conc' (KP concentration "
             "data, all 5 chips, W22+).",
    )
    args = p.parse_args()

    from lacewing.classification.core.seeding import seed_everything
    seed_everything(args.seed)

    X = _load_unlabelled_corpus(corpus=args.corpus)
    print(f"Unlabelled corpus: N={len(X)} traces")
    assert len(X) >= 25_000, f"Need >= 25k traces; got {len(X)}"

    # 90/10 train/val split for pretraining val-loss monitoring
    n = len(X)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(0.1 * n)
    X_va = X[perm[:n_val]]
    X_tr = X[perm[n_val:]]

    encoder = build_ssl_encoder(args.backbone)
    if args.objective == "masked":
        model: nn.Module = MaskedReconstructionModel(encoder)
    else:
        model = ContrastiveModel(encoder)
    model = model.to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []

    from torch.utils.data import DataLoader, TensorDataset
    tr_ds = TensorDataset(torch.from_numpy(X_tr).unsqueeze(1))  # (N, 1, 450)
    va_ds = TensorDataset(torch.from_numpy(X_va).unsqueeze(1))
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0; n_tr = 0
        for (x,) in tr_loader:
            x = x.to(args.device)
            if args.objective == "masked":
                masked_x, mask = mask_chunks(x, mask_ratio=0.15, chunk_len=30)
                recon = model(masked_x)
                loss = ((recon - x) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            else:
                view_a, view_b = augment_pair(x)
                z_a = model(view_a); z_b = model(view_b)
                loss = nt_xent_loss(z_a, z_b, temperature=args.tau)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += float(loss) * x.size(0); n_tr += x.size(0)
        tr_loss /= max(n_tr, 1)

        # Val
        model.eval()
        va_loss = 0.0; n_va = 0
        with torch.no_grad():
            for (x,) in va_loader:
                x = x.to(args.device)
                if args.objective == "masked":
                    masked_x, mask = mask_chunks(x, 0.15, 30)
                    recon = model(masked_x)
                    l = ((recon - x) ** 2 * mask).sum() / mask.sum().clamp(min=1)
                else:
                    view_a, view_b = augment_pair(x)
                    z_a = model(view_a); z_b = model(view_b)
                    l = nt_xent_loss(z_a, z_b, temperature=args.tau)
                va_loss += float(l) * x.size(0); n_va += x.size(0)
        va_loss /= max(n_va, 1)
        history.append({"epoch": epoch, "tr_loss": tr_loss, "va_loss": va_loss})
        print(f"epoch {epoch:02d}/{args.epochs}  tr={tr_loss:.4f}  va={va_loss:.4f}")

    # Save encoder weights only (that's what the fine-tune uses)
    out_dir = LACEWING_PKG_DIR / "quantification" / "methods" / "results" / output_dir_name(
        objective=args.objective, backbone=args.backbone,
        epochs=args.epochs, tau=args.tau, corpus=args.corpus,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), out_dir / "pretrained.pt")
    (out_dir / "config.json").write_text(json.dumps({
        "objective": args.objective, "backbone": args.backbone,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "tau": args.tau, "seed": args.seed, "n_corpus": len(X),
        "corpus": args.corpus,
    }, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[ok] wrote {out_dir}/pretrained.pt")


if __name__ == "__main__":
    main()
