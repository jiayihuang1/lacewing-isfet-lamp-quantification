"""RQ3 · joint classification+quantification framework (all 5 A1-A5 architectures). [Cat A] A1+SSL is the SARS-CoV-2 winner (Report §RQ3 Tab. 7.4). A3+SSL is the KP 5-fold LOCO winner (Report §NewData Tab. 8.3).

P3 joint classification + quantification — 8 architectures on F-B U-Net.

Architectures:
  A1 — Two-heads joint loss: shared U-Net encoder features, cls + quant heads.
  A1s — A1 with shared encoder features (simplified variant).
  A2 — Paper 19 cls-guided: cls gates the quant logit multiplicatively.
  A3 — Sequential routing: cls-first, quant only on cls-positive pixels.
  A3n — A3 without the inference gate (simpler variant for ablation).
  A4 — Attention-gated: cls conf × quant prob at inference.
  A4w — A4 with per-window gating (finer-grained than per-pixel).
  A5 — Prob-space multiplicative gate: p_gated = σ(q) × σ(c) at train + inference.

All architectures share the same F-B U-Net encoder from _fb_backbones (via
build_fb_backbone("unet", input_len=W)).  The cls head is a simple linear
projection off the same encoder features that feed the existing quant head.
"""
from __future__ import annotations

import argparse
import re
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lacewing.quantification.methods._fb_backbones import build_fb_backbone
from lacewing.quantification.methods._windowing import (
    slide_windows,
    aggregate_to_ttp_first_positive,
)
from lacewing.quantification.regression_shared.data import dataset as reg_ds


SAMPLES_PER_MIN = 15
CACHE_STEM = "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3"

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


class JointModel(nn.Module):
    """Shared F-B U-Net encoder + cls & quant heads.

    forward returns (quant_logit, cls_logit) — both (B,) scalars per window.

    For A2, the quant_logit returned is the gated logit:
        quant_logit_gated = quant_logit_raw + log(sigmoid(cls_logit) + eps)
    This means at inference, using quant_logit directly already incorporates
    the cls gate (lower cls confidence suppresses the quant logit).
    """

    def __init__(self, arch: str, window: int, alpha: float):
        super().__init__()
        self.arch = arch
        self.alpha = alpha
        # Existing F-B U-Net backbone: (B, 1, W) → (B,) scalar logit.
        self.backbone = build_fb_backbone("unet", input_len=window)
        # Cls head configuration depends on arch:
        #   a1s = Linear(1, 1) attached to the U-Net's pooled features
        #         (i.e. shares the encoder with the quant head).
        #   otherwise = separate small conv net operating on the raw window
        #               (original A1-A4 behaviour).
        if arch == "a1s":
            self.cls_head = nn.Linear(1, 1)
        else:
            self.cls_head = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(32, 1),
            )

    def load_pretrained_cls(self, ckpt_path: str) -> None:
        """Load a frozen cls-head state_dict (produced by train_cls_stage.py).

        Used by a2p: replaces the (random-init) parallel-conv cls_head with a
        pretrained one, then freezes it so Stage-2 quant training sees a
        stable, calibrated gate. `cls_head` must be the parallel-conv module
        (i.e. arch != 'a1s'); for a1s this method is not applicable.
        """
        if self.arch == "a1s":
            raise ValueError("load_pretrained_cls does not apply to a1s (shared encoder).")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cls_head.load_state_dict(state)
        for p in self.cls_head.parameters():
            p.requires_grad_(False)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        For a1s, both heads consume the SAME pooled U-Net features (shared
        encoder). For all other archs the cls head runs its own conv net on
        the raw window (parameter-disjoint from the U-Net quant head).
        For a2 (legacy), the quant_logit returned is already gated by
        log(sigmoid(cls_logit)); see the original A2 docstring.
        """
        if self.arch == "a1s":
            # SHARED path: pool U-Net encoder features once, feed to both heads.
            feats = self.backbone.forward_features(x)  # (B, 1)
            quant_logit = self.backbone.head(feats).squeeze(-1)
            cls_logit = self.cls_head(feats).squeeze(-1)
        else:
            quant_logit = self.backbone(x)                       # (B,)
            cls_logit = self.cls_head(x).squeeze(-1)             # (B,)
        if self.arch in ("a2", "a2p"):
            gate = torch.log(torch.sigmoid(cls_logit) + 1e-8)
            quant_logit = quant_logit + gate
        return quant_logit, cls_logit


def build_joint_model(arch: str, window: int, alpha: float) -> JointModel:
    if arch not in {"a1", "a2", "a3", "a4", "a1s", "a2p", "a5", "a3n", "a4w"}:
        raise ValueError(f"unknown P3 architecture: {arch!r}")
    return JointModel(arch=arch, window=window, alpha=alpha)


def _joint_loss(
    quant_logit: torch.Tensor,
    cls_logit: torch.Tensor,
    quant_label: torch.Tensor,
    cls_label: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Compute the joint classification + quantification loss.

    Loss = α·BCE(quant_logit, quant_label) + (1-α)·BCE(cls_logit, cls_label)

    Args:
        quant_logit: (B,) raw logits for quantification (pre-amp/post-amp).
        cls_logit: (B,) raw logits for classification (amp-positive/NTC).
        quant_label: (B,) binary labels for quantification (0=pre-amp, 1=post-amp).
        cls_label: (B,) binary labels for classification (0=NTC, 1=amp-positive).
        alpha: Weight on quant loss; (1-alpha) on cls loss.

    Returns:
        Scalar loss tensor.
    """
    bce = nn.BCEWithLogitsLoss()
    quant_loss = bce(quant_logit, quant_label)
    cls_loss = bce(cls_logit, cls_label)
    return alpha * quant_loss + (1.0 - alpha) * cls_loss


class JointWindowsDataset(torch.utils.data.Dataset):
    """Windows dataset with quant + cls labels.

    quant_label per window = 1 if window centre >= pixel's TTP, else 0.
    cls_label per window = 1 if the pixel is amp-positive (has a valid qLAMP TTP), else 0.
    All windows from a pixel share the same cls_label (per-pixel-level cls).

    A3: quant loss is masked to cls-positive windows only during training.
    This is handled externally (in the training loop) by filtering on cls_label.
    """

    def __init__(
        self,
        X: np.ndarray,
        ttp_min: np.ndarray,
        cls_label_per_pixel: np.ndarray,
        window: int,
        stride: int,
        arch: str,
    ):
        assert len(X) == len(ttp_min) == len(cls_label_per_pixel)
        self.windows = slide_windows(X, window, stride)  # (N, W_count, W_samples)
        n_windows = self.windows.shape[1]
        centres = np.arange(n_windows) * stride + (window - 1) / 2.0
        ttp_idx = ttp_min * SAMPLES_PER_MIN
        # (N, W_count) quant labels
        self.quant_labels = (centres[None, :] >= ttp_idx[:, None]).astype(np.float32)
        # (N, W_count) cls labels (broadcast from per-pixel cls label)
        self.cls_labels = np.repeat(
            cls_label_per_pixel[:, None], n_windows, axis=1
        ).astype(np.float32)
        self.arch = arch
        self.n_pixels = self.windows.shape[0]
        self.n_windows = n_windows

    def __len__(self) -> int:
        return self.n_pixels * self.n_windows

    def __getitem__(self, idx: int):
        p = idx // self.n_windows
        w = idx % self.n_windows
        x = torch.from_numpy(self.windows[p, w])[None, :]
        return (
            x,
            torch.tensor(self.quant_labels[p, w], dtype=torch.float32),
            torch.tensor(self.cls_labels[p, w], dtype=torch.float32),
        )


def _load_kp_ntc_pixels_from_cache(cache_stem: str) -> np.ndarray:
    """Load NTC-well pixels from a KP regression cache.

    KP cache layout: each chip has PTC on one of wells {8, 9} and NTC on
    the other, per Chip Order.xlsx. The per-chip NTC well is authoritative
    in conc_data.process_conc_chips._build_chip_configs. This helper reads
    the cache directly (train pool only, split==0) and returns pixels whose
    (chip_id, well_id) matches the per-chip NTC role.

    Returns X_kp_ntc: (M, T) float32. Empty array if the cache is not a
    KP one (chip_id doesn't match any known KP chip).
    """
    from lacewing.quantification.regression_shared.data.dataset import _load_cache
    from lacewing.quantification.conc_data.process_conc_chips import (
        _build_chip_configs,
    )

    data = _load_cache(cache_stem)
    chip_id = data["chip_id"]
    well_id = data["well_id"]
    split = data["split"]
    X = data["X"]

    # Map chip_tag → NTC well index
    ntc_by_chip = {}
    for cfg in _build_chip_configs():
        # ntc_wells is a tuple; here it's always a single well
        if len(cfg.ntc_wells) == 1:
            ntc_by_chip[cfg.chip_tag] = cfg.ntc_wells[0]

    if not any(str(c) in ntc_by_chip for c in np.unique(chip_id)):
        return np.zeros((0, X.shape[1]), dtype=np.float32)

    keep = np.zeros(len(chip_id), dtype=bool)
    for tag, ntc_well in ntc_by_chip.items():
        keep |= (chip_id == tag) & (well_id == ntc_well) & (split == 0)

    return X[keep].astype(np.float32)


def _load_ntc_pixels(
    kp_cache_stem: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a set of NTC pixel traces to serve as classifier negatives.

    Always loads SARS-CoV-2 NTC pixels from the classification cache
    ``dataset_all_filt_abcd_ntcRaw_madk1p5_spatA3`` (capped at 5000, matching
    the SARS-CoV-2 training recipe).

    If ``kp_cache_stem`` is provided AND corresponds to a KP regression
    cache, also loads KP NTC pixels (identified per-chip via the KP well
    role map) from the training pool of that cache, so the classifier has
    same-chemistry negatives available on KP training. Without this, the
    KP classifier could take a chemistry-detection shortcut against the
    SARS-CoV-2 negatives.

    Returns:
        (X_ntc: (M, T) float32, cls_labels: (M,) float32 all zeros).
    """
    from lacewing.classification.core import paths as cls_paths

    # SARS-CoV-2 NTC pixels via the matched-preprocessing cls cache.
    cache_path = cls_paths.cache_path("dataset_all_filt_abcd_ntcRaw_madk1p5_spatA3")
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Classification cache not found: {cache_path}. "
            "Build it with: "
            "python -m lacewing.classification.data.build_filtered_dataset_mad "
            "--scope all --layers ABCD --k 1.5  "
            "&& python -m lacewing.classification.data.build_spatial_filtered_dataset "
            "--input dataset_all_filt_abcd_ntcRaw_madk1p5 --order 3"
        )
    data = np.load(cache_path, allow_pickle=False)
    X_all = data["X"].astype(np.float32)
    y_all = data["y"]

    is_ntc = y_all == 0
    X_cov_ntc = X_all[is_ntc]
    n_cov = len(X_cov_ntc)
    if n_cov > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_cov, 5000, replace=False)
        X_cov_ntc = X_cov_ntc[idx]

    # Optionally add KP NTC pixels alongside the CoV negatives.
    if kp_cache_stem is not None:
        X_kp_ntc = _load_kp_ntc_pixels_from_cache(kp_cache_stem)
        if len(X_kp_ntc) > 0:
            print(
                f"  _load_ntc_pixels: adding {len(X_kp_ntc)} KP NTC pixels "
                f"alongside {len(X_cov_ntc)} SARS-CoV-2 NTC pixels"
            )
            X_ntc = np.concatenate([X_cov_ntc, X_kp_ntc], axis=0)
        else:
            X_ntc = X_cov_ntc
    else:
        X_ntc = X_cov_ntc

    cls_labels = np.zeros(len(X_ntc), dtype=np.float32)
    return X_ntc, cls_labels


def main() -> None:
    """CLI entry point for P3 joint cls+quant training and evaluation."""
    p = argparse.ArgumentParser(
        description="P3 joint cls+quant — train one of 9 architectures (a1/a1s/a2/a2p/a3/a3n/a4/a4w/a5)."
    )
    p.add_argument("--arch", choices=["a1", "a1s", "a2", "a2p", "a3", "a3n", "a4", "a4w", "a5"], required=True,
                   help="Architecture: a1=two-heads, a1s=a1 with shared U-Net encoder, a2=cls-guided, a2p=a2 with pretrained cls, a3=sequential, a3n=a3 without inference gate, a4=attention, a4w=per-window attention, a5=prob-space gate.")
    p.add_argument("--alpha", type=float, required=True,
                   help="Loss weight α: total = α·L_quant + (1-α)·L_cls.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=120)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--k_consecutive", type=int, default=3)
    p.add_argument("--k_thr", type=float, default=0.7)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--max_train_pixels", type=int, default=None,
        help="Subsample training set to at most N pixels (for smoke-testing on CPU).",
    )
    p.add_argument(
        "--max_val_pixels", type=int, default=None,
        help="Subsample val set to at most N pixels (for smoke-testing on CPU).",
    )
    p.add_argument(
        "--pretrained-encoder", type=str, default=None,
        help="Optional path to an SSL-pretrained encoder state_dict "
             "(e.g. p4_ssl_masked_unet/pretrained.pt). Loads via strict=False into "
             "self.backbone.unet — transfers encoder body, skips final projection.",
    )
    p.add_argument(
        "--pretrained-cls", type=str, default=None,
        help="Path to a frozen cls-head state_dict (e.g. p3_cls_pretrained/"
             "seed0/cls.pt). Loaded via load_pretrained_cls() for arch=a2p.",
    )
    p.add_argument(
        "--cache-stem", type=str, default=CACHE_STEM,
        help="Regression cache stem (default = SARS-CoV-2). For new-data "
             "LOCO runs pass e.g. regress_conc_loco1.",
    )
    args = p.parse_args()

    from lacewing.classification.core.seeding import seed_everything
    seed_everything(args.seed)

    # ------------------------------------------------------------------
    # Load amp-positive pixels (quant training data)
    # ------------------------------------------------------------------
    cache_stem = args.cache_stem
    arr = reg_ds.make_split(cache_stem, seed=args.seed, normalise_y=False)
    X_pos = arr.X_tr
    y_pos = arr.y_tr                           # TTP in minutes (not normalised)
    cls_pos = np.ones(len(X_pos), dtype=np.float32)

    # ------------------------------------------------------------------
    # Load NTC pixels (cls training signal). On KP LOCO caches also pulls
    # KP NTC-well pixels so the classifier has same-chemistry negatives.
    # ------------------------------------------------------------------
    kp_stem = cache_stem if cache_stem.startswith("regress_conc_loco") else None
    X_ntc, cls_ntc = _load_ntc_pixels(kp_cache_stem=kp_stem)
    # NTC pixels don't have valid TTPs; set to 30 min (past end of trace
    # so all windows get quant_label=0, which is correct for negatives).
    y_ntc = np.full(len(X_ntc), 30.0, dtype=np.float32)

    # Combine
    X_all = np.concatenate([X_pos, X_ntc], axis=0)
    y_all = np.concatenate([y_pos, y_ntc], axis=0)
    cls_all = np.concatenate([cls_pos, cls_ntc], axis=0)

    # Optional subsampling for fast smoke-tests on CPU
    if args.max_train_pixels is not None and len(X_all) > args.max_train_pixels:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X_all), size=args.max_train_pixels, replace=False)
        idx.sort()
        X_all = X_all[idx]
        y_all = y_all[idx]
        cls_all = cls_all[idx]

    print(
        f"Train: {len(X_pos)} amp-positive + {len(X_ntc)} NTC = {len(X_all)} pixels"
        + (f" [subsampled to {len(X_all)}]" if args.max_train_pixels is not None else "")
    )

    train_ds = JointWindowsDataset(X_all, y_all, cls_all, args.window, args.stride, args.arch)

    # Val: amp-positive only (no NTC in the regression val split)
    X_va = arr.X_va
    y_va = arr.y_va
    if args.max_val_pixels is not None and len(X_va) > args.max_val_pixels:
        rng2 = np.random.default_rng(args.seed + 1)
        idx_va = rng2.choice(len(X_va), size=args.max_val_pixels, replace=False)
        idx_va.sort()
        X_va = X_va[idx_va]
        y_va = y_va[idx_va]
    cls_va = np.ones(len(X_va), dtype=np.float32)
    val_ds = JointWindowsDataset(X_va, y_va, cls_va, args.window, args.stride, args.arch)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    print(
        f"Windows per pixel={train_ds.n_windows}  "
        f"train_windows={len(train_ds)}  val_windows={len(val_ds)}"
    )

    # ------------------------------------------------------------------
    # Build model + output dir
    # ------------------------------------------------------------------
    model = build_joint_model(args.arch, args.window, args.alpha).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built arch={args.arch}  window={args.window}  alpha={args.alpha}  ({n_params:,} params)")

    # ---- P3+P4 combo: optionally load SSL-pretrained encoder weights ----
    pretrained_tag = ""
    if args.pretrained_encoder is not None:
        ckpt_path = Path(args.pretrained_encoder)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No pretrained encoder at {ckpt_path}")
        ssl_state = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        # SSL encoder is a SSLEncoderUNet wrapping a UNet1D at `self.unet`.
        # P3's backbone is a _UNetWindowClassifier wrapping a UNet1D at
        # `self.backbone.unet` (from _fb_backbones.py:48). Load with the "unet."
        # prefix stripped so keys align.
        stripped = {}
        for k, v in ssl_state.items():
            if k.startswith("unet."):
                stripped[k[len("unet."):]] = v
        # `model.backbone.unet` is the target UNet1D. SSL was pretrained with
        # n_classes=8 so its `classifier.{weight,bias}` shape mismatches the
        # F-B backbone's n_classes=1 head. Drop those keys so strict=False can
        # load the encoder body cleanly (PyTorch's strict=False skips MISSING
        # keys but errors on shape mismatches).
        target_shapes = {k: v.shape for k, v in model.backbone.unet.state_dict().items()}
        for k in list(stripped.keys()):
            if k in target_shapes and stripped[k].shape != target_shapes[k]:
                del stripped[k]
        missing, unexpected = model.backbone.unet.load_state_dict(stripped, strict=False)
        print(f"  loaded pretrained encoder from {ckpt_path.name}: "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        # Tag the output dir with the SSL encoder's identity so runs don't collide.
        # Convention:
        #   p4_ssl_{objective}_{backbone}                        → _pt{objective}
        #   p4_ssl_{objective}_{backbone}_{suffix}               → _pt{objective}_{suffix}
        # (suffix comes from W16 sweeps: e.g. tau1, epochs100)
        enc_dir = ckpt_path.parent.name  # e.g. p4_ssl_contrastive_unet_tau1
        m = re.match(r"^p4_ssl_(?P<obj>[a-z]+)_(?P<bb>[a-z_]+?)(?:_(?P<suffix>.+))?$",
                     enc_dir)
        if m and m.group("suffix"):
            pretrained_tag = f"_pt{m.group('obj')}_{m.group('suffix')}"
        elif m:
            pretrained_tag = f"_pt{m.group('obj')}"
        else:
            # Fallback: use the parent dir name minus 'p4_ssl_' prefix
            pretrained_tag = f"_pt{enc_dir.replace('p4_ssl_', '')}"
    if args.pretrained_cls is not None:
        if args.arch != "a2p":
            raise ValueError(
                f"--pretrained-cls only supported for arch=a2p; got {args.arch!r}"
            )
        model.load_pretrained_cls(args.pretrained_cls)
        print(f"  loaded frozen cls state_dict from {args.pretrained_cls}")
        pretrained_tag = pretrained_tag + "_ptcls"


    cache_suffix = ""
    if cache_stem != CACHE_STEM:
        cache_suffix = "_" + cache_stem.replace("regress_", "")
    out_dir = (
        RESULTS_ROOT
        / f"p3_{args.arch}_alpha{args.alpha:g}_unet_w{args.window}_stride{args.stride}{pretrained_tag}{cache_suffix}"
        / f"seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for x, y_q, y_c in train_loader:
            x = x.to(args.device)
            y_q = y_q.to(args.device)
            y_c = y_c.to(args.device)
            quant_logit, cls_logit = model(x)
            if args.arch in ("a3", "a3n"):
                # a3, a3n: mask quant loss on cls-negative windows (label-based).
                mask = y_c > 0.5
                if mask.any():
                    quant_loss = nn.BCEWithLogitsLoss()(quant_logit[mask], y_q[mask])
                else:
                    quant_loss = torch.tensor(0.0, device=args.device)
                cls_loss = nn.BCEWithLogitsLoss()(cls_logit, y_c)
                loss = args.alpha * quant_loss + (1.0 - args.alpha) * cls_loss
            elif args.arch == "a5":
                # a5: prob-space multiplicative gate INSIDE the loss. The quant
                # head sees the gated probability during training so it learns
                # to compensate for the expected scaling at inference.
                p_gated = torch.sigmoid(quant_logit) * torch.sigmoid(cls_logit)
                # BCE from probabilities directly (not logits, since we've
                # already applied sigmoid).
                p_gated = p_gated.clamp(1e-7, 1 - 1e-7)
                quant_loss = -(
                    y_q * torch.log(p_gated)
                    + (1 - y_q) * torch.log(1 - p_gated)
                ).mean()
                cls_loss = nn.BCEWithLogitsLoss()(cls_logit, y_c)
                loss = args.alpha * quant_loss + (1.0 - args.alpha) * cls_loss
            else:
                loss = _joint_loss(quant_logit, cls_logit, y_q, y_c, args.alpha)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.detach().item() * x.size(0)
            n += x.size(0)
        tr_loss = total_loss / max(n, 1)

        # Validation
        model.eval()
        v_loss = 0.0
        nv = 0
        with torch.no_grad():
            for x, y_q, y_c in val_loader:
                x = x.to(args.device)
                y_q = y_q.to(args.device)
                y_c = y_c.to(args.device)
                quant_logit, cls_logit = model(x)
                loss = _joint_loss(quant_logit, cls_logit, y_q, y_c, args.alpha)
                v_loss += loss.item() * x.size(0)
                nv += x.size(0)
        va_loss = v_loss / max(nv, 1)
        history.append({"epoch": epoch, "tr_loss": tr_loss, "va_loss": va_loss})
        print(f"epoch {epoch:02d}/{args.epochs}  tr={tr_loss:.4f}  va={va_loss:.4f}")

        # Checkpointing
        state = {"epoch": epoch, "model_state": model.state_dict()}
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(state, out_dir / "checkpoints" / "best.pt")
        torch.save(state, out_dir / "checkpoints" / "last.pt")

    # ------------------------------------------------------------------
    # Test-time prediction on SD test chip
    # ------------------------------------------------------------------
    X_te = arr.X_te
    test_windows = slide_windows(X_te, args.window, args.stride)  # (N, W_count, W)
    n_te, n_wins, _ = test_windows.shape
    quant_probs = np.zeros((n_te, n_wins), dtype=np.float32)
    cls_probs = np.zeros(n_te, dtype=np.float32)

    # Batch across (pixel × window) pairs for efficient CPU/GPU utilisation.
    # Flatten to (N*W_count, W), run in batches, then reshape.
    flat_windows = test_windows.reshape(-1, args.window)  # (N*W_count, W)
    flat_q = np.zeros(len(flat_windows), dtype=np.float32)
    flat_c = np.zeros(len(flat_windows), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(flat_windows), args.batch_size):
            end = min(start + args.batch_size, len(flat_windows))
            xb = torch.from_numpy(flat_windows[start:end]).unsqueeze(1).to(args.device)
            q_logit, c_logit = model(xb)
            flat_q[start:end] = torch.sigmoid(q_logit).cpu().numpy()
            flat_c[start:end] = torch.sigmoid(c_logit).cpu().numpy()

    quant_probs = flat_q.reshape(n_te, n_wins)
    # Per-window cls prob shape (n_te, n_wins), plus the per-pixel mean
    # used by A4 (kept for backwards compat).
    cls_probs_per_win = flat_c.reshape(n_te, n_wins)
    cls_probs = cls_probs_per_win.mean(axis=1)

    # A5: prob-space gate applied to inference probs (matches training).
    if args.arch == "a5":
        quant_probs = quant_probs * cls_probs_per_win
    # A4: attention-gated — per-pixel cls scalar rescales every window's quant prob
    if args.arch == "a4":
        quant_probs = quant_probs * cls_probs[:, None]
    # A4w: per-window gating — each window's quant prob is scaled by its OWN
    # cls prob, not the pixel mean. Finer-grained; avoids single-window
    # extraction failures caused by uniform per-pixel scaling.
    if args.arch == "a4w":
        quant_probs = quant_probs * cls_probs_per_win

    ttp_pred = aggregate_to_ttp_first_positive(
        quant_probs,
        threshold=args.k_thr,
        k_consecutive=args.k_consecutive,
        window=args.window,
        stride=args.stride,
        samples_per_min=SAMPLES_PER_MIN,
    )

    # A3: pixels with low cls confidence get NaN TTP (fragile on the SD chip).
    # a3n: identical training-time behaviour but SKIPS the inference gate to
    # isolate the training-time contribution of the label-based mask.
    if args.arch == "a3":
        ttp_pred = np.where(cls_probs >= 0.5, ttp_pred, np.nan)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    np.savez_compressed(
        out_dir / "predictions.npz",
        ttp_pred_min=ttp_pred.astype(np.float32),
    )
    np.savez_compressed(
        out_dir / "cls_predictions.npz",
        cls_prob=cls_probs.astype(np.float32),
    )

    # Save labels using the canonical schema
    from lacewing.quantification.eval.schema import write_labels
    from lacewing.quantification.methods.pdf_regression import _log10_concentration_array

    log10_conc = _log10_concentration_array(arr.chip_te, arr.well_te)
    write_labels(
        out_dir / "labels.npz",
        ttp_true_min=arr.y_te.astype(np.float32),
        chip_id=arr.chip_te,
        well_id=arr.well_te.astype(np.int32),
        log10_concentration=log10_conc,
        split=np.full(len(arr.y_te), "test", dtype="U16"),
    )

    cfg = {
        "method": f"P3 joint cls+quant, arch={args.arch}, alpha={args.alpha}",
        "arch": args.arch,
        "alpha": args.alpha,
        "backbone": "unet",
        "window": args.window,
        "stride": args.stride,
        "k_consecutive": args.k_consecutive,
        "k_thr": args.k_thr,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "n_train_pos": len(X_pos),
        "n_train_ntc": len(X_ntc),
        "n_val": len(X_va),
        "n_test": len(X_te),
        "n_params": n_params,
        "samples_per_min": SAMPLES_PER_MIN,
        "cache_stem": cache_stem,
        "pretrained_encoder": str(args.pretrained_encoder) if args.pretrained_encoder else None,
        "pretrained_cls": str(args.pretrained_cls) if args.pretrained_cls else None,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[ok] wrote {out_dir}/")


if __name__ == "__main__":
    main()
