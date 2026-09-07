"""P2 backbone factory: adapt every classification-tier backbone to F-E's framing.

F-E framing: input `(B, 1, T=450)` → per-timestep sigmoid logits `(B, T=450)`.

Each backbone below is adapted to that output shape:

* BiGRU, GRU: per-timestep hidden state → linear head (no pooling).
* CNN-GRU-par: strip the fusion head + global pool; per-timestep linear on
  the concatenation of the CNN branch broadcast + GRU branch per-timestep
  hidden state.  See below for detail.
* transformer_patch: strip the global pool; broadcast each patch's
  attended embedding across its P samples via `repeat_interleave`, then
  per-timestep linear.
* 1D U-Net: already per-timestep; just use `n_classes=1`.
* TCN: use `TCNPerTimestep` (n_classes=1).
* ANN: per-timestep application of a tiny MLP.  Deliberately included as
  a diagnostic — this is the "F1.1 pathology" ceiling (each timestep has
  no context beyond the raw value at t).

Every constructor returns something with `forward(x: (B, 1, T)) -> (B, T)`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# =====================================================================
# BiGRU + GRU: per-timestep linear head on the RNN's hidden state
# =====================================================================

class BiGRUPerTimestepP2(nn.Module):
    """Same as the P1.5 BiGRUPerTimestep (kept here for consistency)."""
    def __init__(self, input_size: int = 1, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x.transpose(1, 2))  # (B, T, 2*hidden)
        return self.head(h).squeeze(-1)


class GRUUniPerTimestep(nn.Module):
    """Unidirectional GRU with per-timestep head."""
    def __init__(self, input_size: int = 1, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=False,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x.transpose(1, 2))  # (B, T, hidden)
        return self.head(h).squeeze(-1)


# =====================================================================
# ANN — deliberately-diagnostic no-context per-timestep MLP
# =====================================================================

class ANNPerTimestep(nn.Module):
    """Tiny MLP applied to EACH timestep independently.

    The whole point is that the timestep-t input is just x[t] (a single
    scalar) — no context.  This is the F1.1 pathology as a baseline:
    if the P2 sweep shows this method underperforms BiGRU / U-Net / TCN
    by a lot, that's confirmation that full-trace context is what
    matters.  If it doesn't underperform, less context is enough.
    """
    def __init__(self, hidden_dims: tuple[int, ...] = (16, 8)):
        super().__init__()
        dims = (1,) + tuple(hidden_dims)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T) → (B*T, 1) → MLP → (B*T, 1) → (B, T)
        b, c, t = x.shape
        assert c == 1, x.shape
        x_flat = x.transpose(1, 2).reshape(b * t, 1)
        y_flat = self.mlp(x_flat).squeeze(-1)  # (B*T,)
        return y_flat.view(b, t)


# =====================================================================
# CNN-GRU-par per-timestep
# =====================================================================

class CNNGRUParPerTimestep(nn.Module):
    """CNN-GRU-par adapted for per-timestep output.

    The GRU branch already gives per-timestep hidden state.
    The CNN branch gives a global feature summary — we broadcast it
    across every timestep by simply concatenating the same CNN feature
    vector to every timestep's GRU output.  This preserves the original
    CNN-GRU-par 'CNN + GRU parallel two-stream' philosophy while
    yielding a per-timestep output.
    """
    def __init__(
        self,
        input_len: int = 450,
        d_branch: int = 64,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
        d_fusion: int = 64,
    ):
        super().__init__()
        # Reuse the classification CNN body.
        from lacewing.classification.models import cnn1d
        self.cnn_features = cnn1d.CNN1D(input_len).features
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            n_flat = self.cnn_features(dummy).shape[1]
        self.cnn_proj = nn.Linear(n_flat, d_branch)

        # BiGRU branch.
        self.gru = nn.GRU(
            input_size=1, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        gru_out_dim = 2 * hidden
        self.gru_proj = nn.Linear(gru_out_dim, d_branch)

        # Per-timestep fusion head.
        self.head = nn.Sequential(
            nn.Linear(2 * d_branch, d_fusion), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fusion, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        # CNN branch: (B, d_branch) → broadcast to (B, T, d_branch).
        cnn_feat = self.cnn_proj(self.cnn_features(x))  # (B, d_branch)
        cnn_bcast = cnn_feat.unsqueeze(1).expand(b, t, cnn_feat.size(-1))

        # GRU branch: per-timestep.
        h, _ = self.gru(x.transpose(1, 2))  # (B, T, 2*hidden)
        gru_feat = self.gru_proj(h)         # (B, T, d_branch)

        # Fuse per timestep.
        fused = torch.cat([cnn_bcast, gru_feat], dim=-1)  # (B, T, 2*d_branch)
        return self.head(fused).squeeze(-1)  # (B, T)


# =====================================================================
# Transformer-patch per-timestep
# =====================================================================

class TransformerPatchPerTimestep(nn.Module):
    """PatchTST-style transformer with per-timestep output.

    The encoder produces one embedding per patch (T/P patches).
    We broadcast each patch embedding to every timestep it covers
    via `repeat_interleave(P)`, then apply a per-timestep linear head.
    """
    def __init__(
        self,
        input_len: int = 450,
        patch_size: int = 15,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_ff: int = 128,
        dropout: float = 0.1,
    ):
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
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        # (B, 1, T) → (B, T/P, P)
        h = x.view(b, c, t // self.patch_size, self.patch_size).squeeze(1)
        h = self.embed(h)               # (B, T/P, d_model)
        h = self.pos(h)
        h = self.encoder(h)             # (B, T/P, d_model)
        # Broadcast each patch embedding to every timestep it covers.
        h = h.repeat_interleave(self.patch_size, dim=1)  # (B, T, d_model)
        return self.head(h).squeeze(-1)  # (B, T)


# =====================================================================
# 1D U-Net + TCN (already per-timestep)
# =====================================================================

class UNet1DPerTimestep(nn.Module):
    """1D U-Net wrapper with n_classes=1 output → (B, T) logits."""
    def __init__(self, in_channels: int = 1, base_channels: int = 8):
        super().__init__()
        from lacewing.classification.models.unet_1d import UNet1D
        self.unet = UNet1D(in_channels=in_channels, n_classes=1, base_channels=base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.unet(x)  # (B, 1, T)
        return h.squeeze(1)


class TCNPerTimestepAdapter(nn.Module):
    """TCNPerTimestep wrapper with n_classes=1 output → (B, T) logits."""
    def __init__(self, input_size: int = 1, num_channels: list[int] | None = None,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        from lacewing.classification.models.tcn import TCNPerTimestep
        if num_channels is None:
            num_channels = [32, 32, 64, 64]
        self.tcn = TCNPerTimestep(
            input_size=input_size, num_channels=num_channels,
            kernel_size=kernel_size, dropout=dropout, n_classes=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x)  # (B, 1, T)
        return h.squeeze(1)


# =====================================================================
# Factory
# =====================================================================

# Registry: (backbone_id → constructor)
BACKBONES = {
    "bigru":              lambda: BiGRUPerTimestepP2(hidden=64, layers=2, dropout=0.1),
    "gru":                lambda: GRUUniPerTimestep(hidden=64, layers=2, dropout=0.1),
    "ann":                lambda: ANNPerTimestep(hidden_dims=(16, 8)),
    "cnn_gru_par":        lambda: CNNGRUParPerTimestep(input_len=450, hidden=64, layers=2, dropout=0.1),
    "transformer_patch":  lambda: TransformerPatchPerTimestep(input_len=450, patch_size=15, d_model=64, n_layers=3, dropout=0.1),
    "unet":               lambda: UNet1DPerTimestep(base_channels=8),
    "tcn":                lambda: TCNPerTimestepAdapter(kernel_size=3, dropout=0.1),
}


def build_p2_backbone(backbone_id: str) -> nn.Module:
    """Build a per-timestep model from a backbone id.  Signature: (B, 1, T) → (B, T) logits."""
    if backbone_id not in BACKBONES:
        raise ValueError(f"Unknown backbone {backbone_id!r}.  Valid: {list(BACKBONES)}")
    return BACKBONES[backbone_id]()
