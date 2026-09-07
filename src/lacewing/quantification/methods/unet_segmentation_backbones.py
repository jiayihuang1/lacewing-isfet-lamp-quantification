"""Backbone factory for F-D 4-class segmentation.

Extends the existing P2 backbone registry so per-timestep multi-class output
(B, 1, T) → (B, n_classes, T) is available for U-Net, GRU, CNN-GRU-par.

Each returned module:
    forward(x: (B, 1, T)) -> (B, n_classes, T)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lacewing.quantification.methods.unet_segmentation import UNet1D


class GRUPerTimestepMultiClass(nn.Module):
    """Unidirectional GRU with per-timestep n_class linear head."""
    def __init__(self, n_classes: int = 4, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=1, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=False,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T) → (B, T, 1)
        h, _ = self.rnn(x.transpose(1, 2))       # (B, T, hidden)
        logits = self.head(h)                     # (B, T, n_classes)
        return logits.transpose(1, 2)             # (B, n_classes, T)


class CNNGRUParPerTimestepMultiClass(nn.Module):
    """CNN-GRU-par with per-timestep n_class output.

    CNN branch: stack of Conv1d + BN + ReLU → global pool → feature vector.
    GRU branch: per-timestep hidden state.
    Fuse: broadcast CNN vector to every timestep + concat with GRU hidden.
    Head: Linear(cnn_feat + gru_hidden, n_classes) per timestep.
    """
    def __init__(
        self,
        n_classes: int = 4,
        cnn_channels: tuple[int, ...] = (16, 32, 64),
        gru_hidden: int = 64,
        gru_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = 1
        for ch in cnn_channels:
            layers += [nn.Conv1d(prev, ch, kernel_size=5, padding=2), nn.BatchNorm1d(ch), nn.ReLU()]
            prev = ch
        self.cnn = nn.Sequential(*layers)
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)   # (B, cnn_channels[-1], 1)
        self.gru = nn.GRU(
            input_size=1, hidden_size=gru_hidden, num_layers=gru_layers,
            batch_first=True, bidirectional=False,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Linear(cnn_channels[-1] + gru_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T = x.shape
        cnn_feat = self.cnn_pool(self.cnn(x)).squeeze(-1)  # (B, cnn_channels[-1])
        gru_out, _ = self.gru(x.transpose(1, 2))            # (B, T, gru_hidden)
        cnn_broadcast = cnn_feat.unsqueeze(1).expand(B, T, cnn_feat.size(-1))
        fused = torch.cat([cnn_broadcast, gru_out], dim=-1)  # (B, T, cnn+gru)
        logits = self.head(fused)                             # (B, T, n_classes)
        return logits.transpose(1, 2)                         # (B, n_classes, T)


def build_fd_backbone(backbone_id: str, n_classes: int = 4) -> nn.Module:
    """Return a per-timestep n_class segmentation model.

    Supported backbones:
        unet          — the existing 1D U-Net (base_channels=8, ~500k params)
        gru           — unidirectional GRU + per-timestep n_class head
        cnn_gru_par   — parallel CNN + GRU with fused per-timestep head
    """
    if backbone_id == "unet":
        return UNet1D(in_channels=1, n_classes=n_classes, base_channels=8)
    if backbone_id == "gru":
        return GRUPerTimestepMultiClass(n_classes=n_classes)
    if backbone_id == "cnn_gru_par":
        return CNNGRUParPerTimestepMultiClass(n_classes=n_classes)
    raise ValueError(f"unknown backbone: {backbone_id!r}")
