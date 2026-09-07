"""CNN + GRU, parallel two-stream fusion.

  Input: (batch, 1, T) (same preprocessed signal fed to both branches)
    CNN branch: existing CNN1D body (Conv1D 1->64 k=3, MaxPool2,
                Conv1D 64->32 k=3, MaxPool2, Flatten) -> Linear -> d
    GRU branch: bidirectional GRU (hidden=64, layers=2)
                -> mean-pool over time -> Linear -> d
  Concat [cnn_feat ; gru_feat] -> FC layers -> logit

Mirrors cnn_transformer_parallel: same CNN branch, same fusion head,
but with a recurrent branch in place of attention.
"""
from __future__ import annotations

import torch
from torch import nn

from . import cnn1d


class CNNGRUParallel(nn.Module):
    def __init__(
        self,
        input_len: int,
        d_branch: int = 64,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        d_fusion: int = 64,
        bidirectional: bool = True,
    ):
        super().__init__()
        # CNN branch: reuse the existing CNN1D feature extractor.
        self.cnn_features = cnn1d.CNN1D(input_len).features
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            n_flat = self.cnn_features(dummy).shape[1]
        self.cnn_proj = nn.Linear(n_flat, d_branch)

        # GRU branch.
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        gru_out_dim = hidden_size * (2 if bidirectional else 1)
        self.gru_proj = nn.Linear(gru_out_dim, d_branch)

        # Fusion head.
        self.head = nn.Sequential(
            nn.Linear(2 * d_branch, d_fusion), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fusion, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn_proj(self.cnn_features(x))

        h, _ = self.gru(x.transpose(1, 2))   # (B, T, 2*H)
        h = h.mean(dim=1)                    # (B, 2*H)
        gru_feat = self.gru_proj(h)

        fused = torch.cat([cnn_feat, gru_feat], dim=1)
        return self.head(fused).squeeze(-1)


def build(input_shape) -> nn.Module:
    return CNNGRUParallel(int(input_shape[-1]))
