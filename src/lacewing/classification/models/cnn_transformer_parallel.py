"""CNN + Transformer, parallel two-stream fusion.

  Input: (batch, 1, T) (same preprocessed signal fed to both branches)
    CNN branch: existing CNN1D body (Conv1D 1->64 k=3, MaxPool2,
                Conv1D 64->32 k=3, MaxPool2, Flatten) -> Linear -> d
    Transformer branch: linear-embed each sample -> sinusoidal PE ->
                3 encoder layers -> mean-pool -> Linear -> d
  Concat [cnn_feat ; xformer_feat] -> FC layers -> logit

Both branches see the same raw input. The CNN branch captures
local pattern features; the transformer branch captures global
dependencies; the FC head learns how to combine them.
"""
from __future__ import annotations

import torch
from torch import nn

from . import cnn1d
from ._positional import SinusoidalPositionalEncoding


class CNNTransformerParallel(nn.Module):
    def __init__(
        self,
        input_len: int,
        d_branch: int = 64,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_ff: int = 128,
        dropout: float = 0.1,
        d_fusion: int = 64,
    ):
        super().__init__()
        # CNN branch: reuse the existing CNN1D feature extractor.
        self.cnn_features = cnn1d.CNN1D(input_len).features
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            n_flat = self.cnn_features(dummy).shape[1]
        self.cnn_proj = nn.Linear(n_flat, d_branch)

        # Transformer branch: same shape as the standalone transformer.
        self.t_embed = nn.Linear(1, d_model)
        self.t_pos = SinusoidalPositionalEncoding(d_model, max_len=input_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.t_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.t_proj = nn.Linear(d_model, d_branch)

        # Fusion head.
        self.head = nn.Sequential(
            nn.Linear(2 * d_branch, d_fusion), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fusion, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn_proj(self.cnn_features(x))

        h = self.t_embed(x.transpose(1, 2))
        h = self.t_pos(h)
        h = self.t_encoder(h).mean(dim=1)
        t_feat = self.t_proj(h)

        fused = torch.cat([cnn_feat, t_feat], dim=1)
        return self.head(fused).squeeze(-1)


def build(input_shape) -> nn.Module:
    return CNNTransformerParallel(int(input_shape[-1]))
