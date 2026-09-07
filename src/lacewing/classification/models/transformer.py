"""Vanilla time-series transformer.

  Input: (batch, 1, T)
  Linear-embed each sample -> (batch, T, d_model)
  + sinusoidal positional encoding
  -> N encoder layers (multi-head self-attention)
  -> mean-pool over time
  -> Linear -> logit

Faithful to Vaswani 2017 encoder, with the standard
"linear-embed each sample" trick used by time-series transformers
(Zerveas 2021, Mao 2023 T-CDAN) since there is no discrete vocabulary.
"""
from __future__ import annotations

import torch
from torch import nn

from ._positional import SinusoidalPositionalEncoding


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=input_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T) -> (B, T, 1) -> (B, T, d_model)
        h = self.embed(x.transpose(1, 2))
        h = self.pos(h)
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h).squeeze(-1)


def build(input_shape) -> nn.Module:
    return TimeSeriesTransformer(int(input_shape[-1]))
