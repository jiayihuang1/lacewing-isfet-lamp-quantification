"""Patch-based time-series transformer (PatchTST-style).

  Input: (batch, 1, T)
  Split into non-overlapping patches of length P (default 15).
    -> sequence of (T/P) patches, each P samples
    -> Linear-embed each patch -> (batch, T/P, d_model)
    + sinusoidal positional encoding
    -> N encoder layers
    -> mean-pool
    -> Linear -> logit

For T=450, P=15 gives a sequence of 30 patches, so attention is
30x30 per head instead of 450x450 (225x faster than the vanilla
transformer).
"""
from __future__ import annotations

import torch
from torch import nn

from ._positional import SinusoidalPositionalEncoding


class PatchTransformer(nn.Module):
    def __init__(
        self,
        input_len: int,
        patch_size: int = 15,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_len % patch_size != 0:
            raise ValueError(
                f"input_len={input_len} not divisible by patch_size={patch_size}"
            )
        self.patch_size = patch_size
        n_patches = input_len // patch_size
        self.embed = nn.Linear(patch_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=n_patches)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, T) -> (B, T/P, P) -> (B, T/P, d_model)
        b, c, t = x.shape
        h = x.view(b, c, t // self.patch_size, self.patch_size).squeeze(1)
        h = self.embed(h)
        h = self.pos(h)
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h).squeeze(-1)


def build(input_shape) -> nn.Module:
    return PatchTransformer(int(input_shape[-1]))
