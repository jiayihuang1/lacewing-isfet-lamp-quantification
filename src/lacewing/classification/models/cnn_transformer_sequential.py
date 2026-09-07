"""CNN -> Transformer, sequential stem-then-attention.

  Input: (batch, 1, T)
    CNN stem: existing CNN1D body (Conv1D 1->64 k=3, MaxPool2,
              Conv1D 64->32 k=3, MaxPool2) -- WITHOUT the Flatten.
              Output: (batch, 32, T') where T' is the shortened
              sequence length (T=450 -> T'=111).
    -> 1x1 Conv to widen channels to d_model (default 64)
    -> permute to (batch, T', d_model)
    -> sinusoidal positional encoding
    -> 3 transformer encoder layers
    -> mean-pool over time
    -> Linear -> logit

The CNN front-end shortens the sequence the transformer attends
over (450 -> 111), so attention is ~16x cheaper than the vanilla
transformer.  The 1x1 projection brings the channel count up to
d_model so the transformer width matches the other variants.
This is the standard 'conv stem + transformer encoder' pattern
(Conformer, ConvViT, ConvFormer for time series).
"""
from __future__ import annotations

import torch
from torch import nn

from . import cnn1d
from ._positional import SinusoidalPositionalEncoding


class CNNTransformerSequential(nn.Module):
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
        # CNN stem: reuse the existing CNN1D body but stop before Flatten.
        cnn = cnn1d.CNN1D(input_len)
        # cnn.features = [Conv, ReLU, Pool, Conv, ReLU, Pool, Flatten]
        # Drop the trailing Flatten so we keep (B, C, T').
        layers = list(cnn.features.children())
        if isinstance(layers[-1], nn.Flatten):
            layers = layers[:-1]
        self.stem = nn.Sequential(*layers)

        # Discover the post-stem channel + sequence dims.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            _, c_out, t_out = self.stem(dummy).shape

        # 1x1 conv projects stem channels (32) up to d_model.
        self.project = nn.Conv1d(c_out, d_model, kernel_size=1)

        self.pos = SinusoidalPositionalEncoding(d_model, max_len=t_out)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)                  # (B, 32, T')
        h = self.project(h)               # (B, d_model, T')
        h = h.transpose(1, 2)             # (B, T', d_model)
        h = self.pos(h)
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h).squeeze(-1)


def build(input_shape) -> nn.Module:
    return CNNTransformerSequential(int(input_shape[-1]))
