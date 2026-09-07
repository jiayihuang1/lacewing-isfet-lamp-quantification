"""CNN -> GRU, sequential stem-then-recurrence.

  Input: (batch, 1, T)
    CNN stem: existing CNN1D body (Conv1D 1->64 k=3, MaxPool2,
              Conv1D 64->32 k=3, MaxPool2) -- WITHOUT the Flatten.
              Output: (batch, 32, T') where T' is the shortened
              sequence length (T=450 -> T'=111).
    -> 1x1 Conv to widen channels to gru_input_size (default 64)
    -> permute to (batch, T', gru_input_size)
    -> bidirectional GRU (hidden=64, layers=2)
    -> mean-pool over time
    -> Linear -> logit

Mirrors cnn_transformer_sequential: same conv stem, same 1x1 projection,
mean-pool head — recurrent encoder in place of attention.  The CNN
front-end shortens the sequence the GRU iterates over (450 -> 111),
which both speeds training and gives the recurrence locally-aggregated
features instead of raw single-sample inputs.
"""
from __future__ import annotations

import torch
from torch import nn

from . import cnn1d


class CNNGRUSequential(nn.Module):
    def __init__(
        self,
        input_len: int,
        gru_input_size: int = 64,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        # CNN stem: reuse the existing CNN1D body but stop before Flatten.
        cnn = cnn1d.CNN1D(input_len)
        layers = list(cnn.features.children())
        if isinstance(layers[-1], nn.Flatten):
            layers = layers[:-1]
        self.stem = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            _, c_out, _ = self.stem(dummy).shape

        # 1x1 conv projects stem channels (32) up to gru_input_size.
        self.project = nn.Conv1d(c_out, gru_input_size, kernel_size=1)

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)                  # (B, 32, T')
        h = self.project(h)               # (B, gru_input_size, T')
        h = h.transpose(1, 2)             # (B, T', gru_input_size)
        h, _ = self.gru(h)                # (B, T', 2*hidden_size)
        h = h.mean(dim=1)                 # (B, 2*hidden_size)
        return self.head(h).squeeze(-1)


def build(input_shape) -> nn.Module:
    return CNNGRUSequential(int(input_shape[-1]))
