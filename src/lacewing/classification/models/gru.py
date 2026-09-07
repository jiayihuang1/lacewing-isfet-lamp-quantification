"""Standalone bidirectional GRU.

  Input: (batch, 1, T)
    -> permute to (batch, T, 1)
    -> bidirectional GRU (hidden=64, layers=2)
    -> mean-pool over time
    -> Linear -> logit

Mirrors the standalone transformer in shape (hidden 64, 2 layers,
mean-pool head) so the GRU vs transformer comparison is apples-to-
apples on this dataset.
"""
from __future__ import annotations

import torch
from torch import nn


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_len: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x.transpose(1, 2))   # (B, T, 2*H)
        h = h.mean(dim=1)                    # (B, 2*H)
        return self.head(h).squeeze(-1)


def build(input_shape) -> nn.Module:
    return GRUClassifier(int(input_shape[-1]))
