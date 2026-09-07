"""BiGRU wrapper around the existing GRU backbone, with per-timestep binary head.

Grounded in Li et al. 2024 (paper 18) — a bidirectional NN classifier
is a strict generalisation of CUSUM (Lemma 3.1), applied to the whole
trace so the per-timestep decision has full-trace context.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BiGRUPerTimestep(nn.Module):
    def __init__(self, input_size: int = 1, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T) → (B, T) per-timestep logit."""
        h, _ = self.rnn(x.transpose(1, 2))  # (B, T, 2*hidden)
        return self.head(h).squeeze(-1)     # (B, T)
