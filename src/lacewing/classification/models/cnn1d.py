"""Paper 3 Model 2: 1D-DCNN.

  Input: (batch, 1, T)
  Conv1D(64, k=3, relu) -> MaxPool(2) ->
  Conv1D(32, k=3, relu) -> MaxPool(2) ->
  Flatten -> Dense(N_dense, relu) -> Dense(1, sigmoid)

  Paper params ~77,513.  N_dense tuned to land near that count.

  T=450 -> after conv/pool: floor((floor((450-2)/2) - 2) / 2) = 111
  Flatten = 111 * 32 = 3552. With Dense(20) we get
    3552*20 + 20 + 20*1 + 1 = 71,061  (close to 77.5k)
  N_dense=22 gives 78,167 - within 1% of paper. Use 22.
"""
from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, input_len: int, n_dense: int = 22):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            n_flat = self.features(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Linear(n_flat, n_dense), nn.ReLU(),
            nn.Linear(n_dense, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


def build(input_shape) -> nn.Module:
    return CNN1D(int(input_shape[-1]))
