"""Paper 3 Model 1: Feed-forward ANN.

  Input: (batch, 1, T)  - flattened to (batch, T)
  Dense(64, relu) -> Dense(32, relu) -> Dense(16, relu) -> Dense(1, sigmoid)

  Paper params: ~9,937; paper acc: 57.58%
"""
from __future__ import annotations

import torch
from torch import nn


class ANN(nn.Module):
    def __init__(self, input_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_len, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build(input_shape) -> nn.Module:
    # input_shape from data is (T,) since DataLoader prepends batch + channel
    input_len = int(input_shape[-1])
    return ANN(input_len)
