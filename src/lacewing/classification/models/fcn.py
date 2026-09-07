"""Paper 3 Model 3: FCN (Wang et al. 2017, time-series classification).

  Conv1D(128, k=8, padding=same, BN, relu)
  Conv1D(256, k=5, padding=same, BN, relu)
  Conv1D(128, k=3, padding=same, BN, relu)
  GlobalAveragePooling1D -> Dense(1, sigmoid)

  No max-pool, no dense (other than the head).
"""
from __future__ import annotations

import torch
from torch import nn


def _conv_bn_relu(in_c: int, out_c: int, k: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, kernel_size=k, padding="same"),
        nn.BatchNorm1d(out_c), nn.ReLU(),
    )


class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            _conv_bn_relu(1, 128, 8),
            _conv_bn_relu(128, 256, 5),
            _conv_bn_relu(256, 128, 3),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        h = h.mean(dim=-1)               # global average pool
        return self.head(h).squeeze(-1)


def build(input_shape) -> nn.Module:
    return FCN()
