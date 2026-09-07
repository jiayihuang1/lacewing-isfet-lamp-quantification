"""Paper 3 Model 7 (HEADLINE, 84.84%): 2D-DCNN on STFT spectrogram.

  Input: (batch, 1, 10, 39)
  Conv2D(N1, k=3) -> ReLU -> MaxPool(2,2)
  Conv2D(N2, k=3) -> ReLU -> MaxPool(2,2)
  Flatten -> Dense(N_dense, relu) -> Dense(1, sigmoid)

  Paper params ~24,265.

  Sizes after each layer (input 10x39):
    Conv N1=16, k=3 -> (16, 8, 37)
    Pool 2x2        -> (16, 4, 18)
    Conv N2=32, k=3 -> (32, 2, 16)
    Pool 2x2        -> (32, 1, 8)
    Flatten         -> 256
  Dense(N_dense=64) -> (16*1*9*3 + 16) + (16*32*9 + 32) + (256*64+64) +
                       (64*1+1) = 144 + 4640 + 16448 + 65 = 21,297
  N_dense=80 lands at ~25,800. Use 78 -> 25,150 ~ paper's 24,265. Use 75
  -> 24,367 - closest. Final choice: N1=16, N2=32, N_dense=75.
"""
from __future__ import annotations

import torch
from torch import nn


class CNN2DSpectrogram(nn.Module):
    def __init__(self, n1: int = 16, n2: int = 32, n_dense: int = 75):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, n1, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(n1, n2, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 10, 39)
            n_flat = self.features(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Linear(n_flat, n_dense), nn.ReLU(),
            nn.Linear(n_dense, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


def build(input_shape) -> nn.Module:
    return CNN2DSpectrogram()
